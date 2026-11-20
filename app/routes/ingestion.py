"""PDF and image upload and background-ingestion endpoints."""

import shutil
import mimetypes
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Literal
from urllib.parse import quote
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from HandwrittenNotesRAG.logger import Logger

BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "data" / "raw" / "pdfs"
IMAGE_UPLOAD_DIR = BASE_DIR / "data" / "raw" / "images"
STAGING_DIR = BASE_DIR / "storage" / "ingestion_jobs"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

router = APIRouter(prefix="/api/documents", tags=["documents"])
_jobs: dict[str, "IngestionJob"] = {}
_jobs_lock = Lock()
_ingestion_lock = Lock()


class IngestionJob(BaseModel):
    job_id: str
    filename: str
    status: Literal["queued", "processing", "completed", "failed"]
    result: dict[str, int] | None = None
    error: str | None = None


class DocumentInfo(BaseModel):
    filename: str
    kind: Literal["pdf", "image"]
    status: Literal["queued", "processing", "indexed", "failed"]
    preview_url: str


async def _save_upload(upload: UploadFile, destination: Path, is_pdf: bool) -> None:
    header = await upload.read(5 if is_pdf else 0)
    if is_pdf and header != b"%PDF-":
        raise HTTPException(status_code=415, detail="The uploaded file is not a valid PDF.")

    size = len(header)
    temporary = destination.with_name(f".{destination.stem}-{uuid4().hex}.upload{destination.suffix}")
    try:
        with temporary.open("xb") as output:
            output.write(header)
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Files must be 25 MB or smaller.",
                    )
                output.write(chunk)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if not is_pdf:
        try:
            with Image.open(temporary) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as error:
            temporary.unlink(missing_ok=True)
            raise HTTPException(status_code=415, detail="The uploaded file is not a valid image.") from error
    temporary.replace(destination)


def _update_job(job_id: str, **updates: object) -> None:
    with _jobs_lock:
        current = _jobs[job_id]
        _jobs[job_id] = current.model_copy(update=updates)


def _ingest_upload(job_id: str, stored_file: Path) -> None:
    _update_job(job_id, status="processing")
    try:
        from HandwrittenNotesRAG.pipelines.ingestion_pipeline import IngestionPipeline

        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        with _ingestion_lock, TemporaryDirectory(
            prefix=f"{job_id}-",
            dir=STAGING_DIR,
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            pdf_dir = temporary_root / "pdfs"
            image_dir = temporary_root / "images"
            pdf_dir.mkdir()
            image_dir.mkdir()
            destination_dir = pdf_dir if stored_file.suffix.lower() == ".pdf" else image_dir
            shutil.copy2(stored_file, destination_dir / stored_file.name)

            result = IngestionPipeline().ingest(
                pdf_dir=pdf_dir,
                image_dir=image_dir,
            )

        if result["stored"] == 0:
            raise ValueError("No text could be extracted from this PDF.")
        _update_job(job_id, status="completed", result=result)
    except Exception:
        Logger.error("PDF ingestion failed for job %s", job_id, exc_info=True)
        _update_job(
            job_id,
            status="failed",
            error="Ingestion failed. Check the application logs for details.",
        )


@router.post("", response_model=IngestionJob, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> IngestionJob:
    """Save a PDF and queue its ingestion into the configured vector index."""
    raw_filename = file.filename or ""
    filename = Path(raw_filename.replace("\\", "/")).name
    suffix = Path(filename).suffix.lower()
    is_pdf = suffix == ".pdf"
    if not filename or filename in {".", ".."} or not (is_pdf or suffix in IMAGE_EXTENSIONS):
        raise HTTPException(status_code=415, detail="Upload a PDF or a supported image file.")

    upload_directory = UPLOAD_DIR if is_pdf else IMAGE_UPLOAD_DIR
    upload_directory.mkdir(parents=True, exist_ok=True)
    destination = upload_directory / filename
    try:
        await _save_upload(file, destination, is_pdf)
    finally:
        await file.close()

    job_id = uuid4().hex
    job = IngestionJob(job_id=job_id, filename=filename, status="queued")
    with _jobs_lock:
        _jobs[job_id] = job
    background_tasks.add_task(_ingest_upload, job_id, destination)
    return job


@router.get("", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    """List uploaded documents and their latest in-memory indexing status."""
    with _jobs_lock:
        jobs_by_name = {job.filename: job for job in _jobs.values()}
    documents: list[DocumentInfo] = []
    for directory, kind in ((UPLOAD_DIR, "pdf"), (IMAGE_UPLOAD_DIR, "image")):
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if not path.is_file() or path.name.startswith("."):
                continue
            if kind == "pdf" and path.suffix.lower() != ".pdf":
                continue
            if kind == "image" and path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            job = jobs_by_name.get(path.name)
            status = "indexed" if job is None or job.status == "completed" else job.status
            documents.append(DocumentInfo(
                filename=path.name,
                kind=kind,
                status=status,
                preview_url=f"/api/documents/{quote(path.name, safe='')}/preview",
            ))
    return sorted(documents, key=lambda document: document.filename.lower())


@router.get("/{filename}/preview")
def preview_document(filename: str) -> FileResponse:
    """Open an uploaded PDF or image in the browser's native viewer."""
    candidates = (UPLOAD_DIR / filename, IMAGE_UPLOAD_DIR / filename)
    path = next((candidate for candidate in candidates if candidate.is_file() and candidate.name == filename), None)
    if path is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name, content_disposition_type="inline")


def _delete_document_vectors(filename: str) -> None:
    """Remove a deleted upload's vectors without blocking the UI response."""
    try:
        from HandwrittenNotesRAG.components.pinecone_store import PineconeStore

        with _ingestion_lock:
            PineconeStore().delete_by_sources([filename])
        Logger.info("Removed search vectors for deleted document %s", filename)
    except Exception:
        Logger.error("Could not remove search vectors for deleted document %s", filename, exc_info=True)


@router.delete("/{filename}")
def delete_document(filename: str, background_tasks: BackgroundTasks) -> dict[str, str]:
    """Remove the uploaded file now, then clean its vectors in the background."""
    if Path(filename.replace("\\", "/")).name != filename or filename in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid document name.")
    candidates = [path for path in (UPLOAD_DIR / filename, IMAGE_UPLOAD_DIR / filename) if path.is_file()]
    if not candidates:
        raise HTTPException(status_code=404, detail="Document not found.")
    with _jobs_lock:
        if any(job.filename == filename and job.status in {"queued", "processing"} for job in _jobs.values()):
            raise HTTPException(status_code=409, detail="Wait for indexing to finish before deleting this document.")
    try:
        for path in candidates:
            path.unlink(missing_ok=True)
    except OSError as error:
        Logger.error("Could not remove uploaded document %s", filename, exc_info=True)
        raise HTTPException(status_code=500, detail="Could not remove the uploaded document.") from error
    background_tasks.add_task(_delete_document_vectors, filename)
    return {"status": "deleted", "filename": filename}


@router.get("/{job_id}", response_model=IngestionJob)
def get_ingestion_status(job_id: str) -> IngestionJob:
    """Return the current status for an uploaded PDF's ingestion job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found.")
    return job
