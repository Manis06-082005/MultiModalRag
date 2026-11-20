"""PDF and image upload and background-ingestion endpoints."""

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
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


@router.get("/{job_id}", response_model=IngestionJob)
def get_ingestion_status(job_id: str) -> IngestionJob:
    """Return the current status for an uploaded PDF's ingestion job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found.")
    return job
