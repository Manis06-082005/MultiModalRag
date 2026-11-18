"""Command-line entry point for note ingestion."""

import sys
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(BASE_DIR / ".env")

from HandwrittenNotesRAG.pipelines.ingestion_pipeline import IngestionPipeline  # noqa: E402


def main() -> None:
    """Ingest PDFs and images found in the project's raw data folders."""
    pdf_dir = BASE_DIR / "data" / "raw" / "pdfs"
    image_dir = BASE_DIR / "data" / "raw" / "images"

    pipeline = IngestionPipeline()
    result = pipeline.ingest(pdf_dir=pdf_dir, image_dir=image_dir)
    if result["files"] == 0:
        print("No PDF or image files found.")
        return
    if result["chunks"] == 0:
        print("No text extracted from the files.")
        return
    print(
        f"Ingestion complete: {result['files']} files, "
        f"{result['documents']} text documents, {result['stored']} chunks stored."
    )


if __name__ == "__main__":
    main()
