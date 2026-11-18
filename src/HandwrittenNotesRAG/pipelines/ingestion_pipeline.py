"""Load, extract, chunk, embed, and store documents for retrieval."""

from pathlib import Path
from typing import Any

from ..components.chunking import TextChunker
from ..components.embedding import EmbeddingGenerator
from ..components.image_loader import ImageLoader
from ..components.image_preprocessing import ImagePreprocessor
from ..components.ocr_extractor import OCRExtractor
from ..components.pdf_loader import PDFLoader
from ..components.pinecone_store import PineconeStore
from ..components.text_cleaner import TextCleaner
from ..logger import Logger


class IngestionPipeline:
    """Ingest PDFs and images into the configured vector store."""

    def __init__(
        self,
        pdf_loader: PDFLoader | None = None,
        image_loader: ImageLoader | None = None,
        preprocessor: ImagePreprocessor | None = None,
        ocr: OCRExtractor | None = None,
        cleaner: TextCleaner | None = None,
        chunker: TextChunker | None = None,
        embedding_model: EmbeddingGenerator | None = None,
        vector_store: PineconeStore | None = None,
    ) -> None:
        self.pdf_loader = pdf_loader or PDFLoader()
        self.image_loader = image_loader or ImageLoader()
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.ocr = ocr or OCRExtractor()
        self.cleaner = cleaner or TextCleaner()
        self.chunker = chunker or TextChunker()
        self.embedding_model = embedding_model or EmbeddingGenerator()
        self.vector_store = vector_store

    def ingest(self, pdf_dir: str | Path, image_dir: str | Path) -> dict[str, int]:
        """Ingest files in the two input directories and return processing counts."""
        pdf_dir, image_dir = Path(pdf_dir), Path(image_dir)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        image_files = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.image_loader.SUPPORTED_EXTENSIONS
        )
        if not pdf_files and not image_files:
            Logger.info("No PDF or image files found to ingest")
            return {"files": 0, "documents": 0, "chunks": 0, "stored": 0}

        records: list[tuple[str, dict[str, Any]]] = []
        for pdf_path in pdf_files:
            Logger.info("Processing PDF: %s", pdf_path.name)
            for page in self.pdf_loader.load_pdf(pdf_path):
                text = page.get("text", "")
                if not text.strip() and page.get("image") is not None:
                    image = self.preprocessor.preprocess(page["image"])
                    text = self.ocr.extract_text(image)
                if text and text.strip():
                    records.append((text, {
                        "source": pdf_path.name,
                        "page": page.get("page_number"),
                    }))

        for image_path in image_files:
            Logger.info("Processing image: %s", image_path.name)
            image = self.preprocessor.preprocess(self.image_loader.load(image_path))
            text = self.ocr.extract_text(image)
            if text and text.strip():
                records.append((text, {"source": image_path.name}))

        chunks: list[str] = []
        metadata: list[dict[str, Any]] = []
        for text, source_metadata in records:
            cleaned = self.cleaner.clean_text(text)
            source_chunks = self.chunker.split_text(cleaned)
            chunks.extend(source_chunks)
            metadata.extend([source_metadata.copy() for _ in source_chunks])

        if not chunks:
            Logger.info("No text chunks were produced from the input files")
            return {"files": len(pdf_files) + len(image_files), "documents": len(records), "chunks": 0, "stored": 0}

        embeddings = self.embedding_model.embed_document(chunks)
        if len(embeddings) != len(chunks):
            raise ValueError("Embedding count does not match chunk count")

        store = self.vector_store or PineconeStore()
        store.store_embeddings(embeddings=embeddings, texts=chunks, metadata=metadata)
        Logger.info("Ingested %d chunks from %d source documents", len(chunks), len(records))
        return {
            "files": len(pdf_files) + len(image_files),
            "documents": len(records),
            "chunks": len(chunks),
            "stored": len(chunks),
        }
