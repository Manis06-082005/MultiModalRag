
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from HandwrittenNotesRAG.components.pdf_loader import PDFLoader
from HandwrittenNotesRAG.components.image_loader import ImageLoader
from HandwrittenNotesRAG.components.image_preprocessing import ImagePreprocessor
from HandwrittenNotesRAG.components.ocr_extractor import OCRExtractor
from HandwrittenNotesRAG.components.text_cleaner import TextCleaner
from HandwrittenNotesRAG.components.chunking import TextChunker
from HandwrittenNotesRAG.components.embedding import EmbeddingGenerator
from HandwrittenNotesRAG.components.pinecone_store import PineconeStore


# Project paths
DATA_DIR = BASE_DIR / "data"

PDF_DIR = DATA_DIR / "raw" / "pdfs"
IMAGE_DIR = DATA_DIR / "raw" / "images"

load_dotenv(BASE_DIR / ".env")


def main():

    # Create data folders if they do not exist
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(PDF_DIR.glob("*.pdf"))
    image_files = [
        path for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in ImageLoader.SUPPORTED_EXTENSIONS
    ]

    if not pdf_files and not image_files:
        print("No PDF or image files found.")
        return

    # Initialize components
    pdf_loader = PDFLoader()
    image_loader = ImageLoader()
    preprocessor = ImagePreprocessor()
    ocr = OCRExtractor()
    cleaner = TextCleaner()
    chunker = TextChunker()
    embedding_model = EmbeddingGenerator()

    all_text = []

    # Process PDF files
    for pdf_path in pdf_files:

        print(f"Processing PDF: {pdf_path.name}")

        pages = pdf_loader.load_pdf(pdf_path)

        for page in pages:

            if page["text"]:
                extracted_text = page["text"]

            elif page["image"] is not None:
                processed_image = preprocessor.preprocess(
                    page["image"]
                )
                extracted_text = ocr.extract_text(processed_image)

            else:
                continue

            if extracted_text.strip():
                all_text.append(extracted_text)

    # Process image files
    for image_path in image_files:

        print(f"Processing image: {image_path.name}")

        image = image_loader.load(image_path)
        processed_image = preprocessor.preprocess(image)

        extracted_text = ocr.extract_text(processed_image)

        if extracted_text.strip():
            all_text.append(extracted_text)

    if not all_text:
        print("No text extracted from the files.")
        return

    # Clean extracted text
    cleaned_text = [
        cleaner.clean_text(text)
        for text in all_text
    ]

    # Split text into chunks
    all_chunks = []

    for text in cleaned_text:
        all_chunks.extend(chunker.split_text(text))

    print(f"Total chunks created: {len(all_chunks)}")

    if not all_chunks:
        print("No chunks available for embedding.")
        return

    # Generate embeddings
    embeddings = embedding_model.embed_document(all_chunks)

    print(f"Total embeddings generated: {len(embeddings)}")

    # Store embeddings in Pinecone
    pinecone_store = PineconeStore()

    pinecone_store.store_embeddings(
        embeddings=embeddings,
        texts=all_chunks
    )

    print("All files processed and stored successfully!")


if __name__ == "__main__":
    main()
