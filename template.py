
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Project Name
project_name = "HandwrittenNotesRAG"

# List of files and folders
list_of_files = [

    # GitHub
    ".github/workflows/.gitkeep",

    # Source Code
    f"src/{project_name}/__init__.py",

    # Components
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/image_loader.py",
    f"src/{project_name}/components/image_preprocessing.py",
    f"src/{project_name}/components/pdf_loader.py",
    f"src/{project_name}/components/ocr_extractor.py",
    f"src/{project_name}/components/vision_extractor.py",
    f"src/{project_name}/components/text_cleaner.py",
    f"src/{project_name}/components/chunking.py",
    f"src/{project_name}/components/embedding.py",
    f"src/{project_name}/components/pinecone_store.py",
    f"src/{project_name}/components/retriever.py",
    f"src/{project_name}/components/generator.py",

    # Multimodal RAG
    f"src/{project_name}/rag/__init__.py",
    f"src/{project_name}/rag/state.py",
    f"src/{project_name}/rag/nodes.py",
    f"src/{project_name}/rag/router.py",
    f"src/{project_name}/rag/graph.py",

    # Pipelines
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/ingestion_pipeline.py",
    f"src/{project_name}/pipelines/retrieval_pipeline.py",
    f"src/{project_name}/pipelines/rag_pipeline.py",

    # Utilities
    f"src/{project_name}/utils.py",
    f"src/{project_name}/Exception.py",
    f"src/{project_name}/logger.py",

    # Configuration
    "config/config.yaml",

    # Data
    "data/raw/images/.gitkeep",
    "data/raw/pdfs/.gitkeep",
    "data/processed/.gitkeep",

    # Original File Storage
    "storage/original_files/.gitkeep",

    # Evaluation
    "evaluation/datasets/.gitkeep",
    "evaluation/results/.gitkeep",

    # Notebooks
    "notebooks/ocr_experiments.ipynb",
    "notebooks/multimodal_rag.ipynb",

    # API
    "app/__init__.py",
    "app/app.py",
    "app/routes/__init__.py",
    "app/routes/ingestion.py",
    "app/routes/query.py",

    # Tests
    "tests/__init__.py",
    "tests/test_ocr.py",
    "tests/test_embeddings.py",
    "tests/test_retriever.py",

    # Project Files
    "main.py",
    "requirements.txt",
    ".env",
    ".env.example",
    ".gitignore",
    "README.md",
    "Dockerfile",
]

# Create folders and files
for filepath in list_of_files:

    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir}")

    if not filepath.exists():
        with open(filepath, "w", encoding="utf-8"):
            pass
        logging.info(f"Creating file: {filepath}")

    else:
        logging.info(f"{filepath} already exists")

logging.info("Project structure created successfully!")
