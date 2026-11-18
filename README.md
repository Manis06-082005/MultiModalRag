# MultiModalRag

A local retrieval-augmented question-answering project for handwritten notes and
other study documents. It extracts text from PDFs and images, stores searchable
chunks in Pinecone, retrieves relevant passages with dense and BM25 search, and
uses Groq to generate answers with source citations.

## Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- Pinecone API key and an index configured for 384-dimensional cosine vectors
- Groq API key

## Setup

Create the environment and install the locked dependencies:

```powershell
uv sync
```

Create a `.env` file in the project root with:

```dotenv
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=your-index-name
GROQ_API_KEY=your-groq-api-key
GENERATOR_MODEL=openai/gpt-oss-20b
```

Keep `.env` private; it is ignored by Git. The default generator model is
`openai/gpt-oss-20b`. Set `GENERATOR_MODEL` to another model available to your
Groq account if needed.

## Add and ingest notes

Put PDF files in `data/raw/pdfs/` and supported images (`.jpg`, `.jpeg`, `.png`,
`.bmp`, `.tiff`, or `.webp`) in `data/raw/images/`. Then run:

```powershell
uv run python main.py
```

PDF text is extracted page by page. Image-only PDF pages and image files use
OCR. Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` and
stored in the configured Pinecone index. Source filename and PDF page are
stored as metadata for citations.

## Ask a question from Python

The RAG pipeline returns an answer and the retrieved source passages:

```python
import sys
from pathlib import Path

from dotenv import load_dotenv

project_root = Path.cwd()
sys.path.insert(0, str(project_root / "src"))
load_dotenv(project_root / ".env")

from HandwrittenNotesRAG.pipelines.rag_pipeline import RAGPipeline

result = RAGPipeline().ask("What is a recurrence relation?")
print(result["answer"])
for source in result["sources"]:
    print(source.get("source"), source.get("page"), source.get("text"))
```

The LangGraph version is also available:

```python
from HandwrittenNotesRAG.rag.graph import RAGGraph

result = RAGGraph().ask("What is a recurrence relation?")
print(result["answer"])
print(result["sources"])
```

Both paths need a reachable Pinecone index and `GROQ_API_KEY`. The current
project provides Python pipeline interfaces; no HTTP query route is included.

## Project layout

```text
main.py                         CLI ingestion entry point
src/HandwrittenNotesRAG/
  components/                   PDF/image processing, OCR, embeddings, search, generation
  pipelines/                    ingestion, retrieval, and RAG orchestration
  rag/                          LangGraph state, nodes, routing, and graph
data/raw/pdfs/                  input PDFs
data/raw/images/                input images
```
