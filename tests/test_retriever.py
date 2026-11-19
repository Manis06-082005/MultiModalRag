"""Integration test for real embeddings and the configured Pinecone index.

Run this test with the project environment active and valid Pinecone
credentials in the project-root .env file. It only searches; it does not write.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from HandwrittenNotesRAG.components.embedding import EmbeddingGenerator
from HandwrittenNotesRAG.components.pinecone_store import PineconeStore
from HandwrittenNotesRAG.components.retriever import Retriever


def test_real_retriever_finds_notes_about_recurrences() -> None:
    """Search Pinecone with a real embedding for a topic in the sample PDF."""
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    assert os.getenv("PINECONE_API_KEY"), "Set PINECONE_API_KEY in .env"
    assert os.getenv("PINECONE_INDEX_NAME"), "Set PINECONE_INDEX_NAME in .env"

    retriever = Retriever(
        embedding_model=EmbeddingGenerator(),
        vector_store=PineconeStore(),
        top_k=3,
    )
    results = retriever.search("What is a recurrence relation?")

    assert results, "Pinecone returned no matching note chunks"
    assert all(result.get("text") for result in results)

    print(f"Retrieved {len(results)} note chunks:")
    for result in results:
        source = result.get("source", "Unknown source")
        page = result.get("page", "")
        print(f"- {source} page {page}: {result['text'][:200]}")
