"""Application pipelines for ingestion, retrieval, and question answering."""

from typing import TYPE_CHECKING

__all__ = ["IngestionPipeline", "RetrievalPipeline", "RAGPipeline"]

if TYPE_CHECKING:
    from .ingestion_pipeline import IngestionPipeline
    from .rag_pipeline import RAGPipeline
    from .retrieval_pipeline import RetrievalPipeline


def __getattr__(name: str):
    if name == "IngestionPipeline":
        from .ingestion_pipeline import IngestionPipeline

        return IngestionPipeline
    if name == "RAGPipeline":
        from .rag_pipeline import RAGPipeline

        return RAGPipeline
    if name == "RetrievalPipeline":
        from .retrieval_pipeline import RetrievalPipeline

        return RetrievalPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
