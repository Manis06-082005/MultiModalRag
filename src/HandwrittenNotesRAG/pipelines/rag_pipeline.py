"""Retrieve note passages and generate a source-grounded answer."""

from typing import Any

from ..components.generator import Generator
from ..components.retriever import Retriever
from .retrieval_pipeline import RetrievalPipeline


class RAGPipeline:
    """Run hybrid retrieval followed by Groq answer generation."""

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline | None = None,
        generator: Generator | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        if retrieval_pipeline is not None and retriever is not None:
            raise ValueError("Pass retrieval_pipeline or retriever, not both")
        self.retrieval_pipeline = retrieval_pipeline
        self.retriever = retriever
        self.generator = generator or Generator()

    def _retrieve(self, question: str) -> list[dict[str, Any]]:
        if self.retrieval_pipeline is not None:
            return self.retrieval_pipeline.retrieve(question)
        if self.retriever is not None:
            return self.retriever.search(question)
        self.retrieval_pipeline = RetrievalPipeline()
        return self.retrieval_pipeline.retrieve(question)

    def ask(self, question: str) -> dict[str, Any]:
        """Return the generated answer alongside the passages used as sources."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        passages = self._retrieve(question.strip())
        answer = self.generator.generate(question.strip(), passages)
        return {"answer": answer, "sources": passages}

    def run(self, question: str) -> dict[str, Any]:
        """Alias for :meth:`ask`."""
        return self.ask(question)
