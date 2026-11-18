"""Node functions used by the note retrieval and generation graph."""

from typing import Any

from ..components.generator import Generator
from ..components.retriever import Retriever
from ..pipelines.retrieval_pipeline import RetrievalPipeline
from .state import RAGState


class RAGNodes:
    """Provide retrieval and generation nodes with injectable dependencies."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        retrieval_pipeline: RetrievalPipeline | None = None,
        generator: Generator | None = None,
    ) -> None:
        if retriever is not None and retrieval_pipeline is not None:
            raise ValueError("Pass retriever or retrieval_pipeline, not both")
        self.retriever = retriever
        self.retrieval_pipeline = retrieval_pipeline
        self.generator = generator or Generator()

    def retrieve(self, state: RAGState) -> dict[str, Any]:
        question = state.get("question", "")
        if not question.strip():
            return {"documents": [], "error": "question must be a non-empty string"}
        if self.retrieval_pipeline is not None:
            documents = self.retrieval_pipeline.retrieve(question)
        elif self.retriever is not None:
            documents = self.retriever.search(question)
        else:
            self.retrieval_pipeline = RetrievalPipeline()
            documents = self.retrieval_pipeline.retrieve(question)
        return {"documents": documents, "error": ""}

    def generate(self, state: RAGState) -> dict[str, str]:
        question = state.get("question", "")
        error = state.get("error", "")
        if error:
            return {"answer": error}
        if not question.strip():
            return {"answer": "question must be a non-empty string"}
        return {
            "answer": self.generator.generate(question, state.get("documents", []))
        }
