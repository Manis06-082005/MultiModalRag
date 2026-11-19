"""Node functions used by the note retrieval and generation graph."""

from typing import Any

from ..components.generator import Generator
from ..pipelines.retrieval_pipeline import RetrievalPipeline
from .state import RAGState


class RAGNodes:
    """Provide retrieval and generation nodes with injectable dependencies."""

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.retrieval_pipeline = retrieval_pipeline
        self.generator = generator or Generator()

    def retrieve(self, state: RAGState) -> dict[str, Any]:
        question = state.get("question", "")
        if not isinstance(question, str) or not question.strip():
            return {"documents": [], "error": "question must be a non-empty string"}
        if self.retrieval_pipeline is None:
            self.retrieval_pipeline = RetrievalPipeline()
        documents = self.retrieval_pipeline.retrieve(question.strip())
        return {"documents": documents, "error": ""}

    @staticmethod
    def no_results(_: RAGState) -> dict[str, str]:
        """Return a grounded response when retrieval found no usable passages."""
        return {
            "answer": "I couldn't find relevant information in your notes to answer that."
        }

    @staticmethod
    def invalid_question(state: RAGState) -> dict[str, str]:
        """Expose invalid input as both a user-facing answer and an error."""
        message = state.get("error") or "question must be a non-empty string"
        return {"answer": message, "error": message}

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
