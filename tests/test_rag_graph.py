from collections.abc import Sequence
from typing import Any

from PIL import Image

from HandwrittenNotesRAG.components.generator import Generator
from HandwrittenNotesRAG.pipelines.retrieval_pipeline import RetrievalPipeline
from HandwrittenNotesRAG.rag.graph import RAGGraph


class FakeRetrievalPipeline(RetrievalPipeline):
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self.queries: list[str] = []

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        self.queries.append(query)
        return self.documents


class FakeGenerator(Generator):
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    def generate(
        self,
        question: str,
        documents: Sequence[dict[str, Any]],
        image: Image.Image | None = None,
        grounded: bool = True,
        web_search: bool = False,
        history: Sequence[dict[str, str]] = (),
    ) -> str:
        self.calls.append((question, list(documents)))
        return "Generated answer"


def test_invalid_question_returns_error_without_retrieval() -> None:
    retrieval = FakeRetrievalPipeline([])
    generator = FakeGenerator()

    result = RAGGraph(retrieval, generator).ask(" ")

    assert result["error"] == "question must be a non-empty string"
    assert retrieval.queries == []
    assert generator.calls == []


def test_no_results_returns_grounded_message_without_generation() -> None:
    retrieval = FakeRetrievalPipeline([])
    generator = FakeGenerator()

    result = RAGGraph(retrieval, generator).ask("What is a recurrence?")

    assert "couldn't find relevant information" in result["answer"]
    assert result["sources"] == []
    assert generator.calls == []


def test_retrieved_documents_are_passed_to_generator() -> None:
    documents = [{"text": "A recurrence uses earlier terms.", "page": 1}]
    retrieval = FakeRetrievalPipeline(documents)
    generator = FakeGenerator()

    result = RAGGraph(retrieval, generator).ask("What is a recurrence?")

    assert result["answer"] == "Generated answer"
    assert result["sources"] == documents
    assert generator.calls == [("What is a recurrence?", documents)]
