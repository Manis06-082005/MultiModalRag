from HandwrittenNotesRAG.rag.router import route_after_retrieval
from HandwrittenNotesRAG.rag.state import RAGState


def test_invalid_question_routes_to_error_handler() -> None:
    assert route_after_retrieval({"error": "question is required"}) == "invalid_question"


def test_empty_results_route_to_no_results_handler() -> None:
    assert route_after_retrieval({"documents": []}) == "no_results"


def test_documents_route_to_generation() -> None:
    state: RAGState = {"documents": [{"text": "A recurrence relation"}]}

    assert route_after_retrieval(state) == "generate"
