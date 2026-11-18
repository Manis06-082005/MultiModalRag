"""Typed state passed between nodes in the note question-answering graph."""

from typing import Any, TypedDict


class RAGState(TypedDict, total=False):
    """State keys produced and consumed by the LangGraph nodes."""

    question: str
    documents: list[dict[str, Any]]
    answer: str
    error: str
