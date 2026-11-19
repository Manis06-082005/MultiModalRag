"""Conditional routing helpers for the RAG graph."""

from .state import RAGState


def route_after_retrieval(state: RAGState) -> str:
    """Select an error, no-evidence, or answer-generation path."""
    if state.get("error"):
        return "invalid_question"
    if not state.get("documents"):
        return "no_results"
    return "generate"
