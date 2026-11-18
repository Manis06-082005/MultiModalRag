"""Conditional routing helpers for the RAG graph."""

from .state import RAGState


def route_after_retrieval(state: RAGState) -> str:
    """Stop generation when retrieval input is invalid; otherwise generate."""
    return "end" if state.get("error") else "generate"
