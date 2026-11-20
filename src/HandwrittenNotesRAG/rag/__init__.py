"""LangGraph-based retrieval-augmented generation workflow."""

__all__ = ["RAGGraph", "RAGState", "build_graph"]


def __getattr__(name: str):
    """Load graph exports on demand to keep model-routing imports acyclic."""
    if name in {"RAGGraph", "build_graph"}:
        from .graph import RAGGraph, build_graph

        return {"RAGGraph": RAGGraph, "build_graph": build_graph}[name]
    if name == "RAGState":
        from .state import RAGState

        return RAGState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
