"""Select an LLM provider from request modality using a small LangGraph."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class LLMRouteState(TypedDict, total=False):
    has_image: bool
    provider: str


def _route_to_groq(state: LLMRouteState) -> dict[str, str]:
    """Mark text-only requests for the Groq model path."""
    return {"provider": "groq"}


def _route_to_openrouter(state: LLMRouteState) -> dict[str, str]:
    """Mark image requests for the OpenRouter vision path."""
    return {"provider": "openrouter"}


def _build_route_graph():
    graph = StateGraph(LLMRouteState)
    graph.add_node("groq", _route_to_groq)
    graph.add_node("openrouter", _route_to_openrouter)
    graph.add_conditional_edges(
        START,
        lambda state: "openrouter" if state.get("has_image") else "groq",
        {"openrouter": "openrouter", "groq": "groq"},
    )
    graph.add_edge("groq", END)
    graph.add_edge("openrouter", END)
    return graph.compile()


class LLMRouter:
    """Route image requests to OpenRouter and text requests to Groq."""

    def __init__(self) -> None:
        self.graph = _build_route_graph()

    def select(self, *, has_image: bool) -> str:
        result = self.graph.invoke({"has_image": has_image})
        return str(result["provider"])
