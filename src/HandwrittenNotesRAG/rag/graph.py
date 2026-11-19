"""Build and invoke the LangGraph note question-answering workflow."""

from langgraph.graph import END, START, StateGraph

from ..components.generator import Generator
from ..pipelines.retrieval_pipeline import RetrievalPipeline
from .nodes import RAGNodes
from .router import route_after_retrieval
from .state import RAGState


def build_graph(
    retrieval_pipeline: RetrievalPipeline | None = None,
    generator: Generator | None = None,
):
    """Compile a graph that routes invalid, no-evidence, and answerable queries."""
    nodes = RAGNodes(
        retrieval_pipeline=retrieval_pipeline,
        generator=generator,
    )
    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("generate", nodes.generate)
    workflow.add_node("no_results", nodes.no_results)
    workflow.add_node("invalid_question", nodes.invalid_question)
    workflow.add_edge(START, "retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "generate": "generate",
            "no_results": "no_results",
            "invalid_question": "invalid_question",
        },
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("no_results", END)
    workflow.add_edge("invalid_question", END)
    return workflow.compile()


class RAGGraph:
    """Convenience wrapper exposing a simple ``ask`` method."""

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.graph = build_graph(
            retrieval_pipeline=retrieval_pipeline,
            generator=generator,
        )

    def ask(self, question: str) -> dict:
        """Execute the graph and return the answer and retrieved documents."""
        result = self.graph.invoke({"question": question})
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("documents", []),
            "error": result.get("error", ""),
        }

    def invoke(self, state: RAGState) -> RAGState:
        """Expose the compiled graph's state-based invocation API."""
        return self.graph.invoke(state)
