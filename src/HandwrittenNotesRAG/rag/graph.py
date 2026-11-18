"""Build and invoke the LangGraph note question-answering workflow."""

from langgraph.graph import END, START, StateGraph

from ..components.generator import Generator
from ..components.retriever import Retriever
from ..pipelines.retrieval_pipeline import RetrievalPipeline
from .nodes import RAGNodes
from .router import route_after_retrieval
from .state import RAGState


def build_graph(
    retriever: Retriever | None = None,
    retrieval_pipeline: RetrievalPipeline | None = None,
    generator: Generator | None = None,
):
    """Compile the retrieval -> generation graph with optional injected parts."""
    nodes = RAGNodes(
        retriever=retriever,
        retrieval_pipeline=retrieval_pipeline,
        generator=generator,
    )
    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("generate", nodes.generate)
    workflow.add_edge(START, "retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {"generate": "generate", "end": END},
    )
    workflow.add_edge("generate", END)
    return workflow.compile()


class RAGGraph:
    """Convenience wrapper exposing a simple ``ask`` method."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        retrieval_pipeline: RetrievalPipeline | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.graph = build_graph(
            retriever=retriever,
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
