"""Retrieve relevant note chunks for a query."""

from ..components.embedding import EmbeddingGenerator
from ..components.pinecone_store import PineconeStore
from ..components.retriever import Retriever
from collections.abc import Sequence


class RetrievalPipeline:
    """Thin pipeline wrapper around the hybrid Retriever component."""

    def __init__(
        self,
        embedding_model: EmbeddingGenerator | None = None,
        vector_store: PineconeStore | None = None,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        self.embedding_model = embedding_model or EmbeddingGenerator()
        self.vector_store = vector_store or PineconeStore()
        self.retriever = Retriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            top_k=top_k,
            rrf_k=rrf_k,
        )

    def retrieve(self, query: str, selected_sources: Sequence[str] | None = None) -> list[dict]:
        """Return ranked text passages and their source metadata."""
        if not isinstance(query, str) or not query.strip():
            return []
        if selected_sources is None:
            return self.retriever.search(query.strip())
        return self.retriever.search(query.strip(), selected_sources=selected_sources)

    def run(self, query: str, selected_sources: Sequence[str] | None = None) -> list[dict]:
        """Alias for :meth:`retrieve`."""
        return self.retrieve(query, selected_sources=selected_sources)
