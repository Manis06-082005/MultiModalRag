"""Retrieve relevant note chunks for a query."""

from ..components.embedding import EmbeddingGenerator
from ..components.pinecone_store import PineconeStore
from ..components.retriever import Retriever


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

    def retrieve(self, query: str) -> list[dict]:
        """Return ranked text passages and their source metadata."""
        if not isinstance(query, str) or not query.strip():
            return []
        return self.retriever.search(query.strip())

    def run(self, query: str) -> list[dict]:
        """Alias for :meth:`retrieve`."""
        return self.retrieve(query)
