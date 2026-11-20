from collections.abc import Sequence


class Retriever:
    """Combine dense and BM25 result rankings with reciprocal rank fusion."""

    def __init__(
        self, embedding_model, vector_store, top_k: int = 5, rrf_k: int = 60
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k
        self.rrf_k = rrf_k

    def search(self, query: str, selected_sources: Sequence[str] | None = None) -> list[dict]:
        if not query or not query.strip():
            return []
        if selected_sources is not None and not selected_sources:
            return []

        candidate_k = self.top_k * 3
        if selected_sources is None:
            semantic_results = self.vector_store.search_embeddings(
                query_vector=self.embedding_model.embed_query(query),
                top_k=candidate_k,
            )
            bm25_results = self.vector_store.search_bm25(query, top_k=candidate_k)
        else:
            semantic_results = self.vector_store.search_embeddings(
                query_vector=self.embedding_model.embed_query(query),
                top_k=candidate_k,
                sources=selected_sources,
            )
            bm25_results = self.vector_store.search_bm25(query, top_k=candidate_k, sources=selected_sources)

        fused = {}
        for results in (semantic_results, bm25_results):
            for rank, result in enumerate(results, start=1):
                result_id = result["id"]
                if result_id not in fused:
                    fused[result_id] = {**result, "score": 0.0}
                fused[result_id]["score"] += 1 / (self.rrf_k + rank)

        return sorted(fused.values(), key=lambda result: result["score"], reverse=True)[
            : self.top_k
        ]

    def retrieve(self, query: str, selected_sources: Sequence[str] | None = None) -> list[dict]:
        return self.search(query, selected_sources=selected_sources)
