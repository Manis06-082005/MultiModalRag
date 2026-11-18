import os
from uuid import uuid4

from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

from ..Exception import CustomException
from ..logger import Logger


class PineconeStore:
    def __init__(self, namespace: str = 'Handwritten-rag') -> None:
        try:
            api_key = os.getenv("PINECONE_API_KEY")
            index_name = os.getenv("PINECONE_INDEX_NAME")
            if not api_key or not index_name:
                raise ValueError('Pinecone credentials are missing')

            self.pc = Pinecone(api_key=api_key)

            if not self.pc.has_index(index_name):
                Logger.warning("Pinecone index '%s' not found. Creating it.", index_name)
                self.pc.create_index(
                    name=index_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )

            self.index = self.pc.Index(index_name)
            self.namespace = namespace
            self.bm25_encoder = BM25Encoder.default()
            Logger.info("Pinecone initialized successfully!")
        except Exception as error:
            raise CustomException(error=error) from error
    
    def store_embeddings(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        metadata: list[dict] | None = None
    ) -> None:

        try:
            if len(embeddings) != len(texts):
                raise ValueError("Embeddings and texts must have equal lengths")

            if metadata is None:
                metadata = [{} for _ in texts]

            if len(metadata) != len(texts):
                raise ValueError("Metadata and texts must have equal lengths")

            vectors = []

            for embedding, text, meta in zip(
                embeddings, texts, metadata
            ):

                vector = {
                    "id": str(uuid4()),
                    "values": embedding,
                    "sparse_values": self.bm25_encoder.encode_documents(text),
                    "metadata": {
                        "text": text,
                        **meta
                    }
                }

                vectors.append(vector)

            if not vectors:
                return

            response = self.index.upsert(
                vectors=vectors,
                namespace=self.namespace
            )

            Logger.info(
                f"Stored {response.upserted_count} vectors in Pinecone"
            )

        except Exception as error:
            Logger.error("Failed to store embeddings", exc_info=True)
            raise CustomException(error) from error

    def search_embeddings(
            self,
            query_vector: list[float],
            top_k: int = 5,
    ) -> list[dict]:
            try:
                results = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    namespace=self.namespace,
                    include_metadata=True
                )
                matches = []

                for match in results.matches:
                    matches.append({
                        "id": match.id,
                        "score": match.score,
                        **dict(match.metadata or {}),
                    })

                Logger.info("Pinecone search completed")
                return matches

            except Exception as error:
                Logger.error("Pinecone search failed", exc_info=True)
                raise CustomException(error) from error

    def search_bm25(self, query: str, top_k: int = 5) -> list[dict]:
        try:
            results = self.index.query(
                vector=[0.0] * 384,
                sparse_vector=self.bm25_encoder.encode_queries(query),
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True,
            )
            return [
                {
                    "id": match.id,
                    "score": match.score,
                    **dict(match.metadata or {}),
                }
                for match in results.matches
            ]
        except Exception as error:
            Logger.error("Pinecone BM25 search failed", exc_info=True)
            raise CustomException(error) from error

