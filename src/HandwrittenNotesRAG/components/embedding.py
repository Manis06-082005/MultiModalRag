from typing import Sequence
from sentence_transformers import SentenceTransformer
from ..Exception import CustomException
from ..logger import Logger

class EmbeddingGenerator:
    def __init__(self,model_name:str="sentence-transformers/all-MiniLM-L6-v2",device:str='cpu')->None:
        try:
            self.model=SentenceTransformer(model_name,device=device)
            Logger.info("Embedding Model initialized successfully!")
        except Exception as error:
            Logger.error("Embedding model initialization failed",exc_info=True)
            raise CustomException(error) from error
    def embed_text(self,text:str|Sequence[str]):
        try:
            if isinstance(text,str):
                vector=self.model.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                return vector.tolist()
            texts=list(text)
            vectors=self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return vectors.tolist()
        except Exception as error:
            Logger.error("failed to convert embeddings",exc_info=True)
            raise CustomException(error) from error
    def embed_query(self,query:str)->list[float]:
        return self.embed_text(query)
    def embed_document(self,documents:Sequence[str])->list[list[float]]:
        return self.embed_text(documents)

#     def embed_query(self, query: str) -> list[float]:
#         return self.embed_text(query)

#     def embed_documents(self, documents: Sequence[str]) -> list[list[float]]:
#         return self.embed_text(documents)