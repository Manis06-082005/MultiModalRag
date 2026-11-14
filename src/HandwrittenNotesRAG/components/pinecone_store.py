import os
from uuid import uuid4
from pinecone import Pinecone
from ..Exception import CustomException
from ..logger import Logger
class PineconeStore:
    def __init__(self,namespace:str='Handwritten-rag')->None:
        try:
            api_key=os.getenv("PINECONE_API_KEY")
            index_name=os.getenv("PINECONE_INDEX_NAME")
            if not api_key or not index_name:
                raise ValueError('Pinecone credentials are missing')
            self.pc=Pinecone(api_key=api_key)
            self.index=self.pc.Pinecone(index_name)
            self.namespace=namespace
            Logger.info("Pincone initialized successfully!")
        except Exception as error:
            raise CustomException(error=error) from error
    def store_embeddings(self,embeddings:list[list[float]],texts:list[str])->None:
        try:
            if len(embeddings)!=len(text):
                raise ValueError("Embeddings length should be equal to len of text")
            vectors=[]
            for embedding,text in zip(embeddings,texts):
                vector={
                    "id":str(uuid4()),
                    "values":embedding,
                    "metadata":{
                        'text':text
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
        
            