
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..Exception import CustomException
from ..logger import Logger


class TextChunker:

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> None:

        try:
            if chunk_size <= 0:
                raise ValueError("Chunk size must be greater than zero")

            if chunk_overlap < 0 or chunk_overlap >= chunk_size:
                raise ValueError(
                    "Chunk overlap must be smaller than chunk size"
                )

            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
                length_function=len
            )

            Logger.info("Text chunker initialized successfully")

        except Exception as error:
            Logger.error("Text chunker initialization failed", exc_info=True)
            raise CustomException(error) from error

    def split_text(self, text: str) -> list[str]:

        try:
            if not text or not text.strip():
                return []

            chunks = self.splitter.split_text(text)

            Logger.info("Text chunking completed")

            return chunks

        except Exception as error:
            Logger.error("Text chunking failed", exc_info=True)
            raise CustomException(error) from error
