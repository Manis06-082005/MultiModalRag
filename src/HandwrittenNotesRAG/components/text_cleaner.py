import re

from ..Exception import CustomException
from ..logger import Logger


class TextCleaner:
    def text_clean(self, text: str) -> str:
        try:
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.strip()
            Logger.info("Text cleaning completed")
            return text
        except Exception as error:
            Logger.error("Text cleaning failed", exc_info=True)
            raise CustomException(error) from error

    def clean_text(self, text: str) -> str:
        return self.text_clean(text)

