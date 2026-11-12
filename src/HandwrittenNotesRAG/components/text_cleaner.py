import re
from ..Exception import CustomException
from ..logger import Logger
class TextCleaner:
    def text_clean(self,text:str)->str:
        try:
            text=re.sub(r"[ \t]+"," ",text)## Remove extra spaces and tabs
            text=re.sub(r"/n{3}","/n/n",text)## Remove excessive blank lines
            text=text.strip()
            Logger.info("Text cleaning completed")

            return text

        except Exception as error:
            Logger.error("Text cleaning failed", exc_info=True)
            raise CustomException(error) from error
            
