
import easyocr
from PIL import Image

from ..Exception import CustomException
from ..logger import Logger


class OCRExtractor:

    def __init__(
        self,
        languages: list[str] = ["en"]
    ) -> None:

        try:
            self.reader = easyocr.Reader(
                languages,
                gpu=False
            )

            Logger.info("OCR model initialized successfully")

        except Exception as error:
            Logger.error("OCR initialization failed", exc_info=True)
            raise CustomException(error) from error

    def extract_text(self, image: Image.Image) -> str:

        try:
            results = self.reader.readtext(
                image,
                detail=1,
                paragraph=False
            )

            extracted_text = []

            for result in results:
                text = result[1]
                confidence = result[2]#(bounding_box, recognized_text, confidence)

                if confidence >= 0.30:
                    extracted_text.append(text)

            final_text = "\n".join(extracted_text)

            Logger.info("OCR text extraction completed")

            return final_text

        except Exception as error:
            Logger.error("OCR extraction failed", exc_info=True)
            raise CustomException(error) from error
