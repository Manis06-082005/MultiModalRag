
import easyocr
import numpy as np
from PIL import Image
from numpy.typing import NDArray
from typing import Any

from ..Exception import CustomException
from ..logger import Logger


class OCRExtractor:

    def __init__(
        self,
        languages: tuple[str, ...] = ("en",)
    ) -> None:

        try:
            self.reader = easyocr.Reader(
                list(languages),
                gpu=False
            )

            Logger.info("OCR model initialized successfully")

        except Exception as error:
            Logger.error("OCR initialization failed", exc_info=True)
            raise CustomException(error) from error

    def extract_text(
        self,
        image: Image.Image | NDArray[Any]
    ) -> str:

        try:
            image_array: NDArray[Any]
            if isinstance(image, Image.Image):
                image_array = np.asarray(image.convert("RGB"))
            else:
                image_array = image

            results = self.reader.readtext(
                image_array,
                detail=1,
                paragraph=False
            )

            extracted_text = []

            for result in results:
                if isinstance(result, (tuple, list)) and len(result) >= 3:
                    text = result[1]
                    confidence = result[2]

                    if (
                        isinstance(text, str)
                        and isinstance(confidence, (int, float))
                        and confidence >= 0.30
                    ):
                        extracted_text.append(text)

            final_text = "\n".join(extracted_text)

            Logger.info("OCR text extraction completed")

            return final_text

        except Exception as error:
            Logger.error("OCR extraction failed", exc_info=True)
            raise CustomException(error) from error
