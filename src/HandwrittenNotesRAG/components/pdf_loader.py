
from pathlib import Path
import pymupdf

from ..Exception import CustomException
from ..logger import Logger


class PDFLoader:

    def load_pdf(self, pdf_path: str | Path) -> list[dict]:

        try:
            pdf_path = Path(pdf_path)

            if not pdf_path.is_file():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            if pdf_path.suffix.lower() != ".pdf":
                raise ValueError("Only PDF files are supported")

            pages = []

            with pymupdf.open(pdf_path) as pdf:

                for page_number in range(1, len(pdf) + 1):
                    page = pdf[page_number - 1]
                    page_text_result = page.get_text("text")
                    page_text = (
                        page_text_result.strip()
                        if isinstance(page_text_result, str)
                        else ""
                    )

                    page_data = {
                        "page_number": page_number,
                        "text": page_text,
                        "image": None
                    }

                    if not page_text:
                        pixmap = page.get_pixmap(dpi=150)
                        page_data["image"] = pixmap.pil_image()

                    pages.append(page_data)

            Logger.info("PDF loaded successfully")

            return pages

        except Exception as error:
            Logger.error("PDF loading failed", exc_info=True)
            raise CustomException(error) from error
