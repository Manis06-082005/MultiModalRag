
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

                for page in pdf:
                    page_text = page.get_text().strip()

                    page_data = {
                        "page_number": page.number + 1,
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
