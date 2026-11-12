import tempfile
from pathlib import Path

import pymupdf

from src.HandwrittenNotesRAG.Exception import CustomException
from src.HandwrittenNotesRAG.components.pdf_loader import PDFLoader


def create_sample_pdf(file_path: Path, content: str) -> None:
    pdf_document = pymupdf.open()
    page = pdf_document.new_page()
    page.insert_text((72, 72), content, fontsize=12)
    pdf_document.save(file_path)
    pdf_document.close()


def test_valid_pdf() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        valid_pdf_path = temp_dir_path / "sample.pdf"
        create_sample_pdf(valid_pdf_path, "Hello from handwritten notes!")

        loader = PDFLoader()
        pages = loader.load_pdf(valid_pdf_path)
        assert isinstance(pages, list), "PDFLoader.load_pdf should return a list of page dictionaries"
        assert len(pages) == 1, "Expected exactly one page in the sample PDF"
        assert pages[0]["page_number"] == 1, "The first page should be numbered 1"
        assert "Hello from handwritten notes!" in pages[0]["text"], "Extracted PDF text did not match the sample content"
        assert pages[0]["image"] is None, "A text page should not contain a rendered image"


def test_missing_file() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        missing_pdf_path = temp_dir_path / "missing.pdf"
        try:
            loader = PDFLoader()
            loader.load_pdf(missing_pdf_path)
            raise AssertionError("Expected a CustomException for a missing PDF path")
        except CustomException:
            pass


def test_invalid_extension() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        invalid_extension_path = temp_dir_path / "notes.txt"
        invalid_extension_path.write_text("This is not a PDF file.", encoding="utf-8")
        try:
            loader = PDFLoader()
            loader.load_pdf(invalid_extension_path)
            raise AssertionError("Expected a CustomException for a non-PDF file")
        except CustomException:
            pass


def main() -> None:
    tests = [
        ("valid_pdf", test_valid_pdf),
        ("missing_file", test_missing_file),
        ("invalid_extension", test_invalid_extension),
    ]

    failures = []
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"PASS: {test_name}")
        except Exception as exc:  # noqa: BLE001 - intentionally capturing failures for test reporting
            failures.append((test_name, str(exc)))
            print(f"FAIL: {test_name}: {exc}")

    if failures:
        print(f"\n{len(failures)} test(s) failed.")
        raise SystemExit(1)

    print("\nAll PDFLoader tests passed successfully.")


if __name__ == "__main__":
    main()
