import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image

from src.HandwrittenNotesRAG.components.ocr_extractor import OCRExtractor


def test_extract_returns_nonempty_recognized_lines(monkeypatch) -> None:
    class FakeReader:
        def __init__(self, languages: list[str], gpu: bool) -> None:
            assert languages == ["en"]
            assert gpu is False

        def readtext(
            self,
            image: np.ndarray,
            detail: int,
            paragraph: bool,
        ) -> list[str]:
            assert image.shape == (4, 5, 3)
            assert detail == 0
            assert paragraph is True
            return ["  Hello world  ", "", "Second line"]

    fake_easyocr = ModuleType("easyocr")
    fake_easyocr.Reader = FakeReader
    monkeypatch.setitem(sys.modules, "easyocr", fake_easyocr)

    extractor = OCRExtractor()

    assert extractor.extract(Image.new("L", (5, 4))) == "Hello world\nSecond line"


def test_extract_returns_empty_string_when_no_text_is_found(monkeypatch) -> None:
    fake_easyocr = ModuleType("easyocr")
    fake_easyocr.Reader = lambda languages, gpu: SimpleNamespace(
        readtext=lambda image, detail, paragraph: []
    )
    monkeypatch.setitem(sys.modules, "easyocr", fake_easyocr)

    extractor = OCRExtractor()

    assert extractor.extract(Image.new("RGB", (2, 2))) == ""