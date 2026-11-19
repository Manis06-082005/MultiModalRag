"""Real OCR test using EasyOCR and a small generated text image."""

from PIL import Image, ImageDraw, ImageFont

from HandwrittenNotesRAG.components.ocr_extractor import OCRExtractor


def test_easyocr_reads_text_from_image() -> None:
    """EasyOCR should recognize a short, clearly printed phrase."""
    image = Image.new("RGB", (700, 180), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 64)
    except OSError:
        font = ImageFont.load_default(size=64)
    draw.text((20, 35), "HELLO WORLD", fill="black", font=font)

    text = OCRExtractor().extract_text(image)

    assert "HELLO" in text.upper()
    assert "WORLD" in text.upper()
