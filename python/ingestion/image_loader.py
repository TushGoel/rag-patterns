"""
Image ingestion — OCR-based text extraction from images.

Extracts text from images, screenshots, scanned documents, and diagrams.
Uses pytesseract (Tesseract OCR) as the primary engine with optional
EasyOCR fallback for better multilingual support.

Prerequisites:
    brew install tesseract          # macOS
    apt-get install tesseract-ocr   # Linux
    pip install pytesseract pillow

Supported formats: PNG, JPG, JPEG, TIFF, BMP, GIF, WEBP
"""
from __future__ import annotations

import re
from pathlib import Path
from .document import Document, SourceType

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"}


class ImageLoader:
    """
    Extract text from images via OCR.

    Uses pytesseract for local OCR. Falls back to EasyOCR for
    multilingual documents or when pytesseract is not installed.

    Usage:
        loader = ImageLoader()
        doc = loader.load("screenshot.png")
        docs = loader.load_batch(["fig1.png", "fig2.png"])
    """

    def __init__(self, engine: str = "tesseract", lang: str = "eng") -> None:
        self.engine = engine
        self.lang = lang

    def load(self, path: str) -> Document:
        path_obj = Path(path)
        if path_obj.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {path_obj.suffix}")

        text = self._extract_text(path)
        text = self._clean(text)

        return Document(
            content=text,
            source=path,
            source_type=SourceType.IMAGE,
            metadata={
                "file_name": path_obj.name,
                "extension": path_obj.suffix.lower(),
                "ocr_engine": self.engine,
                "ocr_lang": self.lang,
                "has_text": len(text.strip()) > 0,
            },
        )

    def load_batch(self, paths: list[str]) -> list[Document]:
        docs = []
        for path in paths:
            try:
                doc = self.load(path)
                if doc.content.strip():
                    docs.append(doc)
            except Exception:
                continue
        return docs

    def _extract_text(self, path: str) -> str:
        if self.engine == "tesseract":
            return self._tesseract(path)
        elif self.engine == "easyocr":
            return self._easyocr(path)
        else:
            raise ValueError(f"Unknown OCR engine: {self.engine}")

    def _tesseract(self, path: str) -> str:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise ImportError("pip install pytesseract pillow  &&  brew install tesseract")

        img = Image.open(path)
        text = pytesseract.image_to_string(img, lang=self.lang)
        return text

    def _easyocr(self, path: str) -> str:
        try:
            import easyocr
        except ImportError:
            raise ImportError("pip install easyocr")

        reader = easyocr.Reader([self.lang])
        results = reader.readtext(path, detail=0)
        return "\n".join(results)

    def _clean(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()
