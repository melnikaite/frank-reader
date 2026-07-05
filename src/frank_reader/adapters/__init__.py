from pathlib import Path

from frank_reader.adapters.base import InlineImage, PageContent, SourceAdapter
from frank_reader.adapters.docx import DocxAdapter
from frank_reader.adapters.image import ImageAdapter
from frank_reader.adapters.pdf import PdfAdapter
from frank_reader.adapters.plaintext import PlainTextAdapter
from frank_reader.adapters.url import UrlAdapter
from frank_reader.config import Settings

_EXTENSION_MAP = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".txt": "text",
}

__all__ = [
    "InlineImage",
    "PageContent",
    "SourceAdapter",
    "detect_source_type",
    "get_adapter",
]


def detect_source_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in _EXTENSION_MAP:
        supported = ", ".join(sorted(set(_EXTENSION_MAP)))
        raise ValueError(f"Unsupported file extension: {ext}. Supported: {supported}")
    return _EXTENSION_MAP[ext]


def get_adapter(source_type: str, settings: Settings, force_vision: bool = False) -> SourceAdapter:
    if source_type == "pdf":
        return PdfAdapter(settings, force_vision=force_vision)
    if source_type == "docx":
        return DocxAdapter(settings)
    if source_type == "image":
        return ImageAdapter(settings)
    if source_type == "url":
        return UrlAdapter(settings)
    if source_type == "text":
        return PlainTextAdapter(settings)
    raise ValueError(f"Unknown source type: {source_type}")
