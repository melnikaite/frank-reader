from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol


@dataclass
class InlineImage:
    image_png: bytes
    position_anchor: int


@dataclass
class PageContent:
    page_number: int
    kind: Literal["text", "image"]
    text: str | None = None
    image_png: bytes | None = None
    inline_images: list[InlineImage] = field(default_factory=list)


class SourceAdapter(Protocol):
    def load(self, source: Path | str) -> list[PageContent]: ...
