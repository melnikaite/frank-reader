from pathlib import Path

from charset_normalizer import from_bytes

from frank_reader.adapters._pseudopage import build_pseudo_pages
from frank_reader.adapters.base import PageContent
from frank_reader.config import Settings


class PlainTextAdapter:
    def __init__(self, settings: Settings):
        self.settings = settings

    def load(self, source: Path | str) -> list[PageContent]:
        if isinstance(source, Path):
            raw = source.read_bytes()
            match = from_bytes(raw).best()
            text = str(match) if match is not None else raw.decode("utf-8", errors="replace")
        else:
            text = source
        return build_pseudo_pages(text, [], self.settings.pseudo_page_chars)
