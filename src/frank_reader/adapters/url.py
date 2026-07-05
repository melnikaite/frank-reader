import re
from typing import Callable

import httpx
import trafilatura

from frank_reader.adapters._pseudopage import build_pseudo_pages
from frank_reader.adapters.base import InlineImage, PageContent
from frank_reader.config import Settings

_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _download_image(url: str) -> bytes | None:
    try:
        resp = httpx.get(url, timeout=20)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


def process_markdown(
    markdown_text: str,
    settings: Settings,
    fetch_image: Callable[[str], bytes | None] | None = None,
) -> list[PageContent]:
    """Strip markdown image links, download their targets, and paginate the
    remaining text with images anchored where the links used to be."""
    fetch = fetch_image or _download_image

    images: list[InlineImage] = []
    out_parts: list[str] = []
    last_end = 0
    out_pos = 0
    for m in _MD_IMAGE_RE.finditer(markdown_text):
        segment = markdown_text[last_end : m.start()]
        out_parts.append(segment)
        out_pos += len(segment)
        data = fetch(m.group(1))
        if data is not None:
            images.append(InlineImage(image_png=data, position_anchor=out_pos))
        last_end = m.end()
    out_parts.append(markdown_text[last_end:])
    text_without_images = "".join(out_parts)

    return build_pseudo_pages(text_without_images, images, settings.pseudo_page_chars)


class UrlAdapter:
    def __init__(self, settings: Settings):
        self.settings = settings

    def load(self, source: str) -> list[PageContent]:
        downloaded = trafilatura.fetch_url(source)
        if not downloaded:
            raise ValueError(f"Failed to fetch URL: {source}")
        extracted = trafilatura.extract(
            downloaded, output_format="markdown", include_images=True, url=source
        )
        if not extracted:
            raise ValueError(f"Failed to extract content from URL: {source}")
        return process_markdown(extracted, self.settings)
