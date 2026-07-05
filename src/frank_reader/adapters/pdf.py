import unicodedata
from pathlib import Path

import fitz

from frank_reader.adapters._pseudopage import build_pseudo_pages
from frank_reader.adapters.base import InlineImage, PageContent
from frank_reader.config import Settings


def _garbage_ratio(text: str) -> float:
    if not text:
        return 1.0
    bad = sum(
        1
        for ch in text
        if ch == "�" or (unicodedata.category(ch) == "Cc" and ch not in "\n\t")
    )
    return bad / len(text)


def _collect_text_blocks(page: "fitz.Page") -> tuple[str, list[tuple["fitz.Rect", int]]]:
    raw_blocks = page.get_text("blocks", sort=True)
    text_blocks = [b for b in raw_blocks if b[6] == 0 and b[4].strip()]
    parts = [b[4].strip() for b in text_blocks]
    full_text = "\n\n".join(parts)
    ranges: list[tuple[fitz.Rect, int]] = []
    pos = 0
    for i, txt in enumerate(parts):
        end = pos + len(txt)
        ranges.append((fitz.Rect(text_blocks[i][:4]), end))
        pos = end + 2
    return full_text, ranges


def _xref_to_png(doc: "fitz.Document", xref: int) -> tuple[bytes | None, int, int]:
    try:
        pix = fitz.Pixmap(doc, xref)
    except Exception:
        return None, 0, 0
    if pix.n - pix.alpha >= 4:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    return pix.tobytes("png"), pix.width, pix.height


def _extract_inline_images(
    page: "fitz.Page",
    doc: "fitz.Document",
    ranges: list[tuple["fitz.Rect", int]],
    settings: Settings,
) -> list[InlineImage]:
    page_area = page.rect.width * page.rect.height
    images: list[InlineImage] = []
    seen_xrefs: set[int] = set()
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        rect = rects[0]
        png_bytes, w, h = _xref_to_png(doc, xref)
        if png_bytes is None:
            continue
        if min(w, h) < settings.inline_image_min_dim:
            continue
        area_fraction = (rect.width * rect.height) / page_area if page_area else 0.0
        if area_fraction < settings.inline_image_min_area:
            continue
        anchor = 0
        best_end = None
        for block_rect, end_offset in ranges:
            if block_rect.y1 <= rect.y0 and (best_end is None or end_offset > best_end):
                best_end = end_offset
        if best_end is not None:
            anchor = best_end
        images.append(InlineImage(image_png=png_bytes, position_anchor=anchor))
    return images


class PdfAdapter:
    def __init__(self, settings: Settings, force_vision: bool = False):
        self.settings = settings
        self.force_vision = force_vision

    def load(self, source: Path) -> list[PageContent]:
        doc = fitz.open(str(source))
        pages: list[PageContent] = []
        next_page_number = 1
        try:
            for page in doc:
                full_text, ranges = _collect_text_blocks(page)
                is_text_ok = (
                    not self.force_vision
                    and len(full_text) >= self.settings.min_text_chars
                    and _garbage_ratio(full_text) < self.settings.garbage_char_ratio
                )
                if is_text_ok:
                    inline_images = _extract_inline_images(page, doc, ranges, self.settings)
                    if len(full_text) <= self.settings.pseudo_page_chars:
                        pages.append(
                            PageContent(
                                page_number=next_page_number,
                                kind="text",
                                text=full_text,
                                inline_images=inline_images,
                            )
                        )
                        next_page_number += 1
                    else:
                        # A single dense PDF page can produce more Frank-method
                        # output (original + per-chunk translation) than the
                        # model reliably completes in one call - split it the
                        # same way long plaintext/DOCX/URL pages are split.
                        for sub_page in build_pseudo_pages(
                            full_text, inline_images, self.settings.pseudo_page_chars
                        ):
                            sub_page.page_number = next_page_number
                            pages.append(sub_page)
                            next_page_number += 1
                else:
                    long_side = max(page.rect.width, page.rect.height, 1.0)
                    zoom = min(self.settings.page_render_max_dim / long_side, 4.0)
                    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                    pages.append(
                        PageContent(page_number=next_page_number, kind="image", image_png=pix.tobytes("png"))
                    )
                    next_page_number += 1
        finally:
            doc.close()
        return pages
