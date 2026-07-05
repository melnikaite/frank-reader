import io
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from PIL import Image

from frank_reader.adapters._pseudopage import build_pseudo_pages
from frank_reader.adapters.base import InlineImage, PageContent
from frank_reader.config import Settings


def _iter_block_items(document: Document):
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield "p", Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield "tbl", Table(child, document)


def _image_to_png(data: bytes) -> tuple[bytes | None, int, int]:
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception:
        return None, 0, 0
    if im.mode not in ("RGB", "RGBA", "L"):
        im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue(), im.width, im.height


class DocxAdapter:
    def __init__(self, settings: Settings):
        self.settings = settings

    def load(self, source: Path) -> list[PageContent]:
        document = Document(str(source))
        parts: list[str] = []
        images: list[InlineImage] = []

        for kind, block in _iter_block_items(document):
            if kind == "p":
                text = block.text
                style_name = block.style.name if block.style is not None else ""
                is_list = "List" in (style_name or "")
                line = f"- {text}" if is_list and text.strip() else text
                blips = block._p.findall(".//" + qn("a:blip"))
                parts.append(line)
                block_end_offset = len("\n\n".join(parts))
                for blip in blips:
                    rid = blip.get(qn("r:embed"))
                    if not rid:
                        continue
                    try:
                        image_part = document.part.related_parts[rid]
                    except KeyError:
                        continue
                    png_bytes, w, h = _image_to_png(image_part.blob)
                    if png_bytes is None:
                        continue
                    if min(w, h) < self.settings.inline_image_min_dim:
                        continue
                    images.append(InlineImage(image_png=png_bytes, position_anchor=block_end_offset))
            else:
                rows_joined = [" | ".join(cell.text for cell in row.cells) for row in block.rows]
                parts.append("\n".join(rows_joined))

        full_text = "\n\n".join(parts)
        return build_pseudo_pages(full_text, images, self.settings.pseudo_page_chars)
