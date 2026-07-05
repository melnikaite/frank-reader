import re

from frank_reader.adapters.base import InlineImage, PageContent

_PARA_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_paragraphs_with_offsets(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int]] = []
    last_end = 0
    for m in _PARA_SPLIT_RE.finditer(text):
        if m.start() > last_end:
            spans.append((last_end, m.start()))
        last_end = m.end()
    if last_end < len(text):
        spans.append((last_end, len(text)))
    return [(s, e, text[s:e]) for s, e in spans if text[s:e].strip()]


def _slice_paragraph(start: int, ptext: str, max_chars: int) -> list[tuple[int, int, str]]:
    pieces: list[tuple[int, int, str]] = []
    n = len(ptext)
    pos = 0
    while pos < n:
        end = min(pos + max_chars, n)
        if end < n:
            ws = ptext.rfind(" ", pos, end)
            if ws > pos:
                end = ws
        pieces.append((start + pos, start + end, ptext[pos:end]))
        pos = end
        while pos < n and ptext[pos] == " ":
            pos += 1
    return pieces


def build_pseudo_pages(
    text: str, images: list[InlineImage], max_chars: int
) -> list[PageContent]:
    """Split text into pseudo-pages on paragraph boundaries (falling back to
    sentence/char slicing for oversized paragraphs), remapping image anchors
    (absolute offsets into `text`) to page-local offsets."""
    paragraphs = _split_paragraphs_with_offsets(text)
    if not paragraphs:
        stripped = text.strip()
        imgs = [InlineImage(image_png=i.image_png, position_anchor=0) for i in images] if stripped else []
        return [PageContent(page_number=1, kind="text", text=stripped, inline_images=imgs)]

    pieces: list[tuple[int, int, str]] = []
    for start, end, ptext in paragraphs:
        if len(ptext) <= max_chars:
            pieces.append((start, end, ptext))
        else:
            pieces.extend(_slice_paragraph(start, ptext, max_chars))

    pages_pieces: list[list[tuple[int, int, str]]] = []
    current: list[tuple[int, int, str]] = []
    current_len = 0
    for piece in pieces:
        plen = piece[1] - piece[0]
        if current and current_len + plen > max_chars:
            pages_pieces.append(current)
            current = []
            current_len = 0
        current.append(piece)
        current_len += plen
    if current:
        pages_pieces.append(current)

    page_ranges = [(pp[0][0], pp[-1][1]) for pp in pages_pieces]
    page_texts = ["\n\n".join(p[2] for p in pp) for pp in pages_pieces]

    pages_images: list[list[InlineImage]] = [[] for _ in pages_pieces]
    for img in images:
        target = 0
        for i, (start, _end) in enumerate(page_ranges):
            if start <= img.position_anchor:
                target = i
            else:
                break
        page_start, page_end = page_ranges[target]
        local_anchor = min(max(img.position_anchor - page_start, 0), max(page_end - page_start, 0))
        pages_images[target].append(InlineImage(image_png=img.image_png, position_anchor=local_anchor))

    return [
        PageContent(page_number=i + 1, kind="text", text=page_texts[i], inline_images=pages_images[i])
        for i in range(len(pages_pieces))
    ]
