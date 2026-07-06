import base64
import html as html_lib
from pathlib import Path
from typing import Any

_STYLE = """
<style>
  :root { color-scheme: light; }
  body {
    font-family: Georgia, "Times New Roman", serif;
    max-width: 46rem;
    margin: 2rem auto;
    padding: 0 1.25rem 4rem;
    line-height: 1.7;
    font-size: 1.05rem;
    color: #1a1a1a;
  }
  .tr { color: #7a7a7a; font-size: 0.92em; }
  .phrase { margin: 0 0 1.1em; }
  .phrase .glossed { margin: 0 0 0.15em; }
  .phrase .plain { margin: 0; color: #444; }
  h2 { margin-top: 2em; }
  .caption { font-style: italic; color: #555; }
  .page-failed { background: #fee; border: 1px solid #c33; color: #900; padding: 0.75rem 1rem;
                 margin: 1rem 0; border-radius: 4px; }
  .inline-image { max-width: 100%; display: block; margin: 1rem 0 0.4rem; }
  .img-labels { color: #7a7a7a; font-size: 0.9em; margin: 0 0 1.2rem 1.2rem; }
  body.translations-hidden .tr { display: none; }
  #toggle-btn {
    position: fixed; top: 1rem; right: 1rem; z-index: 10;
    font-family: sans-serif; font-size: 0.85rem; padding: 0.4rem 0.8rem;
    border: 1px solid #999; border-radius: 4px; background: #fff; cursor: pointer;
  }
  @media print {
    #toggle-btn { display: none; }
    .phrase { page-break-inside: avoid; }
    body { max-width: 100%; margin: 0; padding: 0.5in; }
  }
</style>
"""

_TOGGLE_BUTTON = (
    '<button id="toggle-btn" '
    "onclick=\"document.body.classList.toggle('translations-hidden')\">"
    "Hide translations</button>"
)


def _esc(s: str | None) -> str:
    return html_lib.escape(s or "", quote=False)


def _render_chunk_line(block: dict[str, Any]) -> str:
    pieces = [
        f'{_esc(chunk["original"])} <span class="tr">({_esc(chunk["translation"])})</span>'
        for chunk in block.get("chunks", [])
    ]
    return " ".join(pieces)


def _render_phrase(block: dict[str, Any]) -> str:
    glossed = _render_chunk_line(block)
    plain = _esc(block["original"])
    return f'<div class="phrase"><p class="glossed">{glossed}</p><p class="plain">{plain}</p></div>'


def _render_heading(block: dict[str, Any]) -> str:
    return f'<h2>{_esc(block["original"])} <span class="tr">({_esc(block.get("translation"))})</span></h2>'


def _render_caption(block: dict[str, Any]) -> str:
    return (
        f'<p class="caption">{_esc(block["original"])} '
        f'<span class="tr">({_esc(block.get("translation"))})</span></p>'
    )


def _render_list_items(items: list[dict[str, Any]]) -> str:
    lis = "".join(
        f'<li>{_esc(item["original"])} <span class="tr">({_esc(item.get("translation"))})</span></li>'
        for item in items
    )
    return f"<ul>{lis}</ul>"


def _render_blocks_html(text_blocks: list[dict[str, Any]]) -> list[str]:
    out: list[str] = [""] * len(text_blocks)
    i = 0
    while i < len(text_blocks):
        block = text_blocks[i]
        if block["type"] == "list_item":
            j = i
            items = []
            while j < len(text_blocks) and text_blocks[j]["type"] == "list_item":
                items.append(text_blocks[j])
                j += 1
            out[i] = _render_list_items(items)
            i = j
            continue
        if block["type"] == "phrase":
            out[i] = _render_phrase(block)
        elif block["type"] == "heading":
            out[i] = _render_heading(block)
        elif block["type"] == "caption":
            out[i] = _render_caption(block)
        i += 1
    return out


def _compute_block_spans(text_blocks: list[dict[str, Any]], source_text: str) -> list[tuple[int, int]]:
    spans = []
    cursor = 0
    for block in text_blocks:
        original = block["original"]
        idx = source_text.find(original, cursor)
        if idx == -1:
            idx = cursor
        end = idx + len(original)
        spans.append((idx, end))
        cursor = end
    return spans


def _image_target_index(spans: list[tuple[int, int]], anchor: int) -> int:
    for i, (_start, end) in enumerate(spans):
        if end >= anchor:
            return i
    return len(spans) - 1 if spans else 0


def _render_inline_image_html(img_file: str, job_dir: Path, labels: list[dict[str, Any]]) -> str:
    data = (job_dir / img_file).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    parts = [f'<img class="inline-image" src="data:image/png;base64,{b64}" alt="">']
    if labels:
        items = "".join(
            f'<li>{_esc(lbl["original"])} — <span class="tr">{_esc(lbl["translation"])}</span></li>'
            for lbl in labels
        )
        parts.append(f'<ul class="img-labels">{items}</ul>')
    return "".join(parts)


def _render_page(page: dict[str, Any], job_dir: Path) -> str:
    # Pages are internal processing units (progress/retry granularity), not a
    # feature of the reading text — the output flows continuously. Only a
    # failed fragment surfaces its page number, so it can be matched with the
    # job UI and retried.
    parts: list[str] = []
    if page.get("status") == "failed":
        parts.append(
            f'<div class="page-failed">A fragment was not processed '
            f'(page {page["page_number"]}): {_esc(page.get("error"))}</div>'
        )
        return "".join(parts)

    result = page.get("result") or {}
    text_blocks = result.get("text_blocks", [])
    blocks_html = _render_blocks_html(text_blocks)

    inline_images = page.get("inline_images", [])
    source_text = page.get("source_text")
    images_by_index: dict[int, list[tuple[int, dict]]] = {}
    if inline_images and source_text:
        spans = _compute_block_spans(text_blocks, source_text)
        for img_idx, img in enumerate(inline_images):
            target = _image_target_index(spans, img["position_anchor"])
            images_by_index.setdefault(target, []).append((img_idx, img))

    annotations_by_ref: dict[int, list[dict]] = {
        ann["image_ref"]: ann.get("labels", []) for ann in result.get("image_annotations", [])
    }

    for i, block_html in enumerate(blocks_html):
        parts.append(block_html)
        for img_idx, img in images_by_index.get(i, []):
            labels = annotations_by_ref.get(img_idx, [])
            parts.append(_render_inline_image_html(img["file"], job_dir, labels))

    return "".join(parts)


def render_html(result_doc: dict[str, Any], job_dir: Path) -> str:
    parts = [
        '<!doctype html><html><head><meta charset="utf-8">',
        f'<title>{_esc(result_doc.get("source_name", "Frank Reader"))}</title>',
        _STYLE,
        "</head><body>",
        _TOGGLE_BUTTON,
    ]
    for page in result_doc.get("pages", []):
        parts.append(_render_page(page, job_dir))
    parts.append("</body></html>")
    return "".join(parts)
