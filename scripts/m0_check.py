"""M0 — proof of prompt.

Manual quality check of the Frank-method chunking against the real LocalAI,
without the web UI or the pipeline: one page (PNG or a PDF page) -> LocalAI ->
validated PageResult JSON -> simple HTML next to the source file.

Usage:
    uv run python scripts/m0_check.py document.pdf --page 1 --lang ru
    uv run python scripts/m0_check.py scan.png --lang ru
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fitz  # noqa: E402

from frank_reader.config import Settings  # noqa: E402
from frank_reader.pipeline.llm_client import LocalAIClient, call_structured  # noqa: E402
from frank_reader.pipeline.orchestrator import _sanity_check_chunks  # noqa: E402
from frank_reader.pipeline.prompts import (  # noqa: E402
    build_context_block,
    system_prompt,
    user_vision_page,
)
from frank_reader.pipeline.schema import PageResult  # noqa: E402
from frank_reader.render.html import render_html  # noqa: E402


def _load_page_png(path: Path, page_no: int, settings: Settings) -> bytes:
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(str(path))
        try:
            page = doc[page_no - 1]
            long_side = max(page.rect.width, page.rect.height, 1.0)
            zoom = min(settings.page_render_max_dim / long_side, 4.0)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            return pix.tobytes("png")
        finally:
            doc.close()
    data = path.read_bytes()
    if path.suffix.lower() == ".png":
        return data
    from io import BytesIO

    from PIL import Image

    im = Image.open(BytesIO(data)).convert("RGB")
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="PDF or image file")
    parser.add_argument("--page", type=int, default=1, help="Page number (for PDF), 1-based")
    parser.add_argument("--lang", default="ru", help="Target translation language")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        return 1

    settings = Settings()
    png_bytes = _load_page_png(args.file, args.page, settings)

    system = system_prompt(args.lang)
    context = build_context_block(None, [], {})
    user_text = user_vision_page(context)

    llm = LocalAIClient(settings)
    print(f"LLM: {settings.llm_base_url} model={settings.llm_model}")
    start = time.monotonic()
    try:
        result: PageResult = await call_structured(llm, system, user_text, PageResult, image_png=png_bytes)
    except Exception as exc:
        elapsed = time.monotonic() - start
        print(f"FAILED after {elapsed:.1f}s: {exc}", file=sys.stderr)
        return 1
    finally:
        await llm.aclose()
    elapsed = time.monotonic() - start

    print(f"OK in {elapsed:.1f}s: {len(result.text_blocks)} blocks, language={result.detected_language}")
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))

    for block in result.text_blocks:
        _sanity_check_chunks(block.model_dump())

    out_doc = {
        "source_name": args.file.name,
        "pages": [
            {
                "page_number": args.page,
                "status": "done",
                "result": result.model_dump(),
                "inline_images": [],
            }
        ],
    }
    html = render_html(out_doc, job_dir=args.file.parent)
    out_path = args.file.parent / "m0_out.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"HTML written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
