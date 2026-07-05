import asyncio
import hashlib
import json
import logging

from frank_reader.adapters import PageContent, get_adapter
from frank_reader.config import Settings
from frank_reader.pipeline.llm_client import LLMClient, call_structured
from frank_reader.pipeline.prompts import (
    PROMPT_VERSION,
    build_context_block,
    system_prompt,
    user_inline_image,
    user_text_page,
    user_vision_page,
)
from frank_reader.pipeline.schema import ImageAnnotation, ImageLabelsResult, PageResult
from frank_reader.render.html import render_html
from frank_reader.storage import Storage

logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self) -> None:
        self._queues: dict[str, set[asyncio.Queue]] = {}

    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._queues.setdefault(job_id, set()).add(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        subs = self._queues.get(job_id)
        if subs:
            subs.discard(q)
            if not subs:
                del self._queues[job_id]

    def emit(self, job_id: str, event: dict) -> None:
        for q in self._queues.get(job_id, set()):
            q.put_nowait(event)


def make_cache_key(prompt_version: str, model: str, target_lang: str, kind: str, payload: bytes | str) -> str:
    h = hashlib.sha256()
    for part in (prompt_version, model, target_lang, kind):
        h.update(part.encode("utf-8"))
        h.update(b"|")
    h.update(payload.encode("utf-8") if isinstance(payload, str) else payload)
    return h.hexdigest()


def _sanity_check_chunks(block: dict) -> None:
    if block.get("type") != "phrase":
        return
    covered = sum(len(c["original"]) for c in block.get("chunks", []))
    total = len(block.get("original", ""))
    if total and covered / total < 0.7:
        logger.warning(
            "Low chunk coverage (%.0f%%) for phrase: %r",
            100 * covered / total,
            block.get("original", "")[:80],
        )


def _find_source_file(job_dir):
    candidates = sorted(p for p in job_dir.glob("source.*") if p.name != "source.txt")
    if not candidates:
        candidates = sorted(job_dir.glob("source.*"))
    if not candidates:
        raise FileNotFoundError(f"Source file not found in {job_dir}")
    return candidates[0]


async def process_job(job_id: str, storage: Storage, settings: Settings, llm: LLMClient, events: EventBus) -> None:
    job = storage.get_job(job_id)
    if job is None:
        return

    storage.set_job_status(job_id, "running")
    job_dir = storage.job_dir(job_id)
    source_type = job["source_type"]
    target_lang = job["target_lang"]
    force_vision = bool(job["force_vision"])

    try:
        adapter = get_adapter(source_type, settings, force_vision=force_vision)
        if source_type in ("pdf", "docx", "image"):
            source_arg = _find_source_file(job_dir)
        else:
            source_arg = (job_dir / "source.txt").read_text(encoding="utf-8")
        pages: list[PageContent] = adapter.load(source_arg)
    except Exception as exc:
        logger.exception("Failed to load source for job %s", job_id)
        storage.set_job_status(job_id, "failed", error=str(exc))
        events.emit(job_id, {"event": "job", "status": "failed"})
        return

    total_pages = len(pages)
    storage.set_job_status(job_id, "running", total_pages=total_pages)

    existing = {p["page_number"]: p for p in storage.get_pages(job_id)}
    for page in pages:
        if page.page_number not in existing:
            storage.upsert_page(job_id, page.page_number, "pending")

    glossary: dict[str, str] = {}
    summaries: list[str] = []
    page_records: dict[int, dict] = {}
    for pn in sorted(existing):
        row = existing[pn]
        if row["status"] in ("done", "failed") and row["result_json"]:
            record = json.loads(row["result_json"])
            page_records[pn] = record
            if row["status"] == "done" and record.get("result"):
                pr = PageResult.model_validate(record["result"])
                for term in pr.new_terms:
                    glossary.setdefault(term.term, term.translation)
                summaries.append(pr.page_summary)

    if len(glossary) > settings.glossary_max_terms:
        glossary = dict(list(glossary.items())[: settings.glossary_max_terms])
    first_summary = summaries[0] if summaries else None

    for page in pages:
        pn = page.page_number
        if pn in page_records:
            continue

        if page.kind == "image" and page.image_png:
            pages_dir = job_dir / "pages"
            pages_dir.mkdir(parents=True, exist_ok=True)
            (pages_dir / f"{pn:03d}.png").write_bytes(page.image_png)

        recent = summaries[-settings.context_summaries :]
        context = build_context_block(first_summary, recent, glossary)
        system = system_prompt(target_lang)

        cache_kind = "page_vision" if page.kind == "image" else "page_text"
        payload: bytes | str = page.image_png if page.kind == "image" else (page.text or "")
        cache_key = make_cache_key(PROMPT_VERSION, settings.llm_model, target_lang, cache_kind, payload)

        cached = storage.cache_get(cache_key)
        try:
            if cached is not None:
                page_result = PageResult.model_validate_json(cached)
            elif page.kind == "text":
                user_text = user_text_page(page.text or "", context)
                page_result = await call_structured(llm, system, user_text, PageResult)
                storage.cache_put(cache_key, page_result.model_dump_json())
            else:
                user_text = user_vision_page(context)
                page_result = await call_structured(llm, system, user_text, PageResult, image_png=page.image_png)
                storage.cache_put(cache_key, page_result.model_dump_json())
        except Exception as exc:
            logger.warning("Page %s of job %s failed: %s", pn, job_id, exc)
            record = {"page_number": pn, "status": "failed", "error": str(exc)}
            page_records[pn] = record
            storage.upsert_page(job_id, pn, "failed", result_json=json.dumps(record), error=str(exc))
            events.emit(job_id, {"event": "page", "page_number": pn, "status": "failed"})
            continue

        for block in page_result.text_blocks:
            _sanity_check_chunks(block.model_dump())

        image_annotations = list(page_result.image_annotations)
        inline_meta = []
        for idx, inline_img in enumerate(page.inline_images):
            image_dir = job_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"images/{pn:03d}_{idx:02d}.png"
            (job_dir / file_name).write_bytes(inline_img.image_png)
            inline_meta.append({"file": file_name, "position_anchor": inline_img.position_anchor})

            img_cache_key = make_cache_key(
                PROMPT_VERSION, settings.llm_model, target_lang, "inline_image", inline_img.image_png
            )
            img_cached = storage.cache_get(img_cache_key)
            try:
                if img_cached is not None:
                    labels_result = ImageLabelsResult.model_validate_json(img_cached)
                else:
                    user_text = user_inline_image(context)
                    labels_result = await call_structured(
                        llm, system, user_text, ImageLabelsResult, image_png=inline_img.image_png
                    )
                    storage.cache_put(img_cache_key, labels_result.model_dump_json())
            except Exception as exc:
                logger.warning("Inline image %d on page %s failed: %s", idx, pn, exc)
                labels_result = ImageLabelsResult(labels=[])

            if labels_result.labels:
                image_annotations.append(ImageAnnotation(image_ref=idx, labels=labels_result.labels))

        final_result = page_result.model_copy(update={"image_annotations": image_annotations})
        record: dict = {
            "page_number": pn,
            "status": "done",
            "result": json.loads(final_result.model_dump_json()),
            "inline_images": inline_meta,
        }
        if page.kind == "text" and inline_meta:
            record["source_text"] = page.text

        page_records[pn] = record
        storage.upsert_page(job_id, pn, "done", result_json=json.dumps(record), cache_key=cache_key)

        for term in final_result.new_terms:
            glossary.setdefault(term.term, term.translation)
        if len(glossary) > settings.glossary_max_terms:
            glossary = dict(list(glossary.items())[: settings.glossary_max_terms])
        summaries.append(final_result.page_summary)
        if first_summary is None:
            first_summary = final_result.page_summary

        events.emit(
            job_id,
            {"event": "page", "page_number": pn, "status": "done", "blocks": len(final_result.text_blocks)},
        )

    result_doc = {
        "job_id": job_id,
        "source_name": job["source_name"],
        "target_lang": target_lang,
        "model": settings.llm_model,
        "prompt_version": PROMPT_VERSION,
        "pages": [page_records[pn] for pn in sorted(page_records)],
    }
    (job_dir / "result.json").write_text(
        json.dumps(result_doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    html = render_html(result_doc, job_dir)
    (job_dir / "result.html").write_text(html, encoding="utf-8")

    done_count = sum(1 for r in page_records.values() if r.get("status") == "done")
    if done_count == 0 and total_pages > 0:
        storage.set_job_status(job_id, "failed", error="No page was processed successfully")
        events.emit(job_id, {"event": "job", "status": "failed"})
    else:
        storage.set_job_status(job_id, "done")
        events.emit(job_id, {"event": "job", "status": "done"})
