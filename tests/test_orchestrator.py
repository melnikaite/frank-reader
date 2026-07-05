import json
import shutil

import pytest

from frank_reader.config import Settings
from frank_reader.pipeline.llm_client import FakeLLM
from frank_reader.pipeline.orchestrator import EventBus, process_job
from frank_reader.storage import Storage

PAGE1_JSON = json.dumps(
    {
        "page_summary": "Summary of page 1.",
        "detected_language": "en",
        "text_blocks": [{"order": 1, "type": "heading", "original": "Title", "translation": "Заголовок"}],
        "new_terms": [{"term": "Foo", "translation": "Фу"}],
    }
)

PAGE2_JSON = json.dumps(
    {
        "page_summary": "Summary of page 2.",
        "detected_language": "en",
        "text_blocks": [{"order": 1, "type": "heading", "original": "Title2", "translation": "Заголовок2"}],
    }
)

IMAGE_LABELS_JSON = json.dumps({"labels": [{"original": "Abb", "translation": "Рис"}]})


def _make_two_page_text_job(tmp_path, job_id="job1"):
    storage = Storage(tmp_path)
    storage.create_job(job_id, "text", "notes.txt", "ru")
    job_dir = storage.job_dir(job_id)
    # two paragraphs, each well under 60 chars, forced onto separate pseudo-pages
    text = "Первый параграф текста короткий.\n\nВторой параграф текста тоже короткий."
    (job_dir / "source.txt").write_text(text, encoding="utf-8")
    return storage, job_dir


async def test_happy_path_two_pages_done(tmp_path):
    storage, job_dir = _make_two_page_text_job(tmp_path)
    settings = Settings(pseudo_page_chars=40)
    llm = FakeLLM([PAGE1_JSON, PAGE2_JSON])
    events = EventBus()

    await process_job("job1", storage, settings, llm, events)

    job = storage.get_job("job1")
    assert job["status"] == "done"
    pages = storage.get_pages("job1")
    assert len(pages) == 2
    assert all(p["status"] == "done" for p in pages)
    assert (job_dir / "result.json").exists()
    assert (job_dir / "result.html").exists()
    # glossary from page 1's new_terms threaded into page 2's prompt
    assert len(llm.calls) == 2
    assert "Фу" in llm.calls[1]["user_text"]


async def test_failed_page_does_not_stop_job(tmp_path):
    storage, job_dir = _make_two_page_text_job(tmp_path)
    settings = Settings(pseudo_page_chars=40)
    llm = FakeLLM(["not json", "still not json", PAGE2_JSON])
    events = EventBus()

    await process_job("job1", storage, settings, llm, events)

    job = storage.get_job("job1")
    assert job["status"] == "done"  # at least one page succeeded
    pages = {p["page_number"]: p for p in storage.get_pages("job1")}
    assert pages[1]["status"] == "failed"
    assert pages[2]["status"] == "done"


async def test_cache_hit_avoids_llm_call(tmp_path):
    storage, _ = _make_two_page_text_job(tmp_path, job_id="job1")
    settings = Settings(pseudo_page_chars=40)
    llm1 = FakeLLM([PAGE1_JSON, PAGE2_JSON])
    await process_job("job1", storage, settings, llm1, EventBus())

    # a second job with identical page content should hit the cache entirely
    storage2, _ = _make_two_page_text_job(tmp_path, job_id="job2")
    # job2 shares the same sqlite/data dir as storage (same tmp_path) so the cache is shared
    llm2 = FakeLLM([])  # any call would raise "exhausted"
    await process_job("job2", storage, settings, llm2, EventBus())

    job2 = storage.get_job("job2")
    assert job2["status"] == "done"
    assert len(llm2.calls) == 0


async def test_resume_skips_already_done_pages(tmp_path):
    storage, job_dir = _make_two_page_text_job(tmp_path)
    settings = Settings(pseudo_page_chars=40)

    page1_record = {
        "page_number": 1,
        "status": "done",
        "result": json.loads(PAGE1_JSON),
        "inline_images": [],
    }
    storage.upsert_page("job1", 1, "done", result_json=json.dumps(page1_record))

    llm = FakeLLM([PAGE2_JSON])
    await process_job("job1", storage, settings, llm, EventBus())

    assert len(llm.calls) == 1  # only page 2 was processed
    job = storage.get_job("job1")
    assert job["status"] == "done"
    pages = {p["page_number"]: p for p in storage.get_pages("job1")}
    assert pages[1]["status"] == "done"
    assert pages[2]["status"] == "done"


async def test_inline_image_gets_separate_vision_call_and_annotation(tmp_path, pdf_with_inline_images):
    storage = Storage(tmp_path)
    storage.create_job("job1", "pdf", "doc.pdf", "ru")
    job_dir = storage.job_dir("job1")
    shutil.copy(pdf_with_inline_images, job_dir / "source.pdf")

    def responder(system, user_text, has_image):
        return IMAGE_LABELS_JSON if has_image else PAGE1_JSON

    llm = FakeLLM(responder)
    settings = Settings()

    await process_job("job1", storage, settings, llm, EventBus())

    job = storage.get_job("job1")
    assert job["status"] == "done"
    pages = storage.get_pages("job1")
    record = json.loads(pages[0]["result_json"])
    assert len(record["inline_images"]) == 1
    assert record["result"]["image_annotations"]
    assert record["result"]["image_annotations"][0]["labels"][0]["translation"] == "Рис"


async def test_inline_image_error_is_non_fatal(tmp_path, pdf_with_inline_images):
    storage = Storage(tmp_path)
    storage.create_job("job1", "pdf", "doc.pdf", "ru")
    job_dir = storage.job_dir("job1")
    shutil.copy(pdf_with_inline_images, job_dir / "source.pdf")

    def responder(system, user_text, has_image):
        return "not valid json at all" if has_image else PAGE1_JSON

    llm = FakeLLM(responder)
    settings = Settings()

    await process_job("job1", storage, settings, llm, EventBus())

    job = storage.get_job("job1")
    assert job["status"] == "done"
    pages = storage.get_pages("job1")
    record = json.loads(pages[0]["result_json"])
    assert record["status"] == "done"
    assert record["result"]["image_annotations"] == []
