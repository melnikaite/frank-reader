import json
import time

from fastapi.testclient import TestClient

from frank_reader.config import Settings
from frank_reader.main import create_app
from frank_reader.pipeline.llm_client import FakeLLM

PAGE_JSON = json.dumps(
    {
        "page_summary": "s",
        "detected_language": "en",
        "text_blocks": [{"order": 1, "type": "heading", "original": "Hi", "translation": "Привет"}],
    }
)


def _wait_for_status(client, job_id, target_statuses, timeout=5.0):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        resp = client.get(f"/jobs/{job_id}")
        body = resp.json()
        if body["status"] in target_statuses:
            return body
        time.sleep(0.02)
    raise AssertionError(f"job did not reach {target_statuses} within {timeout}s")


def test_text_job_completes(tmp_path):
    settings = Settings(data_dir=tmp_path)
    app = create_app(settings=settings, llm=FakeLLM([PAGE_JSON]))
    with TestClient(app) as client:
        resp = client.post("/jobs", data={"text": "Короткий текст документа."})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        job = _wait_for_status(client, job_id, {"done", "failed"})
        assert job["status"] == "done"

        html_resp = client.get(f"/jobs/{job_id}/result.html")
        assert html_resp.status_code == 200
        assert "Привет" in html_resp.text

        json_resp = client.get(f"/jobs/{job_id}/result.json")
        assert json_resp.status_code == 200
        assert json_resp.json()["job_id"] == job_id


def test_missing_source_returns_422(tmp_path):
    app = create_app(settings=Settings(data_dir=tmp_path), llm=FakeLLM([]))
    with TestClient(app) as client:
        resp = client.post("/jobs", data={})
        assert resp.status_code == 422


def test_two_sources_returns_422(tmp_path):
    app = create_app(settings=Settings(data_dir=tmp_path), llm=FakeLLM([]))
    with TestClient(app) as client:
        resp = client.post("/jobs", data={"url": "http://example.com", "text": "x"})
        assert resp.status_code == 422


def test_result_not_ready_returns_404(tmp_path):
    app = create_app(settings=Settings(data_dir=tmp_path), llm=FakeLLM([]))
    with TestClient(app) as client:
        resp = client.get("/jobs/does-not-exist/result.html")
        assert resp.status_code == 404


def test_result_pdf_is_generated(tmp_path):
    settings = Settings(data_dir=tmp_path)
    app = create_app(settings=settings, llm=FakeLLM([PAGE_JSON]))
    with TestClient(app) as client:
        resp = client.post("/jobs", data={"text": "Короткий текст документа."})
        job_id = resp.json()["job_id"]
        _wait_for_status(client, job_id, {"done", "failed"})

        pdf_resp = client.get(f"/jobs/{job_id}/result.pdf")
        assert pdf_resp.status_code == 200
        assert pdf_resp.headers["content-type"] == "application/pdf"
        assert pdf_resp.content.startswith(b"%PDF")


def test_health_endpoint_does_not_500_without_preflight(tmp_path):
    app = create_app(settings=Settings(data_dir=tmp_path), llm=FakeLLM([]))
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["llm_reachable"] is False


def test_sse_events_replay_for_completed_job(tmp_path):
    settings = Settings(data_dir=tmp_path)
    app = create_app(settings=settings, llm=FakeLLM([PAGE_JSON]))
    with TestClient(app) as client:
        resp = client.post("/jobs", data={"text": "Короткий текст документа."})
        job_id = resp.json()["job_id"]
        _wait_for_status(client, job_id, {"done", "failed"})

        events_resp = client.get(f"/jobs/{job_id}/events")
        assert events_resp.status_code == 200
        assert "page" in events_resp.text
        assert "job" in events_resp.text
        assert "done" in events_resp.text
