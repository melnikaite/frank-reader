import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from frank_reader.adapters import detect_source_type
from frank_reader.config import Settings
from frank_reader.pipeline.llm_client import LLMClient, LocalAIClient
from frank_reader.pipeline.orchestrator import EventBus, process_job
from frank_reader.render.pdf import PdfNotAvailable, render_pdf
from frank_reader.storage import Storage

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
_STATIC_DIR = Path(__file__).resolve().parent / "static"


async def _worker_loop(app: FastAPI) -> None:
    queue: asyncio.Queue = app.state.queue
    while True:
        job_id = await queue.get()
        try:
            await process_job(job_id, app.state.storage, app.state.settings, app.state.llm, app.state.events)
        except Exception:
            logger.exception("Unhandled error processing job %s", job_id)
        finally:
            queue.task_done()


async def _preflight_result(app: FastAPI) -> dict:
    if hasattr(app.state.llm, "preflight"):
        return await app.state.llm.preflight()
    return {
        "llm_reachable": False,
        "model_loaded": False,
        "model": app.state.settings.llm_model,
        "available_models": [],
    }


def create_app(settings: Settings | None = None, llm: LLMClient | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.storage.mark_running_as_interrupted()
        worker_task = asyncio.create_task(_worker_loop(app))
        try:
            yield
        finally:
            worker_task.cancel()
            if isinstance(app.state.llm, LocalAIClient):
                await app.state.llm.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.settings = settings
    app.state.storage = Storage(settings.data_dir)
    app.state.llm = llm or LocalAIClient(settings)
    app.state.events = EventBus()
    app.state.queue = asyncio.Queue()

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/")
    async def index():
        index_path = _STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return HTMLResponse("<h1>Frank Reader</h1>")

    @app.get("/health")
    async def health():
        return await _preflight_result(app)

    @app.post("/jobs")
    async def create_job_endpoint(
        file: UploadFile | None = File(None),
        url: str | None = Form(None),
        text: str | None = Form(None),
        target_lang: str | None = Form(None),
        force_vision: bool = Form(False),
    ):
        provided = [x for x in (file, url, text) if x]
        if len(provided) != 1:
            raise HTTPException(422, "Provide exactly one source: file, url or text")

        target_lang = target_lang or settings.target_lang_default
        job_id = uuid.uuid4().hex
        storage: Storage = app.state.storage

        if file is not None:
            data = await file.read()
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "File exceeds the 100 MB limit")
            try:
                source_type = detect_source_type(file.filename or "")
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            storage.create_job(job_id, source_type, file.filename or "upload", target_lang, force_vision)
            job_dir = storage.job_dir(job_id)
            ext = Path(file.filename or "").suffix.lower()
            (job_dir / f"source{ext}").write_bytes(data)
        elif url is not None:
            storage.create_job(job_id, "url", url, target_lang, force_vision)
            job_dir = storage.job_dir(job_id)
            (job_dir / "source.txt").write_text(url, encoding="utf-8")
        else:
            storage.create_job(job_id, "text", "text", target_lang, force_vision)
            job_dir = storage.job_dir(job_id)
            (job_dir / "source.txt").write_text(text or "", encoding="utf-8")

        await app.state.queue.put(job_id)
        return {"job_id": job_id}

    @app.get("/jobs")
    async def list_jobs_endpoint():
        jobs = app.state.storage.list_jobs()
        result = []
        for job in jobs:
            pages = app.state.storage.get_pages(job["id"])
            done_pages = sum(1 for p in pages if p["status"] == "done")
            result.append({**job, "done_pages": done_pages})
        return result

    @app.get("/jobs/{job_id}")
    async def get_job_endpoint(job_id: str):
        job = app.state.storage.get_job(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job

    @app.post("/jobs/{job_id}/resume")
    async def resume_job_endpoint(job_id: str):
        job = app.state.storage.get_job(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        if job["status"] not in ("interrupted", "failed"):
            raise HTTPException(409, "Job is not in a resumable state")
        await app.state.queue.put(job_id)
        return JSONResponse({"status": "queued"}, status_code=202)

    @app.get("/jobs/{job_id}/events")
    async def job_events(job_id: str):
        job = app.state.storage.get_job(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")

        async def event_generator():
            pages = app.state.storage.get_pages(job_id)
            for p in pages:
                yield {
                    "event": "page",
                    "data": json.dumps({"page_number": p["page_number"], "status": p["status"]}),
                }
            current = app.state.storage.get_job(job_id)
            if current["status"] in ("done", "failed", "interrupted"):
                yield {"event": "job", "data": json.dumps({"status": current["status"]})}
                return
            q = app.state.events.subscribe(job_id)
            try:
                while True:
                    event = await q.get()
                    yield {"event": event["event"], "data": json.dumps(event)}
                    if event["event"] == "job":
                        break
            finally:
                app.state.events.unsubscribe(job_id, q)

        return EventSourceResponse(event_generator())

    @app.get("/jobs/{job_id}/result.html")
    async def get_result_html(job_id: str):
        path = app.state.storage.job_dir(job_id) / "result.html"
        if not path.exists():
            raise HTTPException(404, "Result is not ready yet")
        return FileResponse(path, media_type="text/html")

    @app.get("/jobs/{job_id}/result.json")
    async def get_result_json(job_id: str):
        path = app.state.storage.job_dir(job_id) / "result.json"
        if not path.exists():
            raise HTTPException(404, "Result is not ready yet")
        return FileResponse(path, media_type="application/json")

    @app.get("/jobs/{job_id}/result.pdf")
    async def get_result_pdf(job_id: str):
        html_path = app.state.storage.job_dir(job_id) / "result.html"
        if not html_path.exists():
            raise HTTPException(404, "Result is not ready yet")
        try:
            pdf_bytes = render_pdf(html_path.read_text(encoding="utf-8"))
        except PdfNotAvailable as exc:
            raise HTTPException(501, str(exc)) from exc
        return Response(content=pdf_bytes, media_type="application/pdf")

    return app


app = create_app()
