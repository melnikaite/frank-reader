# Frank Reader — Implementation Spec v1.0

Self-contained specification for implementation. Read in full before starting work. Implement strictly in the order given in section 13. All decisions have already been made — do not invent alternatives; if something is technically impossible, pick the closest working option and leave a `# DEVIATION:` comment with the reason.

## 0. What this is

A local single-user web tool: the user uploads a document (PDF / DOCX / image / URL / text) → gets HTML following Ilya Frank's reading method: each phrase of the original with translations of its meaningful chunks in parentheses, followed by a repeat of the phrase without translation:

> Die Würde des Menschen (достоинство человека) ist unantastbar (неприкосновенно).
> Die Würde des Menschen ist unantastbar.

LLM inference is local, via an OpenAI-compatible API (LocalAI). Multimodal model: text pages go in as text, scans and images go through vision.

### Principles (important)

1. **The canonical artifact is `result.json`** (validated `PageResult` per page). HTML is a pure function of it. Any new output format is a new renderer over the same JSON, without re-running the LLM.
2. **The pipeline never fails as a whole**: an error on a page → the page is `failed`, processing continues.
3. **Everything survives a server restart**: page statuses in SQLite, LLM response cache keyed by hash, resume re-renders finished work for free and completes the rest.
4. **Tests do not require an LLM**: `FakeLLM` + programmatically generated fixtures. The real LLM is only used in the M0 script and an opt-in integration test.

### Do not build (out of scope)

Image inpainting, handwriting OCR, preserving original layout (columns are flattened into one flow), bbox coordinates of labels on images, auth/multi-user, parallel LLM requests (LocalAI on Metal processes sequentially), EPUB/PPTX/SRT adapters.

## 1. Environment (verified facts)

- macOS, Apple Silicon; uv 0.11+; Python 3.14 on the system.
- LocalAI: `http://127.0.0.1:1240/v1`, model `gemma-4-e4b-it-qat-q4_0` (multimodal, context 131072).
- `response_format={"type": "json_object"}` — **works** (verified with a request), but the repair loop is still mandatory.
- Vision requests: standard OpenAI format, `content` as a list of parts, image as `{"type": "image_url", "image_url": {"url": "data:image/png;base64,...."}}`.

### Known quirks of Gemma 4 via LocalAI (verified in the neighboring tldr / nt / voice-assistant projects)

Mandatory to account for, non-negotiable:

1. **Thinking mode eats the token budget.** Gemma 4 spontaneously enters reasoning even with `reasoning_effort="low"`. Working solution: pass **`reasoning_effort: "none"`** in the body of every request — LocalAI/llama.cpp support this field (the OpenAI API does not; when switching endpoints, drop the field via config).
2. **The answer may arrive in `reasoning_content` with an empty `content`.** Text extraction: first `message.content`; if empty — the last paragraph of `message.reasoning_content`; if that is empty too — treat as an error.
3. **Bare array instead of an object.** LocalAI sometimes ignores json_object and returns `[...]` instead of `{"labels": [...]}`. After stripping fences: if the parsed JSON is a list and the schema expects an object with a single list field (`ImageLabelsResult`) — wrap automatically.
4. **Markdown fences** around JSON — always strip (see `_strip_fences`).
5. **Silent truncation at the context limit.** Exceeding the context window produces no error — the text is simply cut off. Our pages are small (≤ `pseudo_page_chars`), but when assembling context (glossary + summaries), keep total input < 20k tokens (chars/4).
6. **Chat-template markers in input text** (`<start_of_turn>`, `<|im_start|>`, etc.) break the dialogue markup — a document may contain them (or contain a prompt-injection attempt). Sanitize document text before inserting into the prompt (section 9).
7. **Renaming/unloading the model in LocalAI** leads to silent 404s on all calls. At application startup — preflight `GET /v1/models`: if the configured model is missing — `logger.critical` listing the available ones (do not crash: LocalAI may load the model lazily).
8. **Output cut off at `max_tokens` looks like the model "losing" the end of the page.** The Frank method inflates response volume (original + chunk-by-chunk translation + phrase repeat) to ~1.2–1.5x relative to the number of input characters on this model — a low `max_tokens` yields invalid truncated JSON, not an error. Details, measured values, and final defaults — see the "Token budget and chunk size" sidebar in section 4.

## 2. pyproject.toml

```toml
[project]
name = "frank-reader"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi",
    "uvicorn[standard]",
    "httpx",
    "pydantic",
    "pydantic-settings",
    "pymupdf",
    "python-docx",
    "trafilatura",
    "sse-starlette",
    "python-multipart",
    "charset-normalizer",
    "pillow",
    "xhtml2pdf>=0.2.17",
]

[dependency-groups]
dev = ["pytest", "pytest-asyncio", "anyio"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

If some dependency does not resolve under Python 3.14 — `uv python pin 3.12` and continue.

## 3. Project structure

```
frank-reader/
├── pyproject.toml
├── README.md
├── src/frank_reader/
│   ├── __init__.py
│   ├── main.py               # FastAPI app, endpoints, SSE, worker startup
│   ├── config.py             # pydantic-settings
│   ├── adapters/
│   │   ├── __init__.py       # get_adapter(source_type) -> SourceAdapter
│   │   ├── base.py           # PageContent, InlineImage, SourceAdapter protocol
│   │   ├── pdf.py
│   │   ├── docx.py
│   │   ├── image.py
│   │   ├── url.py
│   │   └── plaintext.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── schema.py         # pydantic models of the LLM response
│   │   ├── prompts.py        # prompt texts, PROMPT_VERSION
│   │   ├── llm_client.py     # LLMClient protocol, LocalAIClient, FakeLLM, repair
│   │   └── orchestrator.py   # per-page loop, glossary, cache, resume
│   ├── render/
│   │   ├── __init__.py
│   │   ├── html.py
│   │   ├── pdf.py            # xhtml2pdf (pure Python, no system libs)
│   │   └── fonts/            # Noto Sans (SIL OFL) — Cyrillic for PDF, see 11.2
│   └── storage.py            # SQLite + file layout
├── static/
│   └── index.html            # HTMX UI
├── scripts/
│   └── m0_check.py           # manual quality check against the real LLM
└── tests/
    ├── conftest.py           # PDF fixture generation, FakeLLM, tmp data dir
    ├── test_schema.py
    ├── test_llm_client.py
    ├── test_storage.py
    ├── test_adapters.py
    ├── test_orchestrator.py
    ├── test_render.py
    ├── test_api.py
    └── test_integration_llm.py   # skip unless FRANK_INTEGRATION=1
```

## 4. config.py

`pydantic-settings`, prefix `FRANK_`, `.env` support:

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRANK_", env_file=".env")

    llm_base_url: str = "http://127.0.0.1:1240/v1"
    llm_model: str = "gemma-4-e4b-it-qat-q4_0"
    llm_api_key: str = "not-needed"          # for compatibility with cloud endpoints
    llm_timeout_text: float = 180.0          # seconds
    llm_timeout_vision: float = 420.0
    llm_temperature: float = 0.2
    llm_max_tokens: int = 16384               # see the "Token budget" sidebar below — a low value silently truncates JSON
    llm_reasoning_effort: str | None = "none"  # Gemma 4: suppress thinking; None = do not send the field (OpenAI)

    # optional separate endpoint for vision (None = main one)
    llm_vision_base_url: str | None = None
    llm_vision_model: str | None = None

    data_dir: Path = Path("data")
    target_lang_default: str = "ru"

    # PDF heuristics
    min_text_chars: int = 50                 # fewer → the page is considered a scan
    garbage_char_ratio: float = 0.10         # share of � and non-printables → garbage
    scan_image_coverage: float = 0.80        # image covers >80% → scan

    # images
    page_render_max_dim: int = 1400          # px, long side of the page render
    inline_image_min_dim: int = 64           # px, smaller — skip (decoration)
    inline_image_min_area: float = 0.02      # share of page area, smaller — skip

    # text sources
    pseudo_page_chars: int = 2500             # slicing long text into pseudo-pages, see sidebar below

    glossary_max_terms: int = 200
    context_summaries: int = 3               # how many recent page_summary values to pass
```

Do not make a global `settings = Settings()` singleton — create it in `main.py` and pass explicitly (testability).

### Token budget and chunk size (important, verified on the real model)

`gemma-4-e4b-it-qat-q4_0` via LocalAI is a small quantized model. With the Frank method, the response volume (original + chunk-by-chunk translation + phrase repeat in JSON) is substantially larger than the input text — measured on this model at **~1.2–1.5 output tokens per input character** (legal/formal text closer to 1.2, dialogue-heavy prose to 1.5, due to finer segmentation into chunks). With `llm_max_tokens=4096`, a page of ~4000–6000 input characters is already guaranteed not to fit: the JSON gets cut off midway, the page falls into the repair loop, and often fails even after it (the second attempt also runs out of budget) — the model appears to be "losing" the end of the page, when in fact it is not the model cutting the thought short but `max_tokens` cutting it off.

The fix — both compounding:
1. **`llm_max_tokens=16384`** with plenty of headroom. This does not slow down processing of pages that fit a smaller budget — the model stops at `finish_reason=stop` as soon as it is done; the high ceiling costs time only on pages that actually need it.
2. **`pseudo_page_chars=2500`** — small models are noticeably more reliable with input of this size: in a check against real texts (10 Grundgesetz articles, a real Brothers Grimm fairy tale), chunks of ~2500 characters gave ~99% text coverage on the first attempt almost always without invoking the repair loop, whereas whole pages of 6000+ characters were either cut off at `max_tokens` or the model dropped mandatory JSON fields on dialogue-dense passages (a separate issue unrelated to token budget — it is handled by the already existing repair loop, but only if the budget is also sufficient for a retry).

**Important:** using `pseudo_page_chars` to cut pseudo-pages (plaintext/URL/DOCX) was already in v1. But the same must also be applied to **text extracted from PDF pages** (see 7 → `pdf.py`): previously a PDF page went to the LLM whole without a size limit, and a real dense A4 page easily yields 3000–6000+ characters of text — the same truncation risk. `PdfAdapter` must cut such pages via `build_pseudo_pages` exactly like DOCX/URL, recomputing continuous page numbering across the whole document (one PDF page may become several `PageContent`).

If the model is switched to a more powerful one in the future — these two values (`llm_max_tokens`, `pseudo_page_chars`) are the first candidates for revision towards larger chunks (fewer LLM calls = faster overall, at the cost of the same risk on a weak model).

## 5. Data model (pipeline/schema.py)

Full code, use as is:

```python
from typing import Literal
from pydantic import BaseModel, Field, model_validator

class Chunk(BaseModel):
    original: str
    translation: str

class TextBlock(BaseModel):
    order: int
    type: Literal["phrase", "heading", "list_item", "caption"]
    original: str
    translation: str | None = None   # required for heading/list_item/caption
    chunks: list[Chunk] = Field(default_factory=list)  # required for phrase

    @model_validator(mode="after")
    def _check_by_type(self):
        if self.type == "phrase" and not self.chunks:
            raise ValueError("phrase must have chunks")
        if self.type != "phrase" and not self.translation:
            raise ValueError(f"{self.type} must have translation")
        return self

class ImageLabel(BaseModel):
    original: str
    translation: str

class ImageAnnotation(BaseModel):
    image_ref: int                   # index into PageContent.inline_images
    labels: list[ImageLabel] = Field(default_factory=list)

class Term(BaseModel):
    term: str
    translation: str

class PageResult(BaseModel):
    page_summary: str
    detected_language: str           # ISO code, "de", "en"...
    text_blocks: list[TextBlock]
    image_annotations: list[ImageAnnotation] = Field(default_factory=list)
    new_terms: list[Term] = Field(default_factory=list)
```

Important: **the phrase repeat in the renderer is taken from `TextBlock.original`**, never assembled by concatenating `chunks` (chunks are not required to cover the sentence exactly).

Schema for the vision call on a single inline image (not to be confused with the page):

```python
class ImageLabelsResult(BaseModel):
    labels: list[ImageLabel] = Field(default_factory=list)
```

## 6. Storage (storage.py)

### File layout

```
{data_dir}/
├── frank.db
└── jobs/{job_id}/
    ├── source.{ext}          # source as uploaded (for url/text — source.txt with the URL/text)
    ├── pages/{NNN}.png       # renders of vision pages (debugging and cache keys), NNN = 001…
    ├── images/{NNN}_{MM}.png # inline images: page NNN, image MM
    ├── result.json           # canonical result (see below)
    └── result.html           # latest render
```

`result.json`:

```json
{
  "job_id": "...",
  "source_name": "grundgesetz.pdf",
  "target_lang": "ru",
  "model": "gemma-4-e4b-it-qat-q4_0",
  "prompt_version": "1",
  "pages": [
    {"page_number": 1, "status": "done", "result": { ...PageResult... },
     "inline_images": [{"file": "images/001_00.png", "position_anchor": 1234}]},
    {"page_number": 2, "status": "failed", "error": "invalid JSON after retry"}
  ]
}
```

### SQLite (WAL, `sqlite3` from stdlib, synchronous — calls are short)

Pragmas on every connection (a proven set from a neighboring project):

```sql
PRAGMA journal_mode=WAL;        -- concurrent reads during writes
PRAGMA synchronous=NORMAL;      -- reliability/speed balance for WAL
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=5000;
```

```sql

CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,             -- ISO-8601 UTC
  source_type TEXT NOT NULL,            -- pdf|docx|image|url|text
  source_name TEXT NOT NULL,
  target_lang TEXT NOT NULL,
  force_vision INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,                 -- pending|running|done|failed|interrupted
  total_pages INTEGER,
  error TEXT
);

CREATE TABLE IF NOT EXISTS pages (
  job_id TEXT NOT NULL REFERENCES jobs(id),
  page_number INTEGER NOT NULL,
  status TEXT NOT NULL,                 -- pending|done|failed
  cache_key TEXT,
  result_json TEXT,                     -- serialized PageResult
  error TEXT,
  PRIMARY KEY (job_id, page_number)
);

CREATE TABLE IF NOT EXISTS llm_cache (
  cache_key TEXT PRIMARY KEY,
  result_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
```

`Storage` — a class with methods: `create_job`, `get_job`, `list_jobs`, `set_job_status`, `upsert_page`, `get_pages`, `cache_get`, `cache_put`, `job_dir(job_id)`. One connection per call or one per object with `check_same_thread=False` — either is fine, but must be thread-safe with respect to the background worker.

**At application startup**: all jobs with status `running` → `interrupted` (the server restarted mid-work).

### Cache key

`sha256` of the concatenation: `prompt_version | model | target_lang | payload`, where payload = the page text (for the text path) or PNG bytes (for the vision path). For inline images — the same with the image's PNG bytes. Before an LLM call — lookup; after a successful one — put.

## 7. Adapters (adapters/)

### base.py

```python
@dataclass
class InlineImage:
    image_png: bytes
    position_anchor: int   # offset into PageContent.text where the image goes at render time

@dataclass
class PageContent:
    page_number: int                    # starting at 1
    kind: Literal["text", "image"]
    text: str | None = None             # for kind="text"
    image_png: bytes | None = None      # for kind="image"
    inline_images: list[InlineImage] = field(default_factory=list)

class SourceAdapter(Protocol):
    def load(self, source: Path | str) -> list[PageContent]: ...
```

`adapters/__init__.py`: `get_adapter(source_type: str, settings: Settings) -> SourceAdapter` + `detect_source_type(filename: str) -> str` by extension (`.pdf`→pdf, `.docx`→docx, `.png/.jpg/.jpeg/.webp`→image, `.txt`→text; otherwise ValueError).

### pdf.py — PdfAdapter

Per-page decision:

```python
text = page.get_text("text", sort=True).strip()
garbage = share of characters in the "non-printable + �" category in text
page_area_coverage = max area of a single image on the page / page area

if not force_vision and len(text) >= settings.min_text_chars and garbage < settings.garbage_char_ratio:
    kind = "text"
else:
    kind = "image"
```

- `force_vision` — a constructor parameter of the adapter (from job options); when True, all pages → `image`.
- kind="image": render the page to PNG via `page.get_pixmap(matrix=...)`, choose the scale so the long side ≈ `page_render_max_dim` px.
- kind="text": extract inline images: `page.get_images(full=True)` → for each `xref` pull the bytes (`fitz.Pixmap(doc, xref)`, convert to PNG; CMYK/alpha handled correctly via `fitz.Pixmap(fitz.csRGB, pix)`), filter out small ones (`inline_image_min_dim`, `inline_image_min_area` of page area). `position_anchor`: take the image's bbox on the page (`page.get_image_rects(xref)`), find the text block from `page.get_text("blocks", sort=True)` closest ABOVE the image, and set anchor = the offset of the end of that block within the assembled page text. If there is no block above — anchor = 0. If rects is empty — skip the image.
- Assemble the page text from the same `blocks` (join with `"\n\n"`), so that offsets stay consistent.
- **If the page text is longer than `pseudo_page_chars`** — run it (together with the already extracted inline images) through the same `build_pseudo_pages` from `adapters/_pseudopage.py` used to slice DOCX/URL/plaintext (see the "Token budget" sidebar in section 4). One PDF page then becomes several `PageContent`. `page_number` numbering is continuous across the whole document, not "PDF page = page_number": keep a `next_page_number` counter, increment it for every added `PageContent` (including every sub-page), and reassign `page_number` on the objects returned by `build_pseudo_pages` (their local 1-based numbering is meaningless).

### docx.py — DocxAdapter

`python-docx`. The whole document is **one** PageContent kind="text" (DOCX has no reliable pages), but if the text is longer than `pseudo_page_chars` — slice into pseudo-pages at paragraph boundaries. Text: paragraphs joined with `"\n\n"`, list items prefixed with `"- "`. Inline images: iterate `document.part.related_parts` with `content_type` image/*, anchor — the offset of the end of the paragraph in which the drawing was encountered (walk `document.element.body` in order; if matching fails — anchor at the end of the text). Do not overcomplicate: convert tables to text row by row (cells joined with ` | `).

### image.py — ImageAdapter

One page kind="image", file bytes as is (if not PNG — convert to PNG via PyMuPDF: `fitz.open(stream=...)` → pixmap). Resize to `page_render_max_dim` if larger.

### url.py — UrlAdapter

`trafilatura.fetch_url` + `trafilatura.extract(..., output_format="markdown", include_images=True)`. The result is text; strip markdown image links `![...](url)` from the text, download the images (httpx, 20s timeout, skip on error), anchor = the position where the link was. Then like PlainText: slicing into pseudo-pages. If trafilatura returns None/empty — ValueError with a clear message.

### plaintext.py — PlainTextAdapter

Input — a string or a .txt file (decoding via `charset_normalizer.from_bytes(...).best()`). Slice into pseudo-pages by `pseudo_page_chars`, cut at paragraph boundaries (`"\n\n"`); if a paragraph is gigantic — at a sentence boundary. kind="text", no images.

## 8. LLM client (pipeline/llm_client.py)

```python
class LLMClient(Protocol):
    async def complete(self, system: str, user_text: str,
                       image_png: bytes | None = None) -> str: ...
```

### LocalAIClient

`httpx.AsyncClient`, POST `{base_url}/chat/completions`:

- `messages`: system + user. With `image_png`, user.content = `[{"type": "text", "text": ...}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,<b64>"}}]`, otherwise — just a string. If `llm_vision_base_url/model` are set, vision requests go there.
- `response_format={"type": "json_object"}`, `temperature`, `max_tokens` from settings; **`reasoning_effort` from settings if not None** (quirk #1 from section 1 — without it Gemma burns max_tokens on thinking).
- Text extraction from the response (quirk #2): `message.content`; if empty — the last paragraph of `message.reasoning_content`; otherwise an exception.
- Timeout: `llm_timeout_vision` if there is an image, otherwise `llm_timeout_text`.
- Network errors/5xx: 2 retries with a 5s pause, then propagate the exception.
- After every successful call: `logger.info` with the model, prompt/completion tokens (from `usage`, if LocalAI returned it) and elapsed time — this is the only source of speed data, needed to estimate "how long will a 50-page scan take".
- `preflight()`: `GET {base_url}/models`; if `llm_model` is not in the list — `logger.critical("model X is not loaded, available: …")`, return False (called from lifespan, the app does not crash).

### parse_structured — parsing with a repair loop (a standalone function, works with any LLMClient)

```python
async def call_structured(client, system, user_text, schema: type[T],
                          image_png=None) -> T:
    raw = await client.complete(system, user_text, image_png)
    try:
        return schema.model_validate_json(_strip_fences(raw))
    except (ValidationError, ValueError):
        repair_user = (user_text
            + "\n\nYour previous answer failed validation. The answer was:\n" + raw[:3000]
            + "\n\nError: " + str(err)[:500]
            + "\n\nReturn STRICTLY valid JSON per the schema, no explanations, no markdown.")
        raw2 = await client.complete(system, repair_user, image_png)
        return schema.model_validate_json(_strip_fences(raw2))  # exception → page failed
```

`_strip_fences`: strip surrounding ```json …``` and text before the first `{` or `[` / after the last `}` or `]`.

Additionally in `call_structured`, before validation (quirk #3): if `json.loads` yielded a **list** and the schema is an object with a single list field, wrap it: `[...]` → `{"labels": [...]}` for `ImageLabelsResult`. For `PageResult` — do not wrap (the schema is too heterogeneous, let it go to repair).

### FakeLLM (imported by tests and by the orchestrator in tests)

The constructor takes a list of prepared response strings (returned in order) or a callable `(system, user_text, has_image) -> str`. Keeps a call log `calls: list[dict]` for assertions.

## 9. Prompts (pipeline/prompts.py)

`PROMPT_VERSION = "1"` — bump on any change to prompt texts (it is part of the cache key).

### sanitize_source_text(text) -> str

Applied to document text (and to text extracted by trafilatura/docx/pdf) before inserting into the user prompt — a document may contain chat-template markers or a prompt injection (quirk #6):

- Remove control characters except `\n`, `\t`.
- Neutralize template markers: `<start_of_turn>`, `<end_of_turn>`, `<|im_start|>`, `<|im_end|>`, `<bos>`, `<eos>` → replace with an empty string.
- Wrap the document text in the prompt with `<document>…</document>` tags; escape a closing `</document>` inside the text.

Instructions in the system prompt take priority, but document content is data, not commands; we translate everything, including phrases like "ignore previous instructions" (that is just text to translate).

Describe the schema with compact hand-written text (not a JSON Schema dump). All prompts are constants/functions assembling a string. The texts below are final for v1 (iterate based on M0 results).

### SYSTEM_FRANK

```
You are preparing an instructional text using Ilya Frank's reading method.
Rules of the method:
- The text is split into meaningful phrases (usually a sentence or a coherent part of it).
- Within a phrase, translation is given per meaningful chunk — a piece of the original immediately followed by its translation.
- Chunks are 2-6 words, along the boundaries of sense groups (subject+modifiers, verb+object, set phrases).
- The translation is as literal as clarity allows; preserve the original's order of thought.
- Translate terms consistently: if a term is in the glossary, use the glossary translation.
- Target translation language: {target_lang}. Detect the original language yourself.

Example segmentation (German → Russian):
Original: "Als Gregor Samsa eines Morgens aus unruhigen Träumen erwachte, fand er sich in seinem Bett zu einem ungeheueren Ungeziefer verwandelt."
chunks:
- "Als Gregor Samsa" → "когда Грегор Замза"
- "eines Morgens" → "однажды утром"
- "aus unruhigen Träumen erwachte" → "пробудился от беспокойных снов"
- "fand er sich in seinem Bett" → "обнаружил, что он у себя в постели"
- "zu einem ungeheueren Ungeziefer verwandelt" → "превратился в чудовищное насекомое"

Respond with STRICTLY valid JSON per the given schema. No markdown, no explanations, no text outside the JSON.
```

### SCHEMA_DESCRIPTION (interpolated into user prompts)

```
JSON response schema:
{
  "page_summary": "2-3 sentences about the page content",
  "detected_language": "ISO code of the original language, e.g. de",
  "text_blocks": [
    {"order": 1, "type": "phrase", "original": "<full phrase>",
     "chunks": [{"original": "<chunk>", "translation": "<chunk translation>"}]},
    {"order": 2, "type": "heading", "original": "<heading>", "translation": "<translation>"}
  ],
  "image_annotations": [],
  "new_terms": [{"term": "<term from the original>", "translation": "<chosen translation>"}]
}
Block types: phrase (regular text; chunks required), heading, list_item, caption
(these need no chunks, translation is required).
In new_terms include only recurring domain-specific terms (names, subject-area
concepts), not ordinary words.
```

### user_text_page(page_text, context) / user_vision_page(context)

Common context header:

```
Document context:
{summary of the first pages, or "(beginning of the document)"}
Recent pages: {last N page_summary joined with " / "}
Glossary (term → translation): {term: translation, ... or "(empty)"}
```

Text path — then:

```
Split the following page text into blocks and meaningful chunks, translate using the Frank method.
{SCHEMA_DESCRIPTION}

Page text:
<document>
{sanitize_source_text(page_text)}
</document>
```

Vision path — then:

```
The image shows a document page.
1. Recognize all the text in reading order (mind columns, if any).
2. Split into blocks and meaningful chunks per the Frank method, translate.
3. Compose page_summary.
{SCHEMA_DESCRIPTION}
```

### user_inline_image(context)

```
The image shows an illustration/diagram from the document.
Find all labels (captions, tags, legends) and translate each one.
JSON response schema: {"labels": [{"original": "<label>", "translation": "<translation>"}]}
If there are no labels — {"labels": []}.
```

## 10. Orchestrator (pipeline/orchestrator.py)

```python
async def process_job(job_id, storage, settings, llm, events):  # events: EventBus
    # 1. jobs.status = running; load source via the adapter → pages
    # 2. total_pages into jobs; upsert pending for each page (on resume — leave done alone)
    # 3. glossary: dict[str, str] = {}; summaries: list[str] = []
    #    on resume: restore glossary and summaries from done pages (walk result_json in order)
    # 4. loop over pages in order:
    for page in pages:
        if pages[n].status == "done": continue          # resume
        cache_key = make_cache_key(...)
        cached = storage.cache_get(cache_key)
        if cached: result = PageResult.model_validate_json(cached)
        else:
            try:
                result = await call_structured(llm, SYSTEM, user_prompt, PageResult,
                                               image_png=...)
            except Exception as e:
                pages[n] = failed(str(e)); events.emit(page_failed); continue
            storage.cache_put(cache_key, result)
        # inline images (kind="text"): for each — call_structured(ImageLabelsResult),
        #   separate cache key; an image error does NOT fail the page (labels=[])
        #   results → result.image_annotations (image_ref = image index)
        # glossary: for t in result.new_terms: glossary.setdefault(t.term, t.translation)
        #   (first translation wins — that IS the consistency); cap at glossary_max_terms
        # summaries.append(result.page_summary)
        # next page's context: summaries[0] + last settings.context_summaries
        # pages[n] = done(result); events.emit(page_done)
    # 5. write result.json (full, section 6), render result.html
    # 6. jobs.status = done (or failed, if zero done pages); events.emit(job_done)
```

Quality sanity check (does not fail, warning in the log only): for each phrase block, the concatenation of `chunk.original` (lowercased, whitespace collapsed) must cover ≥ 70% of the characters of `block.original`; otherwise `logger.warning`.

### EventBus (in main.py or standalone)

A dict `{job_id: set[asyncio.Queue]}`. `emit(job_id, event_dict)` puts into all queues. SSE endpoint: first a replay from the DB (all pages with their statuses), then a subscription to live events. Events: `{"event": "page", "page_number": n, "status": "done|failed", "blocks": <count>}`, `{"event": "job", "status": "done|failed"}`.

### Job queue

`asyncio.Queue[str]` (job_id) + one background `asyncio.Task` worker, started in the FastAPI lifespan. Sequential processing. POST /jobs puts into the queue; resume — likewise.

## 11. Rendering (render/html.py)

Function `render_html(result_json: dict, job_dir: Path) -> str`. Do not pull in Jinja — an f-string/template string is enough. Requirements:

- One HTML file; CSS and JS inline; images — **base64 data URIs** (self-contained, the file can be forwarded).
- Structure of a phrase block:
  ```html
  <div class="phrase">
    <p class="glossed">Kусок1 <span class="tr">(перевод1)</span> кусок2 <span class="tr">(перевод2)</span>…</p>
    <p class="plain">Оригинал фразы целиком (из block.original).</p>
  </div>
  ```
  The glossed line is assembled from chunks in order; plain is always `block.original`.
- heading → `<h2>original <span class="tr">(translation)</span></h2>`; list_item → `<li>` in a `<ul>` (group consecutive list_item into one list); caption → `<p class="caption">`.
- Translation styling: gray, slightly smaller than body text. Font — a system serif family (Georgia, serif), line spacing comfortable for reading, max-width ~46rem centered.
- A "hide translations" toggle: a fixed button, 3 lines of JS toggling the `translations-hidden` class on body; `.translations-hidden .tr {display: none}`. Self-testing mode.
- Inline images: insert after the block whose span covers `position_anchor` (algorithm below); `<img>` max-width 100%; below the image — annotations: `<ul class="img-labels"><li>label — <span class="tr">translation</span></li></ul>`.
- Vision pages (scans): do NOT insert the page image (the text is already recognized), blocks only.
- failed pages: a prominent banner `<div class="page-failed">Page N was not processed: {error}</div>` — do not skip silently.
- Page separator: a thin `<hr>` with the page number.
- Print CSS (`@media print`): hide the toggle, `page-break-inside: avoid` on `.phrase`, page margins. **Printing = the browser's ⌘P** — this is the primary path to a paper version.

### Image-to-block anchoring algorithm

```
cursor = 0; spans = []
norm(s) = s with whitespace collapsed
for block in text_blocks (by order):
    idx = page_text.find(block.original, cursor)      # page_text — the original page text
    if idx == -1: idx = cursor                        # fallback: the LLM rephrased
    spans.append((idx, idx + len(block.original), block))
    cursor = idx + len(block.original)
for each image: insert after the first block whose span_end >= position_anchor;
if there is none — after the last block of the page.
```

For this, `page_text` is stored in result.json per page (the `source_text` field, only for kind="text" pages with images; otherwise null — do not bloat the file).

### render/pdf.py

**Do not use WeasyPrint.** WeasyPrint technically installs via pip/uv, but at runtime it looks up native libraries (pango/cairo/gobject) via `ctypes`/`cffi.dlopen` along the system path — on macOS with Homebrew they are not picked up without a manual `DYLD_FALLBACK_LIBRARY_PATH`, which violates the requirement "all dependencies inside uv, no manual system steps". Verified on a real machine: `uv sync` installs WeasyPrint successfully, but `from weasyprint import HTML` fails with `OSError: cannot load library 'libgobject-2.0-0'` until the environment variable is set by hand.

Instead — **`xhtml2pdf`** (the `xhtml2pdf` package, transitively pulls `reportlab`/`pypdf`, pure Python + Pillow, no system libraries, no extra steps after `uv sync`).

**A Cyrillic font must be embedded.** The base `reportlab` fonts (Helvetica etc., as well as the bundled `Vera.ttf`) contain no Cyrillic glyphs — a Russian translation without an explicit font renders as empty rectangles (verified: `pypdf` text extraction from such a PDF yields `■■■■■` instead of Cyrillic). Solution: download and commit **Noto Sans** into the repository (Regular/Bold/Italic, SIL OFL license — free embedding) under `render/fonts/`, put the `OFL.txt` license file there too, and inject the font via `@font-face` with an absolute file path right before rendering (the path is derived from `Path(__file__).resolve().parent / "fonts"`, independent of the process cwd):

```python
from pathlib import Path

_FONTS_DIR = Path(__file__).resolve().parent / "fonts"

_FONT_FACE_CSS = f"""
<style>
  @font-face {{ font-family: "NotoSansFrank"; src: url({_FONTS_DIR / "NotoSans-Regular.ttf"}); }}
  @font-face {{ font-family: "NotoSansFrank"; font-weight: bold; src: url({_FONTS_DIR / "NotoSans-Bold.ttf"}); }}
  @font-face {{ font-family: "NotoSansFrank"; font-style: italic; src: url({_FONTS_DIR / "NotoSans-Italic.ttf"}); }}
  body, * {{ font-family: "NotoSansFrank", sans-serif !important; }}
</style>
"""


class PdfNotAvailable(Exception):
    pass


def render_pdf(html: str) -> bytes:
    from xhtml2pdf import pisa

    patched_html = html.replace("</head>", _FONT_FACE_CSS + "</head>")
    out = BytesIO()
    result = pisa.CreatePDF(patched_html, dest=out)
    if result.err:
        raise PdfNotAvailable(f"xhtml2pdf reported errors while rendering the PDF: {result.err}")
    return out.getvalue()
```

`xhtml2pdf` is always available (not an extra, a regular dependency): `result.pdf` works right after `uv sync` on any platform. `PdfNotAvailable` remains as an exception class (main.py still catches it and returns 501), but it now means "xhtml2pdf reported a render error", not "the backend is not installed" — that path should not normally occur.

`xhtml2pdf`'s CSS capabilities are noticeably more modest than WeasyPrint's (no flexbox/grid, limited support for modern selectors) — the markup in `render/html.py` must stay simple (blocks/lists/images, no complex layout) so the PDF path does not break.

## 12. API and UI

### Endpoints (main.py)

| Method | Path | Input | Output |
|---|---|---|---|
| GET | `/` | — | static/index.html |
| GET | `/health` | — | `{"llm_reachable": bool, "model_loaded": bool, "model": "...", "available_models": [...]}` — a live preflight() call |
| POST | `/jobs` | multipart: `file` (opt.) + form fields `url` (opt.), `text` (opt.), `target_lang` (default from settings), `force_vision` (bool). Exactly one of file/url/text, otherwise 422 | `{"job_id": "..."}` |
| GET | `/jobs` | — | `[{id, created_at, source_name, status, total_pages, done_pages}]`, newest first |
| GET | `/jobs/{id}` | — | same + error |
| POST | `/jobs/{id}/resume` | — | 202; puts into the queue if status ∈ {interrupted, failed} |
| GET | `/jobs/{id}/events` | — | SSE (sse-starlette): replay + live; closes after the job event |
| GET | `/jobs/{id}/result.html` | — | HTML (Content-Disposition inline); 404 if absent |
| GET | `/jobs/{id}/result.json` | — | the canonical JSON |
| GET | `/jobs/{id}/result.pdf` | — | PDF on demand from result.html via xhtml2pdf; always available after `uv sync`. 501 if the render itself failed (`PdfNotAvailable`) |

Upload limit: 100 MB, otherwise 413. Unknown extension — 422 with a list of supported ones.

### static/index.html

A single page, HTMX + ~50 lines of vanilla JS:

- Three tabs: "File" (drag&drop zone + file input), "URL" (input), "Text" (textarea). Common options: target language (text input, default `ru`), a "force vision" checkbox.
- Submit → POST /jobs (fetch/htmx) → got job_id → subscribe `new EventSource('/jobs/{id}/events')` → progress bar "page X/Y" + a list of pages with statuses.
- On the job done event → show links: "Open HTML", "result.json", "Download PDF".
- At the bottom — history: `GET /jobs`, a table with links to results and a resume button for interrupted.
- No build step, no npm: htmx from a CDN `<script src>` + no local fallback needed (the tool is local, but internet access for the CDN is available; plain CDN is acceptable).

## 13. Implementation order (follow the steps strictly)

Each step is complete when its tests are green (`uv run pytest tests/test_<step>.py`), and only then move to the next. At the end of each step, run the full suite.

1. **Skeleton.** `uv init`, pyproject from section 2, directory structure, `config.py`, `pipeline/schema.py`. Tests `test_schema.py`: a valid page parses; phrase without chunks → ValidationError; heading without translation → ValidationError; extra fields are ignored (pydantic default).
2. **LLM client.** `llm_client.py`: LocalAIClient, FakeLLM, `_strip_fences`, `call_structured`, `preflight`. Build the request body with a pure function `build_request_body(...)` — it is tested without a network. Tests `test_llm_client.py` (FakeLLM + build_request_body): valid JSON on the first try; JSON in a ```json fence — parses; invalid → a repair request went out (check calls) → valid on the second; two invalid → exception; a bare array for ImageLabelsResult — wrapped and parsed; the body contains `reasoning_effort="none"` with the default config and lacks the field when None; text extraction: empty content + populated reasoning_content → the last paragraph of reasoning_content is taken.
3. **Storage.** DB schema, methods, layout, `interrupted` at startup, cache. Tests `test_storage.py`: create/get/list job; upsert/get pages; cache put/get; running→interrupted.
4. **Adapters + fixtures.** conftest.py: a fresh SQLite DB per test — **a real file in tmp_path, not `:memory:`** (in-memory breaks when accessed from the background worker/threads); functions generating PDFs via PyMuPDF into a temp directory — (a) 2 text pages with paragraphs, (b) a "scan": a page with one inserted full-size image (`page.insert_image` with a placeholder PNG, no text), (c) mixed: text page + scan, (d) a text page with a small and a large inserted image, (e) a page with text several times longer than `pseudo_page_chars`, plus a separate short second PDF page. Tests `test_adapters.py`: pdf text → kind=text and the text is extracted; scan → kind=image and image_png is non-empty; mixed → [text, image]; force_vision → everything image; the large inline image ends up in inline_images with anchor > 0, the small one is filtered out; **a long PDF page is sliced into several `PageContent` via `build_pseudo_pages`, page numbering is continuous across the whole document (`page_number` goes 1..N without duplicates or gaps), the image from the original long page ends up in the correct sub-page, and the short second PDF page gets a correct (shifted) number**; plaintext slices pseudo-pages at paragraph boundaries; the image adapter returns 1 page; docx (created via python-docx in a fixture: paragraphs + a list) → text with "- " prefixes. UrlAdapter — test only the function processing already-extracted markdown (do not test the network fetch).
5. **Prompts.** prompts.py per section 9, including `sanitize_source_text`. Tests: chat-template markers (`<start_of_turn>`, `<|im_start|>`) are stripped; control characters are stripped, `\n`/`\t` preserved; `</document>` inside text is escaped; `PROMPT_VERSION` exists.
6. **Orchestrator.** Per section 10 + EventBus. Tests `test_orchestrator.py` (FakeLLM, plaintext sources and pdf fixtures): happy path 2 pages → both done, result.json written, the glossary from page one's new_terms is visible in page two's prompt (check FakeLLM.calls); a page with an invalid×2 response → failed, the next one is processed, job done; cache: re-running the same content → 0 LLM calls; resume: page one done in the DB → only page two is processed; an inline image → a separate vision call, an annotation in the result; a vision error on an image → the page is still done with labels=[].
7. **Rendering.** render/html.py. Tests `test_render.py`: the glossed line contains translations in parentheses, the plain line == original; heading has no repeat; consecutive list_item → one ul; a failed page → a banner; the toggle JS is present; `@media print` is present; the image is inserted after the correct block (assemble a result by hand with a known source_text and anchor); img src starts with `data:image/png;base64,`.
8. **API + worker.** main.py in full; in lifespan: preflight() (log, do not crash), running→interrupted, worker startup. Tests `test_api.py` (httpx ASGITransport + FakeLLM, substituted via DI: the client factory lives in `app.state`): POST text job → 200, job_id; the worker finishes processing → GET /jobs/{id} status=done (poll with a timeout); result.html is served; result.pdf → 200 with `content-type: application/pdf`, body starts with `%PDF` (xhtml2pdf is always available, no extra needed); POST without file/url/text → 422; two sources at once → 422; SSE: read a couple of events and see page done; /health responds even with the LLM unreachable (reachable=false, not 500).
9. **UI.** static/index.html per section 12. No automated tests — manual check.
10. **M0 script.** `scripts/m0_check.py`: CLI `uv run python scripts/m0_check.py <file.png|.pdf> [--page N] [--lang ru]` — real LocalAI, prints: the JSON response pretty-printed, timing, validity; writes `m0_out.html` next to it. This is a manual tool for assessing segmentation quality.
11. **README.md**: startup (`uv sync`, `uv run uvicorn frank_reader.main:app`), config via env/.env, how to switch the LLM endpoint/model (LocalAI :1240 / Ollama :11434 / LM Studio :1234 / cloud — and that for OpenAI-compatible clouds `FRANK_LLM_REASONING_EFFORT=` must be set to empty), PDF via xhtml2pdf (no extra, no system libs) and the embedded font for Cyrillic, how to print (⌘P), the M0 script, `/health` for diagnosing "model not loaded", recommendations on `pseudo_page_chars`/`llm_max_tokens` when changing models.
12. **Final**: `uv run pytest` fully green; manual smoke: the server starts, a textarea job with FakeLLM is not needed — run a short text against real LocalAI.

`test_integration_llm.py`: one test — a short German paragraph through real LocalAI → a valid PageResult; `@pytest.mark.skipif(os.environ.get("FRANK_INTEGRATION") != "1", ...)`.

## 14. Acceptance criteria

- [ ] `uv sync && uv run pytest` — green on a clean clone (without LocalAI).
- [ ] Textarea text in German (real LocalAI) → done, the HTML opens, phrases in Frank format, phrase repeat = original.
- [ ] A single-page PDF scan → vision path → result.
- [ ] Killing the server in the middle of a 3-page job → restart → status=interrupted → resume → completed, finished pages were not recomputed (per logs/cache).
- [ ] result.pdf is served right after `uv sync` (no additional extras/system packages), Cyrillic in the PDF is readable (not empty rectangles); printing via the browser looks decent (translations not cut off from phrases).
- [ ] Invalid JSON from the LLM (simulated via FakeLLM) does not bring down the job.
- [ ] The job history in the UI works, re-downloading a result works.
- [ ] A document of 5000+ characters (e.g. 10 Grundgesetz articles as one text job) is processed in full — no pages cut off mid-sentence and no pages with JSON validation errors on real LocalAI.

## 15. Explicitly deferred (do not build, even if easy)

bbox overlays on images, EPUB/FB2/PPTX/SRT adapters, page parallelism, a re-summary of the whole document via a separate LLM call, result editing in the UI, Jinja templates, Docker, auth.
