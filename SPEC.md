# Frank Reader — Spec v0.1

A web application that converts arbitrary documents into text following Ilya Frank's reading method (phrase in the original language → translation → repeat of the phrase without translation), using local LLM inference.

Status: draft for refinement and reassessment in Claude Code. Open questions are marked [?], known risks — [!].

## 1. Goals and non-goals

### Goal

The user uploads a document (PDF/DOCX/image/URL/text) in any supported language → receives HTML output (and optionally PDF) in the Frank-method format:

> Die Würde des Menschen (достоинство человека) ist unantastbar (неприкосновенно).
> Die Würde des Menschen ist unantastbar.

- Translation is context-aware: the model understands what the document is about, and terms are translated consistently throughout the document.
- Images from the document are preserved in their original positions in the output.
- Text/labels inside images are recognized, and translations are added next to the original position (the original is not erased, no inpainting needed).

### Non-goals (v1)

- Erasing/redrawing text on images (inpainting).
- Handwritten text (handwriting OCR) — explicitly out of scope.
- Perfect preservation of the source PDF layout (two-column layouts are "flattened" into a single flow).
- Multi-user support, authentication, cloud — this is a local single-user tool.

## 2. Stack and environment

| Component | Choice | Note |
|---|---|---|
| Dependency manager | uv | already installed |
| Backend | FastAPI + uvicorn | |
| Frontend | HTMX + minimal JS | drag & drop, SSE progress; no SPA frameworks |
| LLM inference | LocalAI (already deployed) with Gemma 4 omnimodal | OpenAI-compatible API, vision via image_url in content |
| PDF | PyMuPDF (fitz) | render pages to PNG, extract the text layer and images with bboxes |
| DOCX | python-docx or mammoth → HTML | [?] decide during implementation |
| Web articles | trafilatura (primary) | main content extraction; fallback readability-lxml |
| EPUB (v2) | ebooklib | deferred |
| Output PDF | WeasyPrint from HTML | optional, HTML is primary |
| Storage | file system + SQLite (WAL) | jobs, statuses, translation cache |

Target machine: MacBook Pro M1 Max 64GB, LocalAI already running, Gemma 4 omnimodal already configured.

## 3. Architecture

```
┌──────────┐   ┌────────────────┐   ┌──────────────┐   ┌───────────────┐
│ Upload   │ → │ Source Adapter │ → │ Pipeline     │ → │ Output Render │
│ (web UI) │   │ (unification)  │   │ (LLM processing)│ │ (HTML/PDF)   │
└──────────┘   └────────────────┘   └──────────────┘   └───────────────┘
                                        ↕
                                 LocalAI (Gemma 4)
```

### 3.1 Source Adapters — input unification

Each adapter converts its source into a common intermediate representation:

```python
@dataclass
class PageContent:
    page_number: int
    kind: Literal["text", "image"]  # text = a reliable text layer exists
    text: str | None                # for kind="text"
    image_png: bytes | None         # page render for kind="image" (vision path)
    inline_images: list[InlineImage]  # images inside a text page

@dataclass
class InlineImage:
    image_png: bytes
    bbox_rel: tuple[float, float, float, float]  # x, y, w, h as fractions 0..1
    position_anchor: int  # position in the text (offset) where to insert at render time
```

v1 adapters:

| Adapter | Input | Logic |
|---|---|---|
| PdfAdapter | .pdf | See 3.2 — scan/text detection per page |
| DocxAdapter | .docx | text + inline images; always kind="text" |
| ImageAdapter | .png/.jpg/.webp | a single "page" with kind="image" |
| UrlAdapter | URL | trafilatura → main content (text + content images); kind="text" |
| PlainTextAdapter | textarea / .txt | trivial; kind="text" |

v2 adapters (interface reserved now, implementation later): EPUB/FB2, PPTX, SRT/VTT.

### 3.2 Detecting "scan vs text PDF" (per page!)

The decision is made per page, not per document (mixed PDFs exist):

```
text = page.get_text().strip()
images = page.get_images()

if len(text) > MIN_TEXT_CHARS (e.g. 50)
   and the text does not look like garbage (share of non-printable/replacement chars < 10%)
→ kind="text" (the text layer is trustworthy)
else if there is an image covering > 80% of the page area
→ kind="image" (scan) → render the page to PNG
else
→ kind="image" (fallback: when in doubt, the vision path is more reliable)
```

[!] Tune the MIN_TEXT_CHARS threshold and the "garbage" heuristic on real scans.
[!] PDFs with an OCR-produced text layer (already recognized scans) contain recognition errors — [?] possibly worth giving the user a manual "force vision path" toggle.

### 3.3 Pipeline — processing via LocalAI

Per-page loop with a rolling summary:

```
rolling_summary = ""
for page in pages:
    if page.kind == "text":
        response = llm_text_request(page.text, rolling_summary)
    else:
        response = llm_vision_request(page.image_png, rolling_summary)
    # response → PageResult (see schema below)
    rolling_summary = response.page_summary
    # every N=10 pages: re-summary of everything processed so far (drift protection)
    yield PageResult  # SSE progress in the UI
```

For inline images inside text pages — a separate vision call per image (recognize labels + translate), the result is merged into PageResult.

[?] Evaluate: if the document fits entirely into the context window (short articles) — a single request instead of the per-page loop would give better term consistency. Possibly two modes: "short doc" (whole document) / "long doc" (per page + rolling summary).

### 3.4 Model response schema (JSON)

```json
{
  "page_summary": "2-3 sentences about the page content (context for the next page)",
  "detected_language": "de",
  "text_blocks": [
    {
      "order": 1,
      "type": "phrase",
      "original": "Die Würde des Menschen ist unantastbar.",
      "chunks": [
        {"original": "Die Würde des Menschen", "translation": "достоинство человека"},
        {"original": "ist unantastbar", "translation": "неприкосновенно"}
      ]
    },
    {
      "order": 2,
      "type": "heading",
      "original": "Artikel 1",
      "translation": "Статья 1"
    }
  ],
  "image_annotations": [
    {
      "image_ref": 0,
      "labels": [
        {"original": "Abbildung 1: Aufbau", "translation": "Рисунок 1: строение", "bbox_rel": [0.1, 0.85, 0.5, 0.05]}
      ]
    }
  ]
}
```

Key decisions:

- chunks within a phrase — the Frank method inserts translations for meaningful pieces within a sentence, then repeats the sentence as a whole. The renderer assembles the `original (translation) ...` line from the chunks, plus the repeat line.
- type: phrase | heading | list_item | caption — headings and captions are translated but not "Frank-ified" (no repeat).
- bbox in relative coordinates 0..1 — DPI independence.

Validation via pydantic. On a parsing error: 1 retry with the message "return strictly valid JSON per the schema" + the invalid response itself. After 2 failures — the page is marked failed and the pipeline continues (we don't crash entirely).

[!] The JSON mode of llama.cpp backends is unstable on multimodal requests — plan for `response_format={"type": "json_object"}` if LocalAI supports it, otherwise a strict prompt + repair loop.

### 3.5 Prompts (drafts, subject to iteration)

System (shared):

```
You are preparing an educational text using Ilya Frank's reading method.
Method rules:
- The text is split into meaningful phrases (usually a sentence or part of one).
- Within a phrase, the translation is given per meaningful chunk, immediately after the original chunk.
- The translation is as literal as clarity allows; take the document context into account.
- Translate terms consistently throughout the document (see context).
- Target translation language: Russian. Detect the original language yourself.
Respond with STRICTLY valid JSON per the given schema, no markdown, no explanations.
```

User (vision path):

```
Context from previous pages: {rolling_summary or "(start of document)"}

The image shows a page of a document.
1. Recognize all text in reading order (account for columns if present).
2. Split it into blocks (phrase/heading/list_item/caption) and meaningful chunks.
3. Translate each chunk.
4. Find labels inside illustrations/diagrams — return them with bboxes (fractions 0..1).
5. Compose a page_summary (2-3 sentences).
JSON schema: {json_schema}
```

[?] Verify on a real page: chunk segmentation quality is the main product-quality risk. A few-shot example in the prompt (an excerpt of a finished Frank text) may be needed.

### 3.6 Output Renderer

- HTML (primary): a single self-contained file (inline styles, images as base64 or in an adjacent folder).
- Text: Frank-method blocks; chunks with translations — translation in gray/small font in parentheses, the phrase repeat — in regular font.
- Inline images: in place within the flow (by position_anchor), below the image — an annotation block: "label → translation" as a list. When bboxes are available — a CSS overlay on top of the image with semi-transparent markers [?] (nice to have, can be v2).
- A toggle in the HTML (pure CSS/JS): hide/show translations (self-testing mode).
- PDF (optional): WeasyPrint from the same HTML, print optimization (the user has a color laser printer, printing is a real scenario).

### 3.7 Web UI

Minimal, single page:

- Drag & drop zone + URL field + textarea for raw text (three tabs).
- Job settings: target language (default ru), mode (short/long document = whole/per page), [?] force vision path.
- Progress: SSE, per page ("page 3/12, phrases: 47"), preview of finished pages as they are processed.
- Result: link to the HTML + "download PDF" button.
- Job history (SQLite): list of recent jobs, re-download.

## 4. Project structure

```
frank-reader/
├── pyproject.toml            # uv, python >= 3.12
├── README.md
├── src/frank_reader/
│   ├── main.py               # FastAPI app, endpoints, SSE
│   ├── config.py             # LocalAI URL, model, heuristic thresholds (pydantic-settings)
│   ├── adapters/
│   │   ├── base.py           # SourceAdapter protocol → list[PageContent]
│   │   ├── pdf.py
│   │   ├── docx.py
│   │   ├── image.py
│   │   ├── url.py
│   │   └── plaintext.py
│   ├── pipeline/
│   │   ├── orchestrator.py   # page loop, rolling summary, retry
│   │   ├── llm_client.py     # LocalAI OpenAI-compatible, text + vision requests
│   │   ├── schema.py         # pydantic: PageResult, TextBlock, Chunk, ImageAnnotation
│   │   └── prompts.py
│   ├── render/
│   │   ├── html.py
│   │   └── pdf.py            # WeasyPrint wrapper
│   └── storage.py            # SQLite: jobs, pages, cache
├── static/
│   └── index.html            # HTMX UI
└── tests/
    ├── fixtures/             # test PDFs: text-based, scan, mixed, 2 columns
    └── ...
```

## 5. API (minimum)

| Method | Path | Description |
|---|---|---|
| POST | /jobs | multipart (file) or JSON (url / text) + options → job_id |
| GET | /jobs/{id}/events | SSE: per-page progress |
| GET | /jobs/{id}/result.html | finished HTML |
| GET | /jobs/{id}/result.pdf | PDF generation on demand |
| GET | /jobs | history |

Queue: in v1, an in-process asyncio queue is enough (single user, long jobs run sequentially). [?] Evaluate whether page-level parallelism is needed (LocalAI on M1 Max processes sequentially anyway — probably not).

## 6. Known risks and open questions (summary)

- [!] Chunk segmentation quality — the heart of the product. First milestone: manual prompt validation on 3-5 real pages BEFORE writing the UI.
- [!] JSON stability from Gemma via LocalAI — a repair loop is mandatory; measure the actual % of invalid responses.
- [!] bboxes from the VLM — multimodal models produce inaccurate coordinates; for v1 it may be more honest to output annotations as a list below the image, with the overlay in v2.
- [!] Speed: a vision request per page on M1 Max — roughly tens of seconds; a 50-page scan = tens of minutes. The UI must show progress honestly, and jobs must survive a server restart (persist in SQLite).
- [?] Two-column text PDFs — block order from PyMuPDF may be scrambled; option: force the vision path for pages suspected of having columns.
- [?] Rolling summary drift on long books — re-summary every 10 pages; verify in practice.
- [?] Caching: hash (page + prompt + model) → result in SQLite, so that reprocessing/retries don't waste time.

## 7. Milestones

- **M0 — Proof of prompt.** Script without UI: one page (PNG) → LocalAI → valid PageResult JSON → simplest HTML. Criterion: Frank-style chunking quality is acceptable on 3 different pages (text PDF, scan, page with a diagram).
- **M1 — Pipeline.** pdf/image/plaintext adapters, per-page loop, rolling summary, SQLite, retry.
- **M2 — Web UI.** FastAPI + HTMX, drag & drop, SSE progress, HTML result.
- **M3 — Polish.** UrlAdapter (trafilatura), DocxAdapter, WeasyPrint PDF, translation-hide toggle, cache.
- **v2 backlog.** EPUB/PPTX/SRT adapters, bbox overlays on images, parallelism, "whole document in one request" mode for short texts.
