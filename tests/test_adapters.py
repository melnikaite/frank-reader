from pathlib import Path

import pytest

from frank_reader.adapters import detect_source_type, get_adapter
from frank_reader.adapters._pseudopage import build_pseudo_pages
from frank_reader.adapters.base import InlineImage
from frank_reader.adapters.docx import DocxAdapter
from frank_reader.adapters.image import ImageAdapter
from frank_reader.adapters.pdf import PdfAdapter
from frank_reader.adapters.plaintext import PlainTextAdapter
from frank_reader.adapters.url import process_markdown
from frank_reader.config import Settings


@pytest.fixture
def settings():
    return Settings()


def test_detect_source_type():
    assert detect_source_type("doc.pdf") == "pdf"
    assert detect_source_type("doc.DOCX") == "docx"
    assert detect_source_type("pic.PNG") == "image"
    assert detect_source_type("pic.jpeg") == "image"
    assert detect_source_type("notes.txt") == "text"
    with pytest.raises(ValueError):
        detect_source_type("archive.zip")


def test_get_adapter_returns_correct_types(settings):
    assert isinstance(get_adapter("pdf", settings), PdfAdapter)
    assert isinstance(get_adapter("docx", settings), DocxAdapter)
    assert isinstance(get_adapter("image", settings), ImageAdapter)
    assert isinstance(get_adapter("text", settings), PlainTextAdapter)
    with pytest.raises(ValueError):
        get_adapter("unknown", settings)


# --- PDF ---


def test_pdf_text_pages_extracted(text_pdf, settings):
    pages = PdfAdapter(settings).load(text_pdf)
    assert len(pages) == 2
    for page in pages:
        assert page.kind == "text"
        assert page.text is not None
        assert len(page.text) >= settings.min_text_chars


def test_pdf_scan_page_becomes_image(scan_pdf, settings):
    pages = PdfAdapter(settings).load(scan_pdf)
    assert len(pages) == 1
    assert pages[0].kind == "image"
    assert pages[0].image_png is not None
    assert len(pages[0].image_png) > 0


def test_pdf_mixed_pages(mixed_pdf, settings):
    pages = PdfAdapter(settings).load(mixed_pdf)
    assert len(pages) == 2
    assert pages[0].kind == "text"
    assert pages[1].kind == "image"


def test_pdf_force_vision_makes_everything_image(text_pdf, settings):
    pages = PdfAdapter(settings, force_vision=True).load(text_pdf)
    assert len(pages) == 2
    assert all(p.kind == "image" for p in pages)


def test_pdf_inline_images_filtered_by_size(pdf_with_inline_images, settings):
    pages = PdfAdapter(settings).load(pdf_with_inline_images)
    assert len(pages) == 1
    page = pages[0]
    assert page.kind == "text"
    assert len(page.inline_images) == 1
    big = page.inline_images[0]
    assert big.position_anchor > 0


def test_pdf_long_text_page_splits_into_pseudo_pages_with_global_renumbering(long_text_pdf):
    settings = Settings(pseudo_page_chars=400)
    pages = PdfAdapter(settings).load(long_text_pdf)

    # page 1's text exceeds pseudo_page_chars and must split into 2+ sub-pages
    assert len(pages) > 2
    assert [p.page_number for p in pages] == list(range(1, len(pages) + 1))
    assert all(p.kind == "text" for p in pages)

    # the trailing image from PDF page 1 belongs to one of the sub-pages carved
    # out of page 1, not to the final (separate, short) PDF page
    last_page_text = pages[-1].text or ""
    assert "Kurzer zweiter Seiteninhalt" in last_page_text
    assert pages[-1].inline_images == []

    total_images = sum(len(p.inline_images) for p in pages)
    assert total_images == 1
    owner = next(p for p in pages if p.inline_images)
    anchor = owner.inline_images[0].position_anchor
    assert 0 <= anchor <= len(owner.text or "")


# --- DOCX ---


def test_docx_extracts_text_and_list_prefix(sample_docx, settings):
    pages = DocxAdapter(settings).load(sample_docx)
    full_text = "\n\n".join(p.text for p in pages)
    assert "Первый абзац" in full_text
    assert "- Пункт списка один" in full_text
    assert "- Пункт списка два" in full_text


# --- Image ---


def test_image_adapter_single_page(sample_image, settings):
    pages = ImageAdapter(settings).load(sample_image)
    assert len(pages) == 1
    assert pages[0].kind == "image"
    assert pages[0].image_png is not None


# --- PlainText ---


def test_plaintext_splits_paragraphs(sample_text_file, settings):
    pages = PlainTextAdapter(settings).load(sample_text_file)
    assert len(pages) == 1
    assert "Первый абзац" in pages[0].text
    assert "Третий абзац" in pages[0].text


def test_plaintext_long_text_splits_into_pseudo_pages(settings):
    settings = Settings(pseudo_page_chars=100)
    paragraph = "Слово " * 30  # ~180 chars, exceeds pseudo_page_chars
    text = "\n\n".join([paragraph, paragraph, paragraph])
    pages = PlainTextAdapter(settings).load(text)
    assert len(pages) > 1


def test_plaintext_from_string_source(settings):
    pages = PlainTextAdapter(settings).load("Просто текст без файла.")
    assert len(pages) == 1
    assert pages[0].text == "Просто текст без файла."


# --- URL (markdown processing only, no network) ---


def test_process_markdown_strips_images_and_downloads(settings, make_png):
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return make_png(20, 20)

    markdown = "Текст до.\n\n![alt](http://example.com/pic.png)\n\nТекст после."
    pages = process_markdown(markdown, settings, fetch_image=fake_fetch)
    full_text = "\n\n".join(p.text for p in pages)
    assert "![alt]" not in full_text
    assert calls == ["http://example.com/pic.png"]
    total_images = sum(len(p.inline_images) for p in pages)
    assert total_images == 1


def test_process_markdown_skips_failed_downloads(settings):
    pages = process_markdown(
        "Текст. ![alt](http://bad)", settings, fetch_image=lambda url: None
    )
    total_images = sum(len(p.inline_images) for p in pages)
    assert total_images == 0


# --- shared pseudo-paging ---


def test_build_pseudo_pages_empty_images():
    pages = build_pseudo_pages("Один абзац.\n\nДругой абзац.", [], max_chars=1000)
    assert len(pages) == 1


def test_build_pseudo_pages_remaps_image_anchor():
    text = "Абзац раз.\n\nАбзац два, тут была картинка.\n\nАбзац три."
    anchor = text.index("Абзац два")
    images = [InlineImage(image_png=b"x", position_anchor=anchor)]
    pages = build_pseudo_pages(text, images, max_chars=15)
    assert sum(len(p.inline_images) for p in pages) == 1
    for p in pages:
        for img in p.inline_images:
            assert 0 <= img.position_anchor <= len(p.text)
