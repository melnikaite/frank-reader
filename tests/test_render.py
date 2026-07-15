from frank_reader.render.html import render_html
from frank_reader.render.pdf import PdfNotAvailable, render_pdf


def _page_with_blocks(blocks, **extra):
    page = {"page_number": 1, "status": "done", "result": {"text_blocks": blocks, "image_annotations": []}}
    page.update(extra)
    return page


def test_phrase_glossed_and_plain():
    blocks = [
        {
            "order": 1,
            "type": "phrase",
            "original": "Die Würde ist unantastbar.",
            "chunks": [
                {"original": "Die Würde", "translation": "достоинство"},
                {"original": "ist unantastbar", "translation": "неприкосновенно"},
            ],
        }
    ]
    doc = {"source_name": "test", "pages": [_page_with_blocks(blocks)]}
    html = render_html(doc, job_dir=None)
    assert "(достоинство)" in html
    assert "(неприкосновенно)" in html
    assert "Die Würde ist unantastbar." in html


def test_heading_has_no_repeat_line():
    blocks = [{"order": 1, "type": "heading", "original": "Titel", "translation": "Заголовок"}]
    doc = {"source_name": "t", "pages": [_page_with_blocks(blocks)]}
    html = render_html(doc, job_dir=None)
    assert "<h2>" in html
    assert html.count("Titel") == 1


def test_consecutive_list_items_grouped_into_one_ul():
    blocks = [
        {"order": 1, "type": "list_item", "original": "eins", "translation": "раз"},
        {"order": 2, "type": "list_item", "original": "zwei", "translation": "два"},
    ]
    doc = {"source_name": "t", "pages": [_page_with_blocks(blocks)]}
    html = render_html(doc, job_dir=None)
    assert html.count("<ul>") == 1
    assert html.count("<li>") == 2


def test_failed_page_shows_banner():
    doc = {
        "source_name": "t",
        "pages": [{"page_number": 3, "status": "failed", "error": "boom"}],
    }
    html = render_html(doc, job_dir=None)
    assert "page-failed" in html
    assert "page 3" in html
    assert "boom" in html


def test_no_page_markers_in_output():
    blocks = [{"order": 1, "type": "phrase", "original": "Hallo",
               "chunks": [{"original": "Hallo", "translation": "привет"}]}]
    doc = {"source_name": "t", "pages": [_page_with_blocks(blocks)]}
    html = render_html(doc, job_dir=None)
    assert "page-marker" not in html
    assert "page-sep" not in html


def test_toggle_present():
    doc = {"source_name": "t", "pages": []}
    html = render_html(doc, job_dir=None)
    assert "toggle-btn" in html
    assert "translations-hidden" in html


def test_print_css_present():
    doc = {"source_name": "t", "pages": []}
    html = render_html(doc, job_dir=None)
    assert "@media print" in html


def test_inline_image_inserted_after_correct_block(tmp_path, make_png):
    img_path = tmp_path / "images"
    img_path.mkdir()
    (img_path / "001_00.png").write_bytes(make_png(10, 10))

    source_text = "Блок один текст.\n\nБлок два текст, тут была картинка.\n\nБлок три текст."
    anchor = source_text.index("Блок два") + len("Блок два текст, тут была картинка.")
    blocks = [
        {"order": 1, "type": "phrase", "original": "Блок один текст.", "chunks": [{"original": "Блок один текст.", "translation": "x"}]},
        {"order": 2, "type": "phrase", "original": "Блок два текст, тут была картинка.", "chunks": [{"original": "Блок два текст, тут была картинка.", "translation": "y"}]},
        {"order": 3, "type": "phrase", "original": "Блок три текст.", "chunks": [{"original": "Блок три текст.", "translation": "z"}]},
    ]
    page = _page_with_blocks(
        blocks,
        source_text=source_text,
        inline_images=[{"file": "images/001_00.png", "position_anchor": anchor}],
    )
    doc = {"source_name": "t", "pages": [page]}
    html = render_html(doc, job_dir=tmp_path)

    img_idx = html.index('src="data:image/png;base64,')
    block2_idx = html.index("Блок два текст")
    block3_idx = html.index("Блок три текст")
    assert block2_idx < img_idx < block3_idx


def test_render_pdf_produces_valid_pdf_bytes():
    pdf_bytes = render_pdf("<html><head></head><body><p>Hello</p></body></html>")
    assert pdf_bytes.startswith(b"%PDF")


def test_render_pdf_preserves_cyrillic_text():
    import io

    import pypdf

    html = '<html><head></head><body><p>Заголовок и перевод</p></body></html>'
    pdf_bytes = render_pdf(html)
    text = pypdf.PdfReader(io.BytesIO(pdf_bytes)).pages[0].extract_text()
    assert "Заголовок" in text


def test_render_pdf_raises_pdf_not_available_on_pisa_error(monkeypatch):
    from xhtml2pdf import pisa

    class FakeResult:
        err = 1

    monkeypatch.setattr(pisa, "CreatePDF", lambda *a, **k: FakeResult())
    try:
        render_pdf("<html><head></head><body></body></html>")
        assert False, "expected PdfNotAvailable"
    except PdfNotAvailable:
        pass


def test_hiding_translations_removes_the_glossed_line_not_just_spans():
    blocks = [
        {
            "order": 1,
            "type": "phrase",
            "original": "Hallo Welt",
            "chunks": [{"original": "Hallo Welt", "translation": "привет мир"}],
        }
    ]
    doc = {"source_name": "t", "pages": [_page_with_blocks(blocks)]}
    html = render_html(doc, job_dir=None)
    assert "body.translations-hidden .phrase .glossed { display: none; }" in html
