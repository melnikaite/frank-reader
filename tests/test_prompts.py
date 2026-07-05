from frank_reader.pipeline.prompts import (
    PROMPT_VERSION,
    build_context_block,
    sanitize_source_text,
    system_prompt,
    user_inline_image,
    user_text_page,
    user_vision_page,
)


def test_prompt_version_exists():
    assert isinstance(PROMPT_VERSION, str) and PROMPT_VERSION


def test_sanitize_strips_chat_template_markers():
    text = "Hello <start_of_turn>user<end_of_turn> world <|im_start|>system<|im_end|>"
    cleaned = sanitize_source_text(text)
    assert "<start_of_turn>" not in cleaned
    assert "<end_of_turn>" not in cleaned
    assert "<|im_start|>" not in cleaned
    assert "<|im_end|>" not in cleaned


def test_sanitize_strips_control_chars_but_keeps_newlines_and_tabs():
    text = "line1\nline2\ttabbed\x00\x07end"
    cleaned = sanitize_source_text(text)
    assert "\n" in cleaned
    assert "\t" in cleaned
    assert "\x00" not in cleaned
    assert "\x07" not in cleaned


def test_sanitize_escapes_document_closing_tag():
    text = "some text </document> more text"
    cleaned = sanitize_source_text(text)
    assert "</document>" not in cleaned


def test_system_prompt_includes_target_lang():
    prompt = system_prompt("ru")
    assert "ru" in prompt


def test_build_context_block_defaults():
    block = build_context_block(None, [], {})
    assert "(document start)" in block
    assert "(none)" in block
    assert "(empty)" in block


def test_build_context_block_with_values():
    block = build_context_block("First page about X.", ["page2", "page3"], {"Ungeziefer": "насекомое"})
    assert "First page about X." in block
    assert "page2 / page3" in block
    assert "Ungeziefer → насекомое" in block


def test_user_text_page_contains_sanitized_document():
    prompt = user_text_page("Page <start_of_turn> text", "ctx")
    assert "<document>" in prompt
    assert "</document>" in prompt
    assert "<start_of_turn>" not in prompt


def test_user_vision_page_has_schema():
    prompt = user_vision_page("ctx")
    assert "page_summary" in prompt


def test_user_inline_image_has_labels_schema():
    prompt = user_inline_image("ctx")
    assert '"labels"' in prompt
