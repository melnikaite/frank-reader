import pytest
from pydantic import ValidationError

from frank_reader.pipeline.schema import PageResult, TextBlock


def test_valid_page_parses():
    result = PageResult.model_validate(
        {
            "page_summary": "О чём-то важном.",
            "detected_language": "de",
            "text_blocks": [
                {
                    "order": 1,
                    "type": "phrase",
                    "original": "Die Würde des Menschen ist unantastbar.",
                    "chunks": [
                        {"original": "Die Würde des Menschen", "translation": "достоинство человека"},
                        {"original": "ist unantastbar", "translation": "неприкосновенно"},
                    ],
                },
                {
                    "order": 2,
                    "type": "heading",
                    "original": "Artikel 1",
                    "translation": "Статья 1",
                },
            ],
        }
    )
    assert result.detected_language == "de"
    assert len(result.text_blocks) == 2


def test_phrase_without_chunks_rejected():
    with pytest.raises(ValidationError):
        TextBlock.model_validate({"order": 1, "type": "phrase", "original": "x"})


def test_heading_without_translation_rejected():
    with pytest.raises(ValidationError):
        TextBlock.model_validate({"order": 1, "type": "heading", "original": "x"})


def test_extra_fields_ignored():
    result = PageResult.model_validate(
        {
            "page_summary": "s",
            "detected_language": "en",
            "text_blocks": [],
            "unexpected_field": "should be ignored",
        }
    )
    assert result.text_blocks == []
