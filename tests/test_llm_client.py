import pytest

from frank_reader.pipeline.llm_client import (
    FakeLLM,
    build_request_body,
    call_structured,
    extract_text,
)
from frank_reader.pipeline.schema import ImageLabelsResult, PageResult

VALID_PAGE = """{
    "page_summary": "s",
    "detected_language": "de",
    "text_blocks": [
        {"order": 1, "type": "heading", "original": "Titel", "translation": "Заголовок"}
    ]
}"""

INVALID_JSON = "not json at all"


async def test_valid_json_first_try():
    llm = FakeLLM([VALID_PAGE])
    result = await call_structured(llm, "sys", "user", PageResult)
    assert result.detected_language == "de"
    assert len(llm.calls) == 1


async def test_json_in_markdown_fence_parses():
    fenced = "```json\n" + VALID_PAGE + "\n```"
    llm = FakeLLM([fenced])
    result = await call_structured(llm, "sys", "user", PageResult)
    assert result.detected_language == "de"


async def test_invalid_then_valid_triggers_repair():
    llm = FakeLLM([INVALID_JSON, VALID_PAGE])
    result = await call_structured(llm, "sys", "user", PageResult)
    assert result.detected_language == "de"
    assert len(llm.calls) == 2
    assert "failed validation" in llm.calls[1]["user_text"]


async def test_two_invalid_raises():
    llm = FakeLLM([INVALID_JSON, INVALID_JSON])
    with pytest.raises(Exception):
        await call_structured(llm, "sys", "user", PageResult)


async def test_bare_array_wrapped_for_single_list_field_schema():
    llm = FakeLLM(['[{"original": "Abb. 1", "translation": "Рис. 1"}]'])
    result = await call_structured(llm, "sys", "user", ImageLabelsResult)
    assert len(result.labels) == 1
    assert result.labels[0].original == "Abb. 1"


def test_build_request_body_includes_reasoning_effort_by_default():
    body = build_request_body("sys", "user", None, "gemma-4", 0.2, 4096, "none")
    assert body["reasoning_effort"] == "none"
    assert body["messages"][1]["content"] == "user"
    assert body["response_format"] == {"type": "json_object"}


def test_build_request_body_omits_reasoning_effort_when_none():
    body = build_request_body("sys", "user", None, "gpt-4o", 0.2, 4096, None)
    assert "reasoning_effort" not in body


def test_build_request_body_with_image_uses_content_list():
    body = build_request_body("sys", "user", b"\x89PNG...", "gemma-4", 0.2, 4096, "none")
    content = body["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "user"}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_extract_text_prefers_content():
    assert extract_text({"content": "hello", "reasoning_content": "ignored"}) == "hello"


def test_extract_text_falls_back_to_reasoning_content_last_paragraph():
    message = {"content": "", "reasoning_content": "First para.\n\nLast para with answer."}
    assert extract_text(message) == "Last para with answer."


def test_extract_text_raises_when_both_empty():
    with pytest.raises(ValueError):
        extract_text({"content": "", "reasoning_content": ""})


# --- streaming ---------------------------------------------------------------

import json

import httpx

from frank_reader.config import Settings
from frank_reader.pipeline.llm_client import LocalAIClient, StreamAccumulator


def _sse(*chunks: str) -> str:
    lines = [f"data: {c}" for c in chunks]
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


def test_stream_accumulator_collects_content():
    acc = StreamAccumulator()
    assert acc.feed('data: {"choices":[{"delta":{"content":"Hel"}}]}') is True
    assert acc.feed('data: {"choices":[{"delta":{"content":"lo"}}]}') is True
    assert acc.feed("data: [DONE]") is False
    assert acc.feed("") is False
    assert acc.feed("garbage") is False
    assert acc.chunks == 2
    assert acc.message()["content"] == "Hello"


def test_stream_accumulator_collects_reasoning_content():
    acc = StreamAccumulator()
    acc.feed('data: {"choices":[{"delta":{"reasoning_content":"think "}}]}')
    acc.feed('data: {"choices":[{"delta":{"content":"answer"}}]}')
    msg = acc.message()
    assert msg["content"] == "answer"
    assert msg["reasoning_content"] == "think "


def test_stream_accumulator_ignores_empty_delta_and_bad_json():
    acc = StreamAccumulator()
    assert acc.feed('data: {"choices":[{"delta":{}}]}') is False
    assert acc.feed('data: {not json') is False
    assert acc.feed('data: {"choices":[]}') is False
    assert acc.chunks == 0


async def test_localai_client_streams_and_reports_progress():
    body = _sse(
        '{"choices":[{"delta":{"content":"{\\"labels\\":"}}]}',
        '{"choices":[{"delta":{"content":" []}"}}]}',
    )

    def handler(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        assert sent["stream"] is True
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = LocalAIClient(Settings(), http_client=http)
    progress: list[int] = []
    raw = await client.complete("sys", "user", on_progress=progress.append)
    assert raw == '{"labels": []}'
    assert progress == [1, 2]
