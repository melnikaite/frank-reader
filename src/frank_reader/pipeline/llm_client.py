import asyncio
import base64
import json
import logging
import re
import time
from typing import Any, Callable, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from frank_reader.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")


class LLMClient(Protocol):
    async def complete(
        self, system: str, user_text: str, image_png: bytes | None = None
    ) -> str: ...


def build_request_body(
    system: str,
    user_text: str,
    image_png: bytes | None,
    model: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    if image_png is not None:
        b64 = base64.b64encode(image_png).decode("ascii")
        user_content: Any = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
    else:
        user_content = user_text

    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    return body


def extract_text(message: dict[str, Any]) -> str:
    content = (message.get("content") or "").strip()
    if content:
        return content
    reasoning = (message.get("reasoning_content") or "").strip()
    if reasoning:
        paragraphs = [p.strip() for p in reasoning.split("\n\n") if p.strip()]
        if paragraphs:
            return paragraphs[-1]
    raise ValueError("LLM response has empty content and reasoning_content")


def _strip_fences(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = _FENCE_RE.sub("", s).strip()
    start_candidates = [i for i in (s.find("{"), s.find("[")) if i != -1]
    if start_candidates:
        start = min(start_candidates)
        end = max(s.rfind("}"), s.rfind("]"))
        if end != -1 and end >= start:
            s = s[start : end + 1]
    return s.strip()


def _parse(raw: str, schema: type[T]) -> T:
    cleaned = _strip_fences(raw)
    parsed = json.loads(cleaned)
    if isinstance(parsed, list):
        fields = schema.model_fields
        if len(fields) == 1:
            field_name = next(iter(fields))
            parsed = {field_name: parsed}
    return schema.model_validate(parsed)


async def call_structured(
    client: LLMClient,
    system: str,
    user_text: str,
    schema: type[T],
    image_png: bytes | None = None,
) -> T:
    raw = await client.complete(system, user_text, image_png)
    try:
        return _parse(raw, schema)
    except (ValidationError, ValueError, json.JSONDecodeError) as err:
        repair_user = (
            user_text
            + "\n\nYour previous answer failed validation. The answer was:\n"
            + raw[:3000]
            + "\n\nError: "
            + str(err)[:500]
            + "\n\nReturn STRICTLY valid JSON per the schema, no explanations, no markdown."
        )
        raw2 = await client.complete(system, repair_user, image_png)
        return _parse(raw2, schema)


class FakeLLM:
    """Test double. Accepts a list of canned responses (consumed in order)
    or a callable(system, user_text, has_image) -> str."""

    def __init__(self, responses: list[str] | Callable[[str, str, bool], str] | None = None):
        self._callable: Callable[[str, str, bool], str] | None = None
        self._responses: list[str] | None = None
        if callable(responses):
            self._callable = responses
        elif responses is not None:
            self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self, system: str, user_text: str, image_png: bytes | None = None
    ) -> str:
        self.calls.append(
            {"system": system, "user_text": user_text, "has_image": image_png is not None}
        )
        if self._callable is not None:
            return self._callable(system, user_text, image_png is not None)
        if self._responses:
            return self._responses.pop(0)
        raise RuntimeError("FakeLLM exhausted: no more canned responses")


class LocalAIClient:
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None):
        self._settings = settings
        self._http = http_client or httpx.AsyncClient()
        self._owns_http = http_client is None

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def complete(
        self, system: str, user_text: str, image_png: bytes | None = None
    ) -> str:
        s = self._settings
        has_image = image_png is not None
        base_url = (s.llm_vision_base_url if has_image and s.llm_vision_base_url else s.llm_base_url)
        model = s.llm_vision_model if has_image and s.llm_vision_model else s.llm_model
        timeout = s.llm_timeout_vision if has_image else s.llm_timeout_text

        body = build_request_body(
            system, user_text, image_png, model, s.llm_temperature, s.llm_max_tokens, s.llm_reasoning_effort
        )
        headers = {"Authorization": f"Bearer {s.llm_api_key}"}

        last_exc: Exception | None = None
        for attempt in range(3):
            start = time.monotonic()
            try:
                resp = await self._http.post(
                    f"{base_url}/chat/completions", json=body, headers=headers, timeout=timeout
                )
                resp.raise_for_status()
                data = resp.json()
                elapsed = time.monotonic() - start
                usage = data.get("usage", {})
                logger.info(
                    "llm complete model=%s image=%s elapsed=%.1fs prompt_tokens=%s completion_tokens=%s",
                    model,
                    has_image,
                    elapsed,
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                )
                message = data["choices"][0]["message"]
                return extract_text(message)
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if attempt < 2:
                    await asyncio.sleep(5)
                    continue
                raise
        raise last_exc  # pragma: no cover

    async def preflight(self) -> dict[str, Any]:
        s = self._settings
        try:
            resp = await self._http.get(f"{s.llm_base_url}/models", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            available = [m.get("id") for m in data.get("data", [])]
            model_loaded = s.llm_model in available
            if not model_loaded:
                logger.critical(
                    "LLM model '%s' not loaded, available: %s", s.llm_model, available
                )
            return {
                "llm_reachable": True,
                "model_loaded": model_loaded,
                "model": s.llm_model,
                "available_models": available,
            }
        except Exception as exc:
            logger.critical("LLM preflight failed: %s", exc)
            return {
                "llm_reachable": False,
                "model_loaded": False,
                "model": s.llm_model,
                "available_models": [],
            }
