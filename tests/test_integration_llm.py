import os

import pytest

from frank_reader.config import Settings
from frank_reader.pipeline.llm_client import LocalAIClient, call_structured
from frank_reader.pipeline.prompts import build_context_block, system_prompt, user_text_page
from frank_reader.pipeline.schema import PageResult

pytestmark = pytest.mark.skipif(
    os.environ.get("FRANK_INTEGRATION") != "1",
    reason="Set FRANK_INTEGRATION=1 to run against a real LocalAI instance",
)


async def test_real_localai_produces_valid_page_result():
    settings = Settings()
    llm = LocalAIClient(settings)
    try:
        system = system_prompt("ru")
        context = build_context_block(None, [], {})
        user_text = user_text_page(
            "Der Regen fiel die ganze Nacht. Am Morgen war der Himmel wieder klar.", context
        )
        result = await call_structured(llm, system, user_text, PageResult)
        assert result.text_blocks
        assert result.detected_language
    finally:
        await llm.aclose()
