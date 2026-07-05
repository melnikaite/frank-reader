from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRANK_", env_file=".env", extra="ignore")

    llm_base_url: str = "http://127.0.0.1:1240/v1"
    llm_model: str = "gemma-4-e4b-it-qat-q4_0"
    llm_api_key: str = "not-needed"
    llm_timeout_text: float = 180.0
    llm_timeout_vision: float = 420.0
    llm_temperature: float = 0.2
    # Frank-method output roughly duplicates+translates the source, so JSON
    # completion tokens run ~1.2-1.5x the input character count on this model
    # class (measured on gemma-4-e4b-it-qat-q4_0). A low ceiling here silently
    # truncates the JSON mid-page rather than erroring - generous headroom
    # costs nothing since generation stops at finish_reason=stop regardless.
    llm_max_tokens: int = 16384
    llm_reasoning_effort: str | None = "none"

    llm_vision_base_url: str | None = None
    llm_vision_model: str | None = None

    # Lives in the home directory (not cwd) so the app works identically when
    # run from a repo checkout and when installed via `uv tool install`.
    data_dir: Path = Path.home() / ".frank-reader"
    target_lang_default: str = "ru"

    host: str = "127.0.0.1"
    port: int = 8200  # 8000 is a common default for other local services

    min_text_chars: int = 50
    garbage_char_ratio: float = 0.10
    scan_image_coverage: float = 0.80

    page_render_max_dim: int = 1400
    inline_image_min_dim: int = 64
    inline_image_min_area: float = 0.02

    # Smaller chunks measurably improve completeness/reliability on this model
    # class (verified empirically: ~99% coverage at ~2500 chars/call vs. JSON
    # truncation or missing required fields on 6000+ char single-shot pages).
    pseudo_page_chars: int = 2500

    glossary_max_terms: int = 200
    context_summaries: int = 3
