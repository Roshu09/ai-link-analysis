"""Application settings, read from environment variables (and a local .env file).

Everything tunable lives here so the rest of the code never calls os.getenv directly.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class Settings(BaseModel):
    # --- LLM (Gemini) ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_fallback_model: str = "gemini-3.1-flash-lite"
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    llm_retry_base_delay: float = 1.5

    # --- Website fetching ---
    fetch_timeout_seconds: float = 12.0
    fetch_max_bytes: int = 2_000_000
    max_redirects: int = 5
    max_extra_pages: int = 2
    max_content_chars: int = 14_000
    user_agent: str = (
        "Mozilla/5.0 (compatible; LinkAnalysisBot/1.0; +https://github.com/) "
        "AppleWebKit/537.36 (KHTML, like Gecko)"
    )

    # --- Caching and abuse protection ---
    cache_ttl_seconds: int = 3600
    cache_max_entries: int = 200
    rate_limit_per_minute: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip(),
        gemini_fallback_model=os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.1-flash-lite").strip(),
        llm_timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 60.0),
        llm_max_retries=_env_int("LLM_MAX_RETRIES", 2),
        fetch_timeout_seconds=_env_float("FETCH_TIMEOUT_SECONDS", 12.0),
        max_extra_pages=_env_int("MAX_EXTRA_PAGES", 2),
        max_content_chars=_env_int("MAX_CONTENT_CHARS", 14_000),
        cache_ttl_seconds=_env_int("CACHE_TTL_SECONDS", 3600),
        rate_limit_per_minute=_env_int("RATE_LIMIT_PER_MINUTE", 30),
    )
