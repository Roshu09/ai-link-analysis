"""Talks to Google's Gemini API over plain HTTPS (using httpx).

Why raw HTTP instead of an SDK? Fewer dependencies, easier to test with a fake
network, and the request/response format is small and stable.

Reliability features:
  * JSON mode (responseMimeType) so the model returns parseable JSON.
  * Automatic retries with backoff for rate limits (429) and server errors (5xx),
    which matter on the free tier.
  * A fallback model if the primary one keeps failing or isn't available.
  * Optional Google Search grounding (`use_search=True`), used when a page could not
    be read. If that call fails for any reason we fall back to the plain JSON call.
  * The API key travels in a header and is never logged or put in a URL.
"""

import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field

import httpx

from app.config import Settings
from app.exceptions import LLMConfigError, LLMError, LLMOutputError, LLMRateLimitError

logger = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
TRANSIENT_STATUS = {429, 500, 502, 503, 504}


@dataclass
class LLMResult:
    data: dict
    model: str
    prompt_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    grounded: bool = False  # True if Google Search was actually used for this answer
    sources: list[str] = field(default_factory=list)  # titles/domains the search used


class _ModelUnavailable(Exception):
    """Internal: this model name isn't usable right now, try the fallback."""

    def __init__(self, error: Exception):
        self.error = error


class GeminiClient:
    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep=asyncio.sleep,
    ):
        self.settings = settings
        self._transport = transport  # tests inject a fake network here
        self._sleep = sleep

    async def generate_json(
        self, system_prompt: str, user_prompt: str, use_search: bool = False
    ) -> LLMResult:
        if not self.settings.gemini_api_key:
            logger.error("GEMINI_API_KEY is not set")
            raise LLMConfigError()

        if use_search:
            # One quick attempt with Google Search. Any problem (tool not supported on this
            # model/plan, quota, bad output) just means we continue with the plain call below.
            try:
                return await self._call_with_retries(
                    self.settings.gemini_model, system_prompt, user_prompt, search=True, attempts=1
                )
            except (_ModelUnavailable, LLMError, LLMRateLimitError, LLMOutputError) as exc:
                detail = getattr(exc, "message", None) or getattr(getattr(exc, "error", None), "message", "")
                logger.warning("Search-grounded call failed (%s: %s); using plain JSON mode", type(exc).__name__, detail)

        models = [self.settings.gemini_model]
        fallback = self.settings.gemini_fallback_model
        if fallback and fallback not in models:
            models.append(fallback)

        last_error: Exception = LLMError()
        for model in models:
            try:
                return await self._call_with_retries(model, system_prompt, user_prompt)
            except _ModelUnavailable as unavailable:
                last_error = unavailable.error
                logger.warning("Model %s unavailable, trying fallback if any", model)
        raise last_error

    # ---------------------------------------------------------------------------------
    async def _call_with_retries(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        search: bool = False,
        attempts: int | None = None,
    ) -> LLMResult:
        attempts = attempts or max(1, self.settings.llm_max_retries + 1)
        last_error: Exception = LLMError()

        for attempt in range(attempts):
            started = time.perf_counter()
            try:
                payload = await self._post(model, system_prompt, user_prompt, search)
                return self._parse(payload, model, int((time.perf_counter() - started) * 1000))
            except (LLMRateLimitError, LLMError) as exc:
                # Only temporary problems land here. Bad credentials (LLMConfigError),
                # unusable output (LLMOutputError) and "model not found" are NOT retried.
                last_error = exc
                if attempt < attempts - 1:
                    delay = self.settings.llm_retry_base_delay * (2**attempt) + random.uniform(0, 0.5)
                    logger.info("Gemini transient error (%s); retry %d in %.1fs", model, attempt + 1, delay)
                    await self._sleep(delay)

        raise _ModelUnavailable(last_error)

    async def _post(self, model: str, system_prompt: str, user_prompt: str, search: bool = False) -> dict:
        config = {"temperature": 0.3, "maxOutputTokens": 8192}
        if not search:
            # JSON mode. Search grounding is used without it (the prompt asks for JSON and we
            # parse it ourselves), because the two features are not allowed together on every model.
            config["responseMimeType"] = "application/json"
        body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": config,
        }
        if search:
            body["tools"] = [{"google_search": {}}]
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.llm_timeout_seconds, transport=self._transport
            ) as client:
                response = await client.post(
                    f"{GEMINI_BASE_URL}/{model}:generateContent",
                    headers={"x-goog-api-key": self.settings.gemini_api_key},
                    json=body,
                )
        except httpx.TimeoutException:
            raise LLMError("The AI provider timed out.") from None
        except httpx.HTTPError:
            raise LLMError("Could not connect to the AI provider.") from None

        status = response.status_code
        if status == 200:
            try:
                return response.json()
            except ValueError:
                raise LLMError("The AI provider returned an unreadable response.") from None
        if status in (401, 403) or (status == 400 and "API key" in response.text):
            logger.error("Gemini rejected the API key (HTTP %s)", status)
            raise LLMConfigError()
        if status == 404:
            logger.error("Gemini model %s not found", model)
            raise _ModelUnavailable(LLMError(f"AI model '{model}' is not available."))
        if status == 429:
            raise LLMRateLimitError()
        if status in TRANSIENT_STATUS:
            raise LLMError(f"The AI provider had a temporary problem (HTTP {status}).")
        logger.error("Gemini returned HTTP %s: %s", status, response.text[:300])
        raise LLMError(f"The AI provider rejected the request (HTTP {status}).")

    # ---------------------------------------------------------------------------------
    def _parse(self, payload: dict, model: str, latency_ms: int) -> LLMResult:
        candidates = payload.get("candidates") or []
        if not candidates:
            reason = (payload.get("promptFeedback") or {}).get("blockReason", "no candidates")
            logger.warning("Gemini returned no candidates (%s)", reason)
            raise LLMOutputError("The AI could not produce an analysis for this content.")

        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
        if not text.strip():
            raise LLMOutputError()

        grounding = candidate.get("groundingMetadata") or {}
        sources: list[str] = []
        for chunk in grounding.get("groundingChunks") or []:
            web = chunk.get("web") or {}
            label = (web.get("title") or "").strip()
            if label and label not in sources:
                sources.append(label)

        usage = payload.get("usageMetadata") or {}
        try:
            data = _load_json_object(text)
        except LLMOutputError:
            logger.warning(
                "Reply was not parseable JSON (finishReason=%s, %d chars). Start: %r ... End: %r",
                candidate.get("finishReason"), len(text), text[:150], text[-150:],
            )
            raise
        return LLMResult(
            data=data,
            model=model,
            prompt_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
            latency_ms=latency_ms,
            grounded=bool(grounding.get("groundingChunks") or grounding.get("webSearchQueries")),
            sources=sources[:5],
        )


def _load_json_object(text: str) -> dict:
    """Parse the model's JSON, repairing the slips models really make.

    Order: as-is, then trailing commas / invalid backslash escapes (e.g. \\' in "Python\\'s"),
    then the json-repair library (unescaped inner quotes, etc.) as a last resort.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    braces = _outermost_braces(cleaned)
    bases = [b for b in (cleaned, braces) if b]
    candidates = list(bases)
    for base in bases:
        fixed = re.sub(r"\\(?![\"\\/bfnrt]|u[0-9a-fA-F]{4})", "", base)  # invalid escapes: drop the backslash
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)  # trailing commas
        candidates.append(fixed)

    first_error = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate, strict=False)
        except ValueError as exc:
            first_error = first_error or exc
            continue
        if isinstance(parsed, dict):
            return parsed

    if first_error is not None:
        pos = getattr(first_error, "pos", 0) or 0
        logger.warning("JSON error: %s | around: %r", first_error, bases[0][max(0, pos - 60): pos + 60] if bases else "")
    try:
        from json_repair import loads as repair_loads

        for base in bases:
            repaired = repair_loads(base)
            if isinstance(repaired, dict) and repaired:
                logger.info("JSON was repaired with json-repair")
                return repaired
    except Exception:  # library missing or it could not help
        pass
    raise LLMOutputError()


def _outermost_braces(text: str) -> str | None:
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else None
