import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.exceptions import LLMConfigError, LLMError, LLMOutputError, LLMRateLimitError
from app.services.llm_service import GeminiClient


async def no_sleep(_):
    return None


def gemini_ok(text: str) -> httpx.Response:
    body = {
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 1500, "candidatesTokenCount": 800},
    }
    return httpx.Response(200, json=body)


def make_client(handler, **overrides) -> GeminiClient:
    settings = Settings(
        gemini_api_key="secret-key", gemini_model="primary-model", gemini_fallback_model="backup-model",
        llm_max_retries=2, llm_retry_base_delay=0, **overrides,
    )
    return GeminiClient(settings, transport=httpx.MockTransport(handler), sleep=no_sleep)


def run(client: GeminiClient):
    return asyncio.run(client.generate_json("system", "user"))


def test_parses_json_and_token_usage():
    result = run(make_client(lambda r: gemini_ok('{"a": 1}')))
    assert result.data == {"a": 1}
    assert result.model == "primary-model"
    assert (result.prompt_tokens, result.output_tokens) == (1500, 800)


def test_accepts_fenced_or_chatty_json():
    assert run(make_client(lambda r: gemini_ok('```json\n{"a": 2}\n```'))).data == {"a": 2}
    assert run(make_client(lambda r: gemini_ok('Sure! Here you go: {"a": 3} Hope it helps'))).data == {"a": 3}


def test_api_key_is_sent_in_header_not_url():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["body"] = json.loads(request.content)
        return gemini_ok("{}")

    run(make_client(handler))
    assert "secret-key" not in seen["url"]
    assert seen["key"] == "secret-key"
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert "primary-model:generateContent" in seen["url"]


def test_retries_rate_limit_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(429, text="slow down") if calls["n"] < 3 else gemini_ok('{"ok": true}')

    assert run(make_client(handler)).data == {"ok": True}
    assert calls["n"] == 3


def test_uses_fallback_model_when_primary_keeps_failing():
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(503, text="overloaded") if "primary-model" in str(request.url) else gemini_ok("{}")

    result = run(make_client(handler))
    assert result.model == "backup-model"
    assert any("backup-model" in u for u in urls)


def test_uses_fallback_when_primary_model_not_found():
    handler = lambda r: httpx.Response(404, text="not found") if "primary-model" in str(r.url) else gemini_ok("{}")
    assert run(make_client(handler)).model == "backup-model"


def test_all_rate_limited_gives_rate_limit_error():
    with pytest.raises(LLMRateLimitError):
        run(make_client(lambda r: httpx.Response(429)))


def test_all_server_errors_give_llm_error():
    with pytest.raises(LLMError):
        run(make_client(lambda r: httpx.Response(500)))


@pytest.mark.parametrize("status", [401, 403])
def test_bad_credentials_are_not_retried(status):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(status, text="denied")

    with pytest.raises(LLMConfigError):
        run(make_client(handler))
    assert calls["n"] == 1


def test_missing_api_key():
    client = GeminiClient(Settings(gemini_api_key=""), transport=httpx.MockTransport(lambda r: gemini_ok("{}")))
    with pytest.raises(LLMConfigError):
        asyncio.run(client.generate_json("s", "u"))


def test_network_error_is_retried_then_reported():
    def handler(request):
        raise httpx.ConnectError("boom", request=request)

    with pytest.raises(LLMError):
        run(make_client(handler))


@pytest.mark.parametrize("text", ["not json at all", "[1, 2, 3]", ""])
def test_unusable_output_raises_output_error(text):
    with pytest.raises(LLMOutputError):
        run(make_client(lambda r: gemini_ok(text)))


def test_blocked_prompt_raises_output_error():
    handler = lambda r: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(LLMOutputError):
        run(make_client(handler))


def test_search_mode_sends_search_tool_without_json_mode_and_reads_sources():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "candidates": [{
                "content": {"parts": [{"text": 'Here you go: {"a": 1}'}]},
                "groundingMetadata": {"groundingChunks": [{"web": {"title": "crestspot.cloud", "uri": "https://x"}}]},
            }],
        })

    result = asyncio.run(make_client(handler).generate_json("s", "u", use_search=True))
    assert seen["body"]["tools"] == [{"google_search": {}}]
    assert "responseMimeType" not in seen["body"]["generationConfig"]
    assert result.data == {"a": 1}
    assert result.grounded is True and result.sources == ["crestspot.cloud"]


def test_search_mode_falls_back_to_plain_json_when_the_tool_is_rejected():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "tools" in body:
            return httpx.Response(400, text="google_search tool not supported here")
        return gemini_ok('{"b": 2}')

    result = asyncio.run(make_client(handler).generate_json("s", "u", use_search=True))
    assert result.data == {"b": 2} and result.grounded is False
    assert len(bodies) == 2 and "tools" not in bodies[1]


def test_search_mode_does_not_hide_a_bad_api_key():
    with pytest.raises(LLMConfigError):
        asyncio.run(make_client(lambda r: httpx.Response(403, text="denied")).generate_json("s", "u", use_search=True))
