"""Tests for the analysis pipeline and the HTTP API, using fake extractor/LLM objects."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.exceptions import (
    LLMError,
    LLMOutputError,
    LLMRateLimitError,
    UnreadableContentError,
    WebsiteUnreachableError,
)
from app.main import app, get_analysis_service
from app.services.analysis_service import AnalysisService
from tests.helpers import FakeExtractor, FakeLLM, good_output


def make_service(llm: FakeLLM, extractor: FakeExtractor | None = None) -> AnalysisService:
    return AnalysisService(extractor or FakeExtractor(), llm, Settings())


def analyze(service: AnalysisService, url: str = "https://acme.example/"):
    return asyncio.run(service.analyze(url))


# ---------------------------------------------------------------------------------------
# Pipeline behaviour
# ---------------------------------------------------------------------------------------
class TestPipeline:
    def test_happy_path_returns_all_five_sections(self):
        result = analyze(make_service(FakeLLM(good_output())))
        analysis = result.analysis
        assert result.success and result.brand.name == "Acme Web Studio"
        assert len(analysis.target_search_queries) == 5
        assert len(analysis.top_services) == 5
        assert analysis.competitor_analysis.competitors
        assert analysis.seo_improvements.primary_keyword
        assert analysis.target_audience.primary_audience
        assert result.meta.model_used == "fake-model"
        assert (result.meta.prompt_tokens, result.meta.output_tokens) == (1200, 900)

    def test_platform_names_are_normalized(self):
        queries = analyze(make_service(FakeLLM(good_output()))).analysis.target_search_queries
        assert queries[3].platforms == ["X"]  # "Twitter" -> "X"
        assert queries[4].platforms == ["Google", "LinkedIn"]  # "google search" -> "Google"

    def test_competitors_are_cleaned_and_labelled(self):
        ca = analyze(make_service(FakeLLM(good_output()))).analysis.competitor_analysis
        names = [c.name for c in ca.competitors]
        assert "Acme Web Studio" not in names  # the brand itself is removed
        assert all(c.source == "ai_suggested" for c in ca.competitors)
        assert ca.competitors[0].website == "https://www.wix.com"
        assert ca.competitors[1].website == "https://squarespace.com"
        assert ca.comparison_table[0].aspect == "Brand Name"
        assert set(ca.comparison_table[1].competitors) == set(names)

    def test_extra_search_queries_are_trimmed_to_five(self):
        data = good_output()
        data["analysis"]["target_search_queries"].append(data["analysis"]["target_search_queries"][0])
        assert len(analyze(make_service(FakeLLM(data))).analysis.target_search_queries) == 5

    def test_long_meta_description_is_shortened(self):
        data = good_output()
        data["analysis"]["seo_improvements"]["meta_description"] = "Great web design for growing teams. " * 10
        seo = analyze(make_service(FakeLLM(data))).analysis.seo_improvements
        assert seo.meta_description_length <= 160

    def test_model_gets_one_corrective_retry(self):
        broken = good_output()
        broken["analysis"]["target_search_queries"] = broken["analysis"]["target_search_queries"][:3]
        llm = FakeLLM(broken, good_output())
        result = analyze(make_service(llm))
        assert llm.calls == 2
        assert len(result.analysis.target_search_queries) == 5

    def test_retries_after_invalid_json_too(self):
        llm = FakeLLM(LLMOutputError(), good_output())
        assert analyze(make_service(llm)).success
        assert llm.calls == 2

    def test_gives_up_after_second_bad_answer(self):
        broken = good_output()
        broken["analysis"]["top_services"] = []
        llm = FakeLLM(broken)
        with pytest.raises(LLMOutputError):
            analyze(make_service(llm))
        assert llm.calls == 2

    def test_second_request_is_served_from_cache(self):
        llm, extractor = FakeLLM(good_output()), FakeExtractor()
        service = make_service(llm, extractor)
        first = analyze(service)
        second = analyze(service, "https://ACME.example")  # same site, different spelling
        assert (llm.calls, extractor.calls) == (1, 1)
        assert not first.meta.cached and second.meta.cached

    def test_failures_are_not_cached(self):
        llm = FakeLLM(LLMError(), good_output())
        service = make_service(llm)
        with pytest.raises(LLMError):
            analyze(service)
        assert analyze(service).success

    def test_thin_site_adds_warnings_but_still_works(self):
        data = good_output()
        for service in data["analysis"]["top_services"]:
            service["source"] = "inferred"
        result = analyze(make_service(FakeLLM(data)))
        assert any("inferred" in w for w in result.meta.warnings)


# ---------------------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------------------
@pytest.fixture
def client():
    def install(llm: FakeLLM, extractor: FakeExtractor | None = None) -> TestClient:
        service = make_service(llm, extractor)
        app.dependency_overrides[get_analysis_service] = lambda: service
        return TestClient(app, raise_server_exceptions=False)

    yield install
    app.dependency_overrides.clear()


class TestApi:
    def test_success_response_shape(self, client):
        response = client(FakeLLM(good_output())).post("/link-analysis", json={"url": "https://acme.example"})
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["website"] == "https://acme.example/"
        assert set(body["analysis"]) == {
            "competitor_analysis", "target_search_queries", "seo_improvements", "top_services", "target_audience",
        }
        assert {"name", "category", "summary"} <= set(body["brand"])

    def test_url_without_scheme_is_accepted(self, client):
        assert client(FakeLLM(good_output())).post("/link-analysis", json={"url": "acme.example"}).status_code == 200

    @pytest.mark.parametrize("payload", [{}, {"url": ""}, {"url": 123}, {"link": "https://x.com"}])
    def test_bad_request_body(self, client, payload):
        response = client(FakeLLM(good_output())).post("/link-analysis", json=payload)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"

    def test_not_json_body(self, client):
        response = client(FakeLLM(good_output())).post("/link-analysis", content="hello")
        assert response.status_code == 422 and response.json()["success"] is False

    @pytest.mark.parametrize("url", ["not a url", "ftp://example.com", "http://example"])
    def test_invalid_urls(self, client, url):
        response = client(FakeLLM(good_output())).post("/link-analysis", json={"url": url})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_url"

    @pytest.mark.parametrize("url", ["http://localhost:4000", "http://127.0.0.1", "http://169.254.169.254/latest"])
    def test_internal_urls_are_blocked_before_any_fetch(self, client, url):
        extractor = FakeExtractor()
        response = client(FakeLLM(good_output()), extractor).post("/link-analysis", json={"url": url})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "blocked_url"
        assert extractor.calls == 0

    @pytest.mark.parametrize(
        "error, status, code",
        [
            (WebsiteUnreachableError(), 502, "website_unreachable"),
            (UnreadableContentError(), 422, "unreadable_content"),
        ],
    )
    def test_website_problems(self, client, error, status, code):
        response = client(FakeLLM(good_output()), FakeExtractor(error=error)).post(
            "/link-analysis", json={"url": "https://acme.example"}
        )
        assert response.status_code == status
        assert response.json() == {"success": False, "error": {"code": code, "message": error.message}}

    @pytest.mark.parametrize(
        "error, status, code",
        [(LLMError(), 502, "llm_error"), (LLMRateLimitError(), 503, "llm_rate_limited")],
    )
    def test_llm_problems(self, client, error, status, code):
        response = client(FakeLLM(error)).post("/link-analysis", json={"url": "https://acme.example"})
        assert response.status_code == status
        assert response.json()["error"]["code"] == code

    def test_unexpected_crash_returns_clean_500(self, client):
        response = client(FakeLLM(RuntimeError("secret internal detail"))).post(
            "/link-analysis", json={"url": "https://acme.example"}
        )
        assert response.status_code == 500
        assert "secret internal detail" not in response.text

    def test_health_and_docs(self, client):
        c = client(FakeLLM(good_output()))
        assert c.get("/health").json() == {"status": "ok"}
        assert c.get("/docs").status_code == 200


# ---------------------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------------------
def test_rate_limiter_blocks_after_limit_and_recovers():
    from app.utils.rate_limit import SlidingWindowLimiter

    now = [0.0]
    limiter = SlidingWindowLimiter(2, 60, clock=lambda: now[0])
    assert limiter.check("a") == 0 and limiter.check("a") == 0
    assert limiter.check("a") > 0  # third request in the window is refused
    assert limiter.check("b") == 0  # other clients are unaffected
    now[0] = 61
    assert limiter.check("a") == 0  # window passed


def test_cache_expires_and_evicts():
    from app.utils.cache import TTLCache

    now = [0.0]
    cache = TTLCache(ttl_seconds=10, max_entries=2, clock=lambda: now[0])
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)  # evicts "a"
    assert cache.get("a") is None and cache.get("c") == 3
    now[0] = 11
    assert cache.get("c") is None
