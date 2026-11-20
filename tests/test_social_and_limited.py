"""Social links and pages that cannot be read.

This is the scenario that failed in review: an Instagram link answered HTTP 429 and the API
returned an error. Now the API returns a clearly labelled, lower-confidence analysis instead.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.exceptions import (
    AccessRestrictedError,
    LLMError,
    UnreadableContentError,
    WebsiteUnreachableError,
)
from app.main import app, get_analysis_service
from app.services.analysis_service import AnalysisService
from app.services.website_extractor import PageContent, WebsiteContent
from app.utils.social import detect_social
from tests.helpers import FakeExtractor, FakeLLM, good_output

INSTAGRAM = "https://www.instagram.com/crestspot"
BLOCKED = AccessRestrictedError(
    "The website refused our request (HTTP 429). It likely blocks automated visitors.", status=429
)


def make_service(llm, extractor) -> AnalysisService:
    return AnalysisService(extractor, llm, Settings())


def analyze(service, url=INSTAGRAM):
    return asyncio.run(service.analyze(url))


# ---------------------------------------------------------------------------------------
class TestDetectSocial:
    @pytest.mark.parametrize(
        "url, platform, handle, kind",
        [
            ("https://www.instagram.com/crestspot", "Instagram", "crestspot", "profile"),
            ("https://instagram.com/crestspot/?hl=en", "Instagram", "crestspot", "profile"),
            ("https://x.com/OpenAI", "X", "OpenAI", "profile"),
            ("https://twitter.com/@openai", "X", "openai", "profile"),
            ("https://x.com/OpenAI/status/12345", "X", "OpenAI", "post"),
            ("https://www.facebook.com/profile.php?id=12345", "Facebook", "12345", "profile"),
            ("https://www.facebook.com/dxoratech", "Facebook", "dxoratech", "profile"),
            ("https://www.linkedin.com/company/dxora/", "LinkedIn", "dxora", "company"),
            ("https://www.linkedin.com/in/roshangupta09", "LinkedIn", "roshangupta09", "person"),
            ("https://www.youtube.com/@Google", "YouTube", "Google", "channel"),
            ("https://www.youtube.com/watch?v=abc", "YouTube", "", "post"),
            ("https://www.instagram.com/p/ABC123/", "Instagram", "", "post"),
            ("https://www.tiktok.com/@someone", "TikTok", "someone", "profile"),
            ("https://www.reddit.com/r/python", "Reddit", "python", "community"),
        ],
    )
    def test_recognizes_platform_account_and_kind(self, url, platform, handle, kind):
        link = detect_social(url)
        assert (link.platform, link.handle, link.kind) == (platform, handle, kind)

    @pytest.mark.parametrize("url", ["https://example.com", "https://notinstagram.com/crest", "https://instagram.com.evil.io/x"])
    def test_ignores_everything_else(self, url):
        assert detect_social(url) is None

    def test_handle_is_sanitized_because_it_goes_into_a_prompt(self):
        link = detect_social("https://www.instagram.com/crest%20spot%0Aignore-instructions<>")
        assert link.handle and all(c.isalnum() or c in "._-" for c in link.handle)


# ---------------------------------------------------------------------------------------
class TestBlockedLinksStillGetAnAnswer:
    def test_instagram_429_returns_a_labelled_link_only_analysis(self):
        llm = FakeLLM(good_output())
        result = analyze(make_service(llm, FakeExtractor(error=BLOCKED)))
        meta = result.meta
        assert result.success
        assert (meta.content_source, meta.platform, meta.account) == ("url_only", "Instagram", "crestspot")
        assert any("could not be read" in w and "429" in w for w in meta.warnings)
        assert llm.search_flags == [True]  # asked the model to look the account up
        assert meta.grounded and meta.grounding_sources == ["example-source.com"]
        assert any("Google Search" in w for w in meta.warnings)
        assert "crestspot" in llm.prompts[0] and "Instagram" in llm.prompts[0]

    def test_unreadable_page_also_gets_an_answer(self):
        result = analyze(make_service(FakeLLM(good_output()), FakeExtractor(error=UnreadableContentError())), "https://docmind.space")
        assert result.success and result.meta.content_source == "url_only"
        assert any("JavaScript" in w for w in result.meta.warnings)

    def test_nothing_is_labelled_website_when_no_page_was_read(self):
        result = analyze(make_service(FakeLLM(good_output()), FakeExtractor(error=BLOCKED)))
        assert {s.source for s in result.analysis.top_services} == {"inferred"}

    def test_empty_competitor_list_is_allowed_when_the_business_is_unknown(self):
        data = good_output()
        data["analysis"]["competitor_analysis"]["competitors"] = []
        result = analyze(make_service(FakeLLM(data), FakeExtractor(error=BLOCKED)))
        assert result.success and any("competitor" in w.lower() for w in result.meta.warnings)

    def test_missing_brand_name_falls_back_to_the_account_name(self):
        data = good_output()
        data["brand"]["name"] = ""
        assert analyze(make_service(FakeLLM(data), FakeExtractor(error=BLOCKED))).brand.name == "crestspot"

    def test_still_asks_the_model_to_fix_an_incomplete_answer(self):
        broken = good_output()
        broken["analysis"]["target_search_queries"] = broken["analysis"]["target_search_queries"][:2]
        llm = FakeLLM(broken, good_output())
        assert analyze(make_service(llm, FakeExtractor(error=BLOCKED))).success
        assert llm.calls == 2

    def test_link_only_results_are_not_cached(self):
        llm = FakeLLM(good_output())
        service = make_service(llm, FakeExtractor(error=BLOCKED))
        analyze(service)
        assert analyze(service).meta.cached is False
        assert llm.calls == 2

    def test_a_login_page_that_never_mentions_the_account_is_treated_as_unreadable(self):
        wall = WebsiteContent(
            requested_url=INSTAGRAM, final_url=INSTAGRAM,
            pages=[PageContent(url=INSTAGRAM, title="Instagram", text="Log in to see photos and videos from friends.")],
        )
        result = analyze(make_service(FakeLLM(good_output()), FakeExtractor(content=wall)))
        assert result.meta.content_source == "url_only"

    def test_a_real_profile_page_is_analyzed_normally(self):
        page = PageContent(
            url=INSTAGRAM, title="CREST (@crestspot) - Instagram photos and videos",
            meta_description="120 Followers - CREST by Dxora", text="CREST " * 40,
        )
        content = WebsiteContent(requested_url=INSTAGRAM, final_url=INSTAGRAM, pages=[page])
        llm = FakeLLM(good_output())
        result = analyze(make_service(llm, FakeExtractor(content=content)))
        assert result.meta.content_source == "page" and result.meta.platform == "Instagram"
        assert llm.search_flags == [False]
        assert "behind the account" in llm.prompts[0]


class TestRealErrorsStayErrors:
    def test_unknown_domain_or_missing_page_is_not_hidden(self):
        with pytest.raises(WebsiteUnreachableError):
            analyze(make_service(FakeLLM(good_output()), FakeExtractor(error=WebsiteUnreachableError())), "https://acme.example")

    def test_llm_failure_is_still_reported(self):
        with pytest.raises(LLMError):
            analyze(make_service(FakeLLM(LLMError()), FakeExtractor(error=BLOCKED)))


class TestApi:
    @pytest.fixture
    def client(self):
        def install(llm, extractor):
            service = make_service(llm, extractor)
            app.dependency_overrides[get_analysis_service] = lambda: service
            return TestClient(app, raise_server_exceptions=False)

        yield install
        app.dependency_overrides.clear()

    def test_instagram_link_returns_200_not_an_error(self, client):
        response = client(FakeLLM(good_output()), FakeExtractor(error=BLOCKED)).post("/link-analysis", json={"url": INSTAGRAM})
        body = response.json()
        assert response.status_code == 200 and body["success"] is True
        assert body["meta"]["content_source"] == "url_only"
        assert len(body["analysis"]["target_search_queries"]) == 5

    @pytest.mark.parametrize("url", [
        "https://x.com/OpenAI", "https://www.facebook.com/dxoratech", "https://www.linkedin.com/company/dxora/",
        "https://www.youtube.com/@Google", "https://www.tiktok.com/@someone",
    ])
    def test_other_platforms_that_refuse_us_also_work(self, client, url):
        response = client(FakeLLM(good_output()), FakeExtractor(error=BLOCKED)).post("/link-analysis", json={"url": url})
        assert response.status_code == 200 and response.json()["success"] is True


class TestOutages:
    def test_social_link_that_times_out_still_gets_an_answer(self):
        from app.exceptions import WebsiteDownError

        err = WebsiteDownError("The website took too long to respond.")
        result = analyze(make_service(FakeLLM(good_output()), FakeExtractor(error=err)), "https://www.linkedin.com/company/dxora-technologies")
        assert result.meta.content_source == "url_only" and result.meta.platform == "LinkedIn"
        assert any("took too long" in w for w in result.meta.warnings)

    def test_ordinary_site_outage_is_still_an_error(self):
        from app.exceptions import WebsiteDownError

        err = WebsiteDownError("The website took too long to respond.")
        with pytest.raises(WebsiteDownError):
            analyze(make_service(FakeLLM(good_output()), FakeExtractor(error=err)), "https://acme.example")

    def test_a_social_page_that_really_does_not_exist_is_still_an_error(self):
        with pytest.raises(WebsiteUnreachableError):
            analyze(make_service(FakeLLM(good_output()), FakeExtractor(error=WebsiteUnreachableError("That page was not found (HTTP 404)."))),
                    "https://www.linkedin.com/company/dxora")
