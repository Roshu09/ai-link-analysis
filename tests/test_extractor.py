import asyncio

import httpx
import pytest

from app.config import Settings
from app.exceptions import (
    BlockedURLError,
    UnreadableContentError,
    UnsupportedContentError,
    WebsiteUnreachableError,
)
from app.services.website_extractor import WebsiteExtractor, parse_page, pick_key_links

HOME_HTML = """
<html lang="en"><head>
  <title>Acme Web Studio | Websites</title>
  <meta name="description" content="We build affordable websites.">
  <meta property="og:site_name" content="Acme">
  <script type="application/ld+json">
    {"@type": "LocalBusiness", "name": "Acme Web Studio", "description": "Web agency",
     "address": {"addressLocality": "Austin", "addressCountry": "US"}}
  </script>
  <script>var secret = "SCRIPT_TEXT";</script>
  <style>.x{color:red}</style>
</head><body>
  <header><h1>Websites that grow your business</h1></header>
  <nav><a href="/about">About us</a><a href="/services">Services</a><a href="/contact">Contact</a></nav>
  <main>
    <h2>What we do</h2>
    <p>We design and build fast, affordable websites and online stores for small businesses.</p>
    <p>We design and build fast, affordable websites and online stores for small businesses.</p>
    <p>Every project includes SEO setup, training, and ongoing support from our friendly team.</p>
  </main>
  <footer>Contact us: hello@acme.example. 123 Main St, Austin.</footer>
</body></html>
"""

ABOUT_HTML = "<html><body><main><h1>About</h1><p>" + "Our story and mission of building great sites. " * 6 + "</p></main></body></html>"
SERVICES_HTML = "<html><body><main><h1>Services</h1><p>" + "Web design, e-commerce and SEO services. " * 6 + "</p></main></body></html>"


def make_extractor(handler) -> WebsiteExtractor:
    return WebsiteExtractor(Settings(max_extra_pages=2), transport=httpx.MockTransport(handler))


def html_response(body: str, status: int = 200, content_type: str = "text/html; charset=utf-8"):
    return httpx.Response(status, content=body.encode(), headers={"content-type": content_type})


class TestParsePage:
    def test_extracts_the_useful_parts(self):
        page = parse_page(HOME_HTML, "https://acme.example/")
        assert page.title == "Acme Web Studio | Websites"
        assert page.meta_description == "We build affordable websites."
        assert page.site_name == "Acme"
        assert page.language == "en"
        assert "Websites that grow your business" in page.headings
        assert page.nav_labels == ["About us", "Services", "Contact"]
        assert any("Austin" in fact for fact in page.structured_data)

    def test_removes_scripts_styles_and_duplicates(self):
        page = parse_page(HOME_HTML, "https://acme.example/")
        assert "SCRIPT_TEXT" not in page.text
        assert "color:red" not in page.text
        assert page.text.count("We design and build fast") == 1
        assert "hello@acme.example" not in page.text  # footer is kept separately
        assert "hello@acme.example" in page.footer_note

    def test_respects_body_budget(self):
        html = "<main>" + "".join(f"<p>Line number {i} with some words</p>" for i in range(500)) + "</main>"
        assert len(parse_page(html, "https://x.example/", body_budget=300).text) <= 300

    def test_survives_garbage_html(self):
        page = parse_page(b"<div><p>unclosed <b>tags", "https://x.example/")
        assert "unclosed" in page.text


class TestPickKeyLinks:
    def test_prefers_about_and_services_pages(self):
        page = parse_page(HOME_HTML, "https://acme.example/")
        links = pick_key_links(page, 2)
        assert set(links) == {"https://acme.example/about", "https://acme.example/services"}

    def test_skips_legal_and_account_pages(self):
        html = (
            '<a href="/legal/terms">Terms of Service</a>'
            '<a href="/privacy-policy">Privacy and services policy</a>'
            '<a href="/login">Log in to our platform</a>'
            '<a href="/pricing">Pricing</a>'
        )
        page = parse_page(html, "https://shop.example/")
        assert pick_key_links(page, 3) == ["https://shop.example/pricing"]

    def test_ignores_files_and_external_sites(self):
        html = '<a href="/brochure-services.pdf">Services PDF</a><a href="https://other.com/about">About</a>'
        page = parse_page(html, "https://acme.example/")
        assert pick_key_links(page, 3) == []


class TestFetching:
    def test_reads_home_and_key_pages(self):
        def handler(request: httpx.Request):
            return html_response({"/": HOME_HTML, "/about": ABOUT_HTML, "/services": SERVICES_HTML}.get(request.url.path, ""),
                                 200 if request.url.path in ("/", "/about", "/services") else 404)

        content = asyncio.run(make_extractor(handler).extract("https://acme.example/"))
        assert [p.url for p in content.pages] == [
            "https://acme.example/", "https://acme.example/about", "https://acme.example/services"
        ]
        assert "Web design, e-commerce" in content.to_prompt_text(20_000)

    def test_missing_extra_page_only_adds_a_warning(self):
        def handler(request: httpx.Request):
            if request.url.path == "/":
                return html_response(HOME_HTML)
            return html_response("nope", 500)

        content = asyncio.run(make_extractor(handler).extract("https://acme.example/"))
        assert len(content.pages) == 1
        assert content.warnings

    def test_follows_safe_redirects(self):
        def handler(request: httpx.Request):
            if request.url.host == "old.example.com":
                return httpx.Response(301, headers={"location": "https://new.example.com/"})
            return html_response(HOME_HTML)

        content = asyncio.run(make_extractor(handler).extract("https://old.example.com/"))
        assert content.final_url == "https://new.example.com/"

    def test_blocks_redirect_to_internal_address(self):
        def handler(request: httpx.Request):
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})

        with pytest.raises(BlockedURLError):
            asyncio.run(make_extractor(handler).extract("https://evil.example.com/"))

    def test_blocks_redirect_to_host_resolving_privately(self):
        def handler(request: httpx.Request):
            return httpx.Response(302, headers={"location": "https://intranet.example.com/"})

        with pytest.raises(BlockedURLError):
            asyncio.run(make_extractor(handler).extract("https://evil.example.com/"))

    def test_redirect_loop_gives_up(self):
        def handler(request: httpx.Request):
            return httpx.Response(302, headers={"location": "https://loop.example.com/"})

        with pytest.raises(WebsiteUnreachableError):
            asyncio.run(make_extractor(handler).extract("https://loop.example.com/"))

    @pytest.mark.parametrize("status", [403, 404, 500])
    def test_http_errors_become_friendly_errors(self, status):
        with pytest.raises(WebsiteUnreachableError):
            asyncio.run(make_extractor(lambda r: html_response("x", status)).extract("https://acme.example/"))

    def test_timeout_is_handled(self):
        def handler(request: httpx.Request):
            raise httpx.ConnectTimeout("slow", request=request)

        with pytest.raises(WebsiteUnreachableError):
            asyncio.run(make_extractor(handler).extract("https://acme.example/"))

    def test_non_html_is_rejected(self):
        handler = lambda r: httpx.Response(200, content=b"%PDF-1.4", headers={"content-type": "application/pdf"})
        with pytest.raises(UnsupportedContentError):
            asyncio.run(make_extractor(handler).extract("https://acme.example/file"))

    def test_empty_javascript_shell_is_reported(self):
        shell = '<html><head><title>App</title></head><body><div id="root"></div><script src="/app.js"></script></body></html>'
        with pytest.raises(UnreadableContentError):
            asyncio.run(make_extractor(lambda r: html_response(shell)).extract("https://spa.example.com/"))


class TestBlockedSites:
    @pytest.mark.parametrize("status", [401, 403, 405, 429, 999])
    def test_refusals_are_flagged_as_access_restricted(self, status):
        from app.exceptions import AccessRestrictedError

        with pytest.raises(AccessRestrictedError) as caught:
            asyncio.run(make_extractor(lambda r: html_response("no", status)).extract("https://www.instagram.com/crestspot"))
        assert caught.value.http_status == status
        assert "refused" in caught.value.message

    @pytest.mark.parametrize("status", [404, 410, 500])
    def test_missing_or_broken_pages_are_not_treated_as_blocked(self, status):
        from app.exceptions import AccessRestrictedError

        with pytest.raises(WebsiteUnreachableError) as caught:
            asyncio.run(make_extractor(lambda r: html_response("x", status)).extract("https://acme.example/"))
        assert not isinstance(caught.value, AccessRestrictedError)


class TestFailureClassification:
    @pytest.mark.parametrize("status", [400, 402, 418, 451])
    def test_other_client_errors_count_as_refusals(self, status):
        from app.exceptions import AccessRestrictedError

        with pytest.raises(AccessRestrictedError):
            asyncio.run(make_extractor(lambda r: html_response("no", status)).extract("https://acme.example/"))

    @pytest.mark.parametrize("status", [500, 502, 503])
    def test_server_errors_are_marked_as_down(self, status):
        from app.exceptions import WebsiteDownError

        with pytest.raises(WebsiteDownError):
            asyncio.run(make_extractor(lambda r: html_response("x", status)).extract("https://acme.example/"))

    def test_timeouts_and_connection_errors_are_marked_as_down(self):
        from app.exceptions import WebsiteDownError

        def timeout(request):
            raise httpx.ConnectTimeout("slow", request=request)

        def refused(request):
            raise httpx.ConnectError("no route", request=request)

        for handler in (timeout, refused):
            with pytest.raises(WebsiteDownError):
                asyncio.run(make_extractor(handler).extract("https://acme.example/"))

    def test_redirect_loop_counts_as_a_refusal(self):
        from app.exceptions import AccessRestrictedError

        handler = lambda r: httpx.Response(302, headers={"location": "https://loop.example.com/"})
        with pytest.raises(AccessRestrictedError):
            asyncio.run(make_extractor(handler).extract("https://loop.example.com/"))

    def test_404_is_a_real_not_found(self):
        from app.exceptions import AccessRestrictedError, WebsiteDownError

        with pytest.raises(WebsiteUnreachableError) as caught:
            asyncio.run(make_extractor(lambda r: html_response("x", 404)).extract("https://acme.example/nope"))
        assert not isinstance(caught.value, (AccessRestrictedError, WebsiteDownError))
        assert "not found" in caught.value.message
