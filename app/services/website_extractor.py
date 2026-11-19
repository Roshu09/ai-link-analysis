"""Fetch a website and boil it down to the text an LLM actually needs.

What we do, in plain words:
  1. Download the home page (safely: every redirect hop is re-checked for SSRF).
  2. Look through its links for "About", "Services", "Products"-style pages and
     fetch up to a couple of those too, because home pages alone are often thin.
  3. Strip scripts, styles, menus and footers; keep title, meta tags, headings,
     structured data (JSON-LD) and the main body text.
  4. Remove duplicate lines and cap the size, which keeps LLM token cost low.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup

from app.config import Settings
from app.exceptions import (
    AppError,
    UnreadableContentError,
    UnsupportedContentError,
    WebsiteUnreachableError,
)
from app.utils.validators import assert_public_url, normalize_url

logger = logging.getLogger(__name__)

REDIRECT_CODES = {301, 302, 303, 307, 308}
KEY_PAGE_HINTS = (
    "about", "service", "solution", "product", "what-we-do", "offering",
    "feature", "pricing", "company", "expertise", "industr", "platform", "capabilit",
)
# Pages that match a hint word by accident (e.g. "Terms of Service" contains "service")
# or that never describe the business. Found when a real run analyzed /legal/terms.
EXCLUDED_PAGES = re.compile(
    r"\b(legal|terms|privacy|cookies?|polic(?:y|ies)|log-?in|sign-?in|sign-?up|register|cart|checkout)\b",
    re.IGNORECASE,
)
SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".zip", ".mp4",
    ".mp3", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".css", ".js", ".xml",
)
HOME_BUDGET = 6000   # characters of body text kept from the home page
EXTRA_BUDGET = 3500  # characters kept from each additional page
MIN_READABLE_CHARS = 150
THIN_CONTENT_CHARS = 600


# ---------------------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------------------
@dataclass
class PageContent:
    url: str
    title: str = ""
    meta_description: str = ""
    site_name: str = ""
    keywords: str = ""
    language: str = ""
    structured_data: list[str] = field(default_factory=list)
    nav_labels: list[str] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    footer_note: str = ""
    text: str = ""
    links: list[tuple[str, str]] = field(default_factory=list)  # (absolute url, anchor text)

    def to_prompt_block(self, label: str) -> str:
        lines = [f"=== {label}: {self.url}"]
        if self.title:
            lines.append(f"Title: {self.title}")
        if self.site_name:
            lines.append(f"Site name: {self.site_name}")
        if self.meta_description:
            lines.append(f"Meta description: {self.meta_description}")
        if self.keywords:
            lines.append(f"Meta keywords: {self.keywords}")
        if self.language:
            lines.append(f"Language: {self.language}")
        if self.structured_data:
            lines.append("Structured data: " + " | ".join(self.structured_data))
        if self.nav_labels:
            lines.append("Navigation: " + " | ".join(self.nav_labels))
        if self.headings:
            lines.append("Headings: " + " | ".join(self.headings))
        if self.footer_note:
            lines.append(f"Footer: {self.footer_note}")
        if self.text:
            lines.append("Body text:\n" + self.text)
        return "\n".join(lines)


@dataclass
class WebsiteContent:
    requested_url: str
    final_url: str
    pages: list[PageContent]
    warnings: list[str] = field(default_factory=list)

    @property
    def total_text_chars(self) -> int:
        return sum(len(p.text) + len(p.meta_description) for p in self.pages)

    def to_prompt_text(self, max_chars: int) -> str:
        blocks = [
            page.to_prompt_block("HOME PAGE" if i == 0 else f"EXTRA PAGE {i}")
            for i, page in enumerate(self.pages)
        ]
        combined = "\n\n".join(blocks)
        if len(combined) > max_chars:
            combined = combined[:max_chars].rsplit("\n", 1)[0] + "\n[...content truncated...]"
        return combined


# ---------------------------------------------------------------------------------------
# HTML parsing (pure functions, easy to test without any network)
# ---------------------------------------------------------------------------------------
def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _unique(items: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            result.append(item)
        if len(result) >= limit:
            break
    return result


def _same_site(a: str, b: str) -> bool:
    host_a = (urlsplit(a).hostname or "").removeprefix("www.")
    host_b = (urlsplit(b).hostname or "").removeprefix("www.")
    return bool(host_a) and host_a == host_b


def _read_json_ld(soup: BeautifulSoup) -> list[str]:
    """Pull a few useful facts (name, description, area served) from schema.org data."""
    facts: list[str] = []

    def walk(node):
        if isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            kind = node.get("@type")
            kinds = kind if isinstance(kind, list) else [kind]
            if any(isinstance(k, str) for k in kinds) and node.get("name"):
                parts = [f"{', '.join(str(k) for k in kinds if k)}: {_clean(str(node['name']))}"]
                if node.get("description"):
                    parts.append(_clean(str(node["description"]))[:250])
                address = node.get("address")
                if isinstance(address, dict):
                    place = ", ".join(
                        _clean(str(address[k]))
                        for k in ("addressLocality", "addressRegion", "addressCountry")
                        if address.get(k)
                    )
                    if place:
                        parts.append(f"location {place}")
                if node.get("areaServed") and isinstance(node["areaServed"], (str, list)):
                    parts.append(f"serves {_clean(str(node['areaServed']))[:100]}")
                facts.append("; ".join(parts))
            for value in node.values():
                if isinstance(value, (list, dict)):
                    walk(value)

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            walk(json.loads(script.string or script.get_text() or ""))
        except (ValueError, TypeError):
            continue  # broken JSON-LD is common; just skip it
    return _unique(facts, 4)


def _meta(soup: BeautifulSoup, **attrs) -> str:
    tag = soup.find("meta", attrs=attrs)
    return _clean(tag.get("content", "")) if tag else ""


def parse_page(raw: bytes | str, page_url: str, body_budget: int = HOME_BUDGET) -> PageContent:
    """Extract the useful parts of one HTML page."""
    soup = BeautifulSoup(raw, "html.parser")
    page = PageContent(url=page_url)

    page.title = _clean(soup.title.get_text()) if soup.title else ""
    page.meta_description = _meta(soup, name="description") or _meta(soup, property="og:description")
    page.site_name = _meta(soup, property="og:site_name")
    page.keywords = _meta(soup, name="keywords")[:200]
    html_tag = soup.find("html")
    page.language = _clean(html_tag.get("lang", "")) if html_tag else ""
    page.structured_data = _read_json_ld(soup)

    # Links first (before we throw navigation away), so we can find About/Services pages.
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(page_url, href)
        if urlsplit(absolute).scheme in ("http", "https") and _same_site(absolute, page_url):
            page.links.append((absolute, _clean(anchor.get_text())))

    nav_texts = [
        _clean(a.get_text()) for nav in soup.find_all("nav") for a in nav.find_all("a")
    ]
    page.nav_labels = _unique([t for t in nav_texts if 2 <= len(t) <= 40], 15)

    footer = soup.find("footer")
    if footer:
        page.footer_note = _clean(footer.get_text(" "))[:300]

    # Remove noise.
    # (We keep <header>: many sites put their main headline and tagline there.)
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "template", "canvas",
                     "nav", "footer", "aside", "form"]):
        tag.decompose()

    body = soup.body or soup
    # Headings come from the whole page, since the main headline often sits in a <header>
    # hero section outside <main>.
    headings = [_clean(h.get_text()) for h in body.find_all(["h1", "h2", "h3"])]
    page.headings = _unique([h for h in headings if 3 <= len(h) <= 150], 25)

    def text_lines(node) -> list[str]:
        lines = [_clean(line) for line in node.get_text("\n").splitlines()]
        return _unique([line for line in lines if len(line) >= 3], 10_000)

    # Prefer the <main>/<article> area (less clutter), but if it's suspiciously thin,
    # some sites only wrap a small part in <main>, so use the whole body instead.
    main = soup.find("main") or soup.find("article")
    lines = text_lines(main) if main else []
    if len("\n".join(lines)) < 400:
        lines = text_lines(body)
    page.text = _truncate_lines(lines, body_budget)
    return page


def _truncate_lines(lines: list[str], budget: int) -> str:
    kept: list[str] = []
    used = 0
    for line in lines:
        if used + len(line) + 1 > budget:
            break
        kept.append(line)
        used += len(line) + 1
    return "\n".join(kept)


def pick_key_links(page: PageContent, limit: int) -> list[str]:
    """Choose the most promising 'About / Services / Products' style internal links."""
    home_path = urlsplit(page.url).path.rstrip("/")
    scored: dict[str, tuple[int, int]] = {}
    for url, anchor_text in page.links:
        parts = urlsplit(url)
        path = parts.path.rstrip("/")
        if path == home_path or path.lower().endswith(SKIP_EXTENSIONS):
            continue
        haystack = f"{path} {anchor_text}".lower()
        if EXCLUDED_PAGES.search(haystack):
            continue
        score = sum(1 for hint in KEY_PAGE_HINTS if hint in haystack)
        if score == 0:
            continue
        clean_url = urlunsplit((parts.scheme, parts.netloc, parts.path or "/", "", ""))
        # higher score first; shorter path breaks ties
        scored.setdefault(clean_url, (-score, len(path)))
    return [url for url, _ in sorted(scored.items(), key=lambda item: item[1])[:limit]]


# ---------------------------------------------------------------------------------------
# The extractor service
# ---------------------------------------------------------------------------------------
class WebsiteExtractor:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self._transport = transport  # tests inject a fake network here

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self.settings.fetch_timeout_seconds,
            follow_redirects=False,  # we follow redirects ourselves, checking each hop
            headers={
                "User-Agent": self.settings.user_agent,
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
                "Accept-Language": "en-US,en;q=0.9",
            },
            transport=self._transport,
        )

    async def extract(self, url: str) -> WebsiteContent:
        warnings: list[str] = []
        async with self._client() as client:
            final_url, raw = await self._fetch(client, url)
            home = parse_page(raw, final_url, HOME_BUDGET)

            extra_pages: list[PageContent] = []
            targets = pick_key_links(home, self.settings.max_extra_pages)
            if targets:
                results = await asyncio.gather(
                    *(self._fetch_extra(client, target) for target in targets)
                )
                extra_pages = [page for page in results if page is not None]
                if len(extra_pages) < len(targets):
                    warnings.append("Some additional pages (About/Services) could not be read.")

        content = WebsiteContent(url, final_url, [home, *extra_pages], warnings)

        if content.total_text_chars < MIN_READABLE_CHARS:
            raise UnreadableContentError()
        if content.total_text_chars < THIN_CONTENT_CHARS:
            warnings.append(
                "Very little text was found on this site (it may be JavaScript-rendered), "
                "so the analysis relies more heavily on inference."
            )
        return content

    async def _fetch_extra(self, client: httpx.AsyncClient, url: str) -> PageContent | None:
        """Extra pages are a bonus: any failure just means we skip that page."""
        try:
            final_url, raw = await self._fetch(client, url)
            return parse_page(raw, final_url, EXTRA_BUDGET)
        except AppError as exc:
            logger.info("Skipping extra page %s (%s)", url, exc.code)
            return None

    async def _fetch(self, client: httpx.AsyncClient, url: str) -> tuple[str, bytes]:
        """GET a page, following redirects manually so every hop is SSRF-checked."""
        current = url
        for _ in range(self.settings.max_redirects + 1):
            await assert_public_url(current)
            try:
                async with client.stream("GET", current) as response:
                    if response.status_code in REDIRECT_CODES:
                        location = response.headers.get("location")
                        if not location:
                            raise WebsiteUnreachableError("The website sent a broken redirect.")
                        current = normalize_url(urljoin(current, location))
                        continue

                    if response.status_code >= 400:
                        raise WebsiteUnreachableError(self._http_error_message(response.status_code))

                    content_type = response.headers.get("content-type", "").lower()
                    if content_type and not any(t in content_type for t in ("html", "xml", "text/plain")):
                        raise UnsupportedContentError()

                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) >= self.settings.fetch_max_bytes:
                            break  # huge page: the first couple of MB is plenty
                    return current, bytes(body)
            except httpx.TimeoutException:
                raise WebsiteUnreachableError(
                    "The website took too long to respond."
                ) from None
            except httpx.HTTPError as exc:
                logger.info("Network error fetching %s: %s", current, type(exc).__name__)
                raise WebsiteUnreachableError(
                    "We couldn't connect to that website. It may be down or blocking us."
                ) from None
        raise WebsiteUnreachableError("The website redirected too many times.")

    @staticmethod
    def _http_error_message(status: int) -> str:
        if status in (401, 403, 429, 451):
            return (
                f"The website refused our request (HTTP {status}). "
                "It likely blocks automated visitors."
            )
        if status == 404:
            return "That page was not found (HTTP 404). Please check the URL."
        return f"The website returned an error (HTTP {status})."
