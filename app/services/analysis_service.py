"""The conductor: runs the whole pipeline for one URL.

    validate URL -> cache? -> fetch + clean website -> ask the LLM -> validate the answer
    -> tidy competitors / SEO -> build the response

Two modes:
  * "page"     the normal case: we read the page and the LLM analyzes its content.
  * "url_only" the page could not be read (login wall, HTTP 403/429, bot protection,
               JavaScript-only). Social platforms do this a lot. Instead of returning an
               error we analyze the account behind the link (using Google Search grounding
               when available) and say clearly, in `meta` and `warnings`, that this is
               lower confidence. Errors are kept for things that cannot be analyzed at all:
               invalid or private URLs, unknown domains, 404 pages, non-HTML files.

If the model's answer is incomplete (say, only 3 search queries instead of 5), we
tell it exactly what was wrong and ask once more. If it is STILL incomplete but contains
something useful, we return what we have with a clear warning instead of an error: a
partial, honest answer is more useful than a 502, and nothing is invented to fill the gaps.
"""

import logging
import time
from urllib.parse import urlsplit

from pydantic import ValidationError

from app.config import Settings
from app.exceptions import (
    AccessRestrictedError,
    LLMOutputError,
    UnreadableContentError,
    WebsiteDownError,
)
from app.schemas import LinkAnalysisResponse, LLMOutput, ResponseMeta
from app.services.competitor_service import refine_competitors
from app.services.llm_service import GeminiClient, LLMResult
from app.services.prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_LIMITED,
    build_limited_prompt,
    build_retry_suffix,
    build_user_prompt,
)
from app.services.website_extractor import WebsiteContent, WebsiteExtractor
from app.utils.cache import TTLCache
from app.utils.social import SocialLink, detect_social
from app.utils.validators import normalize_url

logger = logging.getLogger(__name__)

REQUIRED_QUERIES = 5
MAX_SERVICES = 5
MAX_TITLE_CHARS = 60
MAX_META_CHARS = 160
LOGIN_WALL_MAX_CHARS = 800
MAX_ATTEMPTS = 3


class IncompleteAnalysisError(ValueError):
    """The model's JSON parsed fine but is missing something we require."""


class AnalysisService:
    def __init__(
        self,
        extractor: WebsiteExtractor,
        llm: GeminiClient,
        settings: Settings,
        cache: TTLCache[LinkAnalysisResponse] | None = None,
    ):
        self.extractor = extractor
        self.llm = llm
        self.settings = settings
        self.cache = cache if cache is not None else TTLCache(
            settings.cache_ttl_seconds, settings.cache_max_entries
        )

    async def analyze(self, raw_url: str) -> LinkAnalysisResponse:
        started = time.perf_counter()
        url = normalize_url(raw_url)  # raises InvalidURLError / BlockedURLError
        cache_key = url.rstrip("/").lower()

        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.info("Cache hit for %s", url)
            meta = cached.meta.model_copy(
                update={"cached": True, "elapsed_ms": _elapsed_ms(started)}
            )
            return cached.model_copy(update={"meta": meta})

        social = detect_social(url)
        try:
            content = await self.extractor.extract(url)
            if _is_login_wall(content, social):
                raise UnreadableContentError()
        except (AccessRestrictedError, UnreadableContentError, WebsiteDownError) as exc:
            if isinstance(exc, WebsiteDownError) and social is None:
                raise  # an ordinary website that is down is a real error
            logger.info("Page not readable (%s); using link-only analysis for %s", type(exc).__name__, url)
            content = self._limited_content(url, exc)

        limited = content.mode == "url_only"
        output, llm_result, tokens_in, tokens_out, notes = await self._ask_llm(content, social, url)

        warnings = list(content.warnings)
        warnings.extend(notes)
        if limited:
            warnings.append(_grounding_warning(llm_result))
        output = self._tidy(output, url, warnings, limited)

        response = LinkAnalysisResponse(
            website=url,
            brand=output.brand,
            analysis=output.analysis,
            meta=ResponseMeta(
                model_used=llm_result.model,
                cached=False,
                elapsed_ms=_elapsed_ms(started),
                pages_analyzed=[page.url for page in content.pages],
                prompt_tokens=tokens_in,
                output_tokens=tokens_out,
                content_source="url_only" if limited else "page",
                platform=social.platform if social else None,
                account=(social.handle or None) if social else None,
                grounded=llm_result.grounded,
                grounding_sources=list(llm_result.sources),
                warnings=warnings,
            ),
        )
        if not limited:  # don't cache link-only results, so a later try can read the real page
            self.cache.set(cache_key, response)
        logger.info(
            "Analyzed %s in %dms (mode=%s, model=%s, in=%s, out=%s)",
            url, response.meta.elapsed_ms, response.meta.content_source,
            llm_result.model, tokens_in, tokens_out,
        )
        return response

    # ---------------------------------------------------------------------------------
    @staticmethod
    def _limited_content(url: str, exc: Exception) -> WebsiteContent:
        if isinstance(exc, (AccessRestrictedError, WebsiteDownError)):
            reason = exc.message
        else:
            reason = "the page had almost no readable text (it may need JavaScript or a login)"
        warning = (
            f"The page could not be read ({reason}). This analysis is based on the link itself, "
            "not on the page's content, so treat it as lower confidence."
        )
        return WebsiteContent(
            requested_url=url, final_url=url, pages=[], warnings=[warning],
            mode="url_only", block_reason=reason,
        )

    async def _ask_llm(
        self, content: WebsiteContent, social: SocialLink | None, url: str
    ) -> tuple[LLMOutput, LLMResult, int | None, int | None, list[str]]:
        limited = content.mode == "url_only"
        if limited:
            system = SYSTEM_PROMPT_LIMITED
            subject = social.describe() if social else f"a website on {urlsplit(url).hostname}"
            base_prompt = build_limited_prompt(content.final_url, subject, content.block_reason)
            extra = {"use_search": True}
        else:
            system = SYSTEM_PROMPT
            note = (
                f"This link is a {social.describe()}. Describe the business or person behind "
                "the account, not the social platform itself."
            ) if social else ""
            base_prompt = build_user_prompt(
                content.to_prompt_text(self.settings.max_content_chars), content.final_url, note
            )
            extra = {}

        fallback_name = (social.handle if social and social.handle else "") or urlsplit(url).hostname or ""
        problem = ""
        tokens_in = tokens_out = None
        best: tuple[int, LLMOutput, LLMResult, str] | None = None  # most complete answer that was still short of the bar

        for attempt in range(MAX_ATTEMPTS):  # first try + corrective retries
            prompt = base_prompt if attempt == 0 else base_prompt + build_retry_suffix(problem)
            try:
                result = await self.llm.generate_json(system, prompt, **extra)
            except LLMOutputError as exc:
                problem = "the reply was not valid JSON"
                logger.warning("Attempt %d: %s", attempt + 1, exc.message)
                continue

            tokens_in = (tokens_in or 0) + (result.prompt_tokens or 0)
            tokens_out = (tokens_out or 0) + (result.output_tokens or 0)
            try:
                output = LLMOutput.model_validate(result.data)
                self._normalize_counts(output)
                if not output.brand.name:
                    output.brand.name = fallback_name
                self._check_quality(output, limited)
                return output, result, tokens_in, tokens_out, []
            except ValidationError as exc:
                problem = "; ".join(
                    f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()[:5]
                )
            except IncompleteAnalysisError as exc:
                problem = str(exc)
                score = _completeness(output)
                if score > 0 and (best is None or score >= best[0]):
                    best = (score, output, result, problem)
            logger.warning("Attempt %d rejected: %s", attempt + 1, problem)

        if best is not None:
            _, output, result, problem = best
            logger.warning("Returning an incomplete answer with a warning: %s", problem)
            note = (
                f"The AI's answer was incomplete ({problem}). The missing parts are left as they are "
                "rather than invented."
            )
            return output, result, tokens_in, tokens_out, [note]

        raise LLMOutputError()

    @staticmethod
    def _normalize_counts(output: LLMOutput) -> None:
        """Trim over-long lists, since 'too many' is easy to fix but 'too few' is not."""
        analysis = output.analysis
        analysis.target_search_queries = analysis.target_search_queries[:REQUIRED_QUERIES]
        analysis.top_services = analysis.top_services[:MAX_SERVICES]

    @staticmethod
    def _check_quality(output: LLMOutput, limited: bool = False) -> None:
        analysis = output.analysis
        if not output.brand.name:
            raise IncompleteAnalysisError("brand.name is empty")
        queries = [q for q in analysis.target_search_queries if q.query]
        if len(queries) != REQUIRED_QUERIES:
            raise IncompleteAnalysisError(
                f"target_search_queries must contain exactly {REQUIRED_QUERIES} non-empty queries "
                f"(got {len(queries)})"
            )
        if not analysis.seo_improvements.primary_keyword:
            raise IncompleteAnalysisError("seo_improvements.primary_keyword is empty")
        if not analysis.target_audience.primary_audience:
            raise IncompleteAnalysisError("target_audience.primary_audience is empty")
        if limited:
            return  # with only a link to go on, an empty competitor or service list is honest
        if not [s for s in analysis.top_services if s.name]:
            raise IncompleteAnalysisError("top_services is empty")
        if not analysis.competitor_analysis.competitors:
            raise IncompleteAnalysisError("competitor_analysis.competitors is empty")

    def _tidy(self, output: LLMOutput, url: str, warnings: list[str], limited: bool) -> LLMOutput:
        analysis = output.analysis

        competitor_analysis, competitor_warnings = refine_competitors(
            analysis.competitor_analysis, output.brand, url
        )
        analysis.competitor_analysis = competitor_analysis
        warnings.extend(competitor_warnings)

        seo = analysis.seo_improvements
        if len(seo.meta_description) > MAX_META_CHARS:
            seo.meta_description = _shorten(seo.meta_description, MAX_META_CHARS)
        seo.title_length = len(seo.recommended_title)
        seo.meta_description_length = len(seo.meta_description)
        if seo.title_length > MAX_TITLE_CHARS:
            warnings.append(
                f"The suggested SEO title is {seo.title_length} characters; "
                f"search engines usually show about {MAX_TITLE_CHARS}."
            )

        query_count = len([q for q in analysis.target_search_queries if q.query])
        if query_count < REQUIRED_QUERIES:
            warnings.append(f"Only {query_count} search queries could be produced.")

        if limited:
            for service in analysis.top_services:
                service.source = "inferred"  # nothing was read from a page, so nothing is "website"
        else:
            website_services = [s for s in analysis.top_services if s.source == "website"]
            if not website_services:
                warnings.append(
                    "No services were clearly stated on the website; all listed services are inferred."
                )
        if len(analysis.top_services) < MAX_SERVICES:
            warnings.append(f"Only {len(analysis.top_services)} services could be identified.")
        return output


def _completeness(output: LLMOutput) -> int:
    """How much usable content an answer has: used to pick the better of two incomplete answers."""
    analysis = output.analysis
    return (
        len([q for q in analysis.target_search_queries if q.query])
        + len([s for s in analysis.top_services if s.name])
        + len([c for c in analysis.competitor_analysis.competitors if c.name])
    )


def _is_login_wall(content: WebsiteContent, social: SocialLink | None) -> bool:
    """A 'successful' fetch of a social link that never mentions the account is a login page."""
    if not social or not social.handle or social.kind == "post":
        return False
    haystack = " ".join(
        f"{p.title} {p.meta_description} {p.text} {' '.join(p.headings)}" for p in content.pages
    ).lower()
    return social.handle.lower() not in haystack and content.total_text_chars < LOGIN_WALL_MAX_CHARS


def _grounding_warning(result: LLMResult) -> str:
    if result.grounded:
        return "The AI looked this account up using Google Search. Please verify important details yourself."
    return (
        "AI web search was not available for this request, so the analysis relies on the model's "
        "general knowledge and may be generic or wrong."
    )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _shorten(text: str, limit: int) -> str:
    """Cut at a word boundary and add an ellipsis, staying within `limit` characters."""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:-. ") + "…"
