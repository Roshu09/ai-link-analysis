"""The conductor: runs the whole pipeline for one URL.

    validate URL -> cache? -> fetch + clean website -> ask the LLM -> validate the answer
    -> tidy competitors / SEO -> build the response

If the model's answer is incomplete (say, only 3 search queries instead of 5), we
tell it exactly what was wrong and ask once more before giving up.
"""

import logging
import time

from pydantic import ValidationError

from app.config import Settings
from app.exceptions import LLMOutputError
from app.schemas import LinkAnalysisResponse, LLMOutput, ResponseMeta
from app.services.competitor_service import refine_competitors
from app.services.llm_service import GeminiClient, LLMResult
from app.services.prompts import SYSTEM_PROMPT, build_retry_suffix, build_user_prompt
from app.services.website_extractor import WebsiteExtractor
from app.utils.cache import TTLCache
from app.utils.validators import normalize_url

logger = logging.getLogger(__name__)

REQUIRED_QUERIES = 5
MAX_SERVICES = 5
MAX_TITLE_CHARS = 60
MAX_META_CHARS = 160


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

        content = await self.extractor.extract(url)
        output, llm_result, tokens_in, tokens_out = await self._ask_llm(content)

        warnings = list(content.warnings)
        output = self._tidy(output, url, warnings)

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
                warnings=warnings,
            ),
        )
        self.cache.set(cache_key, response)
        logger.info(
            "Analyzed %s in %dms (model=%s, in=%s, out=%s)",
            url, response.meta.elapsed_ms, llm_result.model, tokens_in, tokens_out,
        )
        return response

    # ---------------------------------------------------------------------------------
    async def _ask_llm(self, content) -> tuple[LLMOutput, LLMResult, int | None, int | None]:
        website_text = content.to_prompt_text(self.settings.max_content_chars)
        base_prompt = build_user_prompt(website_text, content.final_url)

        problem = ""
        tokens_in = tokens_out = None
        for attempt in range(2):  # first try + one corrective retry
            prompt = base_prompt if attempt == 0 else base_prompt + build_retry_suffix(problem)
            try:
                result = await self.llm.generate_json(SYSTEM_PROMPT, prompt)
            except LLMOutputError as exc:
                problem = "the reply was not valid JSON"
                logger.warning("Attempt %d: %s", attempt + 1, exc.message)
                continue

            tokens_in = (tokens_in or 0) + (result.prompt_tokens or 0)
            tokens_out = (tokens_out or 0) + (result.output_tokens or 0)
            try:
                output = LLMOutput.model_validate(result.data)
                self._normalize_counts(output)
                self._check_quality(output)
                return output, result, tokens_in, tokens_out
            except ValidationError as exc:
                problem = "; ".join(
                    f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()[:5]
                )
            except IncompleteAnalysisError as exc:
                problem = str(exc)
            logger.warning("Attempt %d rejected: %s", attempt + 1, problem)

        raise LLMOutputError()

    @staticmethod
    def _normalize_counts(output: LLMOutput) -> None:
        """Trim over-long lists, since 'too many' is easy to fix but 'too few' is not."""
        analysis = output.analysis
        analysis.target_search_queries = analysis.target_search_queries[:REQUIRED_QUERIES]
        analysis.top_services = analysis.top_services[:MAX_SERVICES]

    @staticmethod
    def _check_quality(output: LLMOutput) -> None:
        analysis = output.analysis
        if not output.brand.name:
            raise IncompleteAnalysisError("brand.name is empty")
        queries = [q for q in analysis.target_search_queries if q.query]
        if len(queries) != REQUIRED_QUERIES:
            raise IncompleteAnalysisError(
                f"target_search_queries must contain exactly {REQUIRED_QUERIES} non-empty queries "
                f"(got {len(queries)})"
            )
        if not [s for s in analysis.top_services if s.name]:
            raise IncompleteAnalysisError("top_services is empty")
        if not analysis.competitor_analysis.competitors:
            raise IncompleteAnalysisError("competitor_analysis.competitors is empty")
        if not analysis.seo_improvements.primary_keyword:
            raise IncompleteAnalysisError("seo_improvements.primary_keyword is empty")
        if not analysis.target_audience.primary_audience:
            raise IncompleteAnalysisError("target_audience.primary_audience is empty")

    def _tidy(self, output: LLMOutput, url: str, warnings: list[str]) -> LLMOutput:
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

        website_services = [s for s in analysis.top_services if s.source == "website"]
        if len(analysis.top_services) < MAX_SERVICES:
            warnings.append(f"Only {len(analysis.top_services)} services could be identified.")
        if not website_services:
            warnings.append(
                "No services were clearly stated on the website; all listed services are inferred."
            )
        return output


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _shorten(text: str, limit: int) -> str:
    """Cut at a word boundary and add an ellipsis, staying within `limit` characters."""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:-. ") + "…"
