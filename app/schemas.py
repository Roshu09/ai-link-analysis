"""Pydantic models: the request, the response, and the shape we ask the LLM to produce.

The LLM output is parsed with these same models. They are deliberately forgiving
about small formatting slips (a string where a list was expected, `null` for text)
but strict about the things that matter, which `analysis_service` double-checks.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, field_validator


# ---------------------------------------------------------------------------------------
# Small helpers that make LLM output more forgiving
# ---------------------------------------------------------------------------------------
def _to_str(value) -> str:
    return "" if value is None else str(value).strip()


def _to_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [s for s in (_to_str(item) for item in value) if s]
    return [_to_str(value)]


Text = Annotated[str, BeforeValidator(_to_str)]
StrList = Annotated[list[str], BeforeValidator(_to_str_list)]

Platform = Literal["Google", "Facebook", "Instagram", "LinkedIn", "X"]
_PLATFORM_ALIASES = {
    "google": "Google",
    "google search": "Google",
    "facebook": "Facebook",
    "fb": "Facebook",
    "instagram": "Instagram",
    "ig": "Instagram",
    "linkedin": "LinkedIn",
    "x": "X",
    "twitter": "X",
    "x (twitter)": "X",
    "x/twitter": "X",
    "twitter/x": "X",
}


# ---------------------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------------------
class LinkAnalysisRequest(BaseModel):
    url: str = Field(
        ...,
        min_length=3,
        max_length=2048,
        description="The website to analyze.",
        examples=["https://example.com"],
    )


# ---------------------------------------------------------------------------------------
# Brand
# ---------------------------------------------------------------------------------------
class Brand(BaseModel):
    name: Text = ""
    category: Text = ""
    summary: Text = ""


# ---------------------------------------------------------------------------------------
# Section 1: competitors
# ---------------------------------------------------------------------------------------
class BrandProfile(BaseModel):
    main_service: Text = ""
    target_market: Text = ""
    key_strength: Text = ""
    opportunity: Text = ""


class Competitor(BaseModel):
    name: Text = ""
    website: str | None = None
    main_service: Text = ""
    target_market: Text = ""
    strengths: StrList = []
    weaknesses: StrList = []
    # Competitors come from the model's general knowledge, not live research,
    # so we always say so. (The assignment asks for exactly this honesty.)
    source: Literal["ai_suggested"] = "ai_suggested"

    @field_validator("website", mode="before")
    @classmethod
    def _clean_website(cls, value):
        text = _to_str(value)
        if not text or text.lower() in {"null", "none", "n/a", "unknown"}:
            return None
        return text


class ComparisonRow(BaseModel):
    aspect: str
    your_brand: str
    competitors: dict[str, str]


class CompetitorAnalysis(BaseModel):
    brand_overview: Text = ""
    main_category: Text = ""
    main_products_services: StrList = []
    your_brand_profile: BrandProfile = BrandProfile()
    competitors: list[Competitor] = []
    comparison_table: list[ComparisonRow] = []  # built by code from the fields above
    strengths: StrList = []  # your brand's competitive advantages
    opportunities: StrList = []  # ways to improve market positioning


# ---------------------------------------------------------------------------------------
# Section 2: search queries
# ---------------------------------------------------------------------------------------
class SearchQuery(BaseModel):
    query: Text = ""
    platforms: list[Platform] = []
    intent: Text = ""
    business_relevance: Text = ""
    target_audience: Text = ""

    @field_validator("platforms", mode="before")
    @classmethod
    def _normalize_platforms(cls, value):
        seen: list[str] = []
        for item in _to_str_list(value):
            canonical = _PLATFORM_ALIASES.get(item.lower())
            if canonical and canonical not in seen:
                seen.append(canonical)
        if not seen:
            raise ValueError("platforms must include at least one of Google, Facebook, Instagram, LinkedIn, X")
        return seen


# ---------------------------------------------------------------------------------------
# Section 3: SEO
# ---------------------------------------------------------------------------------------
class SupportingContent(BaseModel):
    type: Text = ""
    title: Text = ""
    reason: Text = ""


class SEOImprovements(BaseModel):
    primary_keyword: Text = ""
    secondary_keywords: StrList = []
    long_tail_keywords: StrList = []
    business_keywords: StrList = []
    recommended_title: Text = ""
    title_structure: Text = ""
    meta_description: Text = ""
    supporting_content: list[SupportingContent] = []
    title_length: int = 0
    meta_description_length: int = 0


# ---------------------------------------------------------------------------------------
# Section 4: services
# ---------------------------------------------------------------------------------------
class Service(BaseModel):
    name: Text = ""
    description: Text = ""
    target_industry: Text = ""
    target_customer: Text = ""
    main_benefit: Text = ""
    # "website" = found on the page; "inferred" = the model's educated guess.
    source: Literal["website", "inferred"] = "inferred"

    @field_validator("source", mode="before")
    @classmethod
    def _normalize_source(cls, value):
        return "website" if _to_str(value).lower() == "website" else "inferred"


# ---------------------------------------------------------------------------------------
# Section 5: audience
# ---------------------------------------------------------------------------------------
class TargetAudience(BaseModel):
    primary_audience: StrList = []
    secondary_audience: StrList = []
    industries: StrList = []
    business_types: StrList = []
    customer_profiles: StrList = []
    geographic_focus: StrList = []
    customer_needs: StrList = []
    customer_interests: StrList = []
    recommended_segments: StrList = []


# ---------------------------------------------------------------------------------------
# Whole analysis + response
# ---------------------------------------------------------------------------------------
class Analysis(BaseModel):
    competitor_analysis: CompetitorAnalysis = CompetitorAnalysis()
    target_search_queries: list[SearchQuery] = []
    seo_improvements: SEOImprovements = SEOImprovements()
    top_services: list[Service] = []
    target_audience: TargetAudience = TargetAudience()


class LLMOutput(BaseModel):
    """Exactly what we ask the model to return."""

    brand: Brand = Brand()
    analysis: Analysis = Analysis()


class ResponseMeta(BaseModel):
    model_used: str
    cached: bool = False
    elapsed_ms: int = 0
    pages_analyzed: list[str] = []
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    warnings: list[str] = []
    notes: list[str] = [
        "Competitors are AI-generated suggestions, not live market research.",
        "No search-volume or ranking data is included because none was measured.",
    ]


class LinkAnalysisResponse(BaseModel):
    success: bool = True
    website: str
    brand: Brand
    analysis: Analysis
    meta: ResponseMeta


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorBody
