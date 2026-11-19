"""Tidies up the competitor section after the LLM has written it.

The model suggests competitors from general knowledge, so we:
  * remove the brand itself and duplicates,
  * keep competitor websites only if they look like real http(s) URLs
    (we never fetch them; they're just labels),
  * mark every competitor as "ai_suggested" (done by the schema),
  * build the side-by-side comparison table from the structured fields.
"""

from urllib.parse import urlsplit

from app.schemas import Brand, Competitor, CompetitorAnalysis, ComparisonRow

MAX_COMPETITORS = 6
MIN_COMPETITORS = 3


def _host(url: str | None) -> str:
    if not url:
        return ""
    value = url if "://" in url else f"https://{url}"
    return (urlsplit(value).hostname or "").removeprefix("www.").lower()


def _clean_website(url: str | None) -> str | None:
    if not url:
        return None
    value = url.strip()
    if "://" not in value:
        value = f"https://{value}"
    parts = urlsplit(value)
    if parts.scheme in ("http", "https") and parts.hostname and "." in parts.hostname:
        return f"{parts.scheme}://{parts.hostname}"
    return None


def refine_competitors(
    analysis: CompetitorAnalysis, brand: Brand, site_url: str
) -> tuple[CompetitorAnalysis, list[str]]:
    """Returns the cleaned analysis plus any warnings worth showing the caller."""
    warnings: list[str] = []
    own_host = _host(site_url)
    own_name = brand.name.strip().lower()

    cleaned: list[Competitor] = []
    seen: set[str] = set()
    for competitor in analysis.competitors:
        name = competitor.name.strip()
        key = name.lower()
        if not name or key in seen or key == own_name:
            continue
        website = _clean_website(competitor.website)
        if website and _host(website) == own_host:
            continue  # that's the brand itself
        seen.add(key)
        cleaned.append(competitor.model_copy(update={"website": website}))

    cleaned = cleaned[:MAX_COMPETITORS]
    if len(cleaned) < MIN_COMPETITORS:
        warnings.append(
            f"Only {len(cleaned)} reliable competitor(s) could be suggested for this business."
        )

    refined = analysis.model_copy(update={"competitors": cleaned})
    refined.comparison_table = build_comparison_table(refined, brand)
    return refined, warnings


def build_comparison_table(analysis: CompetitorAnalysis, brand: Brand) -> list[ComparisonRow]:
    """Side-by-side view: one row per aspect, one column per competitor."""
    profile = analysis.your_brand_profile
    competitors = analysis.competitors
    if not competitors:
        return []

    def per_competitor(getter) -> dict[str, str]:
        return {c.name: getter(c) or "Not available" for c in competitors}

    return [
        ComparisonRow(
            aspect="Brand Name",
            your_brand=brand.name or "Not available",
            competitors={c.name: c.name for c in competitors},
        ),
        ComparisonRow(
            aspect="Main Service",
            your_brand=profile.main_service or "Not available",
            competitors=per_competitor(lambda c: c.main_service),
        ),
        ComparisonRow(
            aspect="Target Market",
            your_brand=profile.target_market or "Not available",
            competitors=per_competitor(lambda c: c.target_market),
        ),
        ComparisonRow(
            aspect="Key Strength",
            your_brand=profile.key_strength or "Not available",
            competitors=per_competitor(lambda c: c.strengths[0] if c.strengths else ""),
        ),
        ComparisonRow(
            aspect="Opportunity / Gap",
            your_brand=profile.opportunity or "Not available",
            competitors=per_competitor(lambda c: c.weaknesses[0] if c.weaknesses else ""),
        ),
    ]
