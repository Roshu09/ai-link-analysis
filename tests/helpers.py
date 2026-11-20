"""Shared fakes and sample data for the tests."""

import copy

from app.services.llm_service import LLMResult
from app.services.website_extractor import PageContent, WebsiteContent

GOOD_LLM_OUTPUT = {
    "brand": {
        "name": "Acme Web Studio",
        "category": "Web Development Agency",
        "summary": "A small agency that builds websites and online stores for local businesses.",
    },
    "analysis": {
        "competitor_analysis": {
            "brand_overview": "Acme Web Studio builds fast, affordable websites for small businesses.",
            "main_category": "Web Development",
            "main_products_services": ["Website design", "E-commerce", "SEO"],
            "your_brand_profile": {
                "main_service": "Website development",
                "target_market": "Small businesses",
                "key_strength": "Affordable pricing",
                "opportunity": "Improve online visibility",
            },
            "competitors": [
                {"name": "Wix", "website": "https://www.wix.com", "main_service": "Website builder",
                 "target_market": "DIY users", "strengths": ["Brand awareness"], "weaknesses": ["Generic templates"]},
                {"name": "Squarespace", "website": "squarespace.com", "main_service": "Website builder",
                 "target_market": "Creatives", "strengths": ["Design quality"], "weaknesses": ["Limited customization"]},
                {"name": "WebFX", "website": None, "main_service": "Digital marketing",
                 "target_market": "Mid-market", "strengths": ["Scale"], "weaknesses": ["Higher prices"]},
                {"name": "Acme Web Studio", "website": "https://acme.example", "main_service": "x",
                 "target_market": "x", "strengths": [], "weaknesses": []},
            ],
            "strengths": ["Personal service", "Affordable pricing"],
            "opportunities": ["Publish case studies"],
        },
        "target_search_queries": [
            {"query": "affordable web design for small business", "platforms": ["Google", "Facebook"],
             "intent": "Service Search", "business_relevance": "Matches core offer", "target_audience": "Owners"},
            {"query": "ecommerce website developer", "platforms": ["Google", "LinkedIn"],
             "intent": "B2B Search", "business_relevance": "Store builds", "target_audience": "Retailers"},
            {"query": "website design ideas", "platforms": ["Instagram"],
             "intent": "Inspiration", "business_relevance": "Portfolio reach", "target_audience": "Startups"},
            {"query": "freelance web developer near me", "platforms": ["Twitter"],
             "intent": "Service Discovery", "business_relevance": "Local leads", "target_audience": "Locals"},
            {"query": "SEO for small business websites", "platforms": ["google search", "LinkedIn"],
             "intent": "Education", "business_relevance": "SEO add-on", "target_audience": "Owners"},
        ],
        "seo_improvements": {
            "primary_keyword": "web design for small business",
            "secondary_keywords": ["affordable web design"],
            "long_tail_keywords": ["web design agency for local shops"],
            "business_keywords": ["e-commerce development"],
            "recommended_title": "Web Design for Small Business | Acme Web Studio",
            "title_structure": "Primary Keyword | Brand",
            "meta_description": "Acme Web Studio builds fast, affordable websites and online stores for small businesses.",
            "supporting_content": [{"type": "FAQ", "title": "How much does a website cost?", "reason": "Captures price searches"}],
        },
        "top_services": [
            {"name": "Website Design", "description": "Custom sites", "target_industry": "Retail",
             "target_customer": "Small businesses", "main_benefit": "Professional look", "source": "website"},
            {"name": "E-commerce", "description": "Online stores", "target_industry": "Retail",
             "target_customer": "Shops", "main_benefit": "Sell online", "source": "website"},
            {"name": "SEO", "description": "Search optimization", "target_industry": "All",
             "target_customer": "Owners", "main_benefit": "More traffic", "source": "website"},
            {"name": "Maintenance", "description": "Updates", "target_industry": "All",
             "target_customer": "Owners", "main_benefit": "Peace of mind", "source": "inferred"},
            {"name": "Branding", "description": "Logos", "target_industry": "All",
             "target_customer": "Startups", "main_benefit": "Recognition", "source": "inferred"},
        ],
        "target_audience": {
            "primary_audience": ["Small business owners"],
            "secondary_audience": ["Startups"],
            "industries": ["Retail", "Services"],
            "business_types": ["Local shops"],
            "customer_profiles": ["Owner with no tech team"],
            "geographic_focus": ["Not specified on the website"],
            "customer_needs": ["Affordable online presence"],
            "customer_interests": ["Growing sales online"],
            "recommended_segments": ["Local retailers"],
        },
    },
}


def good_output() -> dict:
    return copy.deepcopy(GOOD_LLM_OUTPUT)


def sample_content(url: str = "https://acme.example/") -> WebsiteContent:
    page = PageContent(
        url=url,
        title="Acme Web Studio - Websites for small businesses",
        meta_description="We build affordable websites.",
        headings=["Websites that grow your business"],
        text="We design and build fast, affordable websites and online stores. " * 5,
    )
    return WebsiteContent(requested_url=url, final_url=url, pages=[page])


class FakeExtractor:
    def __init__(self, content: WebsiteContent | None = None, error: Exception | None = None):
        self.content = content or sample_content()
        self.error = error
        self.calls = 0

    async def extract(self, url: str) -> WebsiteContent:
        self.calls += 1
        if self.error:
            raise self.error
        return self.content


class FakeLLM:
    """Plays back a list of results (dicts become LLMResults, exceptions are raised)."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0
        self.prompts: list[str] = []
        self.search_flags: list[bool] = []

    async def generate_json(self, system_prompt: str, user_prompt: str, use_search: bool = False) -> LLMResult:
        self.calls += 1
        self.prompts.append(user_prompt)
        self.search_flags.append(use_search)
        item = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return LLMResult(
            data=item, model="fake-model", prompt_tokens=1200, output_tokens=900, latency_ms=10,
            grounded=use_search, sources=["example-source.com"] if use_search else [],
        )
