"""FastAPI entry point.

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 4000
Interactive docs:
    http://localhost:4000/docs
"""

import logging
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.exceptions import AppError, RateLimitExceededError
from app.schemas import ErrorResponse, LinkAnalysisRequest, LinkAnalysisResponse
from app.services.analysis_service import AnalysisService
from app.services.llm_service import GeminiClient
from app.services.website_extractor import WebsiteExtractor
from app.utils.rate_limit import SlidingWindowLimiter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    if not settings.gemini_api_key:
        logger.warning("GEMINI_API_KEY is not set. Requests will fail until you add it to .env")
    logger.info("Using model %s (fallback: %s)", settings.gemini_model, settings.gemini_fallback_model)
    yield


app = FastAPI(
    title="AI Website Link Analysis API",
    version="1.0.0",
    description=(
        "Give it a website URL. It reads the site, asks an LLM (Google Gemini) to analyze it, "
        "and returns competitor, search-query, SEO, services and target-audience insights as JSON."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------------------
def build_service() -> AnalysisService:
    settings = get_settings()
    return AnalysisService(WebsiteExtractor(settings), GeminiClient(settings), settings)


@lru_cache
def _service_singleton() -> AnalysisService:
    return build_service()


def get_analysis_service() -> AnalysisService:
    return _service_singleton()


@lru_cache
def _limiter() -> SlidingWindowLimiter:
    return SlidingWindowLimiter(get_settings().rate_limit_per_minute)


def _client_id(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:  # behind a proxy (Render, Railway, ...) the real client is the first entry
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request) -> None:
    wait = _limiter().check(_client_id(request))
    if wait > 0:
        raise RateLimitExceededError(
            f"Too many requests. Please retry in {int(wait) + 1} seconds.",
            headers={"Retry-After": str(int(wait) + 1)},
        )


# ---------------------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return {"service": "AI Website Link Analysis API", "docs": "/docs", "endpoint": "POST /link-analysis"}


@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok"}


@app.post(
    "/link-analysis",
    response_model=LinkAnalysisResponse,
    tags=["analysis"],
    summary="Analyze a website",
    responses={
        400: {"model": ErrorResponse, "description": "URL points to a private/internal address"},
        422: {"model": ErrorResponse, "description": "Invalid URL or unreadable page"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        502: {"model": ErrorResponse, "description": "Website or AI provider failed"},
        503: {"model": ErrorResponse, "description": "AI provider is rate limited"},
    },
)
async def link_analysis(
    payload: LinkAnalysisRequest,
    request: Request,
    service: AnalysisService = Depends(get_analysis_service),
):
    enforce_rate_limit(request)
    return await service.analyze(payload.url)


# ---------------------------------------------------------------------------------------
# Error handling: every failure uses the same friendly JSON shape
# ---------------------------------------------------------------------------------------
def _error_response(status: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"success": False, "error": {"code": code, "message": message}},
        headers=headers,
    )


@app.exception_handler(AppError)
async def handle_app_error(_: Request, exc: AppError):
    return _error_response(exc.status_code, exc.code, exc.message, exc.headers)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_: Request, exc: RequestValidationError):
    problems = "; ".join(
        f"{'.'.join(str(p) for p in err['loc'][1:]) or 'body'}: {err['msg']}" for err in exc.errors()
    )
    return _error_response(
        422,
        "invalid_request",
        f'Send JSON like {{"url": "https://example.com"}}. Problem: {problems}',
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(_: Request, exc: Exception):
    logger.exception("Unexpected error: %s", type(exc).__name__)
    return _error_response(500, "internal_error", AppError.default_message)
