"""Every error the API can return on purpose.

Each one carries an HTTP status, a short machine-readable code, and a friendly
message, so clients always get the same JSON shape:

    {"success": false, "error": {"code": "...", "message": "..."}}
"""


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"
    default_message: str = "Something went wrong on our side. Please try again."

    def __init__(self, message: str | None = None, *, headers: dict[str, str] | None = None):
        self.message = message or self.default_message
        self.headers = headers or {}
        super().__init__(self.message)


# ---- Problems with what the caller sent -------------------------------------------------
class InvalidURLError(AppError):
    status_code = 422
    code = "invalid_url"
    default_message = "That doesn't look like a valid website URL."


class BlockedURLError(AppError):
    status_code = 400
    code = "blocked_url"
    default_message = "This URL points to a private or internal address and can't be analyzed."


class RateLimitExceededError(AppError):
    status_code = 429
    code = "rate_limited"
    default_message = "Too many requests. Please wait a moment and try again."


# ---- Problems with the website being analyzed ---------------------------------------------
class WebsiteUnreachableError(AppError):
    status_code = 502
    code = "website_unreachable"
    default_message = "We couldn't reach that website."


class UnsupportedContentError(AppError):
    status_code = 422
    code = "unsupported_content"
    default_message = "That URL doesn't return an HTML web page."


class UnreadableContentError(AppError):
    status_code = 422
    code = "unreadable_content"
    default_message = (
        "We couldn't find enough readable text on that page. It may be rendered by "
        "JavaScript, empty, or blocking automated visitors."
    )


# ---- Problems with the LLM provider -----------------------------------------------------
class LLMConfigError(AppError):
    status_code = 500
    code = "llm_not_configured"
    default_message = "The AI provider is not configured correctly on the server."


class LLMRateLimitError(AppError):
    status_code = 503
    code = "llm_rate_limited"
    default_message = "The AI provider is busy right now (rate limit). Please retry in a minute."


class LLMError(AppError):
    status_code = 502
    code = "llm_error"
    default_message = "The AI provider failed to respond. Please try again."


class LLMOutputError(AppError):
    status_code = 502
    code = "llm_invalid_output"
    default_message = "The AI returned an incomplete analysis. Please try again."
