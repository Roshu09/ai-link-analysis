"""Recognize social-media links (Instagram, X, Facebook, LinkedIn, YouTube, ...).

Why this exists: these platforms refuse automated page reads (login walls, HTTP 429,
bot detection). When that happens we still want to analyze the *account behind the
link*, so we need to know which platform it is and what the account name is.
"""

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlsplit

PLATFORM_DOMAINS = {
    "instagram.com": "Instagram",
    "facebook.com": "Facebook",
    "fb.com": "Facebook",
    "fb.watch": "Facebook",
    "x.com": "X",
    "twitter.com": "X",
    "linkedin.com": "LinkedIn",
    "youtube.com": "YouTube",
    "youtu.be": "YouTube",
    "tiktok.com": "TikTok",
    "threads.net": "Threads",
    "threads.com": "Threads",
    "pinterest.com": "Pinterest",
    "reddit.com": "Reddit",
    "t.me": "Telegram",
}

# Path words that mean "a post / video", not an account name.
_POST_SEGMENTS = {
    "p", "reel", "reels", "tv", "status", "statuses", "watch", "shorts", "video", "videos",
    "posts", "photo", "photos", "pin", "comments", "story", "stories", "share",
}
# Path words that are site sections, not account names.
_NON_PROFILE = {
    "explore", "hashtag", "i", "intent", "search", "home", "login", "accounts", "feed",
    "groups", "events", "marketplace", "settings", "help", "about", "privacy", "terms",
    "jobs", "messages", "notifications", "directory", "legal", "policies", "business", "ads",
}
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._\-]")


@dataclass(frozen=True)
class SocialLink:
    platform: str
    handle: str  # may be empty (e.g. a bare post link)
    kind: str  # profile | company | person | channel | community | post | other

    def describe(self) -> str:
        if self.handle:
            return f"{self.platform} {self.kind} with the account name '{self.handle}'"
        return f"{self.platform} {self.kind} link"


def _platform_for(host: str) -> str | None:
    host = host.lower().rstrip(".")
    for domain, name in PLATFORM_DOMAINS.items():
        if host == domain or host.endswith("." + domain):
            return name
    return None


def _clean_handle(value: str) -> str:
    """Keep only safe characters: the handle ends up inside an LLM prompt."""
    return _UNSAFE_CHARS.sub("", unquote(value).lstrip("@"))[:60]


def detect_social(url: str) -> SocialLink | None:
    """Return the platform and account name if `url` is a social-media link, else None."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    platform = _platform_for(host)
    if not platform:
        return None

    segments = [s for s in parts.path.split("/") if s]
    first = segments[0].lower() if segments else ""
    query = parse_qs(parts.query)

    if platform == "LinkedIn":
        if len(segments) >= 2 and first in {"company", "school", "showcase"}:
            return SocialLink(platform, _clean_handle(segments[1]), "company")
        if len(segments) >= 2 and first == "in":
            return SocialLink(platform, _clean_handle(segments[1]), "person")
        return SocialLink(platform, "", "other")

    if platform == "YouTube":
        if host.endswith("youtu.be") or first in {"watch", "shorts", "live"}:
            return SocialLink(platform, "", "post")
        if first.startswith("@"):
            return SocialLink(platform, _clean_handle(segments[0]), "channel")
        if first in {"c", "user", "channel"} and len(segments) >= 2:
            return SocialLink(platform, _clean_handle(segments[1]), "channel")
        return SocialLink(platform, "", "other")

    if platform == "Reddit":
        if len(segments) >= 2 and first in {"r", "u", "user"}:
            return SocialLink(platform, _clean_handle(segments[1]), "community" if first == "r" else "profile")
        return SocialLink(platform, "", "other")

    if platform == "Facebook":
        if first == "profile.php" and query.get("id"):
            return SocialLink(platform, _clean_handle(query["id"][0]), "profile")
        if first == "pages" and len(segments) >= 2:
            return SocialLink(platform, _clean_handle(segments[1]), "profile")

    if not segments:
        return SocialLink(platform, "", "other")
    if first in _POST_SEGMENTS:
        return SocialLink(platform, "", "post")
    if first in _NON_PROFILE:
        return SocialLink(platform, "", "other")

    handle = _clean_handle(segments[0])
    is_post = len(segments) >= 3 and segments[1].lower() in _POST_SEGMENTS
    return SocialLink(platform, handle, "post" if is_post else "profile")
