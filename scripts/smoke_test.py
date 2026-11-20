"""Try a wide variety of links against a running copy of the API and check every result.

    python scripts/smoke_test.py                                   # local server on port 4000
    python scripts/smoke_test.py --base-url https://ai-link-analysis-l9md.onrender.com
    python scripts/smoke_test.py --only social                     # one group
    python scripts/smoke_test.py --delay 8                         # slower, gentler on the free AI quota

Each case says what SHOULD happen:
  "ok"            -> HTTP 200 with success=true and all five analysis sections (5 search queries)
  an error code   -> the API must reject the link with exactly that error code

Groups: normal, social, variants (the same site typed in different ways), errors (links that
must be rejected cleanly). Results of "ok" cases depend on live sites and the live AI provider,
so a failure there means "look at this one", not necessarily "the code is broken".
Exit code is 0 only if every case behaved as expected.
"""

import argparse
import sys
import time

import httpx

EMPTY_BODY = "<empty body>"
EMPTY_URL = "<empty url>"

# (group, url, expectation)
CASES: list[tuple[str, str, str]] = [
    # ---- ordinary websites ---------------------------------------------------------------
    ("normal", "https://www.python.org", "ok"),
    ("normal", "https://www.notion.so", "ok"),
    ("normal", "https://www.shopify.com", "ok"),
    ("normal", "https://stripe.com", "ok"),
    ("normal", "https://en.wikipedia.org/wiki/Python_(programming_language)", "ok"),
    ("normal", "https://www.bbc.com", "ok"),
    ("normal", "https://github.com", "ok"),
    ("normal", "https://example.com", "ok"),  # tiny page: very little text to read
    ("normal", "https://docmind.space", "ok"),  # JavaScript-rendered site
    ("normal", "https://crest.dxora.com", "ok"),  # the reviewer's own site
    # ---- social media and profile links --------------------------------------------------
    ("social", "https://www.instagram.com/crestspot", "ok"),
    ("social", "https://www.instagram.com/nasa", "ok"),
    ("social", "https://x.com/OpenAI", "ok"),
    ("social", "https://twitter.com/NASA", "ok"),
    ("social", "https://www.facebook.com/microsoft", "ok"),
    ("social", "https://www.linkedin.com/company/dxora-technologies", "ok"),
    ("social", "https://www.linkedin.com/in/roshangupta09", "ok"),
    ("social", "https://www.youtube.com/@Google", "ok"),
    ("social", "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "ok"),
    ("social", "https://www.tiktok.com/@nasa", "ok"),
    ("social", "https://www.reddit.com/r/python", "ok"),
    ("social", "https://www.threads.net/@instagram", "ok"),
    ("social", "https://www.pinterest.com/nasa", "ok"),
    ("social", "https://t.me/telegram", "ok"),
    ("social", "https://github.com/Roshu09", "ok"),
    # ---- the same site typed in different ways -------------------------------------------
    ("variants", "python.org", "ok"),
    ("variants", "www.python.org", "ok"),
    ("variants", "HTTP://WWW.PYTHON.ORG/", "ok"),
    ("variants", "https://www.python.org/about/?utm_source=test#top", "ok"),
    ("variants", "   https://www.python.org   ", "ok"),
    # ---- links that must be rejected cleanly ---------------------------------------------
    ("errors", "not a url", "invalid_url"),
    ("errors", "ftp://example.com", "invalid_url"),
    ("errors", "http://example", "invalid_url"),
    ("errors", "http://user:pass@example.com", "invalid_url"),
    ("errors", "javascript:alert(1)", "invalid_url"),
    ("errors", EMPTY_URL, "invalid_request"),
    ("errors", EMPTY_BODY, "invalid_request"),
    ("errors", "http://localhost:4000", "blocked_url"),
    ("errors", "http://127.0.0.1", "blocked_url"),
    ("errors", "http://192.168.1.1", "blocked_url"),
    ("errors", "http://169.254.169.254/latest/meta-data/", "blocked_url"),
    ("errors", "http://[::1]/", "blocked_url"),
    ("errors", "https://thisdomain-does-not-exist-12345.com", "website_unreachable"),
    ("errors", "https://www.python.org/this-page-does-not-exist-xyz", "website_unreachable"),
    ("errors", "https://www.python.org/static/img/python-logo.png", "unsupported_content"),
]


def payload_for(url: str) -> dict:
    if url == EMPTY_BODY:
        return {}
    if url == EMPTY_URL:
        return {"url": ""}
    return {"url": url}


def evaluate(expect: str, status: int, body) -> tuple[bool, str]:
    """Decide whether one response is what we wanted. Returns (passed, short description)."""
    body = body if isinstance(body, dict) else {}
    error = body.get("error") if isinstance(body.get("error"), dict) else {}

    if expect == "ok":
        if status == 200 and body.get("success") is True:
            analysis = body.get("analysis") or {}
            sections = {"competitor_analysis", "target_search_queries", "seo_improvements", "top_services", "target_audience"}
            missing = sections - set(analysis)
            queries = len(analysis.get("target_search_queries") or [])
            if missing or queries != 5:
                return False, f"incomplete result (missing={sorted(missing)}, search queries={queries})"
            meta = body.get("meta") or {}
            who = ""
            if meta.get("platform"):
                who = f", {meta['platform']}" + (f"/{meta['account']}" if meta.get("account") else "")
            cached = ", cached" if meta.get("cached") else ""
            return True, f"ok ({meta.get('content_source', 'page')}{who}, grounded={meta.get('grounded')}{cached})"
        return False, f"expected a result, got HTTP {status} {error.get('code', '?')}: {str(error.get('message', ''))[:80]}"

    if error.get("code") == expect:
        return True, f"correctly rejected: {expect} (HTTP {status})"
    return False, f"expected error '{expect}', got HTTP {status} {error.get('code') or body.get('success')}"


def run_case(client: httpx.Client, base: str, url: str, retries: int = 2) -> tuple[int, object, float]:
    """POST one link, waiting and retrying if the free AI quota or our rate limit says 'slow down'."""
    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            response = client.post(f"{base}/link-analysis", json=payload_for(url))
            status = response.status_code
            try:
                body = response.json()
            except ValueError:
                body = {"error": {"code": "non_json_response", "message": response.text[:80]}}
        except httpx.HTTPError as exc:
            status, body = 0, {"error": {"code": "no_response", "message": type(exc).__name__}}
        elapsed = time.perf_counter() - started
        code = (body.get("error") or {}).get("code") if isinstance(body, dict) else None
        if code in {"rate_limited", "llm_rate_limited"} and attempt < retries:
            wait = 35 if code == "llm_rate_limited" else 20
            print(f"      ...{code}, waiting {wait}s and retrying")
            time.sleep(wait)
            continue
        return status, body, elapsed
    return status, body, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:4000")
    parser.add_argument("--only", choices=["normal", "social", "variants", "errors"], help="run one group only")
    parser.add_argument("--delay", type=float, default=5.0, help="seconds to pause after each analysis (default 5)")
    parser.add_argument("--timeout", type=float, default=150.0)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    cases = [c for c in CASES if not args.only or c[0] == args.only]

    with httpx.Client(timeout=args.timeout) as client:
        print(f"Waking up {base} (a free-plan server can take about a minute)...")
        try:
            health = client.get(f"{base}/health")
            print(f"  /health -> {health.status_code}\n")
        except httpx.HTTPError as exc:
            print(f"  Could not reach the server ({type(exc).__name__}). Is it running?")
            return 2

        failures: list[str] = []
        current_group = None
        for group, url, expect in cases:
            if group != current_group:
                current_group = group
                print(f"== {group} " + "=" * (60 - len(group)))
            status, body, elapsed = run_case(client, base, url)
            passed, note = evaluate(expect, status, body)
            label = url.strip() if not url.startswith("<") else url
            print(f"[{'PASS' if passed else 'FAIL'}] {label}\n       {note}  ({elapsed:.1f}s)")
            if not passed:
                failures.append(f"{label}: {note}")
            if expect == "ok" and args.delay:
                time.sleep(args.delay)

    print("\n" + "=" * 64)
    print(f"{len(cases) - len(failures)} of {len(cases)} behaved as expected")
    if failures:
        print("\nNeeds a look:")
        for line in failures:
            print(f"  - {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
