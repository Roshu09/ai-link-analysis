"""The smoke-test script judges real links, so its own judging logic must be right."""

from scripts.smoke_test import CASES, EMPTY_BODY, EMPTY_URL, evaluate, payload_for

ALL_SECTIONS = ["competitor_analysis", "target_search_queries", "seo_improvements", "top_services", "target_audience"]


def good_body(**meta):
    analysis = {name: {} for name in ALL_SECTIONS}
    analysis["target_search_queries"] = [{}] * 5
    return {"success": True, "analysis": analysis, "meta": meta}


def test_a_complete_result_passes():
    ok, note = evaluate("ok", 200, good_body(content_source="url_only", platform="Instagram", account="crestspot", grounded=True))
    assert ok and "url_only" in note and "Instagram/crestspot" in note


def test_an_incomplete_result_fails_even_with_http_200():
    body = good_body()
    body["analysis"]["target_search_queries"] = [{}] * 3
    assert evaluate("ok", 200, body)[0] is False
    del body["analysis"]["top_services"]
    assert evaluate("ok", 200, body)[0] is False


def test_an_error_where_a_result_was_expected_fails_and_says_why():
    ok, note = evaluate("ok", 502, {"success": False, "error": {"code": "website_unreachable", "message": "HTTP 429"}})
    assert not ok and "website_unreachable" in note


def test_expected_errors_must_match_exactly():
    body = {"success": False, "error": {"code": "blocked_url", "message": "x"}}
    assert evaluate("blocked_url", 400, body)[0] is True
    assert evaluate("invalid_url", 400, body)[0] is False
    assert evaluate("blocked_url", 200, good_body())[0] is False


def test_garbage_responses_do_not_crash_the_judge():
    assert evaluate("ok", 0, None)[0] is False
    assert evaluate("blocked_url", 500, "<html>")[0] is False


def test_payloads_for_the_special_cases():
    assert payload_for(EMPTY_BODY) == {}
    assert payload_for(EMPTY_URL) == {"url": ""}
    assert payload_for("https://x.com/a") == {"url": "https://x.com/a"}


def test_case_list_is_sane():
    groups = {group for group, _, _ in CASES}
    assert groups == {"normal", "social", "variants", "errors"}
    assert len(CASES) == len({(url) for _, url, _ in CASES})  # no duplicates
    assert len(CASES) >= 40
    known_errors = {"invalid_url", "invalid_request", "blocked_url", "website_unreachable", "unsupported_content"}
    assert all(expect == "ok" or expect in known_errors for _, _, expect in CASES)
