"""One odd value from the AI must never turn a whole analysis into an error.

These cases come from real runs: several social links returned HTTP 502 'llm_invalid_output'
because the answer was slightly off (an unknown platform name, no competitors for a person's
profile). A partial, honestly labelled answer is more useful than a 502.
"""

import asyncio

import pytest

from app.config import Settings
from app.exceptions import LLMOutputError
from app.services.analysis_service import AnalysisService
from tests.helpers import FakeExtractor, FakeLLM, good_output


def run(llm, url="https://acme.example/"):
    return asyncio.run(AnalysisService(FakeExtractor(), llm, Settings()).analyze(url))


class TestForgivingSchema:
    def test_unknown_platform_name_falls_back_to_google(self):
        data = good_output()
        data["analysis"]["target_search_queries"][0]["platforms"] = ["Telegram", "YouTube"]
        result = run(FakeLLM(data))
        assert result.analysis.target_search_queries[0].platforms == ["Google"]
        assert len(result.analysis.target_search_queries) == 5

    def test_known_platform_aliases_still_map_correctly(self):
        data = good_output()
        data["analysis"]["target_search_queries"][0]["platforms"] = ["Twitter/X", "linked in", "Meta"]
        platforms = run(FakeLLM(data)).analysis.target_search_queries[0].platforms
        assert platforms == ["Google"] or set(platforms) <= {"X", "LinkedIn", "Facebook", "Google"}

    def test_a_source_label_on_a_competitor_is_ignored(self):
        data = good_output()
        data["analysis"]["competitor_analysis"]["competitors"][0]["source"] = "website"
        assert {c.source for c in run(FakeLLM(data)).analysis.competitor_analysis.competitors} == {"ai_suggested"}

    def test_text_in_a_number_field_is_ignored(self):
        data = good_output()
        data["analysis"]["seo_improvements"]["title_length"] = "46 characters"
        seo = run(FakeLLM(data)).analysis.seo_improvements
        assert seo.title_length == len(seo.recommended_title)

    def test_null_sections_and_stray_list_items_are_repaired(self):
        data = good_output()
        ca = data["analysis"]["competitor_analysis"]
        ca["your_brand_profile"] = None
        ca["competitors"] = ["Wix", None, 7] + ca["competitors"]
        data["analysis"]["seo_improvements"]["supporting_content"] = "none"
        result = run(FakeLLM(data))
        assert result.success and len(result.analysis.competitor_analysis.competitors) >= 3

    def test_a_single_object_where_a_list_was_expected_is_wrapped(self):
        data = good_output()
        data["analysis"]["top_services"] = data["analysis"]["top_services"][0]
        result = run(FakeLLM(data, data))
        assert len(result.analysis.top_services) == 1


class TestPartialAnswersInsteadOfErrors:
    def test_still_short_on_queries_after_the_retry_returns_what_we_have(self):
        data = good_output()
        data["analysis"]["target_search_queries"] = data["analysis"]["target_search_queries"][:3]
        llm = FakeLLM(data)
        result = run(llm)
        assert llm.calls == 3
        assert len(result.analysis.target_search_queries) == 3
        assert any("incomplete" in w for w in result.meta.warnings)
        assert any("Only 3 search queries" in w for w in result.meta.warnings)

    def test_no_competitors_for_a_page_returns_the_rest_with_a_warning(self):
        data = good_output()
        data["analysis"]["competitor_analysis"]["competitors"] = []
        result = run(FakeLLM(data))
        assert result.success and result.analysis.competitor_analysis.competitors == []
        assert any("incomplete" in w and "competitors" in w for w in result.meta.warnings)

    def test_the_more_complete_of_two_attempts_is_kept(self):
        first = good_output()
        first["analysis"]["target_search_queries"] = first["analysis"]["target_search_queries"][:2]
        second = good_output()
        second["analysis"]["target_search_queries"] = second["analysis"]["target_search_queries"][:4]
        result = run(FakeLLM(first, second))
        assert len(result.analysis.target_search_queries) == 4

    def test_a_missing_brand_name_uses_the_hostname(self):
        data = good_output()
        data["brand"]["name"] = ""
        assert run(FakeLLM(data)).brand.name == "acme.example"

    def test_an_answer_with_nothing_useful_is_still_an_error(self):
        data = good_output()
        data["analysis"] = {}
        with pytest.raises(LLMOutputError):
            run(FakeLLM(data))

    def test_invalid_json_twice_is_still_an_error(self):
        with pytest.raises(LLMOutputError):
            run(FakeLLM(LLMOutputError()))

    def test_a_complete_answer_has_no_incompleteness_warning(self):
        assert not any("incomplete" in w for w in run(FakeLLM(good_output())).meta.warnings)


class TestSloppyJson:
    def test_trailing_commas_and_raw_newlines_are_accepted(self):
        from app.services.llm_service import _load_json_object
        raw = '```json\n{"a": [1, 2,], "b": "line1\nline2",}\n```'
        assert _load_json_object(raw) == {"a": [1, 2], "b": "line1\nline2"}

    def test_a_third_attempt_rescues_two_bad_answers(self):
        llm = FakeLLM(LLMOutputError(), LLMOutputError(), good_output())
        assert run(llm).success and llm.calls == 3


class TestRealWorldBrokenJson:
    def test_invalid_backslash_escape_is_fixed(self):
        from app.services.llm_service import _load_json_object
        assert _load_json_object('{"a": "Python\\\'s docs"}') == {"a": "Python's docs"}

    def test_unescaped_inner_quotes_are_repaired(self):
        from app.services.llm_service import _load_json_object
        assert _load_json_object('{"a": "He said "hi" there", "b": [1]}')["b"] == [1]

    def test_garbage_is_still_rejected(self):
        from app.services.llm_service import _load_json_object
        with pytest.raises(LLMOutputError):
            _load_json_object("I cannot help with that.")
