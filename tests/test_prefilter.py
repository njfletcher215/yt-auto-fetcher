"""Phase 4.2: the prefilter.

Boundaries get explicit tests because the guide writes the filter with strict
inequalities (`duration > 300`, not `>=`), and an off-by-one here means
five-minute videos either all pass or all fail -- a whole category of content
appearing or vanishing with no error anywhere.
"""

from __future__ import annotations

import pytest

from ytaf import prefilter, spec
from ytaf.config import PrefilterConfig


def meta(**kwargs):
    base = {"duration": 900, "view_count": 10_000}
    base.update(kwargs)
    return {k: v for k, v in base.items() if v is not None}


class TestDuration:
    @pytest.mark.parametrize("duration", [301, 600, 3600, 7199])
    def test_accepts_in_range(self, duration):
        assert prefilter.check(meta(duration=duration)).passed

    @pytest.mark.parametrize("duration", [0, 1, 59, 299, 300])
    def test_rejects_shorts_and_the_lower_bound(self, duration):
        result = prefilter.check(meta(duration=duration))
        assert not result.passed
        assert result.reason == prefilter.TOO_SHORT

    @pytest.mark.parametrize("duration", [7200, 7201, 29000])
    def test_rejects_long_videos_and_the_upper_bound(self, duration):
        result = prefilter.check(meta(duration=duration))
        assert not result.passed
        assert result.reason == prefilter.TOO_LONG

    def test_bounds_are_exclusive_as_the_guide_writes_them(self):
        # "duration > 300 & duration < 7200"
        assert not prefilter.check(meta(duration=300)).passed
        assert prefilter.check(meta(duration=301)).passed
        assert prefilter.check(meta(duration=7199)).passed
        assert not prefilter.check(meta(duration=7200)).passed

    def test_accepts_a_float_duration(self):
        assert prefilter.check(meta(duration=900.5)).passed

    def test_missing_duration_is_rejected_by_default(self):
        result = prefilter.check(meta(duration=None))
        assert not result.passed
        assert result.reason == prefilter.UNKNOWN_DURATION

    def test_missing_duration_is_allowed_before_enrichment(self):
        # RSS entries genuinely do not know the duration yet.
        assert prefilter.check(meta(duration=None), allow_unknown=True).passed

    def test_null_duration_is_treated_as_unknown(self):
        result = prefilter.check({"duration": None, "view_count": 10_000})
        assert result.reason == prefilter.UNKNOWN_DURATION


class TestViewCount:
    @pytest.mark.parametrize("views", [501, 1000, 10_000_000])
    def test_accepts_above_the_floor(self, views):
        assert prefilter.check(meta(view_count=views)).passed

    @pytest.mark.parametrize("views", [0, 1, 499, 500])
    def test_rejects_at_or_below_the_floor(self, views):
        result = prefilter.check(meta(view_count=views))
        assert not result.passed
        assert result.reason == prefilter.TOO_FEW_VIEWS

    def test_missing_view_count_is_rejected_by_default(self):
        result = prefilter.check(meta(view_count=None))
        assert not result.passed
        assert result.reason == prefilter.UNKNOWN_VIEWS

    def test_missing_view_count_is_allowed_before_enrichment(self):
        assert prefilter.check(meta(view_count=None), allow_unknown=True).passed


class TestLiveStatus:
    def test_rejects_is_live_true(self):
        result = prefilter.check(meta(is_live=True))
        assert not result.passed
        assert result.reason == prefilter.IS_LIVE

    def test_accepts_is_live_false(self):
        assert prefilter.check(meta(is_live=False)).passed

    @pytest.mark.parametrize("status", ["is_live", "is_upcoming", "post_live"])
    def test_rejects_live_statuses(self, status):
        assert prefilter.check(meta(live_status=status)).reason == prefilter.IS_LIVE

    @pytest.mark.parametrize("status", ["not_live", "was_live", None])
    def test_accepts_finished_or_ordinary_videos(self, status):
        # A finished stream is just a long video, and some of the best
        # content on the subscriptions list is archived streams.
        assert prefilter.check(meta(live_status=status)).passed

    def test_was_live_flag_alone_is_not_a_rejection(self):
        assert prefilter.check(meta(was_live=True)).passed

    def test_allow_live_config_overrides(self):
        cfg = PrefilterConfig(allow_live=True)
        assert prefilter.check(meta(is_live=True), cfg).passed

    def test_is_live_helper(self):
        assert prefilter.is_live({"is_live": True}) is True
        assert prefilter.is_live({"live_status": "is_live"}) is True
        assert prefilter.is_live({"live_status": "was_live"}) is False
        assert prefilter.is_live({}) is False


class TestConfigurability:
    def test_respects_custom_bounds(self):
        cfg = PrefilterConfig(min_duration=60, max_duration=1200, min_view_count=0)
        assert prefilter.check(meta(duration=120, view_count=1), cfg).passed
        assert not prefilter.check(meta(duration=1500), cfg).passed

    def test_defaults_to_the_guide_values_when_no_config_given(self):
        assert prefilter.check(meta(duration=120)).reason == prefilter.TOO_SHORT


class TestResultObject:
    def test_is_truthy_when_passed(self):
        assert bool(prefilter.check(meta()))

    def test_is_falsy_when_rejected(self):
        assert not bool(prefilter.check(meta(duration=10)))

    def test_passing_result_has_no_reason(self):
        assert prefilter.check(meta()).reason is None

    def test_reports_the_first_failing_reason_deterministically(self):
        # Short AND unpopular: the reason must be stable so the discovery
        # log's rejection counts mean something.
        result = prefilter.check(meta(duration=10, view_count=1))
        assert result.reason == prefilter.TOO_SHORT


class TestMatchFilterString:
    def test_default_config_reproduces_the_guide_string(self):
        assert prefilter.match_filter_string() == spec.MATCH_FILTERS

    def test_default_config_object_reproduces_the_guide_string(self):
        assert prefilter.match_filter_string(PrefilterConfig()) == spec.MATCH_FILTERS

    def test_reflects_custom_bounds(self):
        out = prefilter.match_filter_string(
            PrefilterConfig(min_duration=60, max_duration=1200, min_view_count=10)
        )
        assert "duration > 60" in out
        assert "duration < 1200" in out
        assert "view_count > 10" in out

    def test_drops_the_live_clause_when_live_is_allowed(self):
        out = prefilter.match_filter_string(PrefilterConfig(allow_live=True))
        assert "is_live" not in out

    def test_python_and_ytdlp_filters_agree(self, fixture_json):
        # The two filters exist in parallel; if they ever disagree, discovery
        # logs candidates the download stage refuses, or vice versa.
        entries = fixture_json("search-flat.json")["entries"]
        expected = {
            "searchHit01": True,
            "searchHit02": False,
            "searchHit03": False,
            "searchHit04": False,
            "searchHit05": True,
        }
        for entry in entries:
            assert prefilter.check(entry).passed is expected[entry["id"]], entry["id"]
