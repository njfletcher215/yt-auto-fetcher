"""Guard the constants the design pins exactly.

These are the cheapest tests here and the ones most worth having: if someone
"improves" the format selector to grab 1440p, the server starts being asked to
transcode VP9 and the whole system quietly stops working. A string comparison
catches that in CI instead of on the couch.
"""

from __future__ import annotations

import re

import pytest

from ytaf import spec


class TestFormatSelector:
    def test_is_h264_1080p_aac(self):
        assert spec.FORMAT_SELECTOR == (
            "bv*[vcodec^=avc1][height<=1080]+ba[acodec^=mp4a]/b[ext=mp4]"
        )

    def test_caps_height_at_1080(self):
        assert "[height<=1080]" in spec.FORMAT_SELECTOR
        assert spec.MAX_HEIGHT == 1080

    def test_requests_h264_video(self):
        assert "vcodec^=avc1" in spec.FORMAT_SELECTOR

    def test_requests_aac_audio(self):
        assert "acodec^=mp4a" in spec.FORMAT_SELECTOR

    def test_falls_back_to_a_progressive_mp4(self):
        assert spec.FORMAT_SELECTOR.endswith("/b[ext=mp4]")

    def test_never_mentions_a_codec_that_would_force_transcoding(self):
        for codec in ("vp9", "vp09", "av01", "av1", "webm", "opus"):
            assert codec not in spec.FORMAT_SELECTOR.lower()

    def test_merges_to_mp4(self):
        assert spec.MERGE_OUTPUT_FORMAT == "mp4"

    def test_h264_fallback_keeps_the_same_constraints(self):
        fallback = spec.FORMAT_SELECTOR_H264_FALLBACK
        assert "vcodec^=h264" in fallback
        assert "[height<=1080]" in fallback
        assert "acodec^=mp4a" in fallback


class TestOutputTemplate:
    def test_is_relative_so_the_library_root_stays_configurable(self):
        assert not spec.OUTPUT_TEMPLATE.startswith("/")

    def test_groups_by_channel(self):
        assert spec.OUTPUT_TEMPLATE.startswith("%(channel)s/")

    def test_has_exactly_one_directory_level(self):
        # A Home Videos library needs no season folders, and a flatter tree is
        # one less thing for retention to have to prune.
        assert spec.OUTPUT_TEMPLATE.count("/") == 1

    def test_carries_no_season_or_episode_numbering(self):
        # Those exist only to satisfy a TV Shows library's episode resolver.
        # Home Videos has no such requirement, and MMDD episode numbers
        # collided whenever a channel posted twice in a day.
        assert "Season" not in spec.OUTPUT_TEMPLATE
        assert "upload_date>" not in spec.OUTPUT_TEMPLATE
        assert not re.search(r"S%\(.*\)sE%\(", spec.OUTPUT_TEMPLATE)

    def test_filename_is_date_then_title_then_id(self):
        filename = spec.OUTPUT_TEMPLATE.split("/")[-1]
        assert filename == "%(upload_date)s - %(title)s [%(id)s].%(ext)s"

    def test_date_prefix_sorts_chronologically(self):
        # YYYYMMDD sorts lexicographically, so a channel folder reads in
        # upload order with no metadata at all.
        assert spec.OUTPUT_TEMPLATE.split("/")[-1].startswith("%(upload_date)s")

    def test_embeds_the_video_id(self):
        # Load-bearing: with no NFO uniqueid, this is how the feedback stage
        # recovers a ledger row from a Jellyfin item path.
        assert "[%(id)s]" in spec.OUTPUT_TEMPLATE

    def test_extension_is_templated_not_hardcoded(self):
        assert spec.OUTPUT_TEMPLATE.endswith(".%(ext)s")


class TestVideoIdInPath:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("20260914 - A Title [aircAruvnKk].mp4", "aircAruvnKk"),
            ("/library/C/20260914 - A [WUvTyaaNkzM].mp4", "WUvTyaaNkzM"),
            ("20260914 - Brackets [in] the title [aircAruvnKk].mp4", "aircAruvnKk"),
            ("20260914 - Dashes [OmJ-4B-mS_Y].mp4", "OmJ-4B-mS_Y"),
        ],
    )
    def test_matches_the_id_suffix(self, name, expected):
        found = re.findall(spec.VIDEO_ID_IN_PATH, name)
        assert found[-1] == expected

    @pytest.mark.parametrize(
        "name",
        [
            "Some Movie (2011).mkv",
            "20260914 - No id here.mp4",
            "20260914 - Too short [abc].mp4",
            "20260914 - Too long [aircAruvnKkExtra].mp4",
        ],
    )
    def test_does_not_match_an_unrelated_filename(self, name):
        assert re.findall(spec.VIDEO_ID_IN_PATH, name) == []


class TestDiscoveryConstants:
    def test_rss_url_is_the_anonymous_feed_endpoint(self):
        url = spec.RSS_FEED_URL.format(channel_id="UCabc")
        assert url == "https://www.youtube.com/feeds/videos.xml?channel_id=UCabc"

    def test_rss_url_is_https(self):
        assert spec.RSS_FEED_URL.startswith("https://")

    def test_search_prefix_is_date_ordered(self):
        # ytsearchdate, not ytsearch: plain ytsearch returns relevance-ranked
        # results that barely change from day to day.
        assert spec.SEARCH_PREFIX == "ytsearchdate"

    def test_match_filters_matches_the_prefilter(self):
        assert spec.MATCH_FILTERS == (
            "duration > 300 & duration < 7200 & !is_live & view_count > 500"
        )

    @pytest.mark.parametrize(
        "clause",
        ["duration > 300", "duration < 7200", "!is_live", "view_count > 500"],
    )
    def test_match_filters_contains_each_clause(self, clause):
        assert clause in spec.MATCH_FILTERS


class TestSelectionConstants:
    def test_five_new_two_repeat_three_random(self):
        assert spec.DEFAULT_N_MODEL_NEW == 5
        assert spec.DEFAULT_N_MODEL_REPEAT == 2
        assert spec.DEFAULT_N_RANDOM == 3

    def test_ten_downloads_per_day(self):
        total = (
            spec.DEFAULT_N_MODEL_NEW
            + spec.DEFAULT_N_MODEL_REPEAT
            + spec.DEFAULT_N_RANDOM
        )
        assert total == 10

    def test_exploration_is_never_zero(self):
        # The random arm is the control group the degradation check measures
        # against. A default of 0 would make that check uncomputable.
        assert spec.DEFAULT_N_RANDOM > 0

    def test_three_named_arms(self):
        assert spec.SELECTIONS == {"model", "repeat", "random"}
        assert (spec.MODEL, spec.REPEAT, spec.RANDOM) == ("model", "repeat", "random")


class TestLabels:
    def test_three_sentiment_values(self):
        assert spec.LABELS == {-1, 1, 2}

    def test_like_is_above_history_is_above_dislike(self):
        # Ordered so a numeric comparison means what it reads like.
        assert spec.LABEL_DISLIKE < spec.LABEL_HISTORY < spec.LABEL_LIKE

    def test_dislike_is_negative_so_it_cannot_be_confused_with_absence(self):
        # 0 would be falsy and would read as "no label" in any truthiness
        # check that slipped through.
        assert spec.LABEL_DISLIKE == -1
        assert spec.LABEL_DISLIKE != 0

    def test_sources_are_provenance_only(self):
        # `source` records how a row arrived; sentiment lives in `label`.
        assert spec.SOURCES == {"history", "like", "rss", "search"}


class TestPlaybackBands:
    def test_watch_threshold_is_sixty_percent(self):
        assert spec.WATCH_THRESHOLD == pytest.approx(0.60)

    def test_partial_floor_is_fifteen_percent(self):
        assert spec.PARTIAL_FLOOR == pytest.approx(0.15)

    def test_the_bands_are_ordered_and_inside_zero_to_one(self):
        assert 0 < spec.PARTIAL_FLOOR < spec.WATCH_THRESHOLD < 1


class TestLedgerSchema:
    def test_declares_the_candidates_table(self):
        assert "candidates" in spec.LEDGER_SCHEMA

    @pytest.mark.parametrize("column", spec.LEDGER_COLUMNS)
    def test_schema_declares_every_documented_column(self, column):
        assert re.search(rf"\b{column}\b", spec.LEDGER_SCHEMA)

    def test_column_tuple_is_the_original_twelve_plus_four(self):
        assert spec.LEDGER_COLUMNS == (
            "video_id",
            "seen_at",
            "source",
            "metadata",
            "score",
            "selected",
            "selection",
            "downloaded",
            "filepath",
            "watched",
            "watch_pct",
            "label",
            "play_count",
            "offer_count",
            "last_offered_at",
            "saved",
        )

    def test_video_id_is_the_primary_key(self):
        assert re.search(r"video_id\s+TEXT\s+PRIMARY KEY", spec.LEDGER_SCHEMA)

    def test_seen_at_is_not_nullable(self):
        assert re.search(r"seen_at\s+TEXT\s+NOT NULL", spec.LEDGER_SCHEMA)

    @pytest.mark.parametrize(
        "column", ["selected", "downloaded", "play_count", "offer_count", "saved"]
    )
    def test_counters_default_to_zero(self, column):
        assert re.search(rf"{column}\s+INTEGER\s+DEFAULT 0", spec.LEDGER_SCHEMA)

    def test_is_idempotent(self):
        assert "IF NOT EXISTS" in spec.LEDGER_SCHEMA


class TestForbiddenFlags:
    @pytest.mark.parametrize(
        "flag",
        ["--cookies", "--cookies-from-browser", "--username", "--password", "--netrc"],
    )
    def test_authentication_flags_are_forbidden(self, flag):
        assert flag in spec.FORBIDDEN_YTDLP_FLAGS

    def test_every_forbidden_flag_is_long_form(self):
        # The check is a prefix match on `--flag`; a bare `-x` entry would
        # match far too much.
        assert all(f.startswith("--") for f in spec.FORBIDDEN_YTDLP_FLAGS)
