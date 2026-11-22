"""Phase 2: Google Takeout import.

This runs exactly once, on data that cannot be regenerated on demand (a new
export takes 30 minutes and a trip to a web UI). So the parser has to be
forgiving of everything Google puts in these files -- removed videos, Music
entries, ads, channel visits, missing `subtitles` -- without silently dropping
real history.
"""

from __future__ import annotations

import json

import pytest

from ytaf import takeout
from ytaf.errors import MetadataError
from ytaf.takeout import Subscription


class TestParseVideoId:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://www.youtube.com/watch?v=aircAruvnKk", "aircAruvnKk"),
            ("http://www.youtube.com/watch?v=aircAruvnKk", "aircAruvnKk"),
            ("https://youtube.com/watch?v=aircAruvnKk", "aircAruvnKk"),
            ("https://m.youtube.com/watch?v=aircAruvnKk", "aircAruvnKk"),
            ("https://www.youtube.com/watch?v=aircAruvnKk&t=42s", "aircAruvnKk"),
            ("https://www.youtube.com/watch?t=42s&v=aircAruvnKk", "aircAruvnKk"),
            ("https://youtu.be/aircAruvnKk", "aircAruvnKk"),
            ("https://youtu.be/aircAruvnKk?t=42", "aircAruvnKk"),
            ("https://www.youtube.com/shorts/aircAruvnKk", "aircAruvnKk"),
            ("https://music.youtube.com/watch?v=aircAruvnKk", "aircAruvnKk"),
            ("https://www.youtube.com/embed/aircAruvnKk", "aircAruvnKk"),
        ],
    )
    def test_extracts_from_every_url_shape(self, url, expected):
        assert takeout.parse_video_id(url) == expected

    def test_preserves_case(self):
        assert takeout.parse_video_id("https://youtu.be/WUvTyaaNkzM") == "WUvTyaaNkzM"

    def test_handles_ids_with_dashes_and_underscores(self):
        assert takeout.parse_video_id("https://youtu.be/OmJ-4B-mS_Y") == "OmJ-4B-mS_Y"

    @pytest.mark.parametrize(
        "url",
        [
            None,
            "",
            "not a url",
            "https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw",
            "https://www.youtube.com/@3blue1brown",
            "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab",
            "https://www.youtube.com/results?search_query=cats",
            "https://example.com/watch?v=aircAruvnKk",
            "https://www.youtube.com/watch",
            "https://www.youtube.com/watch?v=",
        ],
    )
    def test_returns_none_for_anything_that_is_not_a_video(self, url):
        assert takeout.parse_video_id(url) is None


class TestParseChannelId:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw",
            "http://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw",
            "https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw/videos",
        ],
    )
    def test_extracts_a_uc_id(self, url):
        assert takeout.parse_channel_id(url) == "UCYO_jab_esuFRV4b17AJtAw"

    @pytest.mark.parametrize(
        "url",
        [
            None,
            "",
            "https://www.youtube.com/@3blue1brown",
            "https://www.youtube.com/user/3blue1brown",
            "https://www.youtube.com/watch?v=aircAruvnKk",
        ],
    )
    def test_returns_none_without_a_uc_id(self, url):
        # Handle and legacy-user URLs carry no channel id, and resolving them
        # would need a network call during an offline import.
        assert takeout.parse_channel_id(url) is None


class TestParseWatchHistory:
    @pytest.fixture
    def entries(self, takeout_dir):
        path = takeout_dir / "YouTube and YouTube Music" / "history" / "watch-history.json"
        return takeout.parse_watch_history(path)

    def test_extracts_the_real_videos(self, entries):
        ids = {e.video_id for e in entries}
        assert "aircAruvnKk" in ids
        assert "WUvTyaaNkzM" in ids

    def test_skips_removed_videos(self, entries):
        titles = [e.title for e in entries]
        assert not any("has been removed" in t for t in titles)

    def test_skips_channel_visits(self, entries):
        assert "UCYO_jab_esuFRV4b17AJtAw" not in {e.video_id for e in entries}

    def test_keeps_entries_with_no_subtitles(self, entries):
        match = [e for e in entries if e.video_id == "noSubsVid01"]
        assert len(match) == 1
        assert match[0].channel == ""

    def test_reads_the_channel_from_subtitles(self, entries):
        match = next(e for e in entries if e.video_id == "aircAruvnKk")
        assert match.channel == "3Blue1Brown"
        assert match.channel_id == "UCYO_jab_esuFRV4b17AJtAw"

    def test_strips_the_watched_prefix_from_the_title(self, entries):
        match = next(e for e in entries if e.video_id == "WUvTyaaNkzM")
        assert match.title == "The Essence of Calculus"

    def test_deduplicates_by_video_id(self, entries):
        ids = [e.video_id for e in entries]
        assert len(ids) == len(set(ids))

    def test_keeps_the_most_recent_timestamp_for_a_duplicate(self, entries):
        # aircAruvnKk appears twice in the fixture: 09-13 and 09-14.
        match = next(e for e in entries if e.video_id == "aircAruvnKk")
        assert match.timestamp.startswith("2026-09-14")

    def test_handles_shorts_and_short_links(self, entries):
        ids = {e.video_id for e in entries}
        assert "shortFormV1" in ids
        assert "shortLinkI1" in ids

    def test_accepts_raw_bytes(self, takeout_dir):
        path = takeout_dir / "YouTube and YouTube Music" / "history" / "watch-history.json"
        assert takeout.parse_watch_history(path.read_bytes())

    def test_empty_array_yields_nothing(self, tmp_path):
        path = tmp_path / "wh.json"
        path.write_text("[]", encoding="utf-8")
        assert takeout.parse_watch_history(path) == []

    def test_malformed_json_raises(self, tmp_path):
        path = tmp_path / "wh.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(MetadataError):
            takeout.parse_watch_history(path)

    def test_an_entry_missing_time_still_parses(self, tmp_path):
        path = tmp_path / "wh.json"
        path.write_text(
            json.dumps(
                [{"title": "Watched X", "titleUrl": "https://youtu.be/aircAruvnKk"}]
            ),
            encoding="utf-8",
        )
        entries = takeout.parse_watch_history(path)
        assert entries[0].video_id == "aircAruvnKk"
        assert entries[0].timestamp is None


class TestParseLikeHistory:
    @pytest.fixture
    def entries(self, takeout_dir):
        path = takeout_dir / "YouTube and YouTube Music" / "history" / "like-history.json"
        return takeout.parse_like_history(path)

    def test_extracts_liked_videos(self, entries):
        assert {e.video_id for e in entries} == {"aircAruvnKk", "OmJ-4B-mS-Y"}

    def test_strips_the_liked_prefix(self, entries):
        match = next(e for e in entries if e.video_id == "OmJ-4B-mS-Y")
        assert match.title == "The Map of Mathematics"

    def test_skips_removed_videos(self, entries):
        assert len(entries) == 2

    def test_also_reads_the_liked_videos_csv_form(self, tmp_path):
        # Real exports have shipped likes as playlists/Liked videos.csv more
        # than once; the guide names the json path.
        path = tmp_path / "Liked videos.csv"
        path.write_text(
            "Video ID,Playlist Video Creation Timestamp\n"
            "aircAruvnKk,2026-09-13T22:40:00+00:00\n"
            "OmJ-4B-mS-Y,2026-08-30T10:00:00+00:00\n",
            encoding="utf-8",
        )
        entries = takeout.parse_like_history(path)
        assert {e.video_id for e in entries} == {"aircAruvnKk", "OmJ-4B-mS-Y"}


class TestParseSubscriptions:
    @pytest.fixture
    def subs(self, takeout_dir):
        path = (
            takeout_dir / "YouTube and YouTube Music" / "subscriptions" / "subscriptions.csv"
        )
        return takeout.parse_subscriptions(path)

    def test_reads_every_row(self, subs):
        assert len(subs) == 4

    def test_extracts_id_and_title(self, subs):
        first = subs[0]
        assert first.channel_id == "UCYO_jab_esuFRV4b17AJtAw"
        assert first.title == "3Blue1Brown"

    def test_handles_a_quoted_title_containing_a_comma(self, subs):
        match = next(s for s in subs if s.channel_id == "UCxqAWLTk1CmBvZFPzeZMd9A")
        assert match.title == "Domain of Science, Ltd"

    def test_tolerates_lowercase_headers(self, tmp_path):
        path = tmp_path / "subs.csv"
        path.write_text(
            "channel id,channel url,channel title\nUCaaa,http://x,A\n", encoding="utf-8"
        )
        assert takeout.parse_subscriptions(path)[0].channel_id == "UCaaa"

    def test_tolerates_a_bom(self, tmp_path):
        path = tmp_path / "subs.csv"
        path.write_bytes(
            b"\xef\xbb\xbfChannel Id,Channel Url,Channel Title\nUCaaa,http://x,A\n"
        )
        assert takeout.parse_subscriptions(path)[0].channel_id == "UCaaa"

    def test_header_only_file_yields_nothing(self, tmp_path):
        path = tmp_path / "subs.csv"
        path.write_text("Channel Id,Channel Url,Channel Title\n", encoding="utf-8")
        assert takeout.parse_subscriptions(path) == []

    def test_skips_a_row_with_no_channel_id(self, tmp_path):
        path = tmp_path / "subs.csv"
        path.write_text(
            "Channel Id,Channel Url,Channel Title\n,http://x,A\nUCaaa,http://y,B\n",
            encoding="utf-8",
        )
        assert [s.channel_id for s in takeout.parse_subscriptions(path)] == ["UCaaa"]


class TestFindTakeoutFiles:
    def test_finds_all_three(self, takeout_dir):
        found = takeout.find_takeout_files(takeout_dir)
        assert found.watch_history is not None
        assert found.like_history is not None
        assert found.subscriptions is not None

    def test_searches_recursively_from_any_ancestor(self, takeout_dir):
        found = takeout.find_takeout_files(takeout_dir.parent)
        assert found.watch_history is not None

    def test_missing_files_are_none_not_an_error(self, tmp_path):
        found = takeout.find_takeout_files(tmp_path)
        assert found.watch_history is None
        assert found.like_history is None
        assert found.subscriptions is None

    def test_finds_the_liked_videos_csv_alternative(self, tmp_path):
        base = tmp_path / "Takeout" / "YouTube and YouTube Music" / "playlists"
        base.mkdir(parents=True)
        (base / "Liked videos.csv").write_text("Video ID\n", encoding="utf-8")
        found = takeout.find_takeout_files(tmp_path)
        assert found.like_history is not None


class TestImportTakeout:
    def test_inserts_history_as_weak_positives(self, ledger, takeout_dir):
        takeout.import_takeout(ledger, takeout_dir)
        row = ledger.get("WUvTyaaNkzM")
        assert row.source == "history"
        assert row.label == 1

    def test_inserts_likes_as_strong_positives(self, ledger, takeout_dir):
        takeout.import_takeout(ledger, takeout_dir)
        row = ledger.get("OmJ-4B-mS-Y")
        assert row.source == "like"
        assert row.label == 2

    def test_a_liked_video_also_in_history_ends_up_strong(self, ledger, takeout_dir):
        # aircAruvnKk is in both files. The like is the stronger signal.
        takeout.import_takeout(ledger, takeout_dir)
        row = ledger.get("aircAruvnKk")
        assert row.source == "like"
        assert row.label == 2

    def test_stores_title_and_channel_in_metadata(self, ledger, takeout_dir):
        takeout.import_takeout(ledger, takeout_dir)
        row = ledger.get("aircAruvnKk")
        assert row.channel == "3Blue1Brown"
        assert row.title

    def test_report_counts_everything(self, ledger, takeout_dir):
        report = takeout.import_takeout(ledger, takeout_dir)
        assert report.watch_entries > 0
        assert report.like_entries == 2
        assert report.subscriptions == 4
        assert report.rows_inserted == ledger.count()

    def test_report_returns_channel_ids_for_discovery(self, ledger, takeout_dir):
        report = takeout.import_takeout(ledger, takeout_dir)
        assert "UCYO_jab_esuFRV4b17AJtAw" in report.channel_ids
        assert len(report.channel_ids) == 4

    def test_is_idempotent(self, ledger, takeout_dir):
        takeout.import_takeout(ledger, takeout_dir)
        first = ledger.count()
        second_report = takeout.import_takeout(ledger, takeout_dir)
        assert ledger.count() == first
        assert second_report.rows_inserted == 0

    def test_reimport_never_lowers_a_label(self, ledger, takeout_dir):
        takeout.import_takeout(ledger, takeout_dir)
        takeout.import_takeout(ledger, takeout_dir)
        assert ledger.get("aircAruvnKk").label == 2

    def test_does_not_overwrite_measured_feedback(self, ledger, takeout_dir):
        takeout.import_takeout(ledger, takeout_dir)
        ledger.record_watch("aircAruvnKk", 0.2)
        takeout.import_takeout(ledger, takeout_dir)
        assert ledger.get("aircAruvnKk").watch_pct == pytest.approx(0.2)

    def test_nothing_is_marked_downloaded_or_selected(self, ledger, takeout_dir):
        # Takeout rows are training data, not a download queue. Marking them
        # selected would try to fetch a decade of watch history.
        takeout.import_takeout(ledger, takeout_dir)
        assert ledger.count(selected=1) == 0
        assert ledger.count(downloaded=1) == 0

    def test_missing_takeout_directory_raises(self, ledger, tmp_path):
        with pytest.raises(FileNotFoundError):
            takeout.import_takeout(ledger, tmp_path / "nope")

    def test_partial_export_still_imports_what_is_there(self, ledger, tmp_path, fixture_text):
        base = tmp_path / "Takeout" / "YouTube and YouTube Music" / "history"
        base.mkdir(parents=True)
        (base / "watch-history.json").write_text(
            fixture_text("watch-history.json"), encoding="utf-8"
        )
        report = takeout.import_takeout(ledger, tmp_path)
        assert report.watch_entries > 0
        assert report.like_entries == 0
        assert report.subscriptions == 0
        assert report.skipped


class TestMergeChannelIds:
    def test_unions_and_sorts(self):
        merged = takeout.merge_channel_ids(
            ["UCbbb"], [Subscription("UCaaa"), Subscription("UCccc")]
        )
        assert merged == ("UCaaa", "UCbbb", "UCccc")

    def test_deduplicates(self):
        merged = takeout.merge_channel_ids(["UCaaa"], [Subscription("UCaaa")])
        assert merged == ("UCaaa",)

    def test_handles_empty_inputs(self):
        assert takeout.merge_channel_ids([], []) == ()
