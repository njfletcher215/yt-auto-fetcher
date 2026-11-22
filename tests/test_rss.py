"""Phase 4a: channel RSS parsing and fetching.

The feed is the backbone of discovery -- anonymous, unmetered, no cookies.
The thing worth testing hardest is what it *doesn't* carry: no duration and no
live status, which is exactly the information the prefilter needs. If
`as_metadata` invented a duration, every short on your subscriptions would
sail through.
"""

from __future__ import annotations

import pytest

from ytaf import rss, spec
from ytaf.errors import DiscoveryError


class TestFeedUrl:
    def test_builds_the_anonymous_feed_url(self):
        assert rss.feed_url("UCYO_jab_esuFRV4b17AJtAw") == (
            "https://www.youtube.com/feeds/videos.xml?channel_id=UCYO_jab_esuFRV4b17AJtAw"
        )

    def test_uses_the_spec_template(self):
        assert rss.feed_url("UCaaa") == spec.RSS_FEED_URL.format(channel_id="UCaaa")

    @pytest.mark.parametrize(
        "channel_id",
        ["", "3blue1brown", "@3blue1brown", "UC", "not-a-channel", None],
    )
    def test_rejects_anything_that_is_not_a_channel_id(self, channel_id):
        # A handle here produces a 404 feed that looks like an empty channel,
        # which would read as "this channel stopped uploading" forever.
        with pytest.raises(DiscoveryError):
            rss.feed_url(channel_id)

    def test_does_not_url_encode_a_valid_id_into_uselessness(self):
        assert "UCYO_jab_esuFRV4b17AJtAw" in rss.feed_url("UCYO_jab_esuFRV4b17AJtAw")


class TestParseFeed:
    @pytest.fixture
    def entries(self, fixture_text):
        return rss.parse_feed(fixture_text("channel-feed.xml"))

    def test_parses_every_usable_entry(self, entries):
        # Four entries in the fixture, one with no yt:videoId.
        assert len(entries) == 3

    def test_skips_an_entry_with_no_video_id(self, entries):
        assert all(e.video_id for e in entries)

    def test_extracts_the_video_id(self, entries):
        assert entries[0].video_id == "aircAruvnKk"

    def test_extracts_the_title_and_unescapes_entities(self, entries):
        assert entries[0].title == "But what is a neural network? & why does it work?"

    def test_extracts_the_channel_name_and_id(self, entries):
        assert entries[0].channel == "3Blue1Brown"
        assert entries[0].channel_id == "UCYO_jab_esuFRV4b17AJtAw"

    def test_extracts_the_published_timestamp(self, entries):
        assert entries[0].published == "2026-09-13T15:00:11+00:00"

    def test_extracts_the_media_description(self, entries):
        assert entries[0].description.startswith("Chapter 1 on neural networks.")

    def test_extracts_the_view_count_from_media_statistics(self, entries):
        assert entries[0].view_count == 4567890

    def test_extracts_the_thumbnail(self, entries):
        assert entries[0].thumbnail.endswith("hqdefault.jpg")

    def test_preserves_feed_order(self, entries):
        assert [e.video_id for e in entries] == [
            "aircAruvnKk",
            "WUvTyaaNkzM",
            "noStatsVid1",
        ]

    def test_missing_description_becomes_empty_string(self, entries):
        match = next(e for e in entries if e.video_id == "noStatsVid1")
        assert match.description == ""

    def test_missing_statistics_becomes_none(self, entries):
        match = next(e for e in entries if e.video_id == "noStatsVid1")
        assert match.view_count is None

    def test_low_view_count_is_preserved_not_filtered_here(self, entries):
        # Parsing and filtering are separate stages; the parser reports what
        # the feed said and the prefilter decides.
        match = next(e for e in entries if e.video_id == "WUvTyaaNkzM")
        assert match.view_count == 310

    def test_an_empty_feed_yields_no_entries(self, fixture_text):
        assert rss.parse_feed(fixture_text("channel-feed-empty.xml")) == []

    def test_accepts_bytes(self, fixture_text):
        assert rss.parse_feed(fixture_text("channel-feed.xml").encode())

    @pytest.mark.parametrize(
        "payload",
        ["", "not xml at all", "<feed><unclosed>", b"\x00\x01\x02"],
    )
    def test_malformed_input_raises_discovery_error(self, payload):
        with pytest.raises(DiscoveryError):
            rss.parse_feed(payload)

    def test_an_html_error_page_raises_rather_than_parsing_as_empty(self):
        # YouTube serves HTML for a bad channel id; treating that as "no new
        # videos" would hide a broken feed indefinitely.
        with pytest.raises(DiscoveryError):
            rss.parse_feed("<html><body>404 Not Found</body></html>")


class TestAsMetadata:
    @pytest.fixture
    def metadata(self, fixture_text):
        entries = rss.parse_feed(fixture_text("channel-feed.xml"))
        return entries[0].as_metadata()

    def test_uses_ytdlp_key_names(self, metadata):
        assert metadata["id"] == "aircAruvnKk"
        assert metadata["title"]
        assert metadata["channel"] == "3Blue1Brown"
        assert metadata["channel_id"] == "UCYO_jab_esuFRV4b17AJtAw"

    def test_carries_the_view_count_through(self, metadata):
        assert metadata["view_count"] == 4567890

    def test_omits_duration_because_the_feed_does_not_know_it(self, metadata):
        assert "duration" not in metadata

    def test_omits_is_live_because_the_feed_does_not_know_it(self, metadata):
        assert "is_live" not in metadata
        assert "live_status" not in metadata

    def test_sets_upload_date_in_ytdlp_format(self, metadata):
        # The output template reads upload_date; an ISO timestamp would
        # produce "Season 2026-09-13T15".
        assert metadata["upload_date"] == "20260913"


class TestFetchFeed:
    def test_fetches_and_parses(self, fake_http, fixture_text):
        fake_http.route("channel_id=UCYO", body=fixture_text("channel-feed.xml"))
        entries = rss.fetch_feed("UCYO_jab_esuFRV4b17AJtAw")
        assert len(entries) == 3

    def test_requests_the_right_url(self, fake_http, fixture_text):
        fake_http.route("feeds/videos.xml", body=fixture_text("channel-feed.xml"))
        rss.fetch_feed("UCYO_jab_esuFRV4b17AJtAw")
        assert fake_http.called("channel_id=UCYO_jab_esuFRV4b17AJtAw", "GET")

    def test_sends_no_cookies_or_auth(self, fake_http, fixture_text):
        fake_http.route("feeds/videos.xml", body=fixture_text("channel-feed.xml"))
        rss.fetch_feed("UCYO_jab_esuFRV4b17AJtAw")
        _, _, kwargs = fake_http.calls[0]
        headers = kwargs.get("headers") or {}
        assert not any(
            k.lower() in ("cookie", "authorization") for k in headers
        )

    @pytest.mark.parametrize("status", [404, 403, 429, 500, 503])
    def test_non_2xx_raises_discovery_error(self, fake_http, status):
        fake_http.route("feeds/videos.xml", status=status, body="nope")
        with pytest.raises(DiscoveryError):
            rss.fetch_feed("UCYO_jab_esuFRV4b17AJtAw")


class TestFetchFeeds:
    def test_fetches_every_channel(self, fake_http, fixture_text):
        fake_http.route("feeds/videos.xml", body=fixture_text("channel-feed.xml"))
        entries, failures = rss.fetch_feeds(["UCaaa", "UCbbb", "UCccc"])
        assert failures == []
        assert fake_http.call_count("feeds/videos.xml") == 3
        assert len(entries) == 9

    def test_one_dead_channel_does_not_abort_the_run(self, fake_http, fixture_text, monkeypatch):
        good = fixture_text("channel-feed.xml")

        def fake_get(url, **kwargs):
            from ytaf import http

            if "UCdead" in url:
                return http.Response(status=404, body=b"not found", url=url)
            return http.Response(status=200, body=good.encode(), url=url)

        from ytaf import http as http_mod

        monkeypatch.setattr(http_mod, "get", fake_get)
        entries, failures = rss.fetch_feeds(["UCaaa", "UCdead", "UCbbb"])
        assert len(entries) == 6
        assert len(failures) == 1
        assert failures[0][0] == "UCdead"

    def test_reports_the_failing_channel_and_a_reason(self, fake_http):
        fake_http.route("feeds/videos.xml", status=500, body="boom")
        entries, failures = rss.fetch_feeds(["UCaaa"])
        assert entries == []
        assert failures[0][0] == "UCaaa"
        assert failures[0][1]

    def test_handles_an_empty_channel_list(self, fake_http):
        assert rss.fetch_feeds([]) == ([], [])

    def test_sleeps_between_requests_when_asked(self, fake_http, fixture_text, monkeypatch):
        fake_http.route("feeds/videos.xml", body=fixture_text("channel-feed.xml"))
        slept: list[float] = []
        monkeypatch.setattr(rss.time, "sleep", slept.append)
        rss.fetch_feeds(["UCaaa", "UCbbb"], sleep_requests=2)
        assert slept and all(s == 2 for s in slept)

    def test_does_not_sleep_when_the_interval_is_zero(self, fake_http, fixture_text, monkeypatch):
        fake_http.route("feeds/videos.xml", body=fixture_text("channel-feed.xml"))
        slept: list[float] = []
        monkeypatch.setattr(rss.time, "sleep", slept.append)
        rss.fetch_feeds(["UCaaa", "UCbbb"], sleep_requests=0)
        assert slept == []

    def test_deduplicates_nothing_leaving_that_to_the_caller(self, fake_http, fixture_text):
        # Two channels can legitimately both surface a collab video; the
        # ledger's primary key is where that collapses.
        fake_http.route("feeds/videos.xml", body=fixture_text("channel-feed.xml"))
        entries, _ = rss.fetch_feeds(["UCaaa", "UCbbb"])
        assert len(entries) == 6
