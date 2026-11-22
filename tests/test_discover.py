"""The discovery stage end to end.

The thing worth testing hardest: RSS gives you an id, a title and sometimes a
view count, but never a duration or live status -- which is three of the
prefilter's four clauses. So candidates go through a metadata-only yt-dlp call
before filtering, and the drop-held -> enrich -> prefilter ordering is what
keeps that call proportional to genuinely new videos.

Reaped rows are deliberately *not* dropped: they are the repeat arm's pool.
"""

from __future__ import annotations

import json

import pytest

from ytaf import discover, prefilter, ytdlp
from ytaf.config import DiscoveryConfig


def ytdlp_router(*, search_payload=None, metadata_payload=None, failures=()):
    """Dispatch a fake yt-dlp call by inspecting its arguments."""

    def handler(args, **kwargs):
        joined = " ".join(args)
        if "ytsearchdate" in joined:
            body = search_payload or '{"_type": "playlist", "entries": []}'
        else:
            body = metadata_payload or ""
        return ytdlp.YtdlpResult(
            args=tuple(args),
            returncode=1 if failures else 0,
            stdout=body,
            stderr="\n".join(failures),
        )

    return handler


def enriched(video_ids, **overrides):
    """A JSON-lines metadata payload covering `video_ids`."""
    lines = []
    for vid in video_ids:
        entry = {
            "id": vid,
            "title": f"Title {vid}",
            "description": "Description.",
            "channel": "3Blue1Brown",
            "channel_id": "UCYO_jab_esuFRV4b17AJtAw",
            "duration": 900,
            "view_count": 10_000,
            "upload_date": "20260914",
            "live_status": "not_live",
        }
        entry.update(overrides.get(vid, {}))
        lines.append(json.dumps(entry))
    return "\n".join(lines)


@pytest.fixture
def rss_config(config):
    return config.with_(
        discovery=DiscoveryConfig(
            strategy="rss", channel_ids=("UCYO_jab_esuFRV4b17AJtAw",)
        )
    )


@pytest.fixture
def search_config(config):
    return config.with_(
        discovery=DiscoveryConfig(strategy="search", keywords=("computerphile",))
    )


class TestCollectRss:
    def test_logs_entries_from_every_configured_channel(
        self, rss_config, fake_http, fixture_text
    ):
        fake_http.route("feeds/videos.xml", body=fixture_text("channel-feed.xml"))
        report = discover.DiscoveryReport()
        found = discover.collect_rss(rss_config, report)
        assert set(found) == {"aircAruvnKk", "WUvTyaaNkzM", "noStatsVid1"}
        assert report.rss_entries == 3

    def test_tags_the_source_as_rss(self, rss_config, fake_http, fixture_text):
        fake_http.route("feeds/videos.xml", body=fixture_text("channel-feed.xml"))
        found = discover.collect_rss(rss_config, discover.DiscoveryReport())
        assert all(c.source == "rss" for c in found.values())

    def test_a_dead_feed_is_recorded_not_raised(self, rss_config, fake_http):
        fake_http.route("feeds/videos.xml", status=404, body="nope")
        report = discover.DiscoveryReport()
        found = discover.collect_rss(rss_config, report)
        assert found == {}
        assert report.failures
        assert not report.ok

    def test_skipped_when_the_strategy_excludes_rss(self, search_config, fake_http):
        found = discover.collect_rss(search_config, discover.DiscoveryReport())
        assert found == {}
        assert fake_http.calls == []


class TestCollectSearch:
    def test_logs_search_entries(self, search_config, fake_ytdlp, fixture_text):
        fake_ytdlp.handler = ytdlp_router(search_payload=fixture_text("search-flat.json"))
        report = discover.DiscoveryReport()
        found = discover.collect_search(search_config, report)
        assert "searchHit01" in found
        assert report.search_entries == 5

    def test_tags_the_source_as_search(self, search_config, fake_ytdlp, fixture_text):
        fake_ytdlp.handler = ytdlp_router(search_payload=fixture_text("search-flat.json"))
        found = discover.collect_search(search_config, discover.DiscoveryReport())
        assert all(c.source == "search" for c in found.values())

    def test_uses_ytsearchdate(self, search_config, fake_ytdlp, fixture_text):
        fake_ytdlp.handler = ytdlp_router(search_payload=fixture_text("search-flat.json"))
        discover.collect_search(search_config, discover.DiscoveryReport())
        assert fake_ytdlp.invoked_with("ytsearchdate")
        assert not any(
            a.startswith("ytsearch:") for call in fake_ytdlp.calls for a in call
        )

    def test_searches_every_keyword(self, config, fake_ytdlp, fixture_text):
        cfg = config.with_(
            discovery=DiscoveryConfig(strategy="search", keywords=("a", "b", "c"))
        )
        fake_ytdlp.handler = ytdlp_router(search_payload=fixture_text("search-flat.json"))
        discover.collect_search(cfg, discover.DiscoveryReport())
        assert len(fake_ytdlp.calls) == 3

    def test_one_failing_keyword_does_not_abort_the_rest(self, config, fake_ytdlp):
        cfg = config.with_(
            discovery=DiscoveryConfig(strategy="search", keywords=("good", "bad"))
        )

        def handler(args, **kwargs):
            if "ytsearchdate40:bad" in args:
                raise ytdlp.YtdlpError("boom", returncode=1, stderr="ERROR")
            return ytdlp.YtdlpResult(
                args=tuple(args),
                returncode=0,
                stdout='{"_type":"playlist","entries":[{"id":"goodHit0001","duration":900,"view_count":1000,"title":"T","channel":"C"}]}',
            )

        fake_ytdlp.handler = handler
        report = discover.DiscoveryReport()
        found = discover.collect_search(cfg, report)
        assert "goodHit0001" in found
        assert report.failures

    def test_skipped_when_the_strategy_excludes_search(self, rss_config, fake_ytdlp):
        found = discover.collect_search(rss_config, discover.DiscoveryReport())
        assert found == {}
        assert fake_ytdlp.calls == []


class TestDropHeld:
    def test_drops_ids_we_currently_hold(self, config, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", downloaded=1, filepath="/x.mp4"))
        candidates = {
            "aaaaaaaaaa1": make_candidate("aaaaaaaaaa1"),
            "bbbbbbbbbb2": make_candidate("bbbbbbbbbb2"),
        }
        report = discover.DiscoveryReport()
        kept = discover.drop_held(candidates, ledger, config, report)
        assert set(kept) == {"bbbbbbbbbb2"}
        assert report.already_held == 1

    def test_drops_saved_videos(self, config, ledger, make_candidate):
        # Saved videos live outside the managed library, so a naive
        # "is it in library/?" check would re-download them forever.
        ledger.upsert(make_candidate("aaaaaaaaaa1", saved=1, filepath="/perm/x.mp4"))
        candidates = {"aaaaaaaaaa1": make_candidate("aaaaaaaaaa1")}
        kept = discover.drop_held(candidates, ledger, config, discover.DiscoveryReport())
        assert kept == {}

    def test_keeps_a_reaped_row_for_the_repeat_arm(self, config, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", downloaded=1, filepath=None))
        candidates = {"aaaaaaaaaa1": make_candidate("aaaaaaaaaa1")}
        kept = discover.drop_held(candidates, ledger, config, discover.DiscoveryReport())
        assert set(kept) == {"aaaaaaaaaa1"}

    def test_keeps_a_known_but_unfetched_row_so_it_can_be_enriched(
        self, config, ledger, make_candidate
    ):
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        candidates = {"aaaaaaaaaa1": make_candidate("aaaaaaaaaa1")}
        kept = discover.drop_held(candidates, ledger, config, discover.DiscoveryReport())
        assert set(kept) == {"aaaaaaaaaa1"}

    def test_handles_an_empty_input(self, config, ledger):
        assert discover.drop_held({}, ledger, config, discover.DiscoveryReport()) == {}


class TestEnrich:
    def test_fills_in_duration_and_view_count(self, rss_config, fake_ytdlp, make_candidate):
        fake_ytdlp.handler = ytdlp_router(metadata_payload=enriched(["aircAruvnKk"]))
        thin = make_candidate("aircAruvnKk", duration=None, view_count=None)
        thin.metadata.pop("duration", None)
        thin.metadata.pop("view_count", None)
        out = discover.enrich({"aircAruvnKk": thin}, rss_config, discover.DiscoveryReport())
        assert out["aircAruvnKk"].duration == 900
        assert out["aircAruvnKk"].view_count == 10_000

    def test_does_not_refetch_candidates_that_are_already_complete(
        self, search_config, fake_ytdlp, make_candidate
    ):
        # Search results arrive with duration and views; paying for a second
        # round trip per video would multiply the day's request count.
        fake_ytdlp.handler = ytdlp_router(metadata_payload="")
        complete = make_candidate("searchHit01", duration=900, view_count=1000)
        discover.enrich(
            {"searchHit01": complete}, search_config, discover.DiscoveryReport()
        )
        assert fake_ytdlp.calls == []

    def test_batches_the_lookup(self, rss_config, fake_ytdlp, make_candidate):
        ids = [f"vid{i:08d}" for i in range(5)]
        fake_ytdlp.handler = ytdlp_router(metadata_payload=enriched(ids))
        thin = {}
        for vid in ids:
            c = make_candidate(vid)
            c.metadata.pop("duration")
            thin[vid] = c
        discover.enrich(thin, rss_config, discover.DiscoveryReport())
        assert len(fake_ytdlp.calls) == 1

    def test_drops_candidates_yt_dlp_could_not_resolve(
        self, rss_config, fake_ytdlp, make_candidate
    ):
        fake_ytdlp.handler = ytdlp_router(metadata_payload=enriched(["aircAruvnKk"]))
        thin = {}
        for vid in ("aircAruvnKk", "deletedVid1"):
            c = make_candidate(vid)
            c.metadata.pop("duration")
            thin[vid] = c
        report = discover.DiscoveryReport()
        out = discover.enrich(thin, rss_config, report)
        assert set(out) == {"aircAruvnKk"}
        assert report.rejected

    def test_preserves_the_original_source(self, rss_config, fake_ytdlp, make_candidate):
        fake_ytdlp.handler = ytdlp_router(metadata_payload=enriched(["aircAruvnKk"]))
        thin = make_candidate("aircAruvnKk", source="rss")
        thin.metadata.pop("duration")
        out = discover.enrich({"aircAruvnKk": thin}, rss_config, discover.DiscoveryReport())
        assert out["aircAruvnKk"].source == "rss"

    def test_handles_an_empty_input(self, rss_config, fake_ytdlp):
        assert discover.enrich({}, rss_config, discover.DiscoveryReport()) == {}
        assert fake_ytdlp.calls == []


class TestApplyPrefilter:
    def test_keeps_passing_candidates(self, config, make_candidate):
        candidates = {"aaaaaaaaaa1": make_candidate("aaaaaaaaaa1", duration=900)}
        out = discover.apply_prefilter(candidates, config, discover.DiscoveryReport())
        assert set(out) == {"aaaaaaaaaa1"}

    def test_drops_shorts(self, config, make_candidate):
        candidates = {"short000001": make_candidate("short000001", duration=30)}
        report = discover.DiscoveryReport()
        out = discover.apply_prefilter(candidates, config, report)
        assert out == {}
        assert report.rejected[prefilter.TOO_SHORT] == 1

    def test_counts_rejections_by_reason(self, config, make_candidate):
        candidates = {
            "short000001": make_candidate("short000001", duration=30),
            "short000002": make_candidate("short000002", duration=60),
            "long00000001": make_candidate("long00000001", duration=99_999),
            "quiet0000001": make_candidate("quiet0000001", view_count=3),
        }
        report = discover.DiscoveryReport()
        discover.apply_prefilter(candidates, config, report)
        assert report.rejected[prefilter.TOO_SHORT] == 2
        assert report.rejected[prefilter.TOO_LONG] == 1
        assert report.rejected[prefilter.TOO_FEW_VIEWS] == 1
        assert report.rejected_total == 4

    def test_drops_livestreams(self, config, make_candidate):
        candidates = {"live00000001": make_candidate("live00000001", is_live=True)}
        report = discover.DiscoveryReport()
        assert discover.apply_prefilter(candidates, config, report) == {}
        assert report.rejected[prefilter.IS_LIVE] == 1


class TestRun:
    @pytest.fixture
    def wired(self, rss_config, fake_http, fake_ytdlp, fixture_text):
        fake_http.route("feeds/videos.xml", body=fixture_text("channel-feed.xml"))
        fake_ytdlp.handler = ytdlp_router(
            metadata_payload=enriched(
                ["aircAruvnKk", "WUvTyaaNkzM", "noStatsVid1"],
                WUvTyaaNkzM={"view_count": 310},
            )
        )
        return rss_config

    def test_writes_surviving_candidates_to_the_ledger(self, wired, ledger):
        report = discover.run(wired, ledger=ledger)
        assert report.inserted == ledger.count()
        assert ledger.count() > 0

    def test_applies_the_prefilter_before_writing(self, wired, ledger):
        # WUvTyaaNkzM has 310 views in the fixture, under the 500 floor.
        discover.run(wired, ledger=ledger)
        assert ledger.get("WUvTyaaNkzM") is None
        assert ledger.get("aircAruvnKk") is not None

    def test_stores_the_enriched_metadata(self, wired, ledger):
        discover.run(wired, ledger=ledger)
        row = ledger.get("aircAruvnKk")
        assert row.duration == 900
        assert row.view_count == 10_000

    def test_sets_seen_at(self, wired, ledger):
        discover.run(wired, ledger=ledger)
        assert ledger.get("aircAruvnKk").seen_at

    def test_downloads_nothing(self, wired, ledger, fake_ytdlp):
        # Discovery must never fetch a video; that is a separate stage with a
        # separate failure mode and a separate timeout.
        discover.run(wired, ledger=ledger)
        assert not fake_ytdlp.invoked_with("-o")
        assert not fake_ytdlp.invoked_with("--merge-output-format")

    def test_enrichment_fetches_json_not_media(self, wired, ledger, fake_ytdlp):
        # The answer to "how many extra downloads is this?" -- none. One
        # metadata request per new RSS video, no media at all.
        discover.run(wired, ledger=ledger)
        assert fake_ytdlp.calls
        for call in fake_ytdlp.calls:
            assert "-J" in call or "-j" in call
            assert "-f" not in call

    def test_selects_nothing(self, wired, ledger):
        discover.run(wired, ledger=ledger)
        assert ledger.count(selected=1) == 0

    def test_scores_nothing(self, wired, ledger):
        discover.run(wired, ledger=ledger)
        assert all(c.score is None for c in ledger.all())

    def test_running_twice_inserts_nothing_new(self, wired, ledger):
        discover.run(wired, ledger=ledger)
        first = ledger.count()
        second = discover.run(wired, ledger=ledger)
        assert ledger.count() == first
        assert second.inserted == 0

    def test_uses_no_authentication(self, wired, ledger, fake_ytdlp, fake_http):
        discover.run(wired, ledger=ledger)
        for call in fake_ytdlp.calls:
            ytdlp.assert_no_auth_flags(call)
        for _, _, kwargs in fake_http.calls:
            headers = kwargs.get("headers") or {}
            assert not any(k.lower() in ("cookie", "authorization") for k in headers)

    def test_dry_run_writes_nothing(self, wired, ledger):
        report = discover.run(wired, ledger=ledger, dry_run=True)
        assert ledger.count() == 0
        assert report.inserted == 0

    def test_opens_its_own_ledger_when_not_given_one(self, wired):
        discover.run(wired)
        assert wired.paths.ledger.exists()

    def test_a_total_source_failure_reports_not_ok(self, rss_config, fake_http, ledger):
        fake_http.route("feeds/videos.xml", status=500, body="boom")
        report = discover.run(rss_config, ledger=ledger)
        assert not report.ok
        assert report.inserted == 0

    def test_both_strategies_merge_into_one_pool(
        self, config, fake_http, fake_ytdlp, fixture_text, ledger
    ):
        cfg = config.with_(
            discovery=DiscoveryConfig(
                strategy="both",
                channel_ids=("UCYO_jab_esuFRV4b17AJtAw",),
                keywords=("computerphile",),
            )
        )
        fake_http.route("feeds/videos.xml", body=fixture_text("channel-feed.xml"))
        fake_ytdlp.handler = ytdlp_router(
            search_payload=fixture_text("search-flat.json"),
            metadata_payload=enriched(["aircAruvnKk", "noStatsVid1"]),
        )
        discover.run(cfg, ledger=ledger)
        sources = {c.source for c in ledger.all()}
        assert "rss" in sources
        assert "search" in sources

    def test_a_video_in_both_sources_is_stored_once(
        self, config, fake_http, fake_ytdlp, fixture_text, ledger
    ):
        cfg = config.with_(
            discovery=DiscoveryConfig(
                strategy="both",
                channel_ids=("UCYO_jab_esuFRV4b17AJtAw",),
                keywords=("neural",),
            )
        )
        fake_http.route("feeds/videos.xml", body=fixture_text("channel-feed.xml"))
        search_payload = json.dumps(
            {
                "_type": "playlist",
                "entries": [
                    {
                        "id": "aircAruvnKk",
                        "title": "T",
                        "channel": "3Blue1Brown",
                        "duration": 900,
                        "view_count": 10_000,
                        "upload_date": "20260914",
                    }
                ],
            }
        )
        fake_ytdlp.handler = ytdlp_router(
            search_payload=search_payload,
            metadata_payload=enriched(["aircAruvnKk", "noStatsVid1"]),
        )
        discover.run(cfg, ledger=ledger)
        assert ledger.count(video_id="aircAruvnKk") == 1
