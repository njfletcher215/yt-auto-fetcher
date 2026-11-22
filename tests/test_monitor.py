"""Phase 14: the monitoring table.

Each row of the guide's failure table is a pure function over already-gathered
inputs, so the decision logic is testable without a Jellyfin server, a RAID
array or a genuinely broken YouTube.

The false-alarm cases matter as much as the detections. An alert that fires
every morning in week one gets muted, and then the real one is missed too.
"""

from __future__ import annotations

import pytest

from ytaf import monitor
from ytaf.config import JellyfinConfig, NotifyConfig, RetentionConfig
from ytaf.download import DownloadOutcome, DownloadReport
from ytaf.jellyfin import PlaybackItem
from ytaf.ledger import SelectionStats
from ytaf.notify import NullNotifier


def report(*, ok=0, failed=0):
    outcomes = [DownloadOutcome(video_id=f"ok{i}", ok=True) for i in range(ok)]
    outcomes += [
        DownloadOutcome(video_id=f"bad{i}", ok=False, error="ERROR") for i in range(failed)
    ]
    return DownloadReport(outcomes=outcomes)


class TestCheckExtraction:
    def test_passes_when_everything_downloaded(self):
        assert monitor.check_extraction(report(ok=10)).ok

    def test_passes_on_partial_failure(self):
        # One geo-blocked video is normal and must not page anyone.
        assert monitor.check_extraction(report(ok=7, failed=3)).ok

    def test_fails_when_every_attempt_failed(self):
        result = monitor.check_extraction(report(failed=10))
        assert not result.ok
        assert result.alert is not None

    def test_fails_on_a_single_total_failure(self):
        assert not monitor.check_extraction(report(failed=1)).ok

    def test_an_empty_run_is_not_a_failure(self):
        # Nothing was selected. That is the discovery check's problem.
        assert monitor.check_extraction(report()).ok

    def test_the_alert_names_the_check(self):
        result = monitor.check_extraction(report(failed=5))
        assert result.alert.check == monitor.EXTRACTION

    def test_the_alert_includes_the_ytdlp_version(self):
        result = monitor.check_extraction(report(failed=5), ytdlp_version="2026.08.19")
        assert "2026.08.19" in result.alert.message

    def test_the_alert_is_high_priority(self):
        # Extraction breaking stops the whole pipeline until it is fixed.
        result = monitor.check_extraction(report(failed=5))
        assert result.alert.priority >= monitor.notify.PRIORITY_HIGH


class TestCheckTranscoding:
    def test_passes_with_no_sessions(self):
        assert monitor.check_transcoding([]).ok

    def test_passes_when_everything_direct_plays(self):
        items = [PlaybackItem(item_id="a", play_method="DirectPlay")]
        assert monitor.check_transcoding(items).ok

    def test_direct_stream_is_acceptable(self):
        # Remuxing is cheap; it is re-encoding that burns the Pi.
        items = [PlaybackItem(item_id="a", play_method="DirectStream")]
        assert monitor.check_transcoding(items).ok

    def test_fails_on_any_transcode(self):
        items = [
            PlaybackItem(item_id="a", play_method="DirectPlay"),
            PlaybackItem(item_id="b", play_method="Transcode"),
        ]
        result = monitor.check_transcoding(items)
        assert not result.ok
        assert result.alert.check == monitor.TRANSCODING

    def test_the_alert_names_the_offending_item(self):
        items = [PlaybackItem(item_id="b", name="The Bad Episode", play_method="Transcode")]
        assert "The Bad Episode" in monitor.check_transcoding(items).alert.message


class TestCheckModelDegradation:
    def test_passes_when_the_model_leads_comfortably(self):
        stats = SelectionStats(
            model_n=20, model_mean_watch_pct=0.80, random_n=10, random_mean_watch_pct=0.30
        )
        assert monitor.check_model_degradation(stats).ok

    def test_fails_when_the_gap_closes(self):
        stats = SelectionStats(
            model_n=20, model_mean_watch_pct=0.42, random_n=10, random_mean_watch_pct=0.40
        )
        result = monitor.check_model_degradation(stats)
        assert not result.ok
        assert result.alert.check == monitor.MODEL_DEGRADING

    def test_fails_when_random_beats_the_model(self):
        stats = SelectionStats(
            model_n=20, model_mean_watch_pct=0.20, random_n=10, random_mean_watch_pct=0.60
        )
        assert not monitor.check_model_degradation(stats).ok

    def test_the_threshold_is_configurable(self):
        stats = SelectionStats(
            model_n=20, model_mean_watch_pct=0.45, random_n=10, random_mean_watch_pct=0.40
        )
        assert monitor.check_model_degradation(stats, min_advantage=0.01).ok
        assert not monitor.check_model_degradation(stats, min_advantage=0.20).ok

    def test_too_few_samples_is_not_an_alert(self):
        # Week one has three random picks total; comparing means there is
        # noise, and alerting on it trains you to ignore the alert.
        stats = SelectionStats(
            model_n=2, model_mean_watch_pct=0.10, random_n=1, random_mean_watch_pct=0.90
        )
        result = monitor.check_model_degradation(stats, min_samples=5)
        assert result.ok
        assert result.detail

    def test_a_missing_arm_is_not_an_alert(self):
        stats = SelectionStats(
            model_n=20, model_mean_watch_pct=0.8, random_n=0, random_mean_watch_pct=None
        )
        assert monitor.check_model_degradation(stats).ok

    def test_no_data_at_all_is_not_an_alert(self):
        stats = SelectionStats(0, None, 0, None)
        assert monitor.check_model_degradation(stats).ok

    def test_the_detail_reports_both_means(self):
        stats = SelectionStats(
            model_n=20, model_mean_watch_pct=0.80, random_n=10, random_mean_watch_pct=0.30
        )
        detail = monitor.check_model_degradation(stats).detail
        assert "0.8" in detail and "0.3" in detail


class TestCheckDisk:
    def test_passes_with_room_to_spare(self):
        assert monitor.check_disk(500.0, min_free_gb=20).ok

    def test_fails_below_the_threshold(self):
        result = monitor.check_disk(5.0, min_free_gb=20)
        assert not result.ok
        assert result.alert.check == monitor.DISK_FILLING

    def test_the_boundary_passes(self):
        assert monitor.check_disk(20.0, min_free_gb=20).ok

    def test_the_alert_states_the_numbers(self):
        alert = monitor.check_disk(5.0, min_free_gb=20).alert
        assert "5" in alert.message
        assert "20" in alert.message


class TestCheckDiscovery:
    def test_passes_with_new_candidates(self):
        assert monitor.check_discovery(42).ok

    def test_fails_with_none_in_the_window(self):
        result = monitor.check_discovery(0)
        assert not result.ok
        assert result.alert.check == monitor.DISCOVERY_STALLED

    def test_a_single_candidate_is_enough(self):
        assert monitor.check_discovery(1).ok

    def test_the_alert_mentions_feed_failures(self):
        # The guide's suggested next step: check whether the RSS feeds are
        # still reachable.
        alert = monitor.check_discovery(0, channel_failures=4).alert
        assert "4" in alert.message

    def test_the_window_is_configurable(self):
        assert "48" in monitor.check_discovery(0, hours=48).alert.message


class TestRun:
    @pytest.fixture
    def monitored(self, config):
        return config.with_(
            notify=NotifyConfig(enabled=True, ntfy_url="https://ntfy.test", topic="t")
        )

    def test_runs_the_checks_it_can(self, monitored, ledger, fake_http):
        fake_http.route("/t", status=200, json_body={})
        result = monitor.run(monitored, ledger=ledger, notifier=NullNotifier())
        assert result.results

    def test_skips_jellyfin_checks_when_unconfigured(self, monitored, ledger, fake_http):
        notifier = NullNotifier()
        result = monitor.run(monitored, ledger=ledger, notifier=notifier)
        names = {r.name for r in result.results}
        assert monitor.TRANSCODING not in names
        assert not fake_http.called("/Sessions")

    def test_includes_the_transcoding_check_when_jellyfin_is_up(
        self, config, ledger, fake_http, fixture_json
    ):
        cfg = config.with_(
            jellyfin=JellyfinConfig(
                base_url="https://jellyfin.test", api_key="K", user_id="u"
            )
        )
        fake_http.route("/Sessions", json_body=fixture_json("jellyfin-sessions.json"))
        result = monitor.run(cfg, ledger=ledger, notifier=NullNotifier())
        names = {r.name for r in result.results}
        assert monitor.TRANSCODING in names

    def test_skips_the_extraction_check_without_a_download_report(self, monitored, ledger):
        result = monitor.run(monitored, ledger=ledger, notifier=NullNotifier())
        assert monitor.EXTRACTION not in {r.name for r in result.results}

    def test_includes_the_extraction_check_when_given_one(self, monitored, ledger):
        result = monitor.run(
            monitored, ledger=ledger, notifier=NullNotifier(), download_report=report(failed=5)
        )
        extraction = next(r for r in result.results if r.name == monitor.EXTRACTION)
        assert not extraction.ok

    def test_sends_one_alert_per_failing_check(self, monitored, ledger):
        notifier = NullNotifier()
        result = monitor.run(
            monitored, ledger=ledger, notifier=notifier, download_report=report(failed=5)
        )
        assert len(notifier.sent) == len(result.failures)
        assert {a.check for a in notifier.sent} == {r.name for r in result.failures}
        assert any(a.check == monitor.EXTRACTION for a in notifier.sent)

    def test_sends_nothing_when_everything_passes(self, monitored, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        notifier = NullNotifier()
        monitor.run(
            monitored, ledger=ledger, notifier=notifier, download_report=report(ok=10)
        )
        assert notifier.sent == []

    def test_reports_ok_when_nothing_failed(self, monitored, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        result = monitor.run(
            monitored, ledger=ledger, notifier=NullNotifier(), download_report=report(ok=10)
        )
        assert result.ok

    def test_a_failing_notifier_does_not_fail_the_run(self, monitored, ledger, monkeypatch):
        from ytaf import http as http_mod
        from ytaf.errors import YtafError

        monkeypatch.setattr(
            http_mod, "post", lambda *a, **k: (_ for _ in ()).throw(YtafError("down"))
        )
        result = monitor.run(monitored, ledger=ledger, download_report=report(failed=5))
        assert not result.ok

    def test_detects_a_full_disk(self, monitored, ledger, monkeypatch, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        cfg = monitored.with_(retention=RetentionConfig(min_free_gb=10_000_000))
        result = monitor.run(cfg, ledger=ledger, notifier=NullNotifier())
        disk = next(r for r in result.results if r.name == monitor.DISK_FILLING)
        assert not disk.ok

    def test_detects_a_stalled_discovery(self, monitored, ledger):
        result = monitor.run(monitored, ledger=ledger, notifier=NullNotifier())
        stalled = next(r for r in result.results if r.name == monitor.DISCOVERY_STALLED)
        assert not stalled.ok
