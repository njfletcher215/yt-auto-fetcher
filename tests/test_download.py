"""The download stage.

Partial failure is the normal case here -- a geo-blocked video should cost you
that one video, not the other nine. Total failure means something else:
YouTube broke the extractor, which gets the auto-update-and-retry-once
treatment rather than a plain alert.

There is no download archive, so the tests also pin that the ledger alone
prevents re-fetching anything we currently hold -- including saved videos,
whose files sit outside the managed library.
"""

from __future__ import annotations

import json

import pytest

from ytaf import download, ytdlp
from ytaf.config import JellyfinConfig
from ytaf.errors import YtdlpError


def writing_ytdlp(config, *, fail_ids=(), info_overrides=None):
    """A fake yt-dlp that actually writes the files a real run would."""

    def handler(args, **kwargs):
        video_id = args[-1]
        if video_id in fail_ids:
            raise YtdlpError(
                f"failed {video_id}",
                video_id=video_id,
                returncode=1,
                stderr="ERROR: Video unavailable",
            )
        info = {
            "id": video_id,
            "title": f"Title {video_id}",
            "description": "A description.",
            "channel": "Test Channel",
            "channel_id": "UCtest",
            "upload_date": "20260914",
            "duration": 900,
            "view_count": 10_000,
            "ext": "mp4",
            "vcodec": "avc1.640028",
            "acodec": "mp4a.40.2",
            "height": 1080,
        }
        info.update((info_overrides or {}).get(video_id, {}))
        channel_dir = config.paths.library / info["channel"]
        channel_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{info['upload_date']} - {info['title']} [{video_id}]"
        (channel_dir / f"{stem}.mp4").write_bytes(b"\0" * 2048)
        (channel_dir / f"{stem}.info.json").write_text(json.dumps(info), encoding="utf-8")
        (channel_dir / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff")
        return ytdlp.YtdlpResult(args=tuple(args), returncode=0, stdout="", stderr="")

    return handler


@pytest.fixture
def pending(ledger, make_candidate):
    for i in range(3):
        ledger.upsert(
            make_candidate(
                f"vid{i:08d}",
                selected=1,
                selection="model" if i < 2 else "random",
                channel="Test Channel",
                upload_date="20260914",
            )
        )
    return ledger


class TestDownloadOne:
    def test_invokes_yt_dlp_with_the_spec_arguments(self, config, fake_ytdlp, make_candidate):
        fake_ytdlp.handler = writing_ytdlp(config)
        download.download_one(make_candidate("vid00000000"), config)
        expected = ytdlp.build_download_args("vid00000000", config)
        assert fake_ytdlp.last_args == expected

    def test_uses_no_cookies(self, config, fake_ytdlp, make_candidate):
        fake_ytdlp.handler = writing_ytdlp(config)
        download.download_one(make_candidate("vid00000000"), config)
        ytdlp.assert_no_auth_flags(fake_ytdlp.last_args)

    def test_reports_the_written_path(self, config, fake_ytdlp, make_candidate):
        fake_ytdlp.handler = writing_ytdlp(config)
        outcome = download.download_one(make_candidate("vid00000000"), config)
        assert outcome.ok
        assert outcome.path.exists()
        assert outcome.path.is_relative_to(config.paths.library)

    def test_the_written_path_carries_the_video_id(self, config, fake_ytdlp, make_candidate):
        # This is how the feedback stage finds its way back to the row.
        from ytaf import naming

        fake_ytdlp.handler = writing_ytdlp(config)
        outcome = download.download_one(make_candidate("vid00000000"), config)
        assert naming.video_id_from_path(outcome.path) == "vid00000000"

    def test_counts_the_offer(self, config, ledger, fake_ytdlp, make_candidate):
        fake_ytdlp.handler = writing_ytdlp(config)
        ledger.upsert(make_candidate("vid00000000", selected=1, selection="model"))
        download.run(config, ledger=ledger)
        row = ledger.get("vid00000000")
        assert row.offer_count == 1
        assert row.last_offered_at

    def test_a_failure_is_reported_not_raised(self, config, fake_ytdlp, make_candidate):
        fake_ytdlp.handler = writing_ytdlp(config, fail_ids={"vid00000000"})
        outcome = download.download_one(make_candidate("vid00000000"), config)
        assert not outcome.ok
        assert outcome.error
        assert outcome.path is None

    def test_a_missing_output_file_is_a_failure(self, config, fake_ytdlp, make_candidate):
        # yt-dlp exiting 0 without writing anything happens when the archive
        # already contains the id; the ledger must not record a phantom file.
        fake_ytdlp.returncode = 0
        outcome = download.download_one(make_candidate("vid00000000"), config)
        assert not outcome.ok

    def test_dry_run_writes_nothing(self, config, fake_ytdlp, make_candidate):
        fake_ytdlp.handler = writing_ytdlp(config)
        outcome = download.download_one(make_candidate("vid00000000"), config, dry_run=True)
        assert fake_ytdlp.calls == []
        assert outcome.skipped


class TestRun:
    def test_downloads_every_pending_selection(self, config, pending, fake_ytdlp):
        fake_ytdlp.handler = writing_ytdlp(config)
        report = download.run(config, ledger=pending)
        assert report.attempted == 3
        assert report.succeeded == 3

    def test_marks_rows_downloaded(self, config, pending, fake_ytdlp):
        fake_ytdlp.handler = writing_ytdlp(config)
        download.run(config, ledger=pending)
        assert pending.count(downloaded=1) == 3

    def test_records_the_filepath(self, config, pending, fake_ytdlp):
        fake_ytdlp.handler = writing_ytdlp(config)
        download.run(config, ledger=pending)
        for row in pending.downloaded():
            assert row.filepath
            assert row.filepath.startswith(str(config.paths.library))

    def test_skips_rows_that_are_not_selected(self, config, pending, fake_ytdlp, make_candidate):
        pending.upsert(make_candidate("notpicked001"))
        fake_ytdlp.handler = writing_ytdlp(config)
        download.run(config, ledger=pending)
        assert pending.get("notpicked001").downloaded == 0

    def test_skips_rows_we_currently_hold(self, config, pending, fake_ytdlp, make_candidate):
        pending.upsert(
            make_candidate(
                "alreadygot01", selected=1, selection="model", downloaded=1, filepath="/x.mp4"
            )
        )
        fake_ytdlp.handler = writing_ytdlp(config)
        assert download.run(config, ledger=pending).attempted == 3

    def test_skips_saved_rows(self, config, pending, fake_ytdlp, make_candidate):
        # With no download archive, the ledger is the only thing standing
        # between us and re-fetching something already saved.
        pending.upsert(
            make_candidate(
                "savedalready", selected=1, selection="model", saved=1, filepath="/perm/x.mp4"
            )
        )
        fake_ytdlp.handler = writing_ytdlp(config)
        report = download.run(config, ledger=pending)
        assert report.attempted == 3
        assert "savedalready" not in [c[-1] for c in fake_ytdlp.calls]

    def test_re_downloads_a_reaped_row(self, config, ledger, fake_ytdlp, make_candidate):
        # The repeat arm depends on this working; a download archive would
        # have blocked it.
        ledger.upsert(
            make_candidate(
                "reoffered001",
                selected=1,
                selection="repeat",
                downloaded=1,
                filepath=None,
                offer_count=1,
            )
        )
        fake_ytdlp.handler = writing_ytdlp(config)
        report = download.run(config, ledger=ledger)
        assert report.succeeded == 1
        assert ledger.get("reoffered001").offer_count == 2

    def test_one_failure_does_not_stop_the_batch(self, config, pending, fake_ytdlp):
        fake_ytdlp.handler = writing_ytdlp(config, fail_ids={"vid00000001"})
        report = download.run(config, ledger=pending)
        assert report.succeeded == 2
        assert report.failed == 1
        assert pending.get("vid00000001").downloaded == 0

    def test_a_failed_row_stays_selected_for_a_retry_tomorrow(
        self, config, pending, fake_ytdlp
    ):
        fake_ytdlp.handler = writing_ytdlp(config, fail_ids={"vid00000001"})
        download.run(config, ledger=pending)
        assert pending.get("vid00000001").selected == 1

    def test_total_failure_is_flagged(self, config, pending, fake_ytdlp):
        fake_ytdlp.handler = writing_ytdlp(
            config, fail_ids={"vid00000000", "vid00000001", "vid00000002"}
        )
        report = download.run(config, ledger=pending, retry_after_update=False)
        assert report.total_failure

    def test_an_empty_queue_is_not_a_total_failure(self, config, ledger, fake_ytdlp):
        report = download.run(config, ledger=ledger)
        assert not report.total_failure
        assert report.attempted == 0

    def test_dry_run_changes_nothing(self, config, pending, fake_ytdlp):
        fake_ytdlp.handler = writing_ytdlp(config)
        download.run(config, ledger=pending, dry_run=True)
        assert pending.count(downloaded=1) == 0
        assert fake_ytdlp.calls == []

    def test_writes_inside_the_library_root_only(self, config, pending, fake_ytdlp):
        fake_ytdlp.handler = writing_ytdlp(config)
        download.run(config, ledger=pending)
        for row in pending.downloaded():
            assert row.filepath.startswith(str(config.paths.library))


class TestAutoUpdateRetry:
    def test_updates_and_retries_once_on_total_failure(self, config, pending, monkeypatch, fake_ytdlp):
        # Auto-update, retry once, then alert. A stale binary is the most
        # common failure mode in this whole system.
        updates: list[int] = []
        monkeypatch.setattr(ytdlp, "update", lambda: updates.append(1))
        attempts = {"n": 0}

        def handler(args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] <= 3:
                raise YtdlpError("broken", video_id=args[-1], returncode=1, stderr="ERROR")
            return writing_ytdlp(config)(args, **kwargs)

        fake_ytdlp.handler = handler
        report = download.run(config, ledger=pending, retry_after_update=True)
        assert updates == [1]
        assert report.succeeded == 3

    def test_does_not_update_on_partial_failure(self, config, pending, monkeypatch, fake_ytdlp):
        updates: list[int] = []
        monkeypatch.setattr(ytdlp, "update", lambda: updates.append(1))
        fake_ytdlp.handler = writing_ytdlp(config, fail_ids={"vid00000001"})
        download.run(config, ledger=pending, retry_after_update=True)
        assert updates == []

    def test_retries_at_most_once(self, config, pending, monkeypatch, fake_ytdlp):
        updates: list[int] = []
        monkeypatch.setattr(ytdlp, "update", lambda: updates.append(1))
        fake_ytdlp.handler = writing_ytdlp(
            config, fail_ids={"vid00000000", "vid00000001", "vid00000002"}
        )
        report = download.run(config, ledger=pending, retry_after_update=True)
        assert len(updates) == 1
        assert report.total_failure

    def test_can_be_disabled(self, config, pending, monkeypatch, fake_ytdlp):
        updates: list[int] = []
        monkeypatch.setattr(ytdlp, "update", lambda: updates.append(1))
        fake_ytdlp.handler = writing_ytdlp(
            config, fail_ids={"vid00000000", "vid00000001", "vid00000002"}
        )
        download.run(config, ledger=pending, retry_after_update=False)
        assert updates == []


class TestJellyfinScan:
    @pytest.fixture
    def jellyfin_cfg(self, config):
        return config.with_(
            jellyfin=JellyfinConfig(base_url="https://jellyfin.test", api_key="K")
        )

    def test_triggers_one_scan_after_the_batch(
        self, jellyfin_cfg, pending, fake_ytdlp, fake_http
    ):
        fake_ytdlp.handler = writing_ytdlp(jellyfin_cfg)
        fake_http.route("/Library/Refresh", status=204)
        report = download.run(jellyfin_cfg, ledger=pending)
        assert report.scan_triggered
        assert fake_http.call_count("/Library/Refresh") == 1

    def test_does_not_scan_when_nothing_downloaded(
        self, jellyfin_cfg, ledger, fake_ytdlp, fake_http
    ):
        report = download.run(jellyfin_cfg, ledger=ledger)
        assert not report.scan_triggered
        assert fake_http.calls == []

    def test_works_without_jellyfin_configured(self, config, pending, fake_ytdlp, fake_http):
        # Download must not depend on Jellyfin being up or even existing.
        fake_ytdlp.handler = writing_ytdlp(config)
        report = download.run(config, ledger=pending)
        assert report.succeeded == 3
        assert not report.scan_triggered
        assert fake_http.calls == []

    def test_a_failing_scan_does_not_fail_the_downloads(
        self, jellyfin_cfg, pending, fake_ytdlp, fake_http
    ):
        fake_ytdlp.handler = writing_ytdlp(jellyfin_cfg)
        fake_http.route("/Library/Refresh", status=500, body="boom")
        report = download.run(jellyfin_cfg, ledger=pending)
        assert report.succeeded == 3
        assert not report.scan_triggered
        assert report.notes
