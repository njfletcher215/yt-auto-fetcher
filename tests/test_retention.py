"""Retention: 14 days, no exceptions, plus `ytaf save`.

The policy is deliberately blunt -- age is the only criterion. Likes are not
spared, unwatched videos are not spared. That makes the rules easy to reason
about, and it puts the entire "keep this" decision in one explicit action.

So the tests here are mostly about what retention must *not* do: reach outside
the managed library, touch a saved video, or lose the ledger row and the
signal attached to it.
"""

from __future__ import annotations

import datetime as dt
import os
import time

import pytest

from ytaf import retention, spec
from ytaf.config import JellyfinConfig, Paths, RetentionConfig
from ytaf.errors import ConfigError, UnsafePathError

NOW = dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc)


def days_ago(n: int) -> str:
    return (NOW - dt.timedelta(days=n)).isoformat()


@pytest.fixture
def saving_config(config, permanent_library):
    """Config with a permanent library wired up."""
    return config.with_(
        paths=Paths.under(config.paths.root, permanent_library=permanent_library)
    )


@pytest.fixture
def library(config, ledger, make_candidate, library_video):
    """One row per outcome under the flat 14-day policy."""
    cases = {
        "oldwatched01": dict(watched=1, watch_pct=0.95, seen_at=days_ago(45)),
        "oldunwatchd1": dict(seen_at=days_ago(45)),
        "oldliked0001": dict(
            watched=1, watch_pct=0.99, seen_at=days_ago(45), label=spec.LABEL_LIKE
        ),
        "oldpartial01": dict(watched=1, watch_pct=0.35, seen_at=days_ago(45)),
        "recent000001": dict(watched=1, watch_pct=0.95, seen_at=days_ago(5)),
    }
    for video_id, attrs in cases.items():
        video = library_video(video_id=video_id, title=f"Title {video_id}")
        ledger.upsert(
            make_candidate(
                video_id,
                downloaded=1,
                filepath=str(video),
                selected=1,
                selection="model",
                offer_count=1,
                last_offered_at=attrs.get("seen_at"),
                **attrs,
            )
        )
    return ledger


class TestAgeDays:
    def test_measures_from_seen_at(self, make_candidate):
        assert retention.age_days(make_candidate(seen_at=days_ago(10)), now=NOW) == (
            pytest.approx(10, abs=0.01)
        )

    def test_a_future_timestamp_is_zero_not_negative(self, make_candidate):
        future = (NOW + dt.timedelta(days=5)).isoformat()
        assert retention.age_days(make_candidate(seen_at=future), now=NOW) >= 0

    def test_falls_back_to_the_upload_date(self, make_candidate):
        assert retention.age_days(
            make_candidate(seen_at="", upload_date="20260101"), now=NOW
        ) > 200

    def test_none_when_nothing_is_parseable(self, make_candidate):
        c = make_candidate(seen_at="garbage")
        c.metadata.pop("upload_date", None)
        assert retention.age_days(c, now=NOW) is None


class TestIsReapable:
    def test_age_is_the_only_criterion(self, config, make_candidate):
        c = make_candidate(downloaded=1, filepath="/x.mp4", seen_at=days_ago(45))
        assert retention.is_reapable(c, config, now=NOW)[0]

    @pytest.mark.parametrize(
        "overrides,label",
        [
            (dict(label=spec.LABEL_LIKE), "liked"),
            (dict(watched=None, watch_pct=None), "never watched"),
            (dict(watched=1, watch_pct=0.1), "barely watched"),
            (dict(watched=1, watch_pct=1.0), "fully watched"),
            (dict(play_count=12), "rewatched"),
        ],
    )
    def test_nothing_else_grants_a_reprieve(self, config, make_candidate, overrides, label):
        # A like is a taste signal, not a keep-forever instruction. `ytaf
        # save` is the only way to keep something.
        attrs = dict(downloaded=1, filepath="/x.mp4", seen_at=days_ago(45))
        attrs.update(overrides)
        assert retention.is_reapable(make_candidate(**attrs), config, now=NOW)[0], label

    @pytest.mark.parametrize(
        "overrides,reason",
        [
            (dict(seen_at=days_ago(5)), retention.TOO_YOUNG),
            (dict(downloaded=0), retention.NOT_DOWNLOADED),
            (dict(filepath=None), retention.ALREADY_REAPED),
            (dict(saved=1), retention.SAVED),
        ],
    )
    def test_keeps_and_explains(self, config, make_candidate, overrides, reason):
        attrs = dict(downloaded=1, filepath="/x.mp4", seen_at=days_ago(45))
        attrs.update(overrides)
        reapable, got = retention.is_reapable(make_candidate(**attrs), config, now=NOW)
        assert not reapable
        assert got == reason

    def test_the_age_bound_is_exclusive(self, config, make_candidate):
        at = make_candidate(downloaded=1, filepath="/x.mp4", seen_at=days_ago(14))
        over = make_candidate(downloaded=1, filepath="/y.mp4", seen_at=days_ago(15))
        assert not retention.is_reapable(at, config, now=NOW)[0]
        assert retention.is_reapable(over, config, now=NOW)[0]

    def test_the_window_is_configurable(self, config, make_candidate):
        cfg = config.with_(retention=RetentionConfig(max_age_days=60))
        c = make_candidate(downloaded=1, filepath="/x.mp4", seen_at=days_ago(45))
        assert not retention.is_reapable(c, cfg, now=NOW)[0]

    def test_default_window_is_fourteen_days(self):
        assert RetentionConfig().max_age_days == 14


class TestPlan:
    def test_reaps_everything_old(self, config, library):
        reapables, _ = retention.plan(library, config, now=NOW)
        assert {r.video_id for r in reapables} == {
            "oldwatched01",
            "oldunwatchd1",
            "oldliked0001",
            "oldpartial01",
        }

    def test_keeps_the_recent_one(self, config, library):
        _, kept = retention.plan(library, config, now=NOW)
        assert kept.get(retention.TOO_YOUNG) == 1

    def test_flags_unwatched_reaps(self, config, library):
        # Retention is where "we offered it, you ignored it, it aged out"
        # becomes a recorded negative.
        reapables, _ = retention.plan(library, config, now=NOW)
        unwatched = {r.video_id for r in reapables if r.unwatched}
        assert unwatched == {"oldunwatchd1"}

    def test_includes_the_sidecars(self, config, library):
        reapables, _ = retention.plan(library, config, now=NOW)
        suffixes = {p.suffix for r in reapables for p in r.sidecars}
        assert ".jpg" in suffixes
        assert any(p.name.endswith(".info.json") for r in reapables for p in r.sidecars)

    def test_estimates_bytes_freed(self, config, library):
        reapables, _ = retention.plan(library, config, now=NOW)
        assert all(r.bytes_freed > 0 for r in reapables)

    def test_skips_a_row_whose_file_is_already_gone(self, config, library):
        from pathlib import Path

        Path(library.get("oldwatched01").filepath).unlink()
        reapables, kept = retention.plan(library, config, now=NOW)
        assert "oldwatched01" not in {r.video_id for r in reapables}
        assert kept.get(retention.MISSING_FILE) == 1

    def test_an_empty_ledger_plans_nothing(self, config, ledger):
        reapables, _ = retention.plan(ledger, config, now=NOW)
        assert reapables == []


class TestApply:
    def test_deletes_the_videos(self, config, library):
        reapables, _ = retention.plan(library, config, now=NOW)
        paths = [r.path for r in reapables]
        retention.apply(library, config, reapables)
        assert all(not p.exists() for p in paths)

    def test_deletes_the_sidecars(self, config, library):
        reapables, _ = retention.plan(library, config, now=NOW)
        sidecars = [p for r in reapables for p in r.sidecars]
        retention.apply(library, config, reapables)
        assert all(not p.exists() for p in sidecars)

    def test_clears_the_filepath(self, config, library):
        reapables, _ = retention.plan(library, config, now=NOW)
        retention.apply(library, config, reapables)
        assert library.get("oldwatched01").filepath is None

    def test_keeps_the_ledger_row(self, config, library):
        before = library.count()
        reapables, _ = retention.plan(library, config, now=NOW)
        retention.apply(library, config, reapables)
        assert library.count() == before

    def test_keeps_every_training_signal(self, config, library):
        # The bytes are disposable; the signal is not.
        reapables, _ = retention.plan(library, config, now=NOW)
        retention.apply(library, config, reapables)
        row = library.get("oldwatched01")
        assert row.watch_pct == pytest.approx(0.95)
        assert row.watched == 1
        assert row.selection == "model"
        assert library.get("oldliked0001").label == spec.LABEL_LIKE

    def test_keeps_the_offer_history_for_the_repeat_arm(self, config, library):
        reapables, _ = retention.plan(library, config, now=NOW)
        retention.apply(library, config, reapables)
        row = library.get("oldunwatchd1")
        assert row.offer_count == 1
        assert row.last_offered_at
        assert row.downloaded == 1

    def test_a_reaped_unwatched_row_becomes_repeat_eligible(self, config, library):
        reapables, _ = retention.plan(library, config, now=NOW)
        retention.apply(library, config, reapables)
        later = NOW + dt.timedelta(days=200)
        pool = {c.video_id for c in library.repeat_pool(now=later)}
        assert "oldunwatchd1" in pool

    def test_a_reaped_watched_row_does_not_become_repeat_eligible(self, config, library):
        reapables, _ = retention.plan(library, config, now=NOW)
        retention.apply(library, config, reapables)
        later = NOW + dt.timedelta(days=200)
        pool = {c.video_id for c in library.repeat_pool(now=later)}
        assert "oldwatched01" not in pool
        assert "oldliked0001" not in pool

    def test_leaves_the_recent_video_alone(self, config, library):
        from pathlib import Path

        reapables, _ = retention.plan(library, config, now=NOW)
        retention.apply(library, config, reapables)
        assert Path(library.get("recent000001").filepath).exists()

    def test_reports_bytes_freed(self, config, library):
        reapables, _ = retention.plan(library, config, now=NOW)
        assert retention.apply(library, config, reapables).bytes_freed > 0

    def test_prunes_an_emptied_channel_directory(self, config, ledger, make_candidate, library_video):
        video = library_video(video_id="lonely00001", channel="Lonely Channel")
        ledger.upsert(
            make_candidate(
                "lonely00001", downloaded=1, filepath=str(video), seen_at=days_ago(45)
            )
        )
        channel_dir = video.parent
        reapables, _ = retention.plan(ledger, config, now=NOW)
        retention.apply(ledger, config, reapables)
        assert not channel_dir.exists()

    def test_does_not_prune_a_directory_that_still_has_videos(self, config, library):
        from pathlib import Path

        parent = Path(library.get("recent000001").filepath).parent
        reapables, _ = retention.plan(library, config, now=NOW)
        retention.apply(library, config, reapables)
        assert parent.exists()

    def test_dry_run_deletes_nothing(self, config, library):
        reapables, _ = retention.plan(library, config, now=NOW)
        report = retention.apply(library, config, reapables, dry_run=True)
        assert all(r.path.exists() for r in reapables)
        assert library.get("oldwatched01").filepath is not None
        assert report.dry_run

    def test_refuses_a_path_outside_the_library(self, config, library, tmp_path):
        outside = tmp_path / "not-the-library.mp4"
        outside.write_bytes(b"precious")
        with pytest.raises(UnsafePathError):
            retention.apply(
                library, config, [retention.Reapable(video_id="x", path=outside)]
            )
        assert outside.exists()

    def test_refuses_a_traversal_path(self, config, library):
        escape = config.paths.library / ".." / ".." / "etc" / "passwd"
        with pytest.raises(UnsafePathError):
            retention.apply(
                library, config, [retention.Reapable(video_id="x", path=escape)]
            )

    def test_cannot_reach_the_permanent_library(self, saving_config, ledger, make_candidate):
        # Saved videos are out of range by construction, not by a flag check.
        target = saving_config.paths.permanent_library / "kept.mp4"
        target.write_bytes(b"kept")
        with pytest.raises(UnsafePathError):
            retention.apply(
                ledger, saving_config, [retention.Reapable(video_id="x", path=target)]
            )
        assert target.exists()

    def test_a_delete_error_is_recorded_not_raised(self, config, library, monkeypatch):
        reapables, _ = retention.plan(library, config, now=NOW)

        def boom(self, *args, **kwargs):
            raise PermissionError("read only filesystem")

        monkeypatch.setattr("pathlib.Path.unlink", boom)
        report = retention.apply(library, config, reapables)
        assert report.errors
        assert library.get("oldwatched01").filepath is not None


class TestSave:
    def test_moves_the_video_to_the_permanent_library(self, saving_config, library):
        new_path = retention.save(library, saving_config, "oldwatched01")
        assert new_path.exists()
        assert new_path.is_relative_to(saving_config.paths.permanent_library)

    def test_removes_it_from_the_managed_library(self, saving_config, library):
        from pathlib import Path

        original = Path(library.get("oldwatched01").filepath)
        retention.save(library, saving_config, "oldwatched01")
        assert not original.exists()

    def test_preserves_the_channel_subdirectory(self, saving_config, library):
        new_path = retention.save(library, saving_config, "oldwatched01")
        assert new_path.parent.name == "Test Channel"

    def test_moves_the_sidecars_too(self, saving_config, library):
        new_path = retention.save(library, saving_config, "oldwatched01")
        assert (new_path.parent / f"{new_path.stem}.info.json").exists()

    def test_marks_the_row_saved_with_the_new_path(self, saving_config, library):
        new_path = retention.save(library, saving_config, "oldwatched01")
        row = library.get("oldwatched01")
        assert row.saved == 1
        assert row.filepath == str(new_path)

    def test_a_saved_video_survives_retention(self, saving_config, library):
        retention.save(library, saving_config, "oldwatched01")
        report = retention.run(library, saving_config, now=NOW)
        assert "oldwatched01" not in {r.video_id for r in report.reaped}
        from pathlib import Path

        assert Path(library.get("oldwatched01").filepath).exists()

    def test_a_saved_video_is_never_re_downloaded(self, saving_config, library):
        retention.save(library, saving_config, "oldwatched01")
        assert "oldwatched01" in library.held_ids()
        assert "oldwatched01" not in {c.video_id for c in library.new_pool()}
        assert "oldwatched01" not in {
            c.video_id for c in library.repeat_pool(now=NOW + dt.timedelta(days=999))
        }

    def test_refuses_without_a_permanent_library_configured(self, config, library):
        assert config.paths.permanent_library is None
        with pytest.raises(ConfigError):
            retention.save(library, config, "oldwatched01")

    def test_refuses_an_unknown_video(self, saving_config, library):
        from ytaf.errors import LedgerError

        with pytest.raises(LedgerError):
            retention.save(library, saving_config, "ghost000000")

    def test_refuses_a_video_with_nothing_on_disk(self, saving_config, ledger, make_candidate):
        from ytaf.errors import LedgerError

        ledger.upsert(make_candidate("reaped000001", downloaded=1, filepath=None))
        with pytest.raises(LedgerError):
            retention.save(ledger, saving_config, "reaped000001")

    def test_dry_run_moves_nothing(self, saving_config, library):
        from pathlib import Path

        original = Path(library.get("oldwatched01").filepath)
        retention.save(library, saving_config, "oldwatched01", dry_run=True)
        assert original.exists()
        assert library.get("oldwatched01").saved == 0

    def test_saving_twice_is_not_an_error(self, saving_config, library):
        first = retention.save(library, saving_config, "oldwatched01")
        second = retention.save(library, saving_config, "oldwatched01")
        assert first == second


class TestPurgeLogs:
    def _age(self, path, days):
        stale = time.time() - days * 86400
        os.utime(path, (stale, stale))

    def test_removes_old_logs(self, config):
        old = config.paths.logs / "old.log"
        old.write_text("x", encoding="utf-8")
        self._age(old, 100)
        assert old in retention.purge_logs(config, now=NOW)
        assert not old.exists()

    def test_keeps_recent_logs(self, config):
        recent = config.paths.logs / "recent.log"
        recent.write_text("x", encoding="utf-8")
        retention.purge_logs(config, now=NOW)
        assert recent.exists()

    def test_dry_run_keeps_everything(self, config):
        old = config.paths.logs / "old.log"
        old.write_text("x", encoding="utf-8")
        self._age(old, 100)
        assert old in retention.purge_logs(config, now=NOW, dry_run=True)
        assert old.exists()

    def test_handles_a_missing_log_directory(self, config):
        import shutil

        shutil.rmtree(config.paths.logs)
        assert retention.purge_logs(config, now=NOW) == []

    def test_never_touches_the_library(self, config, library_video):
        video = library_video()
        self._age(video, 1000)
        retention.purge_logs(config, now=NOW)
        assert video.exists()


class TestFreeSpace:
    def test_reports_free_bytes(self, config):
        assert retention.free_bytes(config.paths.root) > 0

    def test_reports_free_gigabytes(self, config):
        gb = retention.free_gb(config.paths.root)
        assert gb == pytest.approx(
            retention.free_bytes(config.paths.root) / retention.BYTES_PER_GB, rel=1e-6
        )


class TestRun:
    def test_plans_and_applies(self, config, library):
        report = retention.run(config, ledger=library, now=NOW)
        assert len(report.reaped) == 4
        assert all(not r.path.exists() for r in report.reaped)

    def test_counts_unwatched_reaps(self, config, library):
        assert retention.run(config, ledger=library, now=NOW).reaped_unwatched == 1

    def test_also_purges_logs(self, config, library):
        old = config.paths.logs / "old.log"
        old.write_text("x", encoding="utf-8")
        stale = time.time() - 100 * 86400
        os.utime(old, (stale, stale))
        assert retention.run(config, ledger=library, now=NOW).logs_purged

    def test_dry_run_changes_nothing(self, config, library):
        report = retention.run(config, ledger=library, now=NOW, dry_run=True)
        assert report.reaped
        assert all(r.path.exists() for r in report.reaped)

    def test_does_not_need_jellyfin(self, config, library):
        assert JellyfinConfig().configured is False
        assert retention.run(config, ledger=library, now=NOW).reaped
