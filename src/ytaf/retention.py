"""Retention: everything in the managed library older than 14 days goes.

No exceptions for likes, and none for things never watched. A like is a taste
signal, not a keep-forever instruction, and an unwatched video that has aged
out has already told us what it was going to tell us -- its row survives as a
weak negative and it becomes eligible for a re-offer later.

The one way to keep a video is `ytaf save`, which moves it into the permanent
library. Nothing in this module can reach outside `paths.library`, so a saved
video is out of range by construction rather than by a flag check.

The ledger row always survives. Retention clears `filepath` and removes bytes.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import naming
from .config import Config
from .errors import ConfigError, LedgerError, UnsafePathError
from .ledger import Candidate, Ledger

BYTES_PER_GB = 1024**3

TOO_YOUNG = "too_young"
NOT_DOWNLOADED = "not_downloaded"
ALREADY_REAPED = "already_reaped"
SAVED = "saved"
MISSING_FILE = "missing_file"
UNDATED = "undated"


@dataclass(frozen=True)
class Reapable:
    video_id: str
    path: Path
    sidecars: tuple[Path, ...] = ()
    reason: str = ""
    bytes_freed: int = 0
    # True when the row never recorded any playback -- the weak negative the
    # re-offer policy and the training set both read.
    unwatched: bool = False


@dataclass
class RetentionReport:
    reaped: list[Reapable] = field(default_factory=list)
    kept: dict[str, int] = field(default_factory=dict)
    logs_purged: list[Path] = field(default_factory=list)
    bytes_freed: int = 0
    dry_run: bool = False
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def reaped_unwatched(self) -> int:
        return sum(1 for r in self.reaped if r.unwatched)


def age_days(candidate: Candidate, *, now: datetime | None = None) -> float | None:
    """Age from the ledger's `seen_at`, falling back to the upload date.

    `seen_at` is when the pipeline fetched it, which is what the age rule is
    about -- a 2019 upload downloaded yesterday is one day old here.
    """
    now = now or datetime.now(timezone.utc)
    reference: datetime | None = None

    if candidate.seen_at:
        try:
            reference = datetime.fromisoformat(candidate.seen_at)
        except ValueError:
            reference = None

    if reference is None and candidate.upload_date:
        try:
            reference = datetime(
                int(candidate.upload_date[0:4]),
                int(candidate.upload_date[4:6]),
                int(candidate.upload_date[6:8]),
                tzinfo=timezone.utc,
            )
        except (ValueError, IndexError):
            reference = None

    if reference is None:
        return None

    delta_days = (now - reference).total_seconds() / 86400.0
    return max(0.0, delta_days)


def is_reapable(
    candidate: Candidate, config: Config, *, now: datetime | None = None
) -> tuple[bool, str]:
    """Judge one row. Returns (reapable, reason).

    Reapable requires: downloaded, a filepath, not saved, and an age over
    `max_age_days`. Watch state and label are irrelevant.
    """
    if candidate.saved:
        return False, SAVED
    if not candidate.downloaded:
        return False, NOT_DOWNLOADED
    if not candidate.filepath:
        return False, ALREADY_REAPED

    age = age_days(candidate, now=now)
    if age is None:
        return False, UNDATED
    if age <= config.retention.max_age_days:
        return False, TOO_YOUNG

    return True, ""


def plan(
    ledger: Ledger, config: Config, *, now: datetime | None = None
) -> tuple[list[Reapable], dict[str, int]]:
    """What would be deleted, and a count of why each survivor was kept."""
    reapables: list[Reapable] = []
    kept: dict[str, int] = {}

    for candidate in ledger.all():
        reapable, reason = is_reapable(candidate, config, now=now)
        if not reapable:
            kept[reason] = kept.get(reason, 0) + 1
            continue

        path = Path(candidate.filepath)
        if not path.exists():
            kept[MISSING_FILE] = kept.get(MISSING_FILE, 0) + 1
            continue

        side = naming.sidecars(path)
        size = path.stat().st_size + sum(p.stat().st_size for p in side)
        unwatched = candidate.watch_pct is None and not candidate.watched
        reapables.append(
            Reapable(
                video_id=candidate.video_id,
                path=path,
                sidecars=tuple(side),
                bytes_freed=size,
                unwatched=unwatched,
            )
        )

    return reapables, kept


def apply(
    ledger: Ledger,
    config: Config,
    reapables: list[Reapable],
    *,
    dry_run: bool = False,
) -> RetentionReport:
    """Delete the planned files and their sidecars, then clear `filepath`.

    Every path is re-checked against the library root immediately before
    unlinking; a path that escaped it is an error, not a deletion. Emptied
    channel directories are pruned afterwards. A row whose unlink failed
    keeps its `filepath` so the next run retries it.
    """
    report = RetentionReport(dry_run=dry_run)
    touched_dirs: set[Path] = set()

    for reapable in reapables:
        safe_path = naming.ensure_within(config.paths.library, reapable.path)

        if dry_run:
            report.reaped.append(reapable)
            report.bytes_freed += reapable.bytes_freed
            continue

        try:
            safe_path.unlink()
            for sidecar in reapable.sidecars:
                sidecar_safe = naming.ensure_within(config.paths.library, sidecar)
                if sidecar_safe.exists():
                    sidecar_safe.unlink()
        except OSError as exc:
            report.errors.append((reapable.video_id, str(exc)))
            continue

        ledger.clear_filepath(reapable.video_id)
        report.reaped.append(reapable)
        report.bytes_freed += reapable.bytes_freed
        touched_dirs.add(safe_path.parent)

    if not dry_run:
        for directory in touched_dirs:
            try:
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()
            except OSError:
                pass

    return report


def save(
    ledger: Ledger,
    config: Config,
    video_id: str,
    *,
    dry_run: bool = False,
) -> Path:
    """Move a video out of the managed library and into the permanent one.

    Marks the row `saved=1` with its new path, which both takes it out of
    retention's reach and records the strongest positive signal the system
    has. Raises ConfigError if `paths.permanent_library` is unset, and
    LedgerError if the row is unknown or has nothing on disk.
    """
    if config.paths.permanent_library is None:
        raise ConfigError("paths.permanent_library is not configured")

    row = ledger.get(video_id)
    if row is None:
        raise LedgerError(f"unknown video_id: {video_id}")

    if row.saved and row.filepath:
        return Path(row.filepath)

    if not row.filepath:
        raise LedgerError(f"{video_id} has nothing on disk to save")

    if dry_run:
        return Path(row.filepath)

    new_path = naming.move_to_permanent(config, row.filepath)
    ledger.mark_saved(video_id, new_path)
    return new_path


def purge_logs(
    config: Config, *, now: datetime | None = None, dry_run: bool = False
) -> list[Path]:
    """Remove log files last modified more than `log_retention_days` ago.

    Judged by the file's real mtime against the real wall clock, not `now`
    -- unlike a ledger row, a log file has no injectable timestamp field, so
    there is no way to fake its age other than actually touching it.
    """
    logs_dir = config.paths.logs
    if not logs_dir.is_dir():
        return []

    cutoff = datetime.now(timezone.utc).timestamp() - config.retention.log_retention_days * 86400

    purged = []
    for path in logs_dir.iterdir():
        if not path.is_file():
            continue
        if path.stat().st_mtime < cutoff:
            purged.append(path)
            if not dry_run:
                try:
                    path.unlink()
                except OSError:
                    pass
    return purged


def free_bytes(path: str | os.PathLike[str]) -> int:
    """Free space on the filesystem holding `path`."""
    return shutil.disk_usage(path).free


def free_gb(path: str | os.PathLike[str]) -> float:
    return free_bytes(path) / BYTES_PER_GB


def run(
    *args: Config | Ledger,
    ledger: Ledger | None = None,
    now: datetime | None = None,
    dry_run: bool = False,
) -> RetentionReport:
    config: Config | None = None
    for arg in args:
        if isinstance(arg, Config):
            config = arg
        elif isinstance(arg, Ledger):
            ledger = arg
    if config is None:
        raise TypeError("retention.run() requires a Config argument")

    owns_ledger = ledger is None
    led = ledger or Ledger.open(config.paths.ledger)
    try:
        reapables, kept = plan(led, config, now=now)
        report = apply(led, config, reapables, dry_run=dry_run)
        report.kept = kept
        report.logs_purged = purge_logs(config, now=now, dry_run=dry_run)
        return report
    finally:
        if owns_ledger:
            led.close()
