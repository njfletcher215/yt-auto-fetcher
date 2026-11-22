"""The download stage.

Pulls every selected row we do not already hold, fetches it anonymously with
the H.264 selector, updates the ledger, and triggers one Jellyfin library
scan at the end of the batch.

There is no `--download-archive`: the ledger knows strictly more than
archive.txt did (held, reaped, watched, offer count) and the archive would
block the re-offers the repeat arm exists to make. `Ledger.held_ids` is the
skip test, and it covers saved videos, whose files live outside the managed
library.

A failure on one video is recorded and the batch continues -- a geo-blocked
video should not cost you the other nine. Failure across the *whole* batch is
the "extraction broken" signal and is reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import jellyfin, naming, ytdlp
from .config import Config
from .errors import YtdlpError
from .ledger import Candidate, Ledger


@dataclass
class DownloadOutcome:
    video_id: str
    ok: bool
    path: Path | None = None
    error: str | None = None
    skipped: bool = False


@dataclass
class DownloadReport:
    outcomes: list[DownloadOutcome] = field(default_factory=list)
    scan_triggered: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def attempted(self) -> int:
        return sum(1 for o in self.outcomes if not o.skipped)

    @property
    def succeeded(self) -> int:
        return sum(1 for o in self.outcomes if o.ok)

    @property
    def failed(self) -> int:
        return sum(1 for o in self.outcomes if not o.ok and not o.skipped)

    @property
    def total_failure(self) -> bool:
        """Every attempt failed, and there was at least one.

        Treated as "yt-dlp extraction broken" rather than bad luck:
        auto-update, retry once, then alert.
        """
        return self.attempted > 0 and self.succeeded == 0


def download_one(
    candidate: Candidate,
    config: Config,
    *,
    dry_run: bool = False,
) -> DownloadOutcome:
    """Fetch one video.

    Resolves the written path from the output template and verifies it is
    inside the library root before recording anything. On success the ledger
    row gets `downloaded=1`, its `filepath`, an incremented `offer_count` and
    a fresh `last_offered_at` -- a download is an offer, and the repeat arm
    reads both.
    """
    if dry_run:
        return DownloadOutcome(video_id=candidate.video_id, ok=False, skipped=True)

    try:
        ytdlp.download(candidate.video_id, config)
    except YtdlpError as exc:
        return DownloadOutcome(video_id=candidate.video_id, ok=False, error=str(exc))

    lookup = {
        "id": candidate.video_id,
        "channel": candidate.channel,
        "title": candidate.title,
        "upload_date": candidate.upload_date,
    }
    path = naming.find_downloaded(config, lookup)
    if path is None:
        return DownloadOutcome(
            video_id=candidate.video_id,
            ok=False,
            error="yt-dlp exited successfully but no output file was found",
        )

    path = naming.ensure_within(config.paths.library, path)
    return DownloadOutcome(video_id=candidate.video_id, ok=True, path=path)


def _attempt_batch(
    pending: list[Candidate], config: Config, led: Ledger, report: DownloadReport
) -> None:
    for candidate in pending:
        outcome = download_one(candidate, config)
        report.outcomes.append(outcome)
        if outcome.ok:
            led.mark_downloaded(candidate.video_id, outcome.path)


def run(
    config: Config,
    *,
    ledger: Ledger | None = None,
    dry_run: bool = False,
    retry_after_update: bool = True,
) -> DownloadReport:
    """Download the day's selections.

    On total failure with `retry_after_update`, runs `yt-dlp -U` and retries
    the batch exactly once before giving up and alerting (Phase 14, "Pin
    yt-dlp to auto-update").
    """
    owns_ledger = ledger is None
    led = ledger or Ledger.open(config.paths.ledger)
    try:
        report = DownloadReport()
        pending = led.pending_downloads()

        if dry_run:
            for candidate in pending:
                report.outcomes.append(download_one(candidate, config, dry_run=True))
            return report

        _attempt_batch(pending, config, led, report)

        if report.total_failure and retry_after_update:
            try:
                ytdlp.update()
            except Exception:  # noqa: BLE001 - the retry itself must still run
                pass
            retry_report = DownloadReport()
            _attempt_batch(led.pending_downloads(), config, led, retry_report)
            report = retry_report

        if report.succeeded > 0:
            client = jellyfin.build_client(config.jellyfin)
            if client is not None:
                result = client.trigger_scan()
                if result.ok:
                    report.scan_triggered = True
                else:
                    report.notes.append(f"library scan failed: {result.message}")

        return report
    finally:
        if owns_ledger:
            led.close()
