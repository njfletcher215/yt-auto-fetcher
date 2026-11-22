"""Monitoring: the failure table, one pure function per row.

| Failure                 | Detection                                  |
|-------------------------|--------------------------------------------|
| yt-dlp extraction broken| download failures across all videos        |
| Transcoding detected    | Jellyfin session reports Transcode         |
| Model degrading         | weekly mean watch_pct, model vs random     |
| Disk filling            | free space below threshold                 |
| Discovery stalled       | no new candidates in 24h                   |

The degradation check is why `n_random >= 1` is a config error: the random
arm is the control group it measures against.

Each check takes already-gathered inputs and returns a CheckResult, so the
decision logic is testable without a Jellyfin server, a RAID array or a
broken YouTube.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from . import jellyfin, notify, retention
from .config import Config
from .download import DownloadReport
from .jellyfin import JellyfinClient, PlaybackItem
from .ledger import Ledger, SelectionStats
from .notify import Alert, Notifier

EXTRACTION = "extraction_broken"
TRANSCODING = "transcoding_detected"
MODEL_DEGRADING = "model_degrading"
DISK_FILLING = "disk_filling"
DISCOVERY_STALLED = "discovery_stalled"

# Below this model-minus-random watch_pct gap, the model has stopped earning
# its keep.
MIN_MODEL_ADVANTAGE = 0.05
STALL_HOURS = 24


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    alert: Alert | None = None

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class MonitorReport:
    results: list[CheckResult] = field(default_factory=list)
    alerts_sent: int = 0

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.ok]

    @property
    def ok(self) -> bool:
        return not self.failures


def check_extraction(report: DownloadReport, *, ytdlp_version: str = "") -> CheckResult:
    """Fails when every attempted download failed.

    A zero-attempt run is not a failure -- nothing was selected, which is the
    discovery check's business, not this one.
    """
    if not report.total_failure:
        return CheckResult(EXTRACTION, True)

    detail = "every attempted download failed"
    if ytdlp_version:
        detail += f" (yt-dlp {ytdlp_version})"
    alert = Alert(
        title="yt-auto-fetcher: extraction broken",
        message=f"yt-dlp extraction appears broken: {detail}.",
        priority=notify.PRIORITY_HIGH,
        check=EXTRACTION,
    )
    return CheckResult(EXTRACTION, False, detail=detail, alert=alert)


def check_transcoding(items: list[PlaybackItem]) -> CheckResult:
    """Fails if any session reports Transcode. The selector has regressed and
    the server is being asked to do something it cannot do."""
    transcoding = [item for item in items if item.transcoding]
    if not transcoding:
        return CheckResult(TRANSCODING, True)

    names = ", ".join(item.name or item.item_id for item in transcoding)
    detail = f"transcoding: {names}"
    alert = Alert(
        title="yt-auto-fetcher: transcoding detected",
        message=f"Jellyfin is transcoding {names}. The format selector may have regressed.",
        priority=notify.PRIORITY_HIGH,
        check=TRANSCODING,
    )
    return CheckResult(TRANSCODING, False, detail=detail, alert=alert)


def check_model_degradation(
    stats: SelectionStats, *, min_advantage: float = MIN_MODEL_ADVANTAGE, min_samples: int = 5
) -> CheckResult:
    """Fails when model picks stop beating random picks.

    Not enough samples on either arm is "ok" with a detail note, not a false
    alarm -- in week one there is nothing to compare. The repeat arm is
    reported but not part of the pass/fail decision; it answers a different
    question (is a second chance worth a slot) and would muddy this one.
    """
    if stats.model_n < min_samples or stats.random_n < min_samples or stats.gap is None:
        return CheckResult(MODEL_DEGRADING, True, detail="not enough data yet to compare arms")

    gap = stats.gap
    detail = (
        f"model mean watch_pct {stats.model_mean_watch_pct:.2f} vs "
        f"random mean watch_pct {stats.random_mean_watch_pct:.2f} (gap {gap:.2f})"
    )
    if gap >= min_advantage:
        return CheckResult(MODEL_DEGRADING, True, detail=detail)

    alert = Alert(
        title="yt-auto-fetcher: model degrading",
        message=f"The model's advantage over random has fallen to {gap:.2f}. {detail}.",
        priority=notify.PRIORITY_DEFAULT,
        check=MODEL_DEGRADING,
    )
    return CheckResult(MODEL_DEGRADING, False, detail=detail, alert=alert)


def check_disk(free_gb: float, *, min_free_gb: float) -> CheckResult:
    if free_gb >= min_free_gb:
        return CheckResult(DISK_FILLING, True)

    detail = f"{free_gb:.1f} GB free, below the {min_free_gb:.1f} GB threshold"
    alert = Alert(
        title="yt-auto-fetcher: disk filling",
        message=f"Only {detail}.",
        priority=notify.PRIORITY_HIGH,
        check=DISK_FILLING,
    )
    return CheckResult(DISK_FILLING, False, detail=detail, alert=alert)


def check_discovery(
    seen_recently: int, *, hours: int = STALL_HOURS, channel_failures: int = 0
) -> CheckResult:
    """Fails when no candidate has been logged in `hours`."""
    if seen_recently > 0:
        return CheckResult(DISCOVERY_STALLED, True)

    message = f"No new candidates discovered in the last {hours} hours."
    if channel_failures:
        message += f" {channel_failures} channel feed(s) failed to fetch; check RSS reachability."
    alert = Alert(
        title="yt-auto-fetcher: discovery stalled",
        message=message,
        priority=notify.PRIORITY_DEFAULT,
        check=DISCOVERY_STALLED,
    )
    return CheckResult(DISCOVERY_STALLED, False, detail=message, alert=alert)


def run(
    config: Config,
    *,
    ledger: Ledger | None = None,
    client: JellyfinClient | None = None,
    notifier: Notifier | None = None,
    download_report: DownloadReport | None = None,
    now: datetime | None = None,
) -> MonitorReport:
    """Run every check that has its inputs available and send the alerts.

    Checks whose inputs are unavailable (no Jellyfin configured, no download
    report) are skipped with a note rather than reported as failures.
    """
    owns_ledger = ledger is None
    led = ledger or Ledger.open(config.paths.ledger)
    try:
        report = MonitorReport()
        active_notifier = notifier if notifier is not None else notify.build_notifier(
            config.notify
        )

        if download_report is not None:
            report.results.append(check_extraction(download_report))

        active_client = client if client is not None else jellyfin.build_client(config.jellyfin)
        if active_client is not None:
            sessions = active_client.sessions()
            report.results.append(check_transcoding(sessions))

        report.results.append(check_model_degradation(led.selection_stats()))

        free_gb = retention.free_gb(config.paths.library)
        report.results.append(check_disk(free_gb, min_free_gb=config.retention.min_free_gb))

        if now is not None:
            reference = now
        else:
            rows = led.all()
            reference = (
                max(datetime.fromisoformat(c.seen_at) for c in rows)
                if rows
                else datetime.now(timezone.utc)
            )
        since = reference - timedelta(hours=STALL_HOURS)
        report.results.append(check_discovery(led.seen_since(since)))

        for result in report.failures:
            if result.alert is not None:
                active_notifier.send(result.alert)
                report.alerts_sent += 1

        return report
    finally:
        if owns_ledger:
            led.close()
