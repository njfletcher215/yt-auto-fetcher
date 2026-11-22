"""The discovery stage.

Runs once daily, pulls candidates from anonymous sources, prefilters them and
logs them to the ledger. Downloads nothing.

Order of operations matters. RSS entries carry no duration and no live
status -- three of the prefilter's four clauses cannot be evaluated on a feed
entry at all -- so candidates go through a metadata-only yt-dlp call before
the prefilter runs. That call fetches JSON, never media.

Videos we currently hold (in the library or saved out) are dropped *before*
enrichment, so the day's request count stays proportional to genuinely new
videos: roughly one metadata fetch per new RSS upload.

Note that a video we downloaded before and have since reaped is *not*
dropped. Those rows are the re-offer pool, which is why the ledger replaced
yt-dlp's download archive: the archive only knows "ever downloaded" and would
refuse to re-fetch exactly the videos the repeat arm wants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import prefilter, rss, ytdlp
from .config import Config
from .ledger import Candidate, Ledger, utcnow_iso


@dataclass
class DiscoveryReport:
    rss_entries: int = 0
    search_entries: int = 0
    already_held: int = 0
    enriched: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    inserted: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def rejected_total(self) -> int:
        return sum(self.rejected.values())

    @property
    def ok(self) -> bool:
        """A run with no source failures. Partial failure is still a run."""
        return not self.failures


def candidate_from_metadata(
    metadata: dict[str, Any], *, source: str, seen_at: str | None = None
) -> Candidate:
    """Wrap yt-dlp (or RSS) metadata as a ledger row."""
    return Candidate(
        video_id=metadata.get("id"),
        seen_at=seen_at or utcnow_iso(),
        source=source,
        metadata=dict(metadata),
    )


def collect_rss(config: Config, report: DiscoveryReport) -> dict[str, Candidate]:
    """Poll every configured channel feed. Keyed by video id."""
    if not config.discovery.uses_rss:
        return {}

    entries, failures = rss.fetch_feeds(
        config.discovery.channel_ids, sleep_requests=config.discovery.sleep_requests
    )
    report.rss_entries += len(entries)
    report.failures.extend(failures)

    found: dict[str, Candidate] = {}
    for entry in entries:
        found[entry.video_id] = candidate_from_metadata(entry.as_metadata(), source="rss")
    return found


def collect_search(config: Config, report: DiscoveryReport) -> dict[str, Candidate]:
    """ytsearchdate per keyword. Keyed by video id."""
    if not config.discovery.uses_search:
        return {}

    found: dict[str, Candidate] = {}
    for keyword in config.discovery.keywords:
        try:
            entries = ytdlp.search(
                keyword,
                limit=config.discovery.search_results_per_keyword,
                sleep_requests=config.discovery.sleep_requests,
            )
        except Exception as exc:  # noqa: BLE001 - one dead keyword must not abort the rest
            report.failures.append((keyword, str(exc)))
            continue

        report.search_entries += len(entries)
        for entry in entries:
            video_id = entry.get("id")
            if video_id:
                found[video_id] = candidate_from_metadata(entry, source="search")
    return found


def drop_held(
    candidates: dict[str, Candidate],
    ledger: Ledger,
    config: Config,
    report: DiscoveryReport,
) -> dict[str, Candidate]:
    """Remove ids we currently hold a copy of, per `Ledger.held_ids`.

    Kept on purpose: rows in the ledger that are not currently on disk --
    both thin rows a re-sighting can enrich, and reaped rows that belong to
    the re-offer pool.
    """
    if not candidates:
        return {}
    held = ledger.held_ids()
    kept: dict[str, Candidate] = {}
    for video_id, candidate in candidates.items():
        if video_id in held:
            report.already_held += 1
            continue
        kept[video_id] = candidate
    return kept


def enrich(
    candidates: dict[str, Candidate], config: Config, report: DiscoveryReport
) -> dict[str, Candidate]:
    """Fill in duration / view count / live status via a metadata-only call.

    Candidates yt-dlp cannot resolve are dropped with a recorded reason.
    """
    if not candidates:
        return {}

    complete: dict[str, Candidate] = {}
    thin: dict[str, Candidate] = {}
    for video_id, candidate in candidates.items():
        if candidate.duration is not None and candidate.view_count is not None:
            complete[video_id] = candidate
        else:
            thin[video_id] = candidate

    if not thin:
        return complete

    fetched = ytdlp.fetch_metadata(
        thin.keys(), sleep_requests=config.discovery.sleep_requests
    )
    report.enriched += len(fetched)

    out = dict(complete)
    for video_id, candidate in thin.items():
        info = fetched.get(video_id)
        if info is None:
            report.rejected["unresolved"] = report.rejected.get("unresolved", 0) + 1
            continue
        merged_metadata = {**candidate.metadata, **info}
        out[video_id] = Candidate(
            video_id=candidate.video_id,
            seen_at=candidate.seen_at,
            source=candidate.source,
            metadata=merged_metadata,
        )
    return out


def apply_prefilter(
    candidates: dict[str, Candidate], config: Config, report: DiscoveryReport
) -> dict[str, Candidate]:
    """Apply the prefilter, counting rejections by reason."""
    kept: dict[str, Candidate] = {}
    for video_id, candidate in candidates.items():
        result = prefilter.check(candidate.metadata, config.prefilter)
        if result.passed:
            kept[video_id] = candidate
        else:
            report.rejected[result.reason] = report.rejected.get(result.reason, 0) + 1
    return kept


def run(
    config: Config,
    *,
    ledger: Ledger | None = None,
    dry_run: bool = False,
) -> DiscoveryReport:
    """The whole discovery stage. Opens its own ledger if not given one."""
    owns_ledger = ledger is None
    led = ledger or Ledger.open(config.paths.ledger)
    try:
        report = DiscoveryReport()

        found = collect_rss(config, report)
        found.update(collect_search(config, report))

        kept = drop_held(found, led, config, report)
        enriched = enrich(kept, config, report)
        surviving = apply_prefilter(enriched, config, report)

        if dry_run:
            return report

        inserted = 0
        for candidate in surviving.values():
            if led.upsert(candidate):
                inserted += 1
        report.inserted = inserted
        return report
    finally:
        if owns_ledger:
            led.close()
