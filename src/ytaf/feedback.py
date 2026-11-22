"""The feedback stage: backfill playback and sentiment from Jellyfin.

Two things are synced:

  * playback -- `watch_pct` and `play_count`, the measured signal;
  * favorites -- Jellyfin's heart maps to `label = LABEL_LIKE`, so a like
    made in Streamyfin and one made with `ytaf like` are the same state.

The favorite sync is deliberately asymmetric. An un-hearted item clears a
*like*, but never touches a *dislike* -- there is no dislike control in any
Jellyfin client, so "not favorited" cannot be read as "not disliked". The only
way out of a dislike is `ytaf clear`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime

from . import jellyfin, spec
from .config import Config
from .jellyfin import JellyfinClient, PlaybackItem
from .ledger import Ledger


@dataclass
class FeedbackReport:
    items_seen: int = 0
    matched: int = 0
    playback_updated: int = 0
    likes_added: int = 0
    likes_cleared: int = 0
    unmatched: list[str] = field(default_factory=list)
    transcoding: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def updated(self) -> int:
        return self.playback_updated + self.likes_added + self.likes_cleared


def watch_pct(
    position_ticks: int,
    runtime_ticks: int,
    *,
    played: bool = False,
    play_count: int = 0,
) -> float | None:
    """Fraction watched. See `PlaybackItem.watch_pct` for the full rule.

    `played` forces 1.0; a zero position with a non-zero play count is 0.0;
    never opened is None; non-positive runtime is None.
    """
    if played:
        return 1.0
    if position_ticks == 0 and play_count == 0:
        return None
    if runtime_ticks <= 0:
        return None
    pct = position_ticks / runtime_ticks
    return max(0.0, min(1.0, pct))


def filepath_index(ledger: Ledger) -> dict[str, str]:
    """{filepath: video_id} for rows we hold, for matching Jellyfin items.

    Keys are normalised (resolved, no redundant segments) so a trailing slash
    or a `./` in Jellyfin's path does not miss a match.
    """
    index: dict[str, str] = {}
    for candidate in ledger.all():
        if not candidate.on_disk or not candidate.filepath:
            continue
        index[os.path.normpath(candidate.filepath)] = candidate.video_id
    return index


def apply_playback(
    ledger: Ledger, items: dict[str, PlaybackItem], report: FeedbackReport
) -> None:
    """Write watch_pct and play_count. Never lowers either."""
    for video_id, item in items.items():
        if not ledger.has_video(video_id):
            continue
        pct = item.watch_pct
        if pct is None:
            continue
        ledger.record_watch(video_id, pct, play_count=item.play_count)
        report.playback_updated += 1


def sync_favorites(
    ledger: Ledger,
    favorite_ids: set[str],
    seen_ids: set[str],
    report: FeedbackReport,
) -> None:
    """Reconcile Jellyfin hearts with the ledger's `label`.

    * in `favorite_ids`                        -> set LABEL_LIKE
    * in `seen_ids`, not favorited, now LIKE   -> clear the like
    * in `seen_ids`, not favorited, DISLIKE    -> left alone
    * not in `seen_ids`                        -> left alone (we cannot tell
      the difference between un-hearted and not-looked-at)
    """
    for video_id in favorite_ids:
        if not ledger.has_video(video_id):
            continue
        row = ledger.get(video_id)
        if row.label != spec.LABEL_LIKE:
            ledger.set_label(video_id, spec.LABEL_LIKE)
            report.likes_added += 1

    for video_id in seen_ids - favorite_ids:
        if not ledger.has_video(video_id):
            continue
        row = ledger.get(video_id)
        if row.label == spec.LABEL_LIKE:
            ledger.set_label(video_id, None)
            report.likes_cleared += 1


def run(
    config: Config,
    *,
    ledger: Ledger | None = None,
    client: JellyfinClient | None = None,
    since: datetime | str | None = None,
) -> FeedbackReport:
    """Query recent playback and favorites, then backfill.

    Also records any session found transcoding, which is the signal that the
    format selector has regressed.
    """
    owns_ledger = ledger is None
    led = ledger or Ledger.open(config.paths.ledger)
    try:
        report = FeedbackReport()

        active_client = client or jellyfin.build_client(config.jellyfin)
        if active_client is None:
            report.notes.append("Jellyfin is not configured; nothing to sync")
            return report

        items = active_client.played_items(since=since)
        report.items_seen = len(items)

        index = filepath_index(led)
        matched = jellyfin.match_items_to_ledger(items, index)
        report.matched = len(matched)

        matched_item_ids = {item.item_id for item in matched.values()}
        report.unmatched = [
            item.item_id for item in items if item.item_id not in matched_item_ids
        ]

        seen_ids = set(matched)
        favorite_ids = {video_id for video_id, item in matched.items() if item.is_favorite}

        apply_playback(led, matched, report)

        if config.feedback.sync_favorites:
            sync_favorites(led, favorite_ids, seen_ids, report)

        report.transcoding = [s.item_id for s in active_client.transcoding_sessions()]

        return report
    finally:
        if owns_ledger:
            led.close()
