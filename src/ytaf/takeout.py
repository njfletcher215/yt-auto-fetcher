"""Seed the ledger from a Google Takeout export.

Run once. Watch history is a weak positive (label 1), likes are a strong
positive (label 2), subscriptions become the RSS channel list.

Note on paths: the guide names `history/like-history.json`. Real exports have
moved likes into `playlists/Liked videos.csv` more than once, so the finder
accepts either shape and the test suite covers both.
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import spec
from .errors import MetadataError
from .ledger import Candidate, Ledger, utcnow_iso

WATCH_HISTORY_NAMES = ("watch-history.json",)
LIKE_HISTORY_NAMES = ("like-history.json", "Liked videos.csv")
SUBSCRIPTIONS_NAMES = ("subscriptions.csv",)

# Titles Google writes for entries that carry no usable video reference.
REMOVED_TITLE_MARKERS = (
    "Watched a video that has been removed",
    "Visited YouTube Music",
    "Answered survey question",
)

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YOUTUBE_HOSTS = frozenset(
    {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}
)


@dataclass(frozen=True)
class TakeoutEntry:
    video_id: str
    title: str = ""
    channel: str = ""
    channel_id: str = ""
    timestamp: str | None = None

    def as_metadata(self) -> dict[str, Any]:
        """Ledger `metadata` shape, using yt-dlp's key names."""
        meta: dict[str, Any] = {"id": self.video_id, "title": self.title}
        if self.channel:
            meta["channel"] = self.channel
            meta["uploader"] = self.channel
        if self.channel_id:
            meta["channel_id"] = self.channel_id
        return meta


@dataclass(frozen=True)
class Subscription:
    channel_id: str
    title: str = ""
    url: str = ""


@dataclass(frozen=True)
class TakeoutPaths:
    watch_history: Path | None = None
    like_history: Path | None = None
    subscriptions: Path | None = None


@dataclass
class ImportReport:
    watch_entries: int = 0
    like_entries: int = 0
    subscriptions: int = 0
    rows_inserted: int = 0
    rows_updated: int = 0
    skipped: list[str] = field(default_factory=list)
    channel_ids: tuple[str, ...] = ()


def parse_video_id(url: str | None) -> str | None:
    """Pull the 11-character id out of a Takeout `titleUrl`.

    Handles watch?v=, youtu.be/, shorts/, extra query params, and returns None
    for music.youtube.com playlist links, channel links and malformed input.
    """
    if not url:
        return None
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
        return candidate if _ID_RE.match(candidate) else None

    if host not in _YOUTUBE_HOSTS:
        return None

    if parsed.path == "/watch":
        values = parse_qs(parsed.query).get("v")
        if values and _ID_RE.match(values[0]):
            return values[0]
        return None

    for prefix in ("/shorts/", "/embed/"):
        if parsed.path.startswith(prefix):
            candidate = parsed.path[len(prefix):].split("/")[0]
            return candidate if _ID_RE.match(candidate) else None

    return None


def parse_channel_id(url: str | None) -> str | None:
    """Pull a UC... channel id out of a Takeout channel URL."""
    if not url:
        return None
    marker = "/channel/"
    idx = urlparse(url).path.find(marker)
    if idx == -1:
        return None
    candidate = urlparse(url).path[idx + len(marker):].split("/")[0]
    return candidate if candidate.startswith("UC") and len(candidate) > 2 else None


def _load_json_array(source: str | os.PathLike[str] | bytes) -> list[Any]:
    if isinstance(source, bytes):
        try:
            data = json.loads(source.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise MetadataError(f"malformed Takeout JSON: {exc}") from exc
    else:
        path = Path(source)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise MetadataError(f"could not read {path}: {exc}") from exc
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MetadataError(f"malformed Takeout JSON at {path}: {exc}") from exc
    if not isinstance(data, list):
        raise MetadataError("expected a JSON array of Takeout entries")
    return data


def _entry_from_raw(raw: dict[str, Any], prefix: str) -> TakeoutEntry | None:
    video_id = parse_video_id(raw.get("titleUrl"))
    if not video_id:
        return None

    title = str(raw.get("title") or "")
    if title.startswith(prefix):
        title = title[len(prefix):]

    channel = ""
    channel_id = ""
    subtitles = raw.get("subtitles") or []
    if subtitles:
        first = subtitles[0]
        channel = str(first.get("name") or "")
        channel_id = parse_channel_id(first.get("url")) or ""

    timestamp = raw.get("time")
    return TakeoutEntry(
        video_id=video_id, title=title, channel=channel, channel_id=channel_id,
        timestamp=timestamp,
    )


def _dedupe_keep_latest(entries: Iterable[TakeoutEntry]) -> list[TakeoutEntry]:
    by_id: dict[str, TakeoutEntry] = {}
    for entry in entries:
        existing = by_id.get(entry.video_id)
        if existing is None or (entry.timestamp or "") > (existing.timestamp or ""):
            by_id[entry.video_id] = entry
    return list(by_id.values())


def _parse_entries(data: list[Any], prefix: str) -> list[TakeoutEntry]:
    parsed = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        entry = _entry_from_raw(raw, prefix)
        if entry is not None:
            parsed.append(entry)
    return _dedupe_keep_latest(parsed)


def parse_watch_history(source: str | os.PathLike[str] | bytes) -> list[TakeoutEntry]:
    """Parse watch-history.json.

    Skips entries with no video id (removed videos, surveys, ads), de-dupes by
    video id keeping the most recent timestamp, and reads the channel out of
    the first `subtitles` element when present.
    """
    return _parse_entries(_load_json_array(source), "Watched ")


def _parse_liked_videos_csv(path: str | os.PathLike[str]) -> list[TakeoutEntry]:
    text = Path(path).read_text(encoding="utf-8-sig")
    entries = []
    for row in csv.DictReader(text.splitlines()):
        normalized = {(k or "").strip().lower(): v for k, v in row.items()}
        video_id = (normalized.get("video id") or "").strip()
        if not video_id:
            continue
        timestamp = normalized.get("playlist video creation timestamp")
        entries.append(TakeoutEntry(video_id=video_id, timestamp=timestamp))
    return _dedupe_keep_latest(entries)


def parse_like_history(source: str | os.PathLike[str] | bytes) -> list[TakeoutEntry]:
    """Parse like-history.json, or the `Liked videos.csv` form."""
    if isinstance(source, (str, os.PathLike)) and str(source).lower().endswith(".csv"):
        return _parse_liked_videos_csv(source)
    return _parse_entries(_load_json_array(source), "Liked ")


def parse_subscriptions(source: str | os.PathLike[str] | bytes) -> list[Subscription]:
    """Parse subscriptions.csv. Tolerates both the old and current headers."""
    if isinstance(source, bytes):
        text = source.decode("utf-8-sig")
    else:
        text = Path(source).read_text(encoding="utf-8-sig")

    subs = []
    for row in csv.DictReader(text.splitlines()):
        normalized = {(k or "").strip().lower(): v for k, v in row.items()}
        channel_id = (normalized.get("channel id") or "").strip()
        if not channel_id:
            continue
        title = (normalized.get("channel title") or "").strip()
        url = (normalized.get("channel url") or "").strip()
        subs.append(Subscription(channel_id=channel_id, title=title, url=url))
    return subs


def find_takeout_files(root: str | os.PathLike[str]) -> TakeoutPaths:
    """Locate the three files anywhere under an unpacked Takeout directory."""
    root = Path(root)
    watch = like = subs = None
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            name = path.name
            if watch is None and name in WATCH_HISTORY_NAMES:
                watch = path
            elif like is None and name in LIKE_HISTORY_NAMES:
                like = path
            elif subs is None and name in SUBSCRIPTIONS_NAMES:
                subs = path
    return TakeoutPaths(watch_history=watch, like_history=like, subscriptions=subs)


def _upsert_entry(
    ledger: Ledger, entry: TakeoutEntry, *, source: str, label: int, now: str
) -> bool:
    candidate = Candidate(
        video_id=entry.video_id,
        seen_at=entry.timestamp or now,
        source=source,
        metadata=entry.as_metadata(),
        label=label,
    )
    return ledger.upsert(candidate)


def import_takeout(
    ledger: Ledger,
    root: str | os.PathLike[str],
    *,
    now: str | None = None,
) -> ImportReport:
    """Load an unpacked Takeout export into the ledger.

    Likes win over history: a video in both ends up source='like', label=2.
    Re-running is idempotent and never lowers a label.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"takeout directory not found: {root}")

    now = now or utcnow_iso()
    found = find_takeout_files(root)
    report = ImportReport()

    watch_entries: list[TakeoutEntry] = []
    like_entries: list[TakeoutEntry] = []
    subs: list[Subscription] = []

    if found.watch_history is not None:
        watch_entries = parse_watch_history(found.watch_history)
    else:
        report.skipped.append("watch-history.json")

    if found.like_history is not None:
        like_entries = parse_like_history(found.like_history)
    else:
        report.skipped.append("like-history.json")

    if found.subscriptions is not None:
        subs = parse_subscriptions(found.subscriptions)
    else:
        report.skipped.append("subscriptions.csv")

    report.watch_entries = len(watch_entries)
    report.like_entries = len(like_entries)
    report.subscriptions = len(subs)
    report.channel_ids = merge_channel_ids([], subs)

    inserted = 0
    # Likes first: a video present in both ends up source='like', label=2,
    # because the ledger's merge rule keeps whatever was inserted first.
    for entry in like_entries:
        if _upsert_entry(ledger, entry, source="like", label=spec.LABEL_LIKE, now=now):
            inserted += 1
    for entry in watch_entries:
        if _upsert_entry(ledger, entry, source="history", label=spec.LABEL_HISTORY, now=now):
            inserted += 1

    report.rows_inserted = inserted
    return report


def merge_channel_ids(
    existing: Iterable[str], subscriptions: Iterable[Subscription]
) -> tuple[str, ...]:
    """Union of configured and subscribed channel ids, de-duped, sorted."""
    ids = set(existing)
    ids.update(sub.channel_id for sub in subscriptions)
    return tuple(sorted(ids))
