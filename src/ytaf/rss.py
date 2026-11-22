"""Anonymous channel RSS discovery.

YouTube's per-channel Atom feed carries the last 15 uploads with no
authentication and no rate metering. It does not carry duration or live
status, so entries from here are enriched through yt-dlp before the prefilter
can judge them -- see `ytaf.discover`.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from . import http, spec
from .errors import DiscoveryError

ATOM_NS = "http://www.w3.org/2005/Atom"
YT_NS = "http://www.youtube.com/xml/schemas/2015"
MEDIA_NS = "http://search.yahoo.com/mrss/"

_CHANNEL_ID_RE = re.compile(r"^UC[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class FeedEntry:
    video_id: str
    title: str = ""
    channel: str = ""
    channel_id: str = ""
    published: str | None = None
    description: str = ""
    view_count: int | None = None
    thumbnail: str | None = None

    def as_metadata(self) -> dict[str, Any]:
        """Ledger `metadata` shape, using yt-dlp's key names.

        Deliberately omits `duration` and `is_live`: the feed does not know
        them, and inventing a value would let a short or a livestream slip
        past the prefilter.
        """
        meta: dict[str, Any] = {
            "id": self.video_id,
            "title": self.title,
            "description": self.description,
            "channel": self.channel,
            "channel_id": self.channel_id,
            "uploader": self.channel,
        }
        if self.published and len(self.published) >= 10:
            meta["upload_date"] = self.published[:10].replace("-", "")
        if self.view_count is not None:
            meta["view_count"] = self.view_count
        if self.thumbnail:
            meta["thumbnail"] = self.thumbnail
        return meta


def feed_url(channel_id: str) -> str:
    """The feed URL for a channel id. Rejects anything that is not a UC... id."""
    if not isinstance(channel_id, str) or not _CHANNEL_ID_RE.match(channel_id):
        raise DiscoveryError(f"not a channel id: {channel_id!r}")
    return spec.RSS_FEED_URL.format(channel_id=channel_id)


def _text(element: ET.Element | None) -> str | None:
    if element is None or not element.text:
        return None
    return element.text


def parse_feed(xml: str | bytes) -> list[FeedEntry]:
    """Parse an Atom feed into entries, newest first as YouTube orders them.

    Malformed XML raises DiscoveryError. Entries missing a videoId are
    skipped rather than failing the whole feed.
    """
    try:
        root = ET.fromstring(xml)
    except (ET.ParseError, ValueError, UnicodeDecodeError) as exc:
        raise DiscoveryError(f"malformed feed XML: {exc}") from exc

    if root.tag != f"{{{ATOM_NS}}}feed":
        raise DiscoveryError(f"not an Atom feed (root tag {root.tag!r})")

    entries: list[FeedEntry] = []
    for entry_el in root.findall(f"{{{ATOM_NS}}}entry"):
        video_id = _text(entry_el.find(f"{{{YT_NS}}}videoId"))
        if not video_id:
            continue

        title = _text(entry_el.find(f"{{{ATOM_NS}}}title")) or ""
        channel_id = _text(entry_el.find(f"{{{YT_NS}}}channelId")) or ""
        channel = (
            _text(entry_el.find(f"{{{ATOM_NS}}}author/{{{ATOM_NS}}}name")) or ""
        )
        published = _text(entry_el.find(f"{{{ATOM_NS}}}published"))

        description = ""
        thumbnail: str | None = None
        view_count: int | None = None
        group_el = entry_el.find(f"{{{MEDIA_NS}}}group")
        if group_el is not None:
            description = _text(group_el.find(f"{{{MEDIA_NS}}}description")) or ""
            thumb_el = group_el.find(f"{{{MEDIA_NS}}}thumbnail")
            if thumb_el is not None:
                thumbnail = thumb_el.get("url")
            stats_el = group_el.find(
                f"{{{MEDIA_NS}}}community/{{{MEDIA_NS}}}statistics"
            )
            if stats_el is not None and stats_el.get("views") is not None:
                view_count = int(stats_el.get("views"))

        entries.append(
            FeedEntry(
                video_id=video_id,
                title=title,
                channel=channel,
                channel_id=channel_id,
                published=published,
                description=description,
                view_count=view_count,
                thumbnail=thumbnail,
            )
        )
    return entries


def fetch_feed(channel_id: str, *, timeout: float = 30.0) -> list[FeedEntry]:
    """GET and parse one channel feed. Non-2xx raises DiscoveryError."""
    url = feed_url(channel_id)
    response = http.get(url, timeout=timeout)
    if not response.ok:
        raise DiscoveryError(f"feed request failed ({response.status}): {url}")
    return parse_feed(response.text)


def fetch_feeds(
    channel_ids: Iterable[str], *, sleep_requests: float = 0.0, timeout: float = 30.0
) -> tuple[list[FeedEntry], list[tuple[str, str]]]:
    """Fetch many feeds.

    Returns (entries, failures) where failures are (channel_id, message).
    One dead channel must not abort the day's discovery run.
    """
    entries: list[FeedEntry] = []
    failures: list[tuple[str, str]] = []
    for channel_id in channel_ids:
        try:
            entries.extend(fetch_feed(channel_id, timeout=timeout))
        except Exception as exc:  # noqa: BLE001 - one dead feed must not abort the run
            failures.append((channel_id, str(exc)))
        if sleep_requests:
            time.sleep(sleep_requests)
    return entries, failures


__all__ = [
    "ATOM_NS",
    "MEDIA_NS",
    "YT_NS",
    "FeedEntry",
    "feed_url",
    "fetch_feed",
    "fetch_feeds",
    "parse_feed",
    "spec",
    "time",
]
