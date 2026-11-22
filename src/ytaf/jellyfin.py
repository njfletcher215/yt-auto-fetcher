"""Jellyfin API client: library scans, playback feedback, favorites, and
transcode detection.

Ticks are Jellyfin's unit -- 10,000,000 per second.

`UserData` is where the real signal lives: `Played`, `PlayCount`,
`PlaybackPositionTicks` and `IsFavorite`. The heart in the Jellyfin and
Streamyfin UI is `IsFavorite`, which is what `ytaf like` sets and what the
feedback stage syncs back, so a like made on the phone and a like made at the
CLI are the same thing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlencode

from . import http, naming
from .config import JellyfinConfig
from .errors import JellyfinError

TICKS_PER_SECOND = 10_000_000
DIRECT_PLAY = "DirectPlay"
DIRECT_STREAM = "DirectStream"
TRANSCODE = "Transcode"

_FRAC_RE = re.compile(r"\.(\d+)")


def _coerce_dt(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    text = value
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    text = _FRAC_RE.sub(lambda m: "." + m.group(1)[:6].ljust(6, "0"), text, count=1)
    return datetime.fromisoformat(text)


@dataclass(frozen=True)
class PlaybackItem:
    """A Jellyfin item with whatever user state it carries."""

    item_id: str
    name: str = ""
    path: str = ""
    position_ticks: int = 0
    runtime_ticks: int = 0
    played: bool = False
    play_count: int = 0
    is_favorite: bool = False
    play_method: str | None = None
    last_played: str | None = None

    @property
    def watch_pct(self) -> float | None:
        """Fraction watched, from the four fields Jellyfin exposes.

        * `played` -> 1.0. Jellyfin flips this past its max-resume percentage
          (90% by default) and resets the position to 0, so reading position
          alone would turn every finished video into a hard negative.
        * position > 0 -> position / runtime, clamped to [0, 1].
        * position 0 with `play_count` > 0 -> 0.0. Opened and closed inside
          the first few percent, below Jellyfin's threshold for storing a
          resume point.
        * never opened -> None. No playback signal in either direction.
        """
        if self.played:
            return 1.0
        if self.position_ticks == 0 and self.play_count == 0:
            return None
        if self.runtime_ticks <= 0:
            return None
        pct = self.position_ticks / self.runtime_ticks
        return max(0.0, min(1.0, pct))

    @property
    def opened(self) -> bool:
        """Playback was started at least once."""
        return bool(self.play_count) or self.position_ticks != 0 or self.played

    @property
    def transcoding(self) -> bool:
        return self.play_method == TRANSCODE


@dataclass(frozen=True)
class ScanResult:
    ok: bool
    status: int = 0
    message: str = ""


class JellyfinClient:
    """Thin wrapper over the handful of endpoints this pipeline needs."""

    def __init__(self, config: JellyfinConfig) -> None:
        self.config = config

    # --- plumbing -----------------------------------------------------------
    def url(self, path: str, **query: object) -> str:
        """Join `path` onto base_url. Never puts the API key in the query."""
        if not self.config.base_url:
            raise JellyfinError("Jellyfin is not configured")
        base = self.config.base_url.rstrip("/")
        if not path.startswith("/"):
            path = "/" + path
        full = base + path
        if query:
            full += "?" + urlencode(query)
        return full

    def headers(self) -> dict[str, str]:
        """Auth headers. Jellyfin accepts `X-Emby-Token`; also sends the
        `Authorization: MediaBrowser ...` form newer servers prefer."""
        return {
            "X-Emby-Token": self.config.api_key,
            "Authorization": f'MediaBrowser Token="{self.config.api_key}"',
        }

    def get_json(self, path: str, **query: object) -> Any:
        response = http.get(self.url(path, **query), headers=self.headers())
        if not response.ok:
            raise JellyfinError(f"Jellyfin returned {response.status} for {path}")
        try:
            return response.json()
        except ValueError as exc:
            raise JellyfinError(
                f"Jellyfin returned unparseable JSON for {path}: {exc}"
            ) from exc

    def _item_from_raw(self, raw: dict[str, Any]) -> PlaybackItem:
        user_data = raw.get("UserData") or {}
        return PlaybackItem(
            item_id=raw.get("Id", ""),
            name=raw.get("Name", ""),
            path=raw.get("Path", ""),
            position_ticks=user_data.get("PlaybackPositionTicks", 0) or 0,
            runtime_ticks=raw.get("RunTimeTicks", 0) or 0,
            played=bool(user_data.get("Played", False)),
            play_count=user_data.get("PlayCount", 0) or 0,
            is_favorite=bool(user_data.get("IsFavorite", False)),
            last_played=user_data.get("LastPlayedDate"),
        )

    def _require_user(self) -> str:
        if not self.config.user_id:
            raise JellyfinError("Jellyfin user_id is not configured")
        return self.config.user_id

    # --- endpoints ----------------------------------------------------------
    def trigger_scan(self) -> ScanResult:
        """POST /Library/Refresh. Scoped to library_id if set."""
        try:
            response = http.post(self.url("/Library/Refresh"), headers=self.headers())
        except Exception as exc:  # noqa: BLE001 - a failed scan is cosmetic
            return ScanResult(ok=False, message=str(exc))
        return ScanResult(ok=response.ok, status=response.status, message=response.text)

    def sessions(self) -> list[PlaybackItem]:
        """GET /Sessions -- what is playing right now, with PlayMethod."""
        data = self.get_json("/Sessions")
        items = []
        for session in data:
            now_playing = session.get("NowPlayingItem")
            if not now_playing:
                continue
            play_state = session.get("PlayState") or {}
            user_data = now_playing.get("UserData") or {}
            items.append(
                PlaybackItem(
                    item_id=now_playing.get("Id", ""),
                    name=now_playing.get("Name", ""),
                    path=now_playing.get("Path", ""),
                    position_ticks=play_state.get("PositionTicks", 0) or 0,
                    runtime_ticks=now_playing.get("RunTimeTicks", 0) or 0,
                    played=bool(user_data.get("Played", False)),
                    play_count=user_data.get("PlayCount", 0) or 0,
                    is_favorite=bool(user_data.get("IsFavorite", False)),
                    play_method=play_state.get("PlayMethod"),
                )
            )
        return items

    def played_items(
        self, *, since: datetime | str | None = None, limit: int = 200
    ) -> list[PlaybackItem]:
        """Recently played items for the configured user.

        GET /Users/{user_id}/Items with Recursive and SortBy=DatePlayed,
        reading Played, PlayCount, PlaybackPositionTicks and IsFavorite out of
        UserData. `since` filters client-side on LastPlayedDate, because
        Jellyfin has no server-side filter for it.
        """
        user_id = self._require_user()
        data = self.get_json(
            f"/Users/{user_id}/Items",
            Recursive="true",
            SortBy="DatePlayed",
            SortOrder="Descending",
            Limit=limit,
            Fields="UserData,Path",
        )
        items = [self._item_from_raw(raw) for raw in data.get("Items", [])]
        if since is not None:
            since_dt = _coerce_dt(since)
            items = [
                item
                for item in items
                if item.last_played is None or _coerce_dt(item.last_played) >= since_dt
            ]
        return items

    def favorites(self) -> list[PlaybackItem]:
        """Items the user has hearted.

        Queried separately from `played_items`: you can favorite something
        without playing it, and that still means you like it.
        """
        user_id = self._require_user()
        data = self.get_json(
            f"/Users/{user_id}/Items",
            Recursive="true",
            Filters="IsFavorite",
            Fields="UserData,Path",
        )
        return [self._item_from_raw(raw) for raw in data.get("Items", [])]

    def set_favorite(self, item_id: str, favorite: bool) -> bool:
        """POST/DELETE /Users/{user_id}/FavoriteItems/{item_id}.

        Lets `ytaf like` show up as a heart in Streamyfin, so the CLI and the
        app are not two separate sources of truth. Returns False rather than
        raising when Jellyfin is unreachable -- the ledger write is what
        matters and must not be lost to a network blip.
        """
        user_id = self._require_user()
        try:
            response = http.post(
                self.url(f"/Users/{user_id}/FavoriteItems/{item_id}"),
                headers=self.headers(),
            )
        except Exception:  # noqa: BLE001 - the ledger write must not be lost
            return False
        return response.ok

    def transcoding_sessions(self) -> list[PlaybackItem]:
        """Current sessions whose PlayMethod is Transcode."""
        return [item for item in self.sessions() if item.transcoding]

    def find_by_path(self, path: str) -> PlaybackItem | None:
        """Locate an item by filesystem path, for `ytaf like <video_id>`."""
        for item in self.played_items(limit=1000):
            if item.path == path:
                return item
        return None

    def item(self, item_id: str) -> PlaybackItem | None:
        user_id = self._require_user()
        try:
            raw = self.get_json(f"/Users/{user_id}/Items/{item_id}")
        except JellyfinError:
            return None
        return self._item_from_raw(raw)


def build_client(config: JellyfinConfig) -> JellyfinClient | None:
    """Client, or None when Jellyfin is not configured.

    Feedback, favorites and library scans are all optional legs: the pipeline
    must still discover, score and download on a box with no Jellyfin.
    """
    if not config.configured:
        return None
    return JellyfinClient(config)


def video_id_from_item(item: PlaybackItem) -> str | None:
    """Recover a YouTube id from a Jellyfin item's path.

    Reads the `[video_id]` suffix the output template writes. Returning None
    is normal -- the library can contain items this pipeline did not create.
    """
    if not item.path:
        return None
    return naming.video_id_from_path(item.path)


def match_items_to_ledger(
    items: Iterable[PlaybackItem],
    filepath_index: dict[str, str],
) -> dict[str, PlaybackItem]:
    """Map video_id -> item, using a {filepath: video_id} index from the
    ledger and falling back to the id embedded in the path."""
    matched: dict[str, PlaybackItem] = {}
    for item in items:
        video_id = filepath_index.get(item.path) or video_id_from_item(item)
        if video_id:
            matched[video_id] = item
    return matched
