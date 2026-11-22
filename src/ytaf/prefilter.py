"""Discard obvious non-starters before they reach the ledger.

Mirrors the guide's yt-dlp match-filter in Python so RSS entries -- which
never pass through a yt-dlp filter -- are judged by the same rule:

    duration > 300 & duration < 7200 & !is_live & view_count > 500

Bounds are exclusive on both ends, as the guide writes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import PrefilterConfig


@dataclass(frozen=True)
class PrefilterResult:
    passed: bool
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.passed


# Reason codes, so the discovery log can be counted by cause.
TOO_SHORT = "too_short"
TOO_LONG = "too_long"
IS_LIVE = "is_live"
TOO_FEW_VIEWS = "too_few_views"
UNKNOWN_DURATION = "unknown_duration"
UNKNOWN_VIEWS = "unknown_views"


_LIVE_STATUSES = frozenset({"is_live", "is_upcoming", "post_live"})


def is_live(metadata: dict[str, Any]) -> bool:
    """True for a current livestream.

    Reads `is_live`, then `live_status` ('is_live'/'is_upcoming'/'post_live'
    count, 'was_live' and 'not_live' do not). A finished stream is an
    ordinary video.
    """
    if metadata.get("is_live"):
        return True
    return metadata.get("live_status") in _LIVE_STATUSES


def check(
    metadata: dict[str, Any],
    config: PrefilterConfig | None = None,
    *,
    allow_unknown: bool = False,
) -> PrefilterResult:
    """Judge one candidate's metadata.

    Missing duration or view count fails with an `unknown_*` reason unless
    `allow_unknown=True` -- that flag is for the pre-enrichment pass over RSS
    entries, where the fields genuinely are not known yet and the real check
    happens after yt-dlp fills them in.
    """
    cfg = config or PrefilterConfig()

    duration = metadata.get("duration")
    if duration is None:
        if not allow_unknown:
            return PrefilterResult(False, UNKNOWN_DURATION)
    else:
        duration = float(duration)
        if duration <= cfg.min_duration:
            return PrefilterResult(False, TOO_SHORT)
        if duration >= cfg.max_duration:
            return PrefilterResult(False, TOO_LONG)

    if not cfg.allow_live and is_live(metadata):
        return PrefilterResult(False, IS_LIVE)

    view_count = metadata.get("view_count")
    if view_count is None:
        if not allow_unknown:
            return PrefilterResult(False, UNKNOWN_VIEWS)
    else:
        if view_count <= cfg.min_view_count:
            return PrefilterResult(False, TOO_FEW_VIEWS)

    return PrefilterResult(True)


def match_filter_string(config: PrefilterConfig | None = None) -> str:
    """The equivalent `--match-filters` argument for yt-dlp.

    With default config this is exactly `spec.MATCH_FILTERS`.
    """
    cfg = config or PrefilterConfig()
    clause = f"duration > {cfg.min_duration} & duration < {cfg.max_duration}"
    if not cfg.allow_live:
        clause += " & !is_live"
    clause += f" & view_count > {cfg.min_view_count}"
    return clause
