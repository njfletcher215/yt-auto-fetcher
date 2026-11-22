"""The SQLite ledger.

One table, one row per video, mutated as a candidate walks the pipeline:
discovered -> scored -> selected -> downloaded -> watched -> reaped. It is
also the training set and the only record of what we already have, so nothing
is ever deleted from it: retention clears `filepath` and leaves everything
else behind.

Two columns are easy to confuse and must not be:

  * `source` is provenance -- how the row entered the ledger. Set once, never
    changed by a later sighting.
  * `label` is sentiment -- like, dislike, or in-watch-history. Freely
    mutable in any direction, but *only* by an explicit user action. A
    rediscovery by RSS never touches it.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self

from . import spec
from .errors import LedgerError

# Columns that follow "existing value wins unless it is NULL" merge semantics
# on upsert. Deliberately excludes video_id (the key), seen_at (never
# touched by a merge) and metadata (its own dict-merge logic).
_OPERATIONAL_COLUMNS = (
    "source",
    "score",
    "selected",
    "selection",
    "downloaded",
    "filepath",
    "watched",
    "watch_pct",
    "label",
    "play_count",
    "offer_count",
    "last_offered_at",
    "saved",
)

_LEGACY_COLUMN_DDL = {
    "play_count": "INTEGER DEFAULT 0",
    "offer_count": "INTEGER DEFAULT 0",
    "last_offered_at": "TEXT",
    "saved": "INTEGER DEFAULT 0",
}


@dataclass
class Candidate:
    """One ledger row, with `metadata` already decoded from JSON."""

    video_id: str
    seen_at: str
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    selected: int = 0
    selection: str | None = None
    downloaded: int = 0
    filepath: str | None = None
    watched: int | None = None
    watch_pct: float | None = None
    label: int | None = None
    play_count: int = 0
    offer_count: int = 0
    last_offered_at: str | None = None
    saved: int = 0

    # --- metadata accessors; all tolerate a missing key ---------------------
    @property
    def title(self) -> str:
        return str(self.metadata.get("title") or "")

    @property
    def description(self) -> str:
        return str(self.metadata.get("description") or "")

    @property
    def channel(self) -> str:
        meta = self.metadata
        return str(meta.get("channel") or meta.get("uploader") or "")

    @property
    def channel_id(self) -> str:
        return str(self.metadata.get("channel_id") or "")

    @property
    def duration(self) -> float | None:
        value = self.metadata.get("duration")
        return float(value) if value is not None else None

    @property
    def view_count(self) -> int | None:
        value = self.metadata.get("view_count")
        return int(value) if value is not None else None

    @property
    def upload_date(self) -> str | None:
        """yt-dlp's YYYYMMDD string, or None."""
        value = self.metadata.get("upload_date")
        return str(value) if value else None

    # --- state ------------------------------------------------------------
    @property
    def liked(self) -> bool:
        return self.label == spec.LABEL_LIKE

    @property
    def disliked(self) -> bool:
        return self.label == spec.LABEL_DISLIKE

    @property
    def on_disk(self) -> bool:
        """We currently hold a copy: either in the library or saved out.

        This is what replaces yt-dlp's download archive as the "do not fetch
        again" test. It must be true for saved videos, whose `filepath` points
        outside the managed library.
        """
        return bool(self.saved) or bool(self.downloaded and self.filepath)


@dataclass(frozen=True)
class SelectionStats:
    """Mean watch-percent per selection arm, for the degradation check."""

    model_n: int = 0
    model_mean_watch_pct: float | None = None
    repeat_n: int = 0
    repeat_mean_watch_pct: float | None = None
    random_n: int = 0
    random_mean_watch_pct: float | None = None

    @property
    def gap(self) -> float | None:
        """Model advantage over chance. None if either arm has no data."""
        if self.model_mean_watch_pct is None or self.random_mean_watch_pct is None:
            return None
        return self.model_mean_watch_pct - self.random_mean_watch_pct


def utcnow_iso() -> str:
    """Timestamp format used for every stored time -- ISO 8601, UTC, seconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_dt(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _dt_to_str(value: datetime | str) -> str:
    return value.isoformat() if isinstance(value, datetime) else value


def row_to_candidate(row: sqlite3.Row | Sequence[Any]) -> Candidate:
    """Decode a raw row, parsing `metadata` JSON. Bad JSON becomes {}."""
    raw_metadata = row["metadata"]
    metadata: dict[str, Any] = {}
    if raw_metadata:
        try:
            decoded = json.loads(raw_metadata)
            if isinstance(decoded, dict):
                metadata = decoded
        except (TypeError, ValueError):
            metadata = {}
    return Candidate(
        video_id=row["video_id"],
        seen_at=row["seen_at"],
        source=row["source"],
        metadata=metadata,
        score=row["score"],
        selected=row["selected"],
        selection=row["selection"],
        downloaded=row["downloaded"],
        filepath=row["filepath"],
        watched=row["watched"],
        watch_pct=row["watch_pct"],
        label=row["label"],
        play_count=row["play_count"],
        offer_count=row["offer_count"],
        last_offered_at=row["last_offered_at"],
        saved=row["saved"],
    )


def _validate_candidate(candidate: Candidate) -> None:
    if not candidate.video_id:
        raise LedgerError("video_id must not be empty")
    if not candidate.seen_at:
        raise LedgerError("seen_at must not be empty")
    if candidate.source is not None and candidate.source not in spec.SOURCES:
        raise LedgerError(f"unknown source: {candidate.source!r}")


class Ledger:
    """Owns a connection to ledger.db. Use as a context manager."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # --- lifecycle ----------------------------------------------------------
    @classmethod
    def open(cls, path: str | os.PathLike[str], *, create: bool = True) -> Self:
        """Open (and by default create + migrate) the ledger at `path`.

        Enables foreign keys and WAL. `create=False` on a missing file raises
        LedgerError rather than quietly making an empty database -- a missing
        ledger usually means the storage the library lives on is not mounted.
        """
        path = Path(path)
        if not create and not path.exists():
            raise LedgerError(f"ledger not found: {path}")
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        ledger = cls(conn)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            ledger.initialize()
        except sqlite3.DatabaseError as exc:
            conn.close()
            raise LedgerError(f"corrupt or unusable ledger at {path}: {exc}") from exc
        return ledger

    def initialize(self) -> None:
        """Apply the schema and indices. Idempotent.

        Also migrates a ledger created before `play_count`, `offer_count`,
        `last_offered_at` and `saved` existed, by adding the missing columns
        with their defaults.
        """
        self.conn.execute(spec.LEDGER_SCHEMA)
        self._migrate_legacy_columns()
        self._ensure_indexes()
        self.conn.commit()

    def _migrate_legacy_columns(self) -> None:
        existing = {
            row["name"] for row in self.conn.execute("PRAGMA table_info(candidates)")
        }
        for column, decl in _LEGACY_COLUMN_DDL.items():
            if column not in existing:
                self.conn.execute(f"ALTER TABLE candidates ADD COLUMN {column} {decl}")

    def _ensure_indexes(self) -> None:
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_candidates_score ON candidates(score)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_candidates_selected_downloaded "
            "ON candidates(selected, downloaded)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_candidates_repeat "
            "ON candidates(offer_count, last_offered_at)"
        )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- writes -------------------------------------------------------------
    def _upsert_one(self, candidate: Candidate, overwrite: bool) -> bool:
        _validate_candidate(candidate)
        existing = self.conn.execute(
            "SELECT * FROM candidates WHERE video_id = ?", (candidate.video_id,)
        ).fetchone()

        if existing is None:
            self.conn.execute(
                "INSERT INTO candidates ("
                "video_id, seen_at, source, metadata, score, selected, "
                "selection, downloaded, filepath, watched, watch_pct, label, "
                "play_count, offer_count, last_offered_at, saved"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    candidate.video_id,
                    candidate.seen_at,
                    candidate.source,
                    json.dumps(candidate.metadata),
                    candidate.score,
                    candidate.selected,
                    candidate.selection,
                    candidate.downloaded,
                    candidate.filepath,
                    candidate.watched,
                    candidate.watch_pct,
                    candidate.label,
                    candidate.play_count,
                    candidate.offer_count,
                    candidate.last_offered_at,
                    candidate.saved,
                ),
            )
            return True

        old = row_to_candidate(existing)
        merged = {}
        for column in _OPERATIONAL_COLUMNS:
            old_value = getattr(old, column)
            new_value = getattr(candidate, column)
            merged[column] = old_value if old_value is not None else new_value

        merged_metadata = (
            dict(candidate.metadata)
            if overwrite
            else {**candidate.metadata, **old.metadata}
        )

        self.conn.execute(
            "UPDATE candidates SET source = ?, metadata = ?, score = ?, "
            "selected = ?, selection = ?, downloaded = ?, filepath = ?, "
            "watched = ?, watch_pct = ?, label = ?, play_count = ?, "
            "offer_count = ?, last_offered_at = ?, saved = ? WHERE video_id = ?",
            (
                merged["source"],
                json.dumps(merged_metadata),
                merged["score"],
                merged["selected"],
                merged["selection"],
                merged["downloaded"],
                merged["filepath"],
                merged["watched"],
                merged["watch_pct"],
                merged["label"],
                merged["play_count"],
                merged["offer_count"],
                merged["last_offered_at"],
                merged["saved"],
                candidate.video_id,
            ),
        )
        return False

    def upsert(self, candidate: Candidate, *, overwrite: bool = False) -> bool:
        """Insert a candidate, or merge into the existing row.

        Returns True when the row is new. Merging never destroys accumulated
        signal: `label`, `score`, `watch_pct`, `watched`, `play_count`,
        `offer_count`, `last_offered_at`, `saved`, `filepath` and `downloaded`
        all survive a re-sighting, `seen_at` keeps its original value, and
        `source` is never rewritten. Thin metadata is enriched, not replaced.

        `overwrite=True` replaces metadata wholesale for a deliberate
        re-import, and still preserves every column above.
        """
        try:
            result = self._upsert_one(candidate, overwrite)
        except Exception:
            self.conn.rollback()
            raise
        self.conn.commit()
        return result

    def upsert_many(self, candidates: Iterable[Candidate], *, overwrite: bool = False) -> int:
        """Upsert in one transaction. Returns the count of newly inserted rows."""
        candidates = list(candidates)
        if not candidates:
            return 0
        new_count = 0
        try:
            for candidate in candidates:
                if self._upsert_one(candidate, overwrite):
                    new_count += 1
        except Exception:
            self.conn.rollback()
            raise
        self.conn.commit()
        return new_count

    def set_score(self, video_id: str, score: float) -> None:
        cur = self.conn.execute(
            "UPDATE candidates SET score = ? WHERE video_id = ?", (score, video_id)
        )
        if cur.rowcount == 0:
            raise LedgerError(f"unknown video_id: {video_id}")
        self.conn.commit()

    def set_label(self, video_id: str, label: int | None) -> None:
        """Set sentiment. Any transition is legal, including back to None.

        This is the only thing that writes `label`, and it imposes no
        ordering: like -> dislike, dislike -> like, either -> None. Rejects
        values outside `spec.LABELS`.
        """
        if label is not None and label not in spec.LABELS:
            raise LedgerError(f"unknown label: {label!r}")
        cur = self.conn.execute(
            "UPDATE candidates SET label = ? WHERE video_id = ?", (label, video_id)
        )
        if cur.rowcount == 0:
            raise LedgerError(f"unknown video_id: {video_id}")
        self.conn.commit()

    def mark_selected(self, video_id: str, selection: str) -> None:
        """Set selected=1 and selection to 'model', 'repeat' or 'random'."""
        if selection not in spec.SELECTIONS:
            raise LedgerError(f"unknown selection arm: {selection!r}")
        cur = self.conn.execute(
            "UPDATE candidates SET selected = 1, selection = ? WHERE video_id = ?",
            (selection, video_id),
        )
        if cur.rowcount == 0:
            raise LedgerError(f"unknown video_id: {video_id}")
        self.conn.commit()

    def clear_selections(self) -> None:
        """Reset selected/selection for rows we do not currently hold.

        Selections are per-run; yesterday's unselected leftovers should not
        pile up and arrive a week late. This also covers a row that was
        downloaded and has since been reaped -- its stale `selected=1` must
        not leak into `pending_downloads`, or it would look like an
        outstanding offer forever. A row actually on disk keeps its
        selection untouched, since that is the permanent record the arm
        comparison reads.
        """
        self.conn.execute(
            "UPDATE candidates SET selected = 0, selection = NULL "
            "WHERE saved = 0 AND NOT (downloaded = 1 AND filepath IS NOT NULL)"
        )
        self.conn.commit()

    def mark_downloaded(
        self, video_id: str, filepath: str | os.PathLike[str], *, offered_at: str | None = None
    ) -> None:
        """Record a successful download.

        Also increments `offer_count` and stamps `last_offered_at` -- the two
        fields the re-offer policy reads. A download *is* an offer.
        """
        offered_at = offered_at or utcnow_iso()
        cur = self.conn.execute(
            "UPDATE candidates SET downloaded = 1, filepath = ?, "
            "offer_count = offer_count + 1, last_offered_at = ? WHERE video_id = ?",
            (str(filepath), offered_at, video_id),
        )
        if cur.rowcount == 0:
            raise LedgerError(f"unknown video_id: {video_id}")
        self.conn.commit()

    def record_watch(
        self,
        video_id: str,
        watch_pct: float | None,
        *,
        watched: int = 1,
        play_count: int | None = None,
    ) -> None:
        """Record playback feedback. Never lowers watch_pct or play_count."""
        if watch_pct is not None and not (0.0 <= watch_pct <= 1.0):
            raise LedgerError("watch_pct must be between 0 and 1")
        row = self.get(video_id)
        if row is None:
            raise LedgerError(f"unknown video_id: {video_id}")

        if watch_pct is None:
            new_watch_pct = row.watch_pct
        elif row.watch_pct is None:
            new_watch_pct = watch_pct
        else:
            new_watch_pct = max(watch_pct, row.watch_pct)

        new_play_count = row.play_count
        if play_count is not None:
            new_play_count = max(play_count, row.play_count)

        self.conn.execute(
            "UPDATE candidates SET watched = ?, watch_pct = ?, play_count = ? "
            "WHERE video_id = ?",
            (watched, new_watch_pct, new_play_count, video_id),
        )
        self.conn.commit()

    def mark_saved(self, video_id: str, filepath: str | os.PathLike[str]) -> None:
        """Video moved to the permanent library. Retention can no longer see it."""
        cur = self.conn.execute(
            "UPDATE candidates SET saved = 1, filepath = ? WHERE video_id = ?",
            (str(filepath), video_id),
        )
        if cur.rowcount == 0:
            raise LedgerError(f"unknown video_id: {video_id}")
        self.conn.commit()

    def clear_filepath(self, video_id: str) -> None:
        """Retention reaped the file; the row and all its signal stay."""
        cur = self.conn.execute(
            "UPDATE candidates SET filepath = NULL WHERE video_id = ?", (video_id,)
        )
        if cur.rowcount == 0:
            raise LedgerError(f"unknown video_id: {video_id}")
        self.conn.commit()

    # --- reads --------------------------------------------------------------
    def get(self, video_id: str) -> Candidate | None:
        row = self.conn.execute(
            "SELECT * FROM candidates WHERE video_id = ?", (video_id,)
        ).fetchone()
        return row_to_candidate(row) if row is not None else None

    def all(self) -> list[Candidate]:
        rows = self.conn.execute("SELECT * FROM candidates").fetchall()
        return [row_to_candidate(r) for r in rows]

    def unscored(self) -> list[Candidate]:
        """score IS NULL and not currently held -- the scoring work queue."""
        rows = self.conn.execute(
            "SELECT * FROM candidates WHERE score IS NULL "
            "AND saved = 0 AND NOT (downloaded = 1 AND filepath IS NOT NULL)"
        ).fetchall()
        return [row_to_candidate(r) for r in rows]

    def new_pool(self) -> list[Candidate]:
        """Never downloaded, not disliked, not selected -- the `n_model_new`
        and `n_random` pool."""
        rows = self.conn.execute(
            "SELECT * FROM candidates WHERE downloaded = 0 AND saved = 0 "
            "AND selected = 0 AND (label IS NULL OR label != ?)",
            (spec.LABEL_DISLIKE,),
        ).fetchall()
        return [row_to_candidate(r) for r in rows]

    def repeat_pool(
        self, *, now: datetime | str | None = None, cooldown_days: int = 90, max_offers: int = 4
    ) -> list[Candidate]:
        """Downloaded before, never watched, since reaped, and off cooldown.

        Excludes anything watched, liked, disliked, saved, or currently on
        disk. Eligibility is driven by `last_offered_at` against
        `cooldown_days`; `max_offers` (0 = no cap) is a backstop so a video
        ignored over and over eventually stops returning.
        """
        now_dt = _parse_dt(now) if now is not None else datetime.now(timezone.utc)
        rows = self.conn.execute(
            "SELECT * FROM candidates WHERE downloaded = 1 AND filepath IS NULL "
            "AND saved = 0 AND (watched IS NULL OR watched = 0) "
            "AND (label IS NULL OR label NOT IN (?, ?))",
            (spec.LABEL_LIKE, spec.LABEL_DISLIKE),
        ).fetchall()

        result = []
        for raw_row in rows:
            candidate = row_to_candidate(raw_row)
            if max_offers and candidate.offer_count >= max_offers:
                continue
            if candidate.last_offered_at:
                last = _parse_dt(candidate.last_offered_at)
                if (now_dt - last).days < cooldown_days:
                    continue
            result.append(candidate)
        return result

    def pending_downloads(self) -> list[Candidate]:
        """selected=1 AND we do not currently hold a copy."""
        rows = self.conn.execute(
            "SELECT * FROM candidates WHERE selected = 1 "
            "AND saved = 0 AND NOT (downloaded = 1 AND filepath IS NOT NULL)"
        ).fetchall()
        return [row_to_candidate(r) for r in rows]

    def downloaded(self) -> list[Candidate]:
        """Currently in the managed library (not saved out, not reaped)."""
        rows = self.conn.execute(
            "SELECT * FROM candidates WHERE downloaded = 1 "
            "AND filepath IS NOT NULL AND saved = 0"
        ).fetchall()
        return [row_to_candidate(r) for r in rows]

    def held_ids(self) -> set[str]:
        """Every video we currently hold, saved or in-library.

        This is the replacement for yt-dlp's download archive: discovery and
        the download stage both consult it so a video we already have is
        never fetched twice.
        """
        rows = self.conn.execute(
            "SELECT video_id FROM candidates WHERE saved = 1 "
            "OR (downloaded = 1 AND filepath IS NOT NULL)"
        ).fetchall()
        return {r["video_id"] for r in rows}

    def training_rows(self) -> list[Candidate]:
        """Rows carrying any usable signal: a label, a watch_pct, or an
        ignored offer."""
        rows = self.conn.execute(
            "SELECT * FROM candidates WHERE label IS NOT NULL "
            "OR watch_pct IS NOT NULL OR saved = 1 "
            "OR (downloaded = 1 AND filepath IS NULL)"
        ).fetchall()
        return [row_to_candidate(r) for r in rows]

    def liked(self) -> list[Candidate]:
        """label == LABEL_LIKE."""
        rows = self.conn.execute(
            "SELECT * FROM candidates WHERE label = ?", (spec.LABEL_LIKE,)
        ).fetchall()
        return [row_to_candidate(r) for r in rows]

    def saved_rows(self) -> list[Candidate]:
        rows = self.conn.execute("SELECT * FROM candidates WHERE saved = 1").fetchall()
        return [row_to_candidate(r) for r in rows]

    def count(self, **where: object) -> int:
        """Row count filtered by exact column matches. `count()` is the total."""
        if not where:
            return self.conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        clauses = []
        params: list[object] = []
        for column, value in where.items():
            if column not in spec.LEDGER_COLUMNS:
                raise LedgerError(f"unknown column: {column!r}")
            if value is None:
                clauses.append(f"{column} IS NULL")
            else:
                clauses.append(f"{column} = ?")
                params.append(value)
        sql = "SELECT COUNT(*) FROM candidates WHERE " + " AND ".join(clauses)
        return self.conn.execute(sql, params).fetchone()[0]

    def seen_since(self, since: datetime | str) -> int:
        """Candidates first seen since `since`. The discovery-stall check."""
        since_str = _dt_to_str(since)
        return self.conn.execute(
            "SELECT COUNT(*) FROM candidates WHERE seen_at >= ?", (since_str,)
        ).fetchone()[0]

    def selection_stats(self, since: datetime | str | None = None) -> SelectionStats:
        """Mean watch_pct per arm: model, repeat, random."""
        since_str = _dt_to_str(since) if since is not None else None

        def stats_for(arm: str) -> tuple[int, float | None]:
            sql = (
                "SELECT watch_pct FROM candidates WHERE selected = 1 "
                "AND selection = ? AND watch_pct IS NOT NULL"
            )
            params: list[object] = [arm]
            if since_str is not None:
                sql += " AND seen_at >= ?"
                params.append(since_str)
            values = [r["watch_pct"] for r in self.conn.execute(sql, params).fetchall()]
            n = len(values)
            return n, (sum(values) / n if n else None)

        model_n, model_mean = stats_for(spec.MODEL)
        repeat_n, repeat_mean = stats_for(spec.REPEAT)
        random_n, random_mean = stats_for(spec.RANDOM)
        return SelectionStats(
            model_n=model_n,
            model_mean_watch_pct=model_mean,
            repeat_n=repeat_n,
            repeat_mean_watch_pct=repeat_mean,
            random_n=random_n,
            random_mean_watch_pct=random_mean,
        )

    def has_video(self, video_id: str) -> bool:
        return (
            self.conn.execute(
                "SELECT 1 FROM candidates WHERE video_id = ?", (video_id,)
            ).fetchone()
            is not None
        )

    def existing_ids(self, video_ids: Iterable[str]) -> set[str]:
        """Which of `video_ids` are already in the ledger. One query."""
        ids = list(video_ids)
        if not ids:
            return set()
        found: set[str] = set()
        chunk_size = 500
        for i in range(0, len(ids), chunk_size):
            chunk = ids[i : i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"SELECT video_id FROM candidates WHERE video_id IN ({placeholders})",
                chunk,
            ).fetchall()
            found.update(r["video_id"] for r in rows)
        return found


def open_for(config: object, *, create: bool = True) -> Ledger:
    """`Ledger.open(config.paths.ledger)`, for callers that hold a Config."""
    return Ledger.open(config.paths.ledger, create=create)  # type: ignore[attr-defined]


def path_for(config: object) -> Path:
    return config.paths.ledger  # type: ignore[attr-defined]
