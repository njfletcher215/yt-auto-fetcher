"""The ledger: schema, merge semantics, sentiment transitions, work queues.

Two areas get the most attention.

Merge semantics, because a video can be rediscovered by RSS weeks after you
watched it, and an upsert that clobbers `watch_pct` or `label` destroys
training data that cannot be regenerated.

Sentiment transitions, because `label` must move freely in every direction --
like, dislike, neither, and back -- while a *rediscovery* must never move it
at all. Those two requirements pull in opposite directions and the split is
exactly: `set_label` is free, `upsert` never touches it.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3

import pytest

from ytaf import ledger as ledger_mod
from ytaf import spec
from ytaf.errors import LedgerError
from ytaf.ledger import Candidate, Ledger

NOW = dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc)


def days_ago(n: int) -> str:
    return (NOW - dt.timedelta(days=n)).isoformat()


class TestOpen:
    def test_creates_the_database_and_schema(self, config):
        with Ledger.open(config.paths.ledger) as led:
            assert config.paths.ledger.exists()
            assert led.count() == 0

    def test_creates_parent_directories(self, tmp_path):
        target = tmp_path / "deep" / "state" / "ledger.db"
        with Ledger.open(target):
            assert target.exists()

    def test_create_false_on_a_missing_file_raises(self, config):
        # A missing ledger usually means the storage it lives on is not
        # mounted. Creating a fresh empty one would hide that and start the
        # entire history over.
        with pytest.raises(LedgerError):
            Ledger.open(config.paths.ledger, create=False)

    def test_create_false_opens_an_existing_file(self, config):
        Ledger.open(config.paths.ledger).close()
        with Ledger.open(config.paths.ledger, create=False) as led:
            assert led.count() == 0

    def test_reopening_preserves_rows(self, config, make_candidate):
        with Ledger.open(config.paths.ledger) as led:
            led.upsert(make_candidate("aaaaaaaaaa1"))
        with Ledger.open(config.paths.ledger) as led:
            assert led.get("aaaaaaaaaa1") is not None

    def test_initialize_is_idempotent(self, ledger):
        ledger.initialize()
        ledger.initialize()
        assert ledger.count() == 0

    def test_initialize_does_not_wipe_existing_rows(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        ledger.initialize()
        assert ledger.get("aaaaaaaaaa1") is not None

    def test_corrupt_database_raises(self, config):
        config.paths.ledger.write_bytes(b"this is not a sqlite file" * 10)
        with pytest.raises(LedgerError):
            Ledger.open(config.paths.ledger)

    def test_context_manager_closes(self, config):
        with Ledger.open(config.paths.ledger) as led:
            conn = led.conn
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


class TestMigration:
    """A ledger created before the four added columns must still open."""

    @pytest.fixture
    def legacy(self, config):
        conn = sqlite3.connect(config.paths.ledger)
        conn.execute(
            """
            CREATE TABLE candidates (
                video_id   TEXT PRIMARY KEY,
                seen_at    TEXT NOT NULL,
                source     TEXT,
                metadata   TEXT,
                score      REAL,
                selected   INTEGER DEFAULT 0,
                selection  TEXT,
                downloaded INTEGER DEFAULT 0,
                filepath   TEXT,
                watched    INTEGER,
                watch_pct  REAL,
                label      INTEGER
            )
            """
        )
        conn.execute(
            "INSERT INTO candidates (video_id, seen_at, watch_pct, label) "
            "VALUES ('legacy00001', '2026-01-01T00:00:00+00:00', 0.9, 2)"
        )
        conn.commit()
        conn.close()
        return config.paths.ledger

    def test_adds_the_missing_columns(self, legacy):
        with Ledger.open(legacy) as led:
            row = led.get("legacy00001")
            assert row is not None
            assert row.play_count == 0
            assert row.offer_count == 0
            assert row.last_offered_at is None
            assert row.saved == 0

    def test_preserves_existing_data(self, legacy):
        with Ledger.open(legacy) as led:
            row = led.get("legacy00001")
            assert row.watch_pct == pytest.approx(0.9)
            assert row.label == 2

    def test_is_idempotent(self, legacy):
        with Ledger.open(legacy) as led:
            led.initialize()
            assert led.count() == 1


class TestSchema:
    @pytest.fixture
    def columns(self, ledger, raw_conn):
        rows = raw_conn.execute("PRAGMA table_info(candidates)").fetchall()
        return {r["name"]: r for r in rows}

    def test_has_exactly_the_documented_columns(self, columns):
        assert set(columns) == set(spec.LEDGER_COLUMNS)

    @pytest.mark.parametrize(
        "column,sql_type",
        [
            ("video_id", "TEXT"),
            ("seen_at", "TEXT"),
            ("source", "TEXT"),
            ("metadata", "TEXT"),
            ("score", "REAL"),
            ("selected", "INTEGER"),
            ("selection", "TEXT"),
            ("downloaded", "INTEGER"),
            ("filepath", "TEXT"),
            ("watched", "INTEGER"),
            ("watch_pct", "REAL"),
            ("label", "INTEGER"),
            ("play_count", "INTEGER"),
            ("offer_count", "INTEGER"),
            ("last_offered_at", "TEXT"),
            ("saved", "INTEGER"),
        ],
    )
    def test_column_types(self, columns, column, sql_type):
        assert columns[column]["type"].upper() == sql_type

    def test_video_id_is_the_primary_key(self, columns):
        assert columns["video_id"]["pk"] == 1

    def test_seen_at_is_not_null(self, columns):
        assert columns["seen_at"]["notnull"] == 1

    @pytest.mark.parametrize(
        "column", ["selected", "downloaded", "play_count", "offer_count", "saved"]
    )
    def test_counters_default_to_zero(self, columns, column):
        assert str(columns[column]["dflt_value"]) == "0"

    def test_duplicate_video_id_is_rejected_at_the_sql_level(self, ledger, raw_conn):
        raw_conn.execute(
            "INSERT INTO candidates (video_id, seen_at) VALUES ('dupe', '2026-01-01')"
        )
        raw_conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            raw_conn.execute(
                "INSERT INTO candidates (video_id, seen_at) VALUES ('dupe', '2026-01-02')"
            )

    def test_has_indexes_supporting_the_work_queues(self, ledger, raw_conn):
        indexes = raw_conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='candidates'"
        ).fetchall()
        sql = " ".join((r["sql"] or "") for r in indexes)
        assert "score" in sql
        assert "selected" in sql or "downloaded" in sql
        # The re-offer pool filters on these two every run.
        assert "last_offered_at" in sql or "offer_count" in sql

    def test_foreign_keys_are_on(self, ledger):
        assert ledger.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    def test_journal_mode_is_wal(self, ledger):
        assert ledger.conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


class TestTimestamps:
    def test_utcnow_iso_is_parseable(self):
        assert dt.datetime.fromisoformat(ledger_mod.utcnow_iso()).tzinfo is not None

    def test_utcnow_iso_is_utc(self):
        assert dt.datetime.fromisoformat(ledger_mod.utcnow_iso()).utcoffset() == (
            dt.timedelta(0)
        )

    def test_utcnow_iso_sorts_lexicographically(self):
        # seen_since, retention and the repeat cooldown all compare these as
        # strings.
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", ledger_mod.utcnow_iso())

    def test_utcnow_iso_has_no_microseconds(self):
        assert "." not in ledger_mod.utcnow_iso().split("+")[0]


class TestUpsertInsert:
    def test_returns_true_for_a_new_row(self, ledger, make_candidate):
        assert ledger.upsert(make_candidate("aaaaaaaaaa1")) is True

    def test_returns_false_for_an_existing_row(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        assert ledger.upsert(make_candidate("aaaaaaaaaa1")) is False

    def test_round_trips_every_field(self, ledger, make_candidate):
        ledger.upsert(
            make_candidate(
                "aaaaaaaaaa1",
                source="search",
                score=0.75,
                selected=1,
                selection="model",
                downloaded=1,
                filepath="/x/y.mp4",
                watched=1,
                watch_pct=0.95,
                label=spec.LABEL_LIKE,
                play_count=3,
                offer_count=2,
                last_offered_at="2026-10-01T00:00:00+00:00",
                saved=1,
            )
        )
        row = ledger.get("aaaaaaaaaa1")
        assert row.source == "search"
        assert row.score == pytest.approx(0.75)
        assert (row.selected, row.selection) == (1, "model")
        assert (row.downloaded, row.filepath) == (1, "/x/y.mp4")
        assert (row.watched, row.watch_pct) == (1, pytest.approx(0.95))
        assert row.label == spec.LABEL_LIKE
        assert (row.play_count, row.offer_count, row.saved) == (3, 2, 1)
        assert row.last_offered_at == "2026-10-01T00:00:00+00:00"

    def test_metadata_is_stored_as_json_text(self, ledger, make_candidate, raw_conn):
        ledger.upsert(make_candidate("aaaaaaaaaa1", title="Stored As JSON"))
        raw = raw_conn.execute(
            "SELECT metadata FROM candidates WHERE video_id='aaaaaaaaaa1'"
        ).fetchone()[0]
        assert json.loads(raw)["title"] == "Stored As JSON"

    def test_metadata_survives_unicode_and_quotes(self, ledger, make_candidate):
        title = 'Ünïcode "quoted" & <angled> — em dash 日本語'
        ledger.upsert(make_candidate("aaaaaaaaaa1", title=title))
        assert ledger.get("aaaaaaaaaa1").title == title

    def test_metadata_survives_nested_structures(self, ledger, make_candidate):
        chapters = [{"start_time": 0.0, "title": "Intro"}]
        ledger.upsert(make_candidate("aaaaaaaaaa1", metadata={"chapters": chapters}))
        assert ledger.get("aaaaaaaaaa1").metadata["chapters"] == chapters

    def test_empty_metadata_is_allowed(self, ledger):
        ledger.upsert(Candidate(video_id="bare", seen_at="2026-09-14T00:00:00+00:00"))
        assert ledger.get("bare").metadata == {}

    @pytest.mark.parametrize("source", sorted(spec.SOURCES))
    def test_accepts_every_documented_source(self, ledger, make_candidate, source):
        ledger.upsert(make_candidate("aaaaaaaaaa1", source=source))
        assert ledger.get("aaaaaaaaaa1").source == source

    @pytest.mark.parametrize("source", ["recommended", "RSS", "", "youtube"])
    def test_rejects_an_undocumented_source(self, ledger, make_candidate, source):
        with pytest.raises(LedgerError):
            ledger.upsert(make_candidate("aaaaaaaaaa1", source=source))

    def test_rejects_an_empty_video_id(self, ledger):
        with pytest.raises(LedgerError):
            ledger.upsert(Candidate(video_id="", seen_at="2026-09-14T00:00:00+00:00"))

    def test_rejects_a_missing_seen_at(self, ledger):
        with pytest.raises(LedgerError):
            ledger.upsert(Candidate(video_id="aaaaaaaaaa1", seen_at=""))


class TestUpsertMerge:
    """Rediscovering a known video must never cost us signal."""

    @pytest.fixture
    def established(self, ledger, make_candidate):
        ledger.upsert(
            make_candidate(
                "aaaaaaaaaa1",
                source="like",
                score=0.9,
                selected=1,
                selection="model",
                downloaded=1,
                filepath="/library/x.mp4",
                watched=1,
                watch_pct=0.97,
                label=spec.LABEL_LIKE,
                play_count=4,
                offer_count=2,
                last_offered_at="2026-10-01T00:00:00+00:00",
            )
        )
        return ledger

    @pytest.mark.parametrize(
        "field,expected",
        [
            ("score", 0.9),
            ("watch_pct", 0.97),
            ("label", spec.LABEL_LIKE),
            ("filepath", "/library/x.mp4"),
            ("downloaded", 1),
            ("watched", 1),
            ("selection", "model"),
            ("selected", 1),
            ("play_count", 4),
            ("offer_count", 2),
            ("last_offered_at", "2026-10-01T00:00:00+00:00"),
        ],
    )
    def test_rediscovery_preserves(self, established, make_candidate, field, expected):
        established.upsert(make_candidate("aaaaaaaaaa1", source="rss"))
        actual = getattr(established.get("aaaaaaaaaa1"), field)
        if isinstance(expected, float):
            assert actual == pytest.approx(expected)
        else:
            assert actual == expected

    def test_rediscovery_preserves_the_original_seen_at(self, established, make_candidate):
        # seen_at is what retention ages against; bumping it on every RSS
        # poll would mean nothing ever got old enough to reap.
        established.upsert(
            make_candidate("aaaaaaaaaa1", seen_at="2026-10-01T00:00:00+00:00")
        )
        assert established.get("aaaaaaaaaa1").seen_at == "2026-09-14T04:00:00+00:00"

    def test_rediscovery_never_rewrites_source(self, established, make_candidate):
        # `source` is provenance: it records how this row first arrived and is
        # not a ranking. Sentiment lives in `label`.
        established.upsert(make_candidate("aaaaaaaaaa1", source="rss"))
        assert established.get("aaaaaaaaaa1").source == "like"

    def test_rediscovery_never_touches_a_like(self, established, make_candidate):
        established.upsert(make_candidate("aaaaaaaaaa1", source="rss", label=None))
        assert established.get("aaaaaaaaaa1").label == spec.LABEL_LIKE

    def test_rediscovery_never_touches_a_dislike(self, ledger, make_candidate):
        # Critical: RSS re-surfacing a video you explicitly rejected must not
        # quietly clear the rejection and put it back in the pool.
        ledger.upsert(make_candidate("aaaaaaaaaa1", label=spec.LABEL_DISLIKE))
        ledger.upsert(make_candidate("aaaaaaaaaa1", source="rss", label=spec.LABEL_LIKE))
        assert ledger.get("aaaaaaaaaa1").label == spec.LABEL_DISLIKE

    def test_merge_fills_in_a_missing_field(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", source="rss"))
        ledger.upsert(make_candidate("aaaaaaaaaa1", source="rss", score=0.5))
        assert ledger.get("aaaaaaaaaa1").score == pytest.approx(0.5)

    def test_merge_enriches_thin_metadata(self, ledger):
        ledger.upsert(
            Candidate(
                video_id="aaaaaaaaaa1",
                seen_at="2026-09-14T04:00:00+00:00",
                source="rss",
                metadata={"id": "aaaaaaaaaa1", "title": "From RSS"},
            )
        )
        ledger.upsert(
            Candidate(
                video_id="aaaaaaaaaa1",
                seen_at="2026-09-14T05:00:00+00:00",
                source="rss",
                metadata={"id": "aaaaaaaaaa1", "duration": 900, "view_count": 1000},
            )
        )
        row = ledger.get("aaaaaaaaaa1")
        assert row.duration == 900
        assert row.title == "From RSS"

    def test_overwrite_replaces_metadata_wholesale(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", title="Old"))
        ledger.upsert(
            Candidate(
                video_id="aaaaaaaaaa1",
                seen_at="2026-09-14T04:00:00+00:00",
                source="rss",
                metadata={"title": "New"},
            ),
            overwrite=True,
        )
        row = ledger.get("aaaaaaaaaa1")
        assert row.title == "New"
        assert "duration" not in row.metadata

    def test_overwrite_still_keeps_signal(self, established, make_candidate):
        # Even a deliberate re-import is not a reason to lose watch history.
        established.upsert(make_candidate("aaaaaaaaaa1", title="New"), overwrite=True)
        row = established.get("aaaaaaaaaa1")
        assert row.watch_pct == pytest.approx(0.97)
        assert row.label == spec.LABEL_LIKE


class TestUpsertMany:
    def test_counts_new_rows_only(self, ledger, make_candidate):
        batch = [make_candidate(f"vid{i:08d}") for i in range(5)]
        assert ledger.upsert_many(batch) == 5
        assert ledger.upsert_many(batch) == 0

    def test_is_atomic(self, ledger, make_candidate):
        # A bad row mid-batch must not leave the earlier rows committed and
        # the rest missing; discovery would silently lose a day's candidates.
        batch = [
            make_candidate("good00000001"),
            make_candidate("bad000000001", source="nonsense"),
            make_candidate("good00000002"),
        ]
        with pytest.raises(LedgerError):
            ledger.upsert_many(batch)
        assert ledger.count() == 0

    def test_handles_an_empty_batch(self, ledger):
        assert ledger.upsert_many([]) == 0

    def test_deduplicates_within_a_batch(self, ledger, make_candidate):
        assert ledger.upsert_many(
            [make_candidate("same00000001"), make_candidate("same00000001")]
        ) == 1


class TestSentimentTransitions:
    """`label` must move freely in every direction."""

    @pytest.fixture
    def row(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        return ledger

    @pytest.mark.parametrize(
        "sequence",
        [
            [spec.LABEL_LIKE, None],
            [spec.LABEL_LIKE, spec.LABEL_DISLIKE],
            [spec.LABEL_DISLIKE, spec.LABEL_LIKE],
            [spec.LABEL_DISLIKE, None],
            [None, spec.LABEL_LIKE, None, spec.LABEL_DISLIKE, None],
            [spec.LABEL_HISTORY, spec.LABEL_LIKE, spec.LABEL_HISTORY],
            [spec.LABEL_LIKE, spec.LABEL_LIKE],
        ],
        ids=[
            "unlike",
            "like-to-dislike",
            "dislike-to-like",
            "un-dislike",
            "full-cycle",
            "history-like-history",
            "idempotent-like",
        ],
    )
    def test_every_transition_is_legal(self, row, sequence):
        for label in sequence:
            row.set_label("aaaaaaaaaa1", label)
            assert row.get("aaaaaaaaaa1").label == label

    def test_liked_and_disliked_helpers(self, row):
        row.set_label("aaaaaaaaaa1", spec.LABEL_LIKE)
        assert row.get("aaaaaaaaaa1").liked
        assert not row.get("aaaaaaaaaa1").disliked
        row.set_label("aaaaaaaaaa1", spec.LABEL_DISLIKE)
        assert row.get("aaaaaaaaaa1").disliked
        assert not row.get("aaaaaaaaaa1").liked

    @pytest.mark.parametrize("label", [0, 3, -2, 99])
    def test_rejects_an_undefined_label(self, row, label):
        with pytest.raises(LedgerError):
            row.set_label("aaaaaaaaaa1", label)

    def test_unknown_video_raises(self, ledger):
        with pytest.raises(LedgerError):
            ledger.set_label("ghost000000", spec.LABEL_LIKE)

    def test_a_label_change_does_not_disturb_playback(self, row):
        row.record_watch("aaaaaaaaaa1", 0.8, play_count=2)
        row.set_label("aaaaaaaaaa1", spec.LABEL_DISLIKE)
        got = row.get("aaaaaaaaaa1")
        assert got.watch_pct == pytest.approx(0.8)
        assert got.play_count == 2


class TestFieldUpdates:
    def test_set_score(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        ledger.set_score("aaaaaaaaaa1", 0.42)
        assert ledger.get("aaaaaaaaaa1").score == pytest.approx(0.42)

    def test_set_score_on_an_unknown_id_raises(self, ledger):
        with pytest.raises(LedgerError):
            ledger.set_score("ghost000000", 0.5)

    @pytest.mark.parametrize("selection", sorted(spec.SELECTIONS))
    def test_mark_selected_accepts_every_arm(self, ledger, make_candidate, selection):
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        ledger.mark_selected("aaaaaaaaaa1", selection)
        row = ledger.get("aaaaaaaaaa1")
        assert (row.selected, row.selection) == (1, selection)

    @pytest.mark.parametrize("selection", ["manual", "", "MODEL", "explore"])
    def test_mark_selected_rejects_an_unknown_arm(self, ledger, make_candidate, selection):
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        with pytest.raises(LedgerError):
            ledger.mark_selected("aaaaaaaaaa1", selection)

    def test_mark_downloaded_records_the_path(self, ledger, make_candidate, tmp_path):
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        ledger.mark_downloaded("aaaaaaaaaa1", tmp_path / "a.mp4")
        row = ledger.get("aaaaaaaaaa1")
        assert row.downloaded == 1
        assert row.filepath == str(tmp_path / "a.mp4")

    def test_mark_downloaded_counts_the_offer(self, ledger, make_candidate):
        # A download is an offer; the re-offer policy reads both fields.
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        ledger.mark_downloaded("aaaaaaaaaa1", "/x.mp4")
        row = ledger.get("aaaaaaaaaa1")
        assert row.offer_count == 1
        assert row.last_offered_at

    def test_mark_downloaded_increments_across_re_offers(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        ledger.mark_downloaded("aaaaaaaaaa1", "/x.mp4", offered_at=days_ago(300))
        ledger.clear_filepath("aaaaaaaaaa1")
        ledger.mark_downloaded("aaaaaaaaaa1", "/x.mp4", offered_at=days_ago(1))
        row = ledger.get("aaaaaaaaaa1")
        assert row.offer_count == 2
        assert row.last_offered_at == days_ago(1)

    def test_mark_downloaded_stores_a_string_not_a_path(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        ledger.mark_downloaded("aaaaaaaaaa1", "/x/y.mp4")
        assert isinstance(ledger.get("aaaaaaaaaa1").filepath, str)

    def test_record_watch(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        ledger.record_watch("aaaaaaaaaa1", 0.55)
        row = ledger.get("aaaaaaaaaa1")
        assert (row.watched, row.watch_pct) == (1, pytest.approx(0.55))

    def test_record_watch_never_lowers(self, ledger, make_candidate):
        # Jellyfin resets the position after a rewatch; taking the lower
        # number would turn a finished video into a mid-band negative.
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        ledger.record_watch("aaaaaaaaaa1", 0.95)
        ledger.record_watch("aaaaaaaaaa1", 0.10)
        assert ledger.get("aaaaaaaaaa1").watch_pct == pytest.approx(0.95)

    def test_record_watch_raises_the_value(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        ledger.record_watch("aaaaaaaaaa1", 0.10)
        ledger.record_watch("aaaaaaaaaa1", 0.95)
        assert ledger.get("aaaaaaaaaa1").watch_pct == pytest.approx(0.95)

    def test_record_watch_tracks_play_count(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        ledger.record_watch("aaaaaaaaaa1", 0.95, play_count=3)
        assert ledger.get("aaaaaaaaaa1").play_count == 3

    def test_record_watch_never_lowers_play_count(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        ledger.record_watch("aaaaaaaaaa1", 0.95, play_count=5)
        ledger.record_watch("aaaaaaaaaa1", 0.95, play_count=1)
        assert ledger.get("aaaaaaaaaa1").play_count == 5

    def test_record_watch_accepts_none_for_unknown_runtime(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        ledger.record_watch("aaaaaaaaaa1", None, play_count=1)
        row = ledger.get("aaaaaaaaaa1")
        assert row.watch_pct is None
        assert row.play_count == 1

    @pytest.mark.parametrize("pct", [-0.5, 1.5, 2.0])
    def test_record_watch_rejects_out_of_range(self, ledger, make_candidate, pct):
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        with pytest.raises(LedgerError):
            ledger.record_watch("aaaaaaaaaa1", pct)

    def test_mark_saved(self, ledger, make_candidate, tmp_path):
        ledger.upsert(make_candidate("aaaaaaaaaa1", downloaded=1, filepath="/library/x.mp4"))
        ledger.mark_saved("aaaaaaaaaa1", tmp_path / "permanent" / "x.mp4")
        row = ledger.get("aaaaaaaaaa1")
        assert row.saved == 1
        assert row.filepath == str(tmp_path / "permanent" / "x.mp4")

    def test_clear_filepath_keeps_everything_else(self, ledger, make_candidate):
        ledger.upsert(
            make_candidate(
                "aaaaaaaaaa1",
                downloaded=1,
                filepath="/x.mp4",
                watched=1,
                watch_pct=0.95,
                play_count=2,
                offer_count=1,
                label=spec.LABEL_LIKE,
            )
        )
        ledger.clear_filepath("aaaaaaaaaa1")
        row = ledger.get("aaaaaaaaaa1")
        assert row.filepath is None
        assert row.watch_pct == pytest.approx(0.95)
        assert (row.watched, row.play_count, row.offer_count) == (1, 2, 1)
        assert row.label == spec.LABEL_LIKE

    def test_clear_filepath_keeps_downloaded_as_the_historical_record(
        self, ledger, make_candidate
    ):
        # `downloaded` means "we fetched this once", which is what makes the
        # row eligible for the repeat arm. `filepath` is what means "we have
        # it right now".
        ledger.upsert(make_candidate("aaaaaaaaaa1", downloaded=1, filepath="/x.mp4"))
        ledger.clear_filepath("aaaaaaaaaa1")
        assert ledger.get("aaaaaaaaaa1").downloaded == 1

    def test_clear_selections_resets_undownloaded_rows(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", selected=1, selection="model"))
        ledger.clear_selections()
        row = ledger.get("aaaaaaaaaa1")
        assert (row.selected, row.selection) == (0, None)

    def test_clear_selections_leaves_downloaded_rows_alone(self, ledger, make_candidate):
        # The selection is a permanent record of why this video is on disk;
        # it is what the arm comparison reads.
        ledger.upsert(
            make_candidate(
                "aaaaaaaaaa1", selected=1, selection="random", downloaded=1, filepath="/x"
            )
        )
        ledger.clear_selections()
        row = ledger.get("aaaaaaaaaa1")
        assert (row.selected, row.selection) == (1, "random")


class TestOnDisk:
    def test_a_downloaded_row_with_a_path_is_held(self, make_candidate):
        assert make_candidate(downloaded=1, filepath="/x.mp4").on_disk

    def test_a_reaped_row_is_not_held(self, make_candidate):
        assert not make_candidate(downloaded=1, filepath=None).on_disk

    def test_a_saved_row_is_held(self, make_candidate):
        # Saved videos live outside the managed library, so `downloaded` and
        # `filepath` alone would not cover them -- and re-downloading
        # something already saved is exactly the bug to avoid.
        assert make_candidate(saved=1, downloaded=0, filepath=None).on_disk

    def test_a_fresh_candidate_is_not_held(self, make_candidate):
        assert not make_candidate().on_disk


class TestHeldIds:
    def test_includes_downloaded_rows(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", downloaded=1, filepath="/x.mp4"))
        assert ledger.held_ids() == {"aaaaaaaaaa1"}

    def test_includes_saved_rows(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", saved=1, filepath="/permanent/x.mp4"))
        assert "aaaaaaaaaa1" in ledger.held_ids()

    def test_excludes_reaped_rows(self, ledger, make_candidate):
        # This is what makes the repeat arm possible, and it is why the
        # yt-dlp download archive had to go: the archive could not tell the
        # difference.
        ledger.upsert(make_candidate("aaaaaaaaaa1", downloaded=1, filepath=None))
        assert ledger.held_ids() == set()

    def test_excludes_never_downloaded_rows(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        assert ledger.held_ids() == set()

    def test_empty_on_an_empty_ledger(self, ledger):
        assert ledger.held_ids() == set()


class TestNewPool:
    def test_includes_never_downloaded_candidates(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", score=0.5))
        assert [c.video_id for c in ledger.new_pool()] == ["aaaaaaaaaa1"]

    def test_excludes_anything_downloaded_before(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", downloaded=1, filepath=None))
        assert ledger.new_pool() == []

    def test_excludes_disliked_candidates(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", label=spec.LABEL_DISLIKE))
        assert ledger.new_pool() == []

    def test_excludes_already_selected_candidates(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", selected=1, selection="model"))
        assert ledger.new_pool() == []

    def test_includes_liked_candidates_not_yet_fetched(self, ledger, make_candidate):
        # A Takeout like with nothing on disk is a perfectly good download
        # candidate.
        ledger.upsert(make_candidate("aaaaaaaaaa1", label=spec.LABEL_LIKE))
        assert [c.video_id for c in ledger.new_pool()] == ["aaaaaaaaaa1"]


class TestRepeatPool:
    @pytest.fixture
    def stocked(self, ledger, make_candidate):
        cases = {
            "eligible0001": dict(
                downloaded=1, filepath=None, offer_count=1, last_offered_at=days_ago(200)
            ),
            "cooling00001": dict(
                downloaded=1, filepath=None, offer_count=1, last_offered_at=days_ago(7)
            ),
            "watched00001": dict(
                downloaded=1,
                filepath=None,
                watched=1,
                watch_pct=0.9,
                offer_count=1,
                last_offered_at=days_ago(200),
            ),
            "liked0000001": dict(
                downloaded=1,
                filepath=None,
                label=spec.LABEL_LIKE,
                offer_count=1,
                last_offered_at=days_ago(200),
            ),
            "disliked0001": dict(
                downloaded=1,
                filepath=None,
                label=spec.LABEL_DISLIKE,
                offer_count=1,
                last_offered_at=days_ago(200),
            ),
            "saved0000001": dict(
                downloaded=1, filepath="/permanent/x.mp4", saved=1, offer_count=1,
                last_offered_at=days_ago(200),
            ),
            "onhand00001x": dict(
                downloaded=1, filepath="/library/x.mp4", offer_count=1,
                last_offered_at=days_ago(200),
            ),
            "exhausted001": dict(
                downloaded=1, filepath=None, offer_count=9, last_offered_at=days_ago(900)
            ),
            "fresh0000001": dict(),
        }
        for video_id, attrs in cases.items():
            ledger.upsert(make_candidate(video_id, **attrs))
        return ledger

    def test_includes_a_stale_unwatched_reaped_row(self, stocked):
        ids = {c.video_id for c in stocked.repeat_pool(now=NOW)}
        assert "eligible0001" in ids

    def test_respects_the_cooldown(self, stocked):
        # Ignored a week ago: leave it alone. Recency is the signal.
        ids = {c.video_id for c in stocked.repeat_pool(now=NOW, cooldown_days=90)}
        assert "cooling00001" not in ids

    def test_a_shorter_cooldown_admits_it(self, stocked):
        ids = {c.video_id for c in stocked.repeat_pool(now=NOW, cooldown_days=1)}
        assert "cooling00001" in ids

    @pytest.mark.parametrize(
        "video_id",
        ["watched00001", "liked0000001", "disliked0001", "saved0000001", "onhand00001x"],
    )
    def test_excludes_rows_that_are_done_or_still_held(self, stocked, video_id):
        assert video_id not in {c.video_id for c in stocked.repeat_pool(now=NOW)}

    def test_excludes_never_downloaded_rows(self, stocked):
        assert "fresh0000001" not in {c.video_id for c in stocked.repeat_pool(now=NOW)}

    def test_the_max_offers_cap_retires_a_persistent_snub(self, stocked):
        ids = {c.video_id for c in stocked.repeat_pool(now=NOW, max_offers=4)}
        assert "exhausted001" not in ids

    def test_zero_max_offers_disables_the_cap(self, stocked):
        ids = {c.video_id for c in stocked.repeat_pool(now=NOW, max_offers=0)}
        assert "exhausted001" in ids

    def test_a_row_with_no_last_offered_at_is_eligible(self, ledger, make_candidate):
        # Downloaded by an older build that did not stamp the field; treat it
        # as maximally stale rather than permanently ineligible.
        ledger.upsert(
            make_candidate("legacy00001", downloaded=1, filepath=None, offer_count=1)
        )
        assert "legacy00001" in {c.video_id for c in ledger.repeat_pool(now=NOW)}

    def test_empty_on_an_empty_ledger(self, ledger):
        assert ledger.repeat_pool(now=NOW) == []


class TestQueries:
    def test_get_returns_none_for_an_unknown_id(self, ledger):
        assert ledger.get("ghost000000") is None

    def test_has_video(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        assert ledger.has_video("aaaaaaaaaa1") is True
        assert ledger.has_video("ghost000000") is False

    def test_existing_ids(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        ledger.upsert(make_candidate("bbbbbbbbbb2"))
        assert ledger.existing_ids(["aaaaaaaaaa1", "bbbbbbbbbb2", "ghost000000"]) == {
            "aaaaaaaaaa1",
            "bbbbbbbbbb2",
        }

    def test_existing_ids_handles_an_empty_input(self, ledger):
        assert ledger.existing_ids([]) == set()

    def test_existing_ids_handles_a_large_input(self, ledger, make_candidate):
        # SQLite caps variables per statement at 999 by default; a year of
        # discovery will exceed that.
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        probe = [f"id{i:09d}" for i in range(2500)] + ["aaaaaaaaaa1"]
        assert ledger.existing_ids(probe) == {"aaaaaaaaaa1"}

    def test_unscored_finds_rows_without_a_score(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", score=None))
        ledger.upsert(make_candidate("bbbbbbbbbb2", score=0.5))
        assert [c.video_id for c in ledger.unscored()] == ["aaaaaaaaaa1"]

    def test_unscored_excludes_what_we_hold(self, ledger, make_candidate):
        ledger.upsert(
            make_candidate("aaaaaaaaaa1", score=None, downloaded=1, filepath="/x.mp4")
        )
        assert ledger.unscored() == []

    def test_unscored_treats_a_zero_score_as_scored(self, ledger, make_candidate):
        # 0.0 is a real cosine similarity, not a missing value.
        ledger.upsert(make_candidate("aaaaaaaaaa1", score=0.0))
        assert ledger.unscored() == []

    def test_pending_downloads(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", selected=1, selection="model"))
        ledger.upsert(
            make_candidate(
                "bbbbbbbbbb2", selected=1, selection="model", downloaded=1, filepath="/x"
            )
        )
        ledger.upsert(make_candidate("cccccccccc3"))
        assert [c.video_id for c in ledger.pending_downloads()] == ["aaaaaaaaaa1"]

    def test_pending_downloads_includes_a_reaped_re_offer(self, ledger, make_candidate):
        ledger.upsert(
            make_candidate(
                "aaaaaaaaaa1", selected=1, selection="repeat", downloaded=1, filepath=None
            )
        )
        assert [c.video_id for c in ledger.pending_downloads()] == ["aaaaaaaaaa1"]

    def test_pending_downloads_excludes_saved(self, ledger, make_candidate):
        ledger.upsert(
            make_candidate(
                "aaaaaaaaaa1", selected=1, selection="model", saved=1, filepath="/perm/x"
            )
        )
        assert ledger.pending_downloads() == []

    def test_downloaded_lists_what_is_in_the_managed_library(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", downloaded=1, filepath="/x.mp4"))
        ledger.upsert(make_candidate("bbbbbbbbbb2", downloaded=1, filepath=None))
        ledger.upsert(make_candidate("cccccccccc3", saved=1, filepath="/perm/x.mp4"))
        assert [c.video_id for c in ledger.downloaded()] == ["aaaaaaaaaa1"]

    def test_liked(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", label=spec.LABEL_LIKE))
        ledger.upsert(make_candidate("bbbbbbbbbb2", label=spec.LABEL_HISTORY))
        ledger.upsert(make_candidate("cccccccccc3", label=spec.LABEL_DISLIKE))
        assert [c.video_id for c in ledger.liked()] == ["aaaaaaaaaa1"]

    def test_saved_rows(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", saved=1, filepath="/perm/x.mp4"))
        ledger.upsert(make_candidate("bbbbbbbbbb2"))
        assert [c.video_id for c in ledger.saved_rows()] == ["aaaaaaaaaa1"]

    def test_training_rows_cover_every_signal_kind(self, ledger, make_candidate):
        ledger.upsert(make_candidate("liked0000001", label=spec.LABEL_LIKE))
        ledger.upsert(make_candidate("disliked0001", label=spec.LABEL_DISLIKE))
        ledger.upsert(make_candidate("watched00001", watch_pct=0.3, watched=1))
        ledger.upsert(make_candidate("saved0000001", saved=1, filepath="/perm/x"))
        ledger.upsert(
            make_candidate("ignored00001", downloaded=1, filepath=None, offer_count=1)
        )
        ledger.upsert(make_candidate("nosignal0001"))
        found = {c.video_id for c in ledger.training_rows()}
        assert "nosignal0001" not in found
        assert len(found) == 5

    def test_all_returns_everything(self, populated_ledger):
        assert len(populated_ledger.all()) == 12


class TestCount:
    def test_counts_everything_with_no_filter(self, populated_ledger):
        assert populated_ledger.count() == 12

    def test_filters_on_one_column(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", downloaded=1, filepath="/x"))
        ledger.upsert(make_candidate("bbbbbbbbbb2"))
        assert ledger.count(downloaded=1) == 1

    def test_filters_on_several_columns(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", selected=1, selection="model"))
        ledger.upsert(make_candidate("bbbbbbbbbb2", selected=1, selection="random"))
        assert ledger.count(selected=1, selection="model") == 1

    def test_filters_on_null(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", score=0.5))
        ledger.upsert(make_candidate("bbbbbbbbbb2", score=None))
        assert ledger.count(score=None) == 1

    def test_rejects_an_unknown_column(self, ledger):
        with pytest.raises(LedgerError):
            ledger.count(nonexistent=1)


class TestSeenSince:
    @pytest.fixture
    def dated(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", seen_at="2026-09-14T04:00:00+00:00"))
        ledger.upsert(make_candidate("bbbbbbbbbb2", seen_at="2026-09-13T04:00:00+00:00"))
        ledger.upsert(make_candidate("cccccccccc3", seen_at="2026-09-01T04:00:00+00:00"))
        return ledger

    def test_counts_rows_since_a_datetime(self, dated):
        assert dated.seen_since(dt.datetime(2026, 9, 13, 12, tzinfo=dt.timezone.utc)) == 1

    def test_counts_rows_since_an_iso_string(self, dated):
        assert dated.seen_since("2026-09-13T00:00:00+00:00") == 2

    def test_zero_when_nothing_is_recent(self, dated):
        assert dated.seen_since("2026-10-01T00:00:00+00:00") == 0

    def test_counts_everything_for_an_early_cutoff(self, dated):
        assert dated.seen_since("2020-01-01T00:00:00+00:00") == 3


class TestSelectionStats:
    @pytest.fixture
    def armed(self, ledger, make_candidate):
        for arm, pcts in {
            "model": [0.9, 0.8, 0.7],
            "repeat": [0.5, 0.5],
            "random": [0.3, 0.1],
        }.items():
            for i, pct in enumerate(pcts):
                ledger.upsert(
                    make_candidate(
                        f"{arm}{i:07d}",
                        selected=1,
                        selection=arm,
                        watched=1,
                        watch_pct=pct,
                        downloaded=1,
                        filepath=f"/{arm}{i}",
                    )
                )
        return ledger

    def test_reports_all_three_arms(self, armed):
        stats = armed.selection_stats()
        assert (stats.model_n, stats.repeat_n, stats.random_n) == (3, 2, 2)

    def test_computes_each_mean(self, armed):
        stats = armed.selection_stats()
        assert stats.model_mean_watch_pct == pytest.approx(0.8)
        assert stats.repeat_mean_watch_pct == pytest.approx(0.5)
        assert stats.random_mean_watch_pct == pytest.approx(0.2)

    def test_the_gap_compares_model_against_random(self, armed):
        # Deliberately not against the repeat arm: that answers a different
        # question and would muddy the model-vs-chance measurement.
        assert armed.selection_stats().gap == pytest.approx(0.6)

    def test_ignores_rows_with_no_feedback_yet(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", selected=1, selection="model"))
        stats = ledger.selection_stats()
        assert stats.model_n == 0
        assert stats.model_mean_watch_pct is None

    def test_gap_is_none_when_an_arm_is_empty(self, ledger, make_candidate):
        ledger.upsert(
            make_candidate(
                "aaaaaaaaaa1", selected=1, selection="model", watched=1, watch_pct=0.9
            )
        )
        assert ledger.selection_stats().gap is None

    def test_respects_a_since_cutoff(self, ledger, make_candidate):
        ledger.upsert(
            make_candidate(
                "old00000001",
                selected=1,
                selection="model",
                watched=1,
                watch_pct=0.9,
                seen_at="2026-01-01T00:00:00+00:00",
            )
        )
        ledger.upsert(
            make_candidate(
                "new00000001",
                selected=1,
                selection="model",
                watched=1,
                watch_pct=0.5,
                seen_at="2026-09-14T00:00:00+00:00",
            )
        )
        stats = ledger.selection_stats(since="2026-09-01T00:00:00+00:00")
        assert stats.model_n == 1
        assert stats.model_mean_watch_pct == pytest.approx(0.5)


class TestRowDecoding:
    def test_bad_metadata_json_degrades_to_empty(self, ledger, raw_conn):
        # A truncated write should cost one row's metadata, not every query.
        raw_conn.execute(
            "INSERT INTO candidates (video_id, seen_at, metadata) "
            "VALUES ('broken', '2026-09-14T00:00:00+00:00', '{not json')"
        )
        raw_conn.commit()
        assert ledger.get("broken").metadata == {}

    def test_null_metadata_decodes_to_empty(self, ledger, raw_conn):
        raw_conn.execute(
            "INSERT INTO candidates (video_id, seen_at) "
            "VALUES ('nometa', '2026-09-14T00:00:00+00:00')"
        )
        raw_conn.commit()
        assert ledger.get("nometa").metadata == {}


class TestCandidateAccessors:
    def test_reads_fields_from_metadata(self, make_candidate):
        c = make_candidate("aaaaaaaaaa1", title="T", channel="C", duration=600)
        assert (c.title, c.channel, c.duration) == ("T", "C", 600.0)

    def test_falls_back_from_channel_to_uploader(self):
        c = Candidate(
            video_id="x", seen_at="2026-09-14T00:00:00+00:00", metadata={"uploader": "U"}
        )
        assert c.channel == "U"

    def test_missing_fields_are_empty_not_none(self):
        c = Candidate(video_id="x", seen_at="2026-09-14T00:00:00+00:00")
        assert c.title == ""
        assert c.channel == ""
        assert c.duration is None
        assert c.view_count is None
        assert c.upload_date is None

    def test_numeric_strings_are_coerced(self):
        c = Candidate(
            video_id="x",
            seen_at="2026-09-14T00:00:00+00:00",
            metadata={"duration": "900", "view_count": "1234"},
        )
        assert c.duration == 900.0
        assert c.view_count == 1234


class TestConfigHelpers:
    def test_open_for_uses_the_configured_path(self, config):
        with ledger_mod.open_for(config) as led:
            assert config.paths.ledger.exists()
            assert led.count() == 0

    def test_path_for(self, config):
        assert ledger_mod.path_for(config) == config.paths.ledger
