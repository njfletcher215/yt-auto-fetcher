"""The feedback stage: playback and sentiment, both from Jellyfin.

This is the only stage producing the gold label, and the one where a silent
bug stays invisible for weeks -- the pipeline keeps running, the model keeps
retraining, and it is learning from numbers that mean nothing.

The favorite sync is asymmetric on purpose, and that asymmetry gets its own
tests: an un-hearted item clears a like but never a dislike, because no
Jellyfin client has a dislike control and "not favorited" therefore cannot be
read as "not disliked".
"""

from __future__ import annotations

import pytest

from ytaf import feedback, spec
from ytaf.config import FeedbackConfig, JellyfinConfig
from ytaf.errors import JellyfinError
from ytaf.jellyfin import TICKS_PER_SECOND, JellyfinClient, PlaybackItem


@pytest.fixture
def jf_config(config):
    return config.with_(
        jellyfin=JellyfinConfig(
            base_url="https://jellyfin.test", api_key="K", user_id="user-1"
        )
    )


@pytest.fixture
def downloaded(ledger, make_candidate):
    """Rows whose filepaths match the Jellyfin items fixture."""
    rows = {
        "aircAruvnKk": "/library/3Blue1Brown/20260914 - But what is a neural network [aircAruvnKk].mp4",
        "searchHit01": "/library/Computerphile/20260913 - How Computers Compress Text [searchHit01].mp4",
        "searchHit05": "/library/Some Streamer/20260911 - Post-Stream VOD [searchHit05].mp4",
        "favNoPlay01": "/library/3Blue1Brown/20260910 - Hearted But Never Opened [favNoPlay01].mp4",
    }
    for video_id, path in rows.items():
        ledger.upsert(
            make_candidate(
                video_id, selected=1, selection="model", downloaded=1, filepath=path
            )
        )
    return ledger


class TestWatchPct:
    def test_computes_the_ratio(self):
        assert feedback.watch_pct(50, 100, play_count=1) == pytest.approx(0.5)

    def test_played_forces_one(self):
        # Jellyfin zeroes the position once an item is marked played; reading
        # it literally would make every finished video a negative.
        assert feedback.watch_pct(0, 100, played=True, play_count=1) == pytest.approx(1.0)

    def test_a_zero_position_with_plays_is_a_quick_exit(self):
        # Opened and closed below Jellyfin's threshold for storing a resume
        # point -- a real signal, just a weak one.
        assert feedback.watch_pct(0, 100, played=False, play_count=1) == pytest.approx(0.0)

    def test_never_opened_is_none(self):
        assert feedback.watch_pct(0, 100, played=False, play_count=0) is None

    def test_clamps_to_one(self):
        assert feedback.watch_pct(150, 100, play_count=1) == pytest.approx(1.0)

    def test_clamps_to_zero(self):
        assert feedback.watch_pct(-10, 100, play_count=1) == pytest.approx(0.0)

    @pytest.mark.parametrize("runtime", [0, -1])
    def test_nonpositive_runtime_is_none(self, runtime):
        assert feedback.watch_pct(50, runtime, play_count=1) is None

    def test_works_in_real_ticks(self):
        assert feedback.watch_pct(
            90 * TICKS_PER_SECOND, 100 * TICKS_PER_SECOND, play_count=1
        ) == pytest.approx(0.9)


class TestFilepathIndex:
    def test_maps_filepath_to_video_id(self, downloaded):
        index = feedback.filepath_index(downloaded)
        assert set(index.values()) == {
            "aircAruvnKk",
            "searchHit01",
            "searchHit05",
            "favNoPlay01",
        }

    def test_omits_rows_with_no_filepath(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", downloaded=1, filepath=None))
        assert feedback.filepath_index(ledger) == {}

    def test_includes_saved_rows(self, ledger, make_candidate):
        # A saved video still gets watched, and its playback is still signal.
        ledger.upsert(make_candidate("aaaaaaaaaa1", saved=1, filepath="/perm/C/x.mp4"))
        assert "aaaaaaaaaa1" in feedback.filepath_index(ledger).values()

    def test_normalises_redundant_path_segments(self, ledger, make_candidate):
        ledger.upsert(
            make_candidate("aaaaaaaaaa1", downloaded=1, filepath="/library/./C/x.mp4")
        )
        assert "/library/C/x.mp4" in feedback.filepath_index(ledger)


class TestApplyPlayback:
    def test_writes_watch_pct(self, downloaded):
        item = PlaybackItem(
            item_id="jf-1",
            position_ticks=90 * TICKS_PER_SECOND,
            runtime_ticks=100 * TICKS_PER_SECOND,
            play_count=1,
        )
        report = feedback.FeedbackReport()
        feedback.apply_playback(downloaded, {"aircAruvnKk": item}, report)
        assert downloaded.get("aircAruvnKk").watch_pct == pytest.approx(0.9)
        assert report.playback_updated == 1

    def test_writes_play_count(self, downloaded):
        item = PlaybackItem(
            item_id="jf-1",
            position_ticks=0,
            runtime_ticks=100 * TICKS_PER_SECOND,
            played=True,
            play_count=4,
        )
        feedback.apply_playback(downloaded, {"aircAruvnKk": item}, feedback.FeedbackReport())
        assert downloaded.get("aircAruvnKk").play_count == 4

    def test_marks_watched(self, downloaded):
        item = PlaybackItem(
            item_id="jf-1",
            position_ticks=10 * TICKS_PER_SECOND,
            runtime_ticks=100 * TICKS_PER_SECOND,
            play_count=1,
        )
        feedback.apply_playback(downloaded, {"aircAruvnKk": item}, feedback.FeedbackReport())
        assert downloaded.get("aircAruvnKk").watched == 1

    def test_never_lowers_an_existing_value(self, downloaded):
        downloaded.record_watch("aircAruvnKk", 0.95)
        item = PlaybackItem(
            item_id="jf-1",
            position_ticks=0,
            runtime_ticks=100 * TICKS_PER_SECOND,
            play_count=1,
        )
        feedback.apply_playback(downloaded, {"aircAruvnKk": item}, feedback.FeedbackReport())
        assert downloaded.get("aircAruvnKk").watch_pct == pytest.approx(0.95)

    def test_skips_items_with_no_usable_runtime(self, downloaded):
        item = PlaybackItem(item_id="jf-1", position_ticks=10, runtime_ticks=0, play_count=1)
        report = feedback.FeedbackReport()
        feedback.apply_playback(downloaded, {"aircAruvnKk": item}, report)
        assert downloaded.get("aircAruvnKk").watch_pct is None
        assert report.playback_updated == 0

    def test_ignores_an_id_that_is_not_in_the_ledger(self, downloaded):
        item = PlaybackItem(
            item_id="jf-9", position_ticks=50, runtime_ticks=100, play_count=1
        )
        report = feedback.FeedbackReport()
        feedback.apply_playback(downloaded, {"notInLedger": item}, report)
        assert report.playback_updated == 0


class TestSyncFavorites:
    def test_a_heart_becomes_a_like(self, downloaded):
        report = feedback.FeedbackReport()
        feedback.sync_favorites(
            downloaded, {"aircAruvnKk"}, {"aircAruvnKk"}, report
        )
        assert downloaded.get("aircAruvnKk").label == spec.LABEL_LIKE
        assert report.likes_added == 1

    def test_un_hearting_clears_a_like(self, downloaded):
        downloaded.set_label("aircAruvnKk", spec.LABEL_LIKE)
        report = feedback.FeedbackReport()
        feedback.sync_favorites(downloaded, set(), {"aircAruvnKk"}, report)
        assert downloaded.get("aircAruvnKk").label is None
        assert report.likes_cleared == 1

    def test_un_hearting_never_clears_a_dislike(self, downloaded):
        # There is no dislike control in any Jellyfin client, so an
        # un-hearted item says nothing about a dislike. Clearing one here
        # would make `ytaf dislike` undo itself on the next feedback run.
        downloaded.set_label("aircAruvnKk", spec.LABEL_DISLIKE)
        feedback.sync_favorites(
            downloaded, set(), {"aircAruvnKk"}, feedback.FeedbackReport()
        )
        assert downloaded.get("aircAruvnKk").label == spec.LABEL_DISLIKE

    def test_hearting_a_disliked_video_promotes_it_to_a_like(self, downloaded):
        # An affirmative act beats a prior one; this is how you change your
        # mind from the phone.
        downloaded.set_label("aircAruvnKk", spec.LABEL_DISLIKE)
        feedback.sync_favorites(
            downloaded, {"aircAruvnKk"}, {"aircAruvnKk"}, feedback.FeedbackReport()
        )
        assert downloaded.get("aircAruvnKk").label == spec.LABEL_LIKE

    def test_items_not_seen_are_left_alone(self, downloaded):
        # We cannot distinguish "un-hearted" from "not in this response".
        downloaded.set_label("searchHit01", spec.LABEL_LIKE)
        feedback.sync_favorites(
            downloaded, set(), {"aircAruvnKk"}, feedback.FeedbackReport()
        )
        assert downloaded.get("searchHit01").label == spec.LABEL_LIKE

    def test_a_history_label_is_restored_rather_than_cleared(self, downloaded):
        # Un-liking something from your Takeout history returns it to
        # history, not to nothing.
        downloaded.set_label("aircAruvnKk", spec.LABEL_HISTORY)
        downloaded.set_label("aircAruvnKk", spec.LABEL_LIKE)
        feedback.sync_favorites(
            downloaded, set(), {"aircAruvnKk"}, feedback.FeedbackReport()
        )
        assert downloaded.get("aircAruvnKk").label in (None, spec.LABEL_HISTORY)

    def test_is_idempotent(self, downloaded):
        report = feedback.FeedbackReport()
        feedback.sync_favorites(downloaded, {"aircAruvnKk"}, {"aircAruvnKk"}, report)
        feedback.sync_favorites(downloaded, {"aircAruvnKk"}, {"aircAruvnKk"}, report)
        assert downloaded.get("aircAruvnKk").label == spec.LABEL_LIKE
        assert report.likes_added == 1

    def test_can_be_disabled(self, jf_config, downloaded, fake_http, fixture_json):
        cfg = jf_config.with_(feedback=FeedbackConfig(sync_favorites=False))
        fake_http.route("/Items", json_body=fixture_json("jellyfin-items.json"))
        fake_http.route("/Sessions", json_body=[])
        feedback.run(cfg, ledger=downloaded)
        assert downloaded.get("searchHit01").label is None


class TestRun:
    @pytest.fixture
    def wired(self, fake_http, fixture_json):
        fake_http.route("/Items", json_body=fixture_json("jellyfin-items.json"))
        fake_http.route("/Sessions", json_body=[])
        return fake_http

    def test_backfills_playback(self, jf_config, downloaded, wired):
        report = feedback.run(jf_config, ledger=downloaded)
        assert report.playback_updated >= 3

    def test_computes_the_right_percentages(self, jf_config, downloaded, wired):
        feedback.run(jf_config, ledger=downloaded)
        assert downloaded.get("aircAruvnKk").watch_pct == pytest.approx(0.95)
        assert downloaded.get("searchHit01").watch_pct == pytest.approx(1.0)
        assert downloaded.get("searchHit05").watch_pct == pytest.approx(0.30)

    def test_a_played_item_scores_one_despite_a_zero_position(
        self, jf_config, downloaded, wired
    ):
        assert downloaded.get("searchHit01").watch_pct is None
        feedback.run(jf_config, ledger=downloaded)
        assert downloaded.get("searchHit01").watch_pct == pytest.approx(1.0)

    def test_records_play_count(self, jf_config, downloaded, wired):
        feedback.run(jf_config, ledger=downloaded)
        assert downloaded.get("searchHit01").play_count == 3

    def test_a_never_opened_item_gets_no_watch_pct(self, jf_config, downloaded, wired):
        feedback.run(jf_config, ledger=downloaded)
        assert downloaded.get("favNoPlay01").watch_pct is None

    def test_syncs_hearts_into_labels(self, jf_config, downloaded, wired):
        feedback.run(jf_config, ledger=downloaded)
        assert downloaded.get("searchHit01").label == spec.LABEL_LIKE

    def test_hearts_an_unplayed_item(self, jf_config, downloaded, wired):
        # You can favorite something without playing it, and that still means
        # you like it.
        feedback.run(jf_config, ledger=downloaded)
        assert downloaded.get("favNoPlay01").label == spec.LABEL_LIKE

    def test_leaves_unhearted_items_unlabelled(self, jf_config, downloaded, wired):
        feedback.run(jf_config, ledger=downloaded)
        assert downloaded.get("aircAruvnKk").label is None

    def test_ignores_library_items_this_pipeline_did_not_create(
        self, jf_config, downloaded, wired
    ):
        report = feedback.run(jf_config, ledger=downloaded)
        assert report.unmatched

    def test_records_transcoding_sessions(
        self, jf_config, downloaded, fake_http, fixture_json
    ):
        fake_http.route("/Items", json_body=fixture_json("jellyfin-items.json"))
        fake_http.route("/Sessions", json_body=fixture_json("jellyfin-sessions.json"))
        assert feedback.run(jf_config, ledger=downloaded).transcoding

    def test_no_transcoding_when_everything_direct_plays(self, jf_config, downloaded, wired):
        assert feedback.run(jf_config, ledger=downloaded).transcoding == []

    def test_is_idempotent(self, jf_config, downloaded, wired):
        feedback.run(jf_config, ledger=downloaded)
        before = {
            c.video_id: (c.watch_pct, c.play_count, c.label) for c in downloaded.all()
        }
        feedback.run(jf_config, ledger=downloaded)
        after = {
            c.video_id: (c.watch_pct, c.play_count, c.label) for c in downloaded.all()
        }
        assert before == after

    def test_passes_the_since_window_through(self, jf_config, downloaded, wired):
        report = feedback.run(
            jf_config, ledger=downloaded, since="2026-09-14T00:00:00+00:00"
        )
        # searchHit05 was last played 09-13 and falls outside the window.
        assert downloaded.get("searchHit05").watch_pct is None
        assert report.playback_updated >= 1

    def test_does_nothing_without_jellyfin_configured(self, config, downloaded, fake_http):
        report = feedback.run(config, ledger=downloaded)
        assert report.updated == 0
        assert report.notes
        assert fake_http.calls == []

    def test_a_server_error_does_not_corrupt_the_ledger(
        self, jf_config, downloaded, fake_http
    ):
        fake_http.route("/Items", status=500, body="boom")
        fake_http.route("/Sessions", json_body=[])
        with pytest.raises(JellyfinError):
            feedback.run(jf_config, ledger=downloaded)
        assert all(c.watch_pct is None for c in downloaded.all())

    def test_accepts_an_injected_client(self, jf_config, downloaded, wired):
        client = JellyfinClient(jf_config.jellyfin)
        assert feedback.run(jf_config, ledger=downloaded, client=client).updated > 0
