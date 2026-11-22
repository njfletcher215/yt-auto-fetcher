"""The whole chain, with fakes at the two seams.

discover → score → download → (watch) → feedback → retrain → retain, on a
scratch root, asserting the state transitions each stage owns. These are the
tests that catch a stage which works perfectly in isolation and hands the next
one something it cannot use.

The repeat arm gets a full end-to-end case here, because it is the one
behaviour that spans every stage: download records the offer, retention reaps
it unwatched, and scoring offers it again once the cooldown lapses.
"""

from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path

import pytest

from ytaf import discover, download, feedback, retention, score, spec, training, ytdlp
from ytaf.config import (
    Config,
    DiscoveryConfig,
    JellyfinConfig,
    ModelConfig,
    Paths,
    RetentionConfig,
    SelectionConfig,
)
from ytaf.jellyfin import TICKS_PER_SECOND
from ytaf.ledger import Ledger

CHANNELS = ["3Blue1Brown", "Computerphile", "Veritasium", "Some Streamer"]
# Relative to the real clock, not a hardcoded calendar date, so this stays
# "later" no matter when the suite actually runs.
LATER = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=365)


def build_feed(channel, channel_id, video_ids, *, views=10_000):
    entries = "".join(
        f"""
  <entry>
    <id>yt:video:{vid}</id>
    <yt:videoId>{vid}</yt:videoId>
    <yt:channelId>{channel_id}</yt:channelId>
    <title>Video {vid} about mathematics</title>
    <author><name>{channel}</name></author>
    <published>2026-09-14T15:00:00+00:00</published>
    <media:group>
      <media:description>A description of video {vid}.</media:description>
      <media:community>
        <media:statistics views="{views}"/>
      </media:community>
    </media:group>
  </entry>"""
        for vid in video_ids
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns:yt="http://www.youtube.com/xml/schemas/2015"
      xmlns:media="http://search.yahoo.com/mrss/"
      xmlns="http://www.w3.org/2005/Atom">
  <yt:channelId>{channel_id}</yt:channelId>
  <title>{channel}</title>
  <author><name>{channel}</name></author>
{entries}
</feed>
"""


@pytest.fixture
def pipeline(tmp_path, fake_http, fake_ytdlp):
    """A configured pipeline with 20 discoverable videos across 4 channels."""
    cfg = Config.for_root(
        tmp_path / "youtube",
        discovery=DiscoveryConfig(
            strategy="rss",
            channel_ids=tuple(f"UCchannel{i:014d}" for i in range(4)),
        ),
        model=ModelConfig(embedder="hashing", embedding_dim=64),
        selection=SelectionConfig(),
    )
    cfg.paths.ensure()

    videos: dict[str, dict] = {}
    for c, channel in enumerate(CHANNELS):
        channel_id = f"UCchannel{c:014d}"
        ids = [f"c{c}vid{i:06d}" for i in range(5)]  # 11 chars, like a real YouTube id
        for vid in ids:
            videos[vid] = {
                "id": vid,
                "title": f"Video {vid} about mathematics",
                "description": f"A description of video {vid}.",
                "channel": channel,
                "channel_id": channel_id,
                "duration": 900,
                "view_count": 10_000,
                "like_count": 500,
                "upload_date": "20260914",
                "live_status": "not_live",
                "ext": "mp4",
                "vcodec": "avc1.640028",
                "acodec": "mp4a.40.2",
                "height": 1080,
            }
        fake_http.route(
            f"channel_id={channel_id}", body=build_feed(channel, channel_id, ids)
        )

    def handler(args, **kwargs):
        joined = " ".join(args)
        if "-o" in args:  # a download
            video_id = args[-1]
            info = videos[video_id]
            channel_dir = cfg.paths.library / info["channel"]
            channel_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{info['upload_date']} - {info['title']} [{video_id}]"
            (channel_dir / f"{stem}.mp4").write_bytes(b"\0" * 4096)
            (channel_dir / f"{stem}.info.json").write_text(
                json.dumps(info), encoding="utf-8"
            )
            (channel_dir / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff")
            return ytdlp.YtdlpResult(args=tuple(args), returncode=0)
        if "ytsearchdate" in joined:
            return ytdlp.YtdlpResult(
                args=tuple(args), returncode=0, stdout='{"_type":"playlist","entries":[]}'
            )
        # metadata enrichment
        wanted = [a for a in args if a in videos]
        payload = "\n".join(json.dumps(videos[v]) for v in wanted)
        return ytdlp.YtdlpResult(args=tuple(args), returncode=0, stdout=payload)

    fake_ytdlp.handler = handler
    return cfg, videos


@pytest.fixture
def day_one(pipeline):
    """discover → score → download, once."""
    cfg, videos = pipeline
    led = Ledger.open(cfg.paths.ledger)
    discover.run(cfg, ledger=led)
    score.run(cfg, ledger=led, rng=random.Random(1))
    download.run(cfg, ledger=led)
    yield cfg, led, videos
    led.close()


class TestDiscoverToScore:
    def test_discovery_fills_the_scoring_queue(self, pipeline):
        cfg, _ = pipeline
        with Ledger.open(cfg.paths.ledger) as led:
            discover.run(cfg, ledger=led)
            assert led.count() == 20
            assert len(led.unscored()) == 20

    def test_cold_start_scoring_leaves_the_queue_unscored_and_explores_only(self, pipeline):
        cfg, _ = pipeline
        with Ledger.open(cfg.paths.ledger) as led:
            discover.run(cfg, ledger=led)
            score.run(cfg, ledger=led, rng=random.Random(1))
            # No model exists yet, so nothing gets a real score -- the queue
            # stays unscored until there is a model to score it with.
            assert len(led.unscored()) == 20
            # Pure exploration: no model picks, no repeats (nothing reaped
            # yet), just the mandatory random arm.
            assert len(led.pending_downloads()) == 3

    def test_scoring_marks_zero_model_and_three_random(self, pipeline):
        cfg, _ = pipeline
        with Ledger.open(cfg.paths.ledger) as led:
            discover.run(cfg, ledger=led)
            score.run(cfg, ledger=led, rng=random.Random(1))
            assert led.count(selection="model") == 0
            assert led.count(selection="random") == 3
            assert led.count(selection="repeat") == 0


class TestFullChain:
    def test_runs_end_to_end(self, day_one):
        cfg, led, _ = day_one
        # Cold start: no model yet, so day one is random-exploration only.
        assert led.count(downloaded=1) == 3

    def test_every_downloaded_file_exists_with_its_info_json(self, day_one):
        cfg, led, _ = day_one
        for row in led.downloaded():
            video = Path(row.filepath)
            assert video.exists()
            assert (video.parent / f"{video.stem}.info.json").exists()

    def test_the_library_is_one_folder_per_channel(self, day_one):
        cfg, led, _ = day_one
        videos = list(cfg.paths.library.rglob("*.mp4"))
        assert videos
        for video in videos:
            # channel/file.mp4 -- no season directory in a Home Videos layout.
            assert video.parent.parent == cfg.paths.library

    def test_every_filename_carries_its_video_id(self, day_one):
        from ytaf import naming

        cfg, led, _ = day_one
        for row in led.downloaded():
            assert naming.video_id_from_path(row.filepath) == row.video_id

    def test_every_download_recorded_an_offer(self, day_one):
        cfg, led, _ = day_one
        for row in led.downloaded():
            assert row.offer_count == 1
            assert row.last_offered_at

    def test_nothing_is_written_outside_the_root(self, day_one):
        cfg, led, _ = day_one
        for row in led.all():
            if row.filepath:
                assert row.filepath.startswith(str(cfg.paths.root))

    def test_no_authentication_is_used_anywhere(self, day_one, fake_ytdlp, fake_http):
        for call in fake_ytdlp.calls:
            ytdlp.assert_no_auth_flags(call)
        for _, _, kwargs in fake_http.calls:
            headers = kwargs.get("headers") or {}
            assert not any(k.lower() in ("cookie", "authorization") for k in headers)

    def test_every_download_used_the_h264_selector(self, day_one, fake_ytdlp):
        downloads = [c for c in fake_ytdlp.calls if "-o" in c]
        assert len(downloads) == 3
        for call in downloads:
            assert call[call.index("-f") + 1] == spec.FORMAT_SELECTOR

    def test_no_download_archive_was_used(self, day_one, fake_ytdlp):
        assert not fake_ytdlp.invoked_with("--download-archive")


class TestIdempotency:
    def test_successive_runs_work_through_the_pool(self, pipeline):
        cfg, _ = pipeline
        with Ledger.open(cfg.paths.ledger) as led:
            for _ in range(2):
                discover.run(cfg, ledger=led)
                score.run(cfg, ledger=led, rng=random.Random(1))
                download.run(cfg, ledger=led)
            # Cold start: 3 random picks per pass, no model ever trained here.
            assert led.count(downloaded=1) == 6

    def test_eventually_there_is_nothing_left_to_do(self, pipeline):
        cfg, _ = pipeline
        with Ledger.open(cfg.paths.ledger) as led:
            # 3 random picks per pass against a pool of 20 (no model is ever
            # trained in this test, so every pass is pure exploration):
            # 3,3,3,3,3,3,2,0 -- 8 passes to fully drain the pool and see a
            # pass with nothing left to attempt.
            for _ in range(8):
                discover.run(cfg, ledger=led)
                score.run(cfg, ledger=led, rng=random.Random(1))
                report = download.run(cfg, ledger=led)
            assert report.attempted == 0
            assert led.count(downloaded=1) == 20

    def test_no_video_is_downloaded_twice_in_one_pass(self, pipeline, fake_ytdlp):
        cfg, _ = pipeline
        with Ledger.open(cfg.paths.ledger) as led:
            for _ in range(3):
                discover.run(cfg, ledger=led)
                score.run(cfg, ledger=led, rng=random.Random(1))
                download.run(cfg, ledger=led)
        fetched = [c[-1] for c in fake_ytdlp.calls if "-o" in c]
        assert len(fetched) == len(set(fetched))

    def test_rediscovery_does_not_disturb_downloaded_rows(self, day_one):
        cfg, led, _ = day_one
        before = {
            r.video_id: (r.filepath, r.offer_count, r.selection) for r in led.downloaded()
        }
        discover.run(cfg, ledger=led)
        after = {
            r.video_id: (r.filepath, r.offer_count, r.selection) for r in led.downloaded()
        }
        assert before == after


class TestFeedbackToRetrain:
    @pytest.fixture
    def watched(self, day_one, fake_http):
        """Report Jellyfin playback for everything downloaded on day one."""
        cfg, led, _ = day_one
        cfg = cfg.with_(
            jellyfin=JellyfinConfig(
                base_url="https://jellyfin.test", api_key="K", user_id="u"
            )
        )
        items = []
        for index, row in enumerate(led.downloaded()):
            # Alternate finished and abandoned mid-video, so training gets
            # both classes and the strong negative band is exercised.
            pct = 0.95 if index % 2 == 0 else 0.35
            items.append(
                {
                    "Id": f"jf-{index}",
                    "Name": row.title,
                    "Path": row.filepath,
                    "RunTimeTicks": 900 * TICKS_PER_SECOND,
                    "Type": "Video",
                    "UserData": {
                        "PlaybackPositionTicks": int(900 * pct * TICKS_PER_SECOND),
                        "PlayCount": 1,
                        "Played": False,
                        "IsFavorite": index == 0,
                        "LastPlayedDate": "2026-09-15T20:00:00.0000000Z",
                    },
                }
            )
        fake_http.route(
            "/Items", json_body={"Items": items, "TotalRecordCount": len(items)}
        )
        fake_http.route("/Sessions", json_body=[])
        fake_http.route("/Library/Refresh", status=204)
        return cfg, led

    def test_feedback_labels_every_downloaded_video(self, watched):
        cfg, led = watched
        feedback.run(cfg, ledger=led)
        assert all(r.watch_pct is not None for r in led.downloaded())

    def test_feedback_syncs_a_heart_into_a_like(self, watched):
        cfg, led = watched
        feedback.run(cfg, ledger=led)
        assert any(r.liked for r in led.downloaded())

    def test_feedback_produces_both_classes(self, watched):
        cfg, led = watched
        feedback.run(cfg, ledger=led)
        signals = [training.derive_signal(r) for r in led.downloaded()]
        assert any(s and s.positive for s in signals)
        assert any(s and not s.positive for s in signals)

    def test_a_mid_video_bail_is_the_strongest_negative(self, watched):
        cfg, led = watched
        feedback.run(cfg, ledger=led)
        negatives = [
            training.derive_signal(r)
            for r in led.downloaded()
            if (s := training.derive_signal(r)) and not s.positive
        ]
        assert negatives
        assert all(s.reason == training.PARTIAL for s in negatives)

    def test_retrain_consumes_the_feedback(self, watched):
        cfg, led = watched
        feedback.run(cfg, ledger=led)
        training.retrain(cfg, ledger=led)
        assert cfg.paths.centroid_path.exists()

    def test_the_retrained_model_scores_the_next_days_candidates(self, watched):
        cfg, led = watched
        feedback.run(cfg, ledger=led)
        training.retrain(cfg, ledger=led)
        led.clear_selections()
        assert score.run(cfg, ledger=led, rng=random.Random(2)).model_kind == "centroid"

    def test_selection_stats_can_compare_model_against_random(self, watched):
        cfg, led = watched
        feedback.run(cfg, ledger=led)
        # Day one was cold-start (random only); retrain on its feedback so a
        # second day actually produces model-arm picks to compare against.
        training.retrain(cfg, ledger=led)
        led.clear_selections()
        discover.run(cfg, ledger=led)
        score.run(cfg, ledger=led, rng=random.Random(5))
        download.run(cfg, ledger=led)
        for row in led.downloaded():
            if row.watch_pct is None:
                led.record_watch(row.video_id, 0.8, play_count=1)
        stats = led.selection_stats()
        assert stats.model_n > 0
        assert stats.random_n > 0


class TestRetentionAndTheRepeatArm:
    def test_reaps_everything_old_regardless_of_watch_state(self, day_one):
        cfg, led, _ = day_one
        for index, row in enumerate(led.downloaded()):
            if index % 2 == 0:
                led.record_watch(row.video_id, 0.95, play_count=1)
        held = led.count(downloaded=1)
        report = retention.run(cfg, ledger=led, now=LATER)
        assert len(report.reaped) == held

    def test_reaps_a_liked_video_too(self, day_one):
        cfg, led, _ = day_one
        keeper = led.downloaded()[0].video_id
        led.set_label(keeper, spec.LABEL_LIKE)
        report = retention.run(cfg, ledger=led, now=LATER)
        assert keeper in {r.video_id for r in report.reaped}

    def test_the_ledger_keeps_every_row(self, day_one):
        cfg, led, _ = day_one
        for row in led.downloaded():
            led.record_watch(row.video_id, 0.99, play_count=1)
        before = led.count()
        retention.run(cfg, ledger=led, now=LATER)
        assert led.count() == before
        assert all(r.watch_pct is not None for r in led.all() if r.watched)

    def test_a_reaped_watched_video_is_never_downloaded_again(self, day_one):
        cfg, led, _ = day_one
        for row in led.downloaded():
            led.record_watch(row.video_id, 0.99, play_count=1)
        reaped = {r.video_id for r in retention.run(cfg, ledger=led, now=LATER).reaped}
        assert reaped
        discover.run(cfg, ledger=led)
        score.run(cfg, ledger=led, rng=random.Random(3))
        assert {c.video_id for c in led.pending_downloads()}.isdisjoint(reaped)

    def test_a_reaped_unwatched_video_returns_once_the_cooldown_lapses(self, day_one):
        # The repeat arm end to end -- the behaviour a download archive would
        # have made impossible.
        cfg, led, _ = day_one
        reaped = {
            r.video_id
            for r in retention.run(cfg, ledger=led, now=LATER).reaped
            if r.unwatched
        }
        assert reaped
        led.clear_selections()
        discover.run(cfg, ledger=led)
        # Simulate the same passage of time for the cooldown check that
        # retention just used, rather than letting it fall back to the real
        # wall clock (last_offered_at was stamped moments ago in real time).
        score.run(cfg, ledger=led, rng=random.Random(3), now=LATER)
        offered = {c.video_id for c in led.pending_downloads()}
        assert offered & reaped

    def test_a_returning_video_is_marked_as_a_repeat(self, day_one):
        cfg, led, _ = day_one
        retention.run(cfg, ledger=led, now=LATER)
        led.clear_selections()
        discover.run(cfg, ledger=led)
        score.run(cfg, ledger=led, rng=random.Random(3), now=LATER)
        assert led.count(selection="repeat") > 0

    def test_a_second_offer_increments_the_count(self, day_one):
        cfg, led, _ = day_one
        retention.run(cfg, ledger=led, now=LATER)
        led.clear_selections()
        discover.run(cfg, ledger=led)
        score.run(cfg, ledger=led, rng=random.Random(3), now=LATER)
        download.run(cfg, ledger=led)
        repeats = [r for r in led.downloaded() if r.selection == "repeat"]
        assert repeats
        assert all(r.offer_count == 2 for r in repeats)

    def test_a_recently_snubbed_video_does_not_come_back(self, pipeline):
        cfg, _ = pipeline
        with Ledger.open(cfg.paths.ledger) as led:
            discover.run(cfg, ledger=led)
            score.run(cfg, ledger=led, rng=random.Random(1))
            download.run(cfg, ledger=led)
            # A couple of days after the real discovery/download moment --
            # enough to clear a 1-day retention window, derived from the
            # real clock rather than a calendar date that drifts stale.
            soon = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=2)
            short = cfg.with_(retention=RetentionConfig(max_age_days=1))
            reaped = {
                r.video_id for r in retention.run(short, ledger=led, now=soon).reaped
            }
            assert reaped
            led.clear_selections()
            score.run(cfg, ledger=led, rng=random.Random(3))
            assert {c.video_id for c in led.pending_downloads()}.isdisjoint(reaped)


class TestSave:
    @pytest.fixture
    def saving(self, day_one, tmp_path):
        cfg, led, videos = day_one
        permanent = tmp_path / "permanent"
        permanent.mkdir()
        cfg = cfg.with_(
            paths=Paths.under(cfg.paths.root, permanent_library=permanent)
        )
        return cfg, led, permanent

    def test_a_saved_video_survives_retention(self, saving):
        cfg, led, permanent = saving
        keeper = led.downloaded()[0].video_id
        retention.save(led, cfg, keeper)
        report = retention.run(cfg, ledger=led, now=LATER)
        assert keeper not in {r.video_id for r in report.reaped}
        assert Path(led.get(keeper).filepath).exists()

    def test_a_saved_video_lands_in_the_permanent_library(self, saving):
        cfg, led, permanent = saving
        keeper = led.downloaded()[0].video_id
        new_path = retention.save(led, cfg, keeper)
        assert new_path.is_relative_to(permanent)
        assert list(permanent.rglob("*.info.json"))

    def test_a_saved_video_is_never_refetched(self, saving):
        cfg, led, permanent = saving
        keeper = led.downloaded()[0].video_id
        retention.save(led, cfg, keeper)
        retention.run(cfg, ledger=led, now=LATER)
        discover.run(cfg, ledger=led)
        score.run(cfg, ledger=led, rng=random.Random(4))
        assert keeper not in {c.video_id for c in led.pending_downloads()}

    def test_a_saved_video_is_the_strongest_training_signal(self, saving):
        cfg, led, _ = saving
        keeper = led.downloaded()[0].video_id
        retention.save(led, cfg, keeper)
        signal = training.derive_signal(led.get(keeper))
        assert signal.positive
        assert signal.reason == training.SAVED


class TestColdStart:
    def test_day_one_with_no_model_still_downloads(self, pipeline):
        # Before any labels exist the system must still do something, and
        # what it does is pure exploration.
        cfg, _ = pipeline
        with Ledger.open(cfg.paths.ledger) as led:
            discover.run(cfg, ledger=led)
            report = score.run(cfg, ledger=led, rng=random.Random(1))
            assert report.model_picks == ()
            assert len(report.random_picks) == 3
            assert download.run(cfg, ledger=led).succeeded == 3

    def test_takeout_seeded_likes_give_the_model_something_to_work_with(
        self, pipeline, takeout_dir
    ):
        from ytaf import takeout

        cfg, _ = pipeline
        with Ledger.open(cfg.paths.ledger) as led:
            takeout.import_takeout(led, takeout_dir)
            assert led.liked()
            discover.run(cfg, ledger=led)
            assert len(score.run(cfg, ledger=led, rng=random.Random(1)).model_picks) == 5

    def test_sentiment_steers_the_next_run(self, pipeline):
        cfg, _ = pipeline
        with Ledger.open(cfg.paths.ledger) as led:
            discover.run(cfg, ledger=led)
            score.run(cfg, ledger=led, rng=random.Random(1))
            unwanted = led.new_pool()[0].video_id
            led.set_label(unwanted, spec.LABEL_DISLIKE)
            led.clear_selections()
            score.run(cfg, ledger=led, rng=random.Random(1))
            assert unwanted not in {c.video_id for c in led.pending_downloads()}


class TestPartialFailure:
    def test_a_dead_channel_does_not_stop_the_day(self, pipeline, fake_http):
        cfg, _ = pipeline
        fake_http.route("channel_id=UCchannel00000000000001", status=500, body="boom")
        with Ledger.open(cfg.paths.ledger) as led:
            report = discover.run(cfg, ledger=led)
            assert report.failures
            assert led.count() == 15

    def test_a_failed_download_is_retried_the_next_day(self, pipeline, fake_ytdlp):
        cfg, _ = pipeline
        original = fake_ytdlp.handler
        failed_once = {"done": False}

        def flaky(args, **kwargs):
            if "-o" in args and args[-1] == "c0vid0000000" and not failed_once["done"]:
                failed_once["done"] = True
                raise ytdlp.YtdlpError(
                    "transient", video_id=args[-1], returncode=1, stderr="ERROR: 503"
                )
            return original(args, **kwargs)

        fake_ytdlp.handler = flaky
        with Ledger.open(cfg.paths.ledger) as led:
            discover.run(cfg, ledger=led)
            score.run(cfg, ledger=led, rng=random.Random(1))
            first = download.run(cfg, ledger=led)
            if first.failed:
                assert download.run(cfg, ledger=led).succeeded >= 1

    def test_a_failed_download_does_not_count_as_an_offer(self, pipeline, fake_ytdlp):
        # Otherwise a run of transient failures would burn through max_offers
        # and retire a video nobody ever actually saw.
        cfg, _ = pipeline

        def always_fail(args, **kwargs):
            if "-o" in args:
                raise ytdlp.YtdlpError(
                    "boom", video_id=args[-1], returncode=1, stderr="ERROR"
                )
            return fake_ytdlp.handler(args, **kwargs)

        original = fake_ytdlp.handler
        with Ledger.open(cfg.paths.ledger) as led:
            discover.run(cfg, ledger=led)
            score.run(cfg, ledger=led, rng=random.Random(1))
            fake_ytdlp.handler = lambda a, **k: (
                always_fail(a, **k) if "-o" in a else original(a, **k)
            )
            download.run(cfg, ledger=led, retry_after_update=False)
            assert all(c.offer_count == 0 for c in led.all())
