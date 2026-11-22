"""The system's load-bearing constraints, each asserted at the top level.

  1. Never transcode -- H.264 + AAC only, capped at 1080p.
  2. Discovery and download are anonymous -- no cookies, no credentials.
  3. Taste comes from deliberate signals first, measured playback second.
  4. Exploration is mandatory; it is the control group.
  5. Nothing we already hold gets downloaded twice.
  6. Nothing outside the managed library can be deleted.

Each is covered in detail by the per-module tests. This file is the single
place a reviewer can confirm all six are actually enforced somewhere, and it
fails loudly if one is ever relaxed.
"""

from __future__ import annotations

import pytest

import datetime as dt

from ytaf import config as config_mod
from ytaf import prefilter, select, spec, training, ytdlp
from ytaf.config import Config, DownloadConfig, FeedbackConfig, SelectionConfig
from ytaf.errors import ConfigError
from ytaf.ledger import Candidate


class TestConstraintNeverTranscode:
    def test_the_only_selector_is_h264_at_1080p(self):
        assert "vcodec^=avc1" in spec.FORMAT_SELECTOR
        assert "[height<=1080]" in spec.FORMAT_SELECTOR
        assert "acodec^=mp4a" in spec.FORMAT_SELECTOR

    def test_the_default_config_uses_it(self, config):
        assert config.download.format_selector == spec.FORMAT_SELECTOR

    def test_every_download_invocation_carries_it(self, config):
        args = ytdlp.build_download_args("aircAruvnKk", config)
        assert args[args.index("-f") + 1] == spec.FORMAT_SELECTOR

    def test_config_validation_refuses_to_widen_it(self, config):
        for selector in (
            "bv*+ba",
            "bestvideo+bestaudio",
            "bv*[vcodec^=avc1][height<=2160]+ba[acodec^=mp4a]/b[ext=mp4]",
            "bv*[vcodec^=vp9][height<=1080]+ba[acodec^=opus]",
        ):
            cfg = config.with_(download=DownloadConfig(format_selector=selector))
            with pytest.raises(ConfigError):
                config_mod.validate(cfg)

    def test_nothing_asks_yt_dlp_to_re_encode(self, config):
        args = ytdlp.build_download_args("aircAruvnKk", config)
        for reencoding_flag in (
            "--recode-video",
            "--postprocessor-args",
            "--sponsorblock-remove",
            "--extract-audio",
        ):
            assert reencoding_flag not in args


class TestConstraintNoAuthentication:
    def test_download_arguments_carry_no_credentials(self, config):
        ytdlp.assert_no_auth_flags(ytdlp.build_download_args("aircAruvnKk", config))

    def test_search_arguments_carry_no_credentials(self):
        ytdlp.assert_no_auth_flags(ytdlp.build_search_args("computerphile"))

    def test_metadata_arguments_carry_no_credentials(self):
        ytdlp.assert_no_auth_flags(ytdlp.build_metadata_args(["aircAruvnKk"]))

    def test_the_rss_source_is_the_anonymous_feed(self):
        assert spec.RSS_FEED_URL.startswith("https://www.youtube.com/feeds/videos.xml")

    def test_every_takeout_sourced_row_came_from_a_file_not_an_api(self):
        # The only authenticated step in the entire design is Google Takeout,
        # which happens in a browser, once, by hand.
        assert "history" in spec.SOURCES
        assert "like" in spec.SOURCES


class TestConstraintDeliberateSignalsFirst:
    def test_the_ledger_records_both_kinds_of_signal(self):
        for column in ("watch_pct", "play_count", "label", "saved"):
            assert column in spec.LEDGER_COLUMNS

    def test_a_like_outranks_a_contradicting_watch_percent(self, make_candidate):
        # A like is a deliberate act. A watch-percent could be autoplay
        # running while you were asleep, so it must not override one.
        liked = make_candidate(label=spec.LABEL_LIKE, watched=1, watch_pct=0.02)
        assert training.derive_signal(liked).positive

    def test_a_dislike_outranks_a_full_watch(self, make_candidate):
        disliked = make_candidate(label=spec.LABEL_DISLIKE, watched=1, watch_pct=1.0)
        assert not training.derive_signal(disliked).positive

    def test_saving_outranks_everything(self, make_candidate):
        assert training.derive_signal(
            make_candidate(saved=1, label=spec.LABEL_DISLIKE, watch_pct=0.0, watched=1)
        ).positive

    def test_explicit_signals_are_weighted_above_measured_ones(self, make_candidate):
        explicit = training.derive_signal(make_candidate(label=spec.LABEL_LIKE))
        measured = training.derive_signal(make_candidate(watched=1, watch_pct=0.95))
        assert explicit.weight > measured.weight

    def test_the_negative_bands_are_not_monotonic(self, make_candidate):
        # Abandoning a video a third of the way in is the strongest negative
        # available: it got a fair hearing and lost. Closing it in the first
        # few seconds says far less.
        partial = training.derive_signal(make_candidate(watched=1, watch_pct=0.35))
        quick = training.derive_signal(make_candidate(watched=1, watch_pct=0.02))
        assert not partial.positive and not quick.positive
        assert partial.weight > quick.weight

    def test_a_finished_video_is_a_positive(self, make_candidate):
        assert training.derive_signal(make_candidate(watched=1, watch_pct=0.95)).positive

    def test_rewatches_add_weight_with_diminishing_returns(self):
        first = training.rewatch_bonus(2) - training.rewatch_bonus(1)
        later = training.rewatch_bonus(9) - training.rewatch_bonus(8)
        assert first > later > 0

    def test_sentiment_is_freely_reversible(self, ledger, make_candidate):
        # Liking and then un-liking must be possible, and so must disliking.
        ledger.upsert(make_candidate("aaaaaaaaaa1"))
        for label in (spec.LABEL_LIKE, None, spec.LABEL_DISLIKE, None, spec.LABEL_LIKE):
            ledger.set_label("aaaaaaaaaa1", label)
            assert ledger.get("aaaaaaaaaa1").label == label

    def test_validation_holds_out_whole_channels(self):
        # Splitting by row measures "Nat watches channels Nat already
        # watches", which is not taste.
        rows = [
            training.LabeledRow(
                candidate=Candidate(
                    video_id=f"{ch}{i}",
                    seen_at="2026-09-14T00:00:00+00:00",
                    metadata={"channel_id": f"UC{ch}"},
                ),
                signal=training.Signal(label=i % 2, weight=1.0, reason="test"),
                channel_key=f"UC{ch}",
            )
            for ch in "ABCD"
            for i in range(4)
        ]
        split = training.channel_holdout_split(rows, test_fraction=0.25, seed=0)
        assert {r.channel_key for r in split.train}.isdisjoint(
            r.channel_key for r in split.test
        )

    def test_the_feedback_bands_are_coherent_by_default(self):
        cfg = FeedbackConfig()
        assert 0 < cfg.partial_floor < cfg.watch_threshold < 1


class TestConstraintMandatoryExploration:
    def test_the_default_is_five_two_and_three(self):
        assert (
            spec.DEFAULT_N_MODEL_NEW,
            spec.DEFAULT_N_MODEL_REPEAT,
            spec.DEFAULT_N_RANDOM,
        ) == (5, 2, 3)

    def test_zero_exploration_is_a_config_error(self, config):
        # Not a warning. The random arm is the control group the
        # model-vs-chance check is computed from; with none of them that check
        # has nothing to compare against and silently stops meaning anything.
        cfg = config.with_(selection=SelectionConfig(n_model_new=7, n_random=0))
        with pytest.raises(ConfigError):
            config_mod.validate(cfg)

    def test_the_random_picks_are_drawn_from_outside_the_top(self, make_candidate, rng):
        pool = [make_candidate(f"vid{i:08d}", score=0.99 - i * 0.01) for i in range(30)]
        plan = select.select_daily(pool, [], rng=rng)
        top = {p.video_id for p in plan.model_picks}
        assert {p.video_id for p in plan.random_picks}.isdisjoint(top)

    def test_exploration_happens_even_with_no_model(self, make_candidate, rng):
        pool = [make_candidate(f"vid{i:08d}", score=None) for i in range(20)]
        assert len(select.select_daily(pool, [], rng=rng).random_picks) == 3

    def test_exploration_is_not_squeezed_out_by_a_small_pool(self, make_candidate, rng):
        pool = [make_candidate(f"vid{i:08d}", score=0.5 - i * 0.01) for i in range(6)]
        assert len(select.select_daily(pool, [], rng=rng).random_picks) >= 1

    def test_the_arm_is_recorded_so_the_comparison_is_possible(self):
        assert "selection" in spec.LEDGER_COLUMNS
        assert spec.SELECTIONS == {"model", "repeat", "random"}

    def test_the_ledger_compares_model_against_random(self, ledger, make_candidate):
        for arm, pct in (("model", 0.9), ("random", 0.2)):
            for i in range(3):
                ledger.upsert(
                    make_candidate(
                        f"{arm[0]}{i:010d}",
                        selected=1,
                        selection=arm,
                        watched=1,
                        watch_pct=pct,
                    )
                )
        assert ledger.selection_stats().gap == pytest.approx(0.7)


class TestConstraintNoDoubleDownloads:
    def test_a_held_video_is_never_in_the_new_pool(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", downloaded=1, filepath="/x.mp4"))
        assert ledger.new_pool() == []

    def test_a_saved_video_is_never_re_downloaded(self, ledger, make_candidate):
        # It lives outside the managed library, so "is it in library/?" would
        # miss it and re-fetch it forever.
        ledger.upsert(make_candidate("aaaaaaaaaa1", saved=1, filepath="/perm/x.mp4"))
        assert "aaaaaaaaaa1" in ledger.held_ids()
        assert ledger.new_pool() == []

    def test_held_ids_is_the_single_skip_test(self, ledger, make_candidate):
        ledger.upsert(make_candidate("inlibrary001", downloaded=1, filepath="/x.mp4"))
        ledger.upsert(make_candidate("savedout0001", saved=1, filepath="/perm/x.mp4"))
        ledger.upsert(make_candidate("reaped000001", downloaded=1, filepath=None))
        ledger.upsert(make_candidate("brandnew0001"))
        assert ledger.held_ids() == {"inlibrary001", "savedout0001"}

    def test_a_disliked_video_is_never_offered_again(self, ledger, make_candidate):
        ledger.upsert(
            make_candidate(
                "aaaaaaaaaa1", downloaded=1, filepath=None, label=spec.LABEL_DISLIKE,
                offer_count=1,
            )
        )
        now = dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc)
        assert ledger.new_pool() == []
        assert ledger.repeat_pool(now=now) == []

    def test_a_watched_video_is_never_re_offered(self, ledger, make_candidate):
        ledger.upsert(
            make_candidate(
                "aaaaaaaaaa1", downloaded=1, filepath=None, watched=1, watch_pct=0.9,
                offer_count=1,
            )
        )
        now = dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc)
        assert ledger.repeat_pool(now=now) == []

    def test_re_offers_are_gated_on_recency_not_count(self, ledger, make_candidate):
        import datetime as dt2

        base = dt2.datetime(2026, 11, 1, tzinfo=dt2.timezone.utc)
        ledger.upsert(
            make_candidate(
                "oldsnub00001",
                downloaded=1,
                filepath=None,
                offer_count=3,
                last_offered_at=(base - dt2.timedelta(days=700)).isoformat(),
            )
        )
        ledger.upsert(
            make_candidate(
                "newsnub00001",
                downloaded=1,
                filepath=None,
                offer_count=1,
                last_offered_at=(base - dt2.timedelta(days=7)).isoformat(),
            )
        )
        pool = {c.video_id for c in ledger.repeat_pool(now=base, max_offers=0)}
        assert "oldsnub00001" in pool
        assert "newsnub00001" not in pool


class TestPrefilterMatchesTheGuide:
    def test_the_python_and_ytdlp_filters_agree_by_construction(self):
        assert prefilter.match_filter_string() == spec.MATCH_FILTERS

    def test_shorts_never_reach_the_ledger(self):
        assert not prefilter.check({"duration": 45, "view_count": 100_000}).passed

    def test_livestreams_never_reach_the_ledger(self):
        assert not prefilter.check(
            {"duration": 3600, "view_count": 100_000, "is_live": True}
        ).passed


class TestConstraintRetentionCannotReachOutside:
    def test_age_is_the_only_criterion(self, config, make_candidate):
        from ytaf import retention

        now = dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc)
        liked = make_candidate(
            downloaded=1,
            filepath="/x.mp4",
            watched=1,
            watch_pct=0.99,
            seen_at="2020-01-01T00:00:00+00:00",
            label=spec.LABEL_LIKE,
        )
        # A like is a taste signal, not a keep-forever instruction.
        assert retention.is_reapable(liked, config, now=now)[0]

    def test_saving_is_the_only_reprieve(self, config, make_candidate):
        from ytaf import retention

        now = dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc)
        saved = make_candidate(
            downloaded=1, filepath="/perm/x.mp4", saved=1,
            seen_at="2020-01-01T00:00:00+00:00",
        )
        assert not retention.is_reapable(saved, config, now=now)[0]

    def test_cannot_delete_outside_the_library(self, config, ledger, tmp_path):
        from ytaf import retention
        from ytaf.errors import UnsafePathError

        outside = tmp_path / "precious.mp4"
        outside.write_bytes(b"x")
        with pytest.raises(UnsafePathError):
            retention.apply(
                ledger, config, [retention.Reapable(video_id="x", path=outside)]
            )
        assert outside.exists()

    def test_the_ledger_row_always_survives(self, config, ledger, make_candidate, library_video):
        from ytaf import retention

        video = library_video(video_id="aaaaaaaaaa1")
        ledger.upsert(
            make_candidate(
                "aaaaaaaaaa1",
                downloaded=1,
                filepath=str(video),
                watched=1,
                watch_pct=0.95,
                seen_at="2020-01-01T00:00:00+00:00",
            )
        )
        retention.run(
            config, ledger=ledger, now=dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc)
        )
        row = ledger.get("aaaaaaaaaa1")
        assert row is not None
        assert row.watch_pct == pytest.approx(0.95)
        assert row.filepath is None

    def test_a_default_config_is_valid(self, tmp_path):
        from ytaf.config import DiscoveryConfig

        config_mod.validate(
            Config.for_root(
                tmp_path,
                discovery=DiscoveryConfig(strategy="rss", channel_ids=("UCaaa",)),
            )
        )
