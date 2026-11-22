"""Signals, channel hold-out, training, and the weekly retrain.

Two things carry the design and get the most attention.

`derive_signal` has to put an explicit like above a measured watch-percent. A
like is a deliberate act; a video watched to the end might just mean autoplay
ran while you were asleep. And the playback bands are deliberately
non-monotonic -- abandoning a video a third of the way in is the strongest
negative available, because it got a fair hearing and lost, while closing it
in the first few seconds says very little.

`channel_holdout_split` splits by whole channel, because channel identity
dominates every feature available and a random row split measures only "Nat
watches channels Nat already watches".
"""

from __future__ import annotations

import numpy as np
import pytest

from ytaf import spec, training
from ytaf.errors import ModelError
from ytaf.ledger import Candidate
from ytaf.training import LabeledRow, Signal


def row(video_id, channel, label, weight=1.0, reason="test"):
    return LabeledRow(
        candidate=Candidate(
            video_id=video_id,
            seen_at="2026-09-14T00:00:00+00:00",
            metadata={"channel": channel, "channel_id": f"UC{channel}", "title": video_id},
        ),
        signal=Signal(label=label, weight=weight, reason=reason),
        channel_key=f"UC{channel}",
    )


class TestChannelKey:
    def test_prefers_the_channel_id(self, make_candidate):
        c = make_candidate(channel="Name", channel_id="UCstable")
        assert training.channel_key(c) == "UCstable"

    def test_falls_back_to_the_channel_name(self):
        c = Candidate(
            video_id="x", seen_at="2026-09-14T00:00:00+00:00", metadata={"channel": "Name"}
        )
        assert training.channel_key(c) == "Name"

    def test_uses_a_sentinel_when_nothing_is_known(self):
        assert (
            training.channel_key(Candidate(video_id="x", seen_at="2026-09-14T00:00:00+00:00"))
            == training.UNKNOWN_CHANNEL
        )

    def test_a_renamed_channel_keeps_its_key(self, make_candidate):
        # Channels rename constantly; grouping on the display name would split
        # one channel across the train/test boundary.
        a = make_candidate(channel="Old Name", channel_id="UCsame")
        b = make_candidate(channel="New Name", channel_id="UCsame")
        assert training.channel_key(a) == training.channel_key(b)


class TestRewatchBonus:
    @pytest.mark.parametrize("count", [0, 1])
    def test_no_bonus_for_a_single_viewing(self, count):
        assert training.rewatch_bonus(count) == pytest.approx(0.0)

    def test_a_second_viewing_earns_a_bonus(self):
        assert training.rewatch_bonus(2) > 0

    def test_is_monotonic(self):
        values = [training.rewatch_bonus(n) for n in range(1, 12)]
        assert values == sorted(values)

    def test_has_diminishing_returns(self):
        # A rewatch is a strong signal; a tenth rewatch is not ten times one.
        first = training.rewatch_bonus(2) - training.rewatch_bonus(1)
        later = training.rewatch_bonus(9) - training.rewatch_bonus(8)
        assert later < first

    def test_is_capped(self):
        assert training.rewatch_bonus(10_000) <= training.REWATCH_BONUS_CAP


class TestDeriveSignalPrecedence:
    def test_a_like_beats_a_contradicting_watch_percent(self, make_candidate):
        # The whole point: an explicit like is deliberate, a watch-percent
        # could be autoplay running while you were asleep.
        c = make_candidate(label=spec.LABEL_LIKE, watched=1, watch_pct=0.02)
        signal = training.derive_signal(c)
        assert signal.positive
        assert signal.reason == training.LIKED

    def test_a_dislike_beats_a_full_watch(self, make_candidate):
        c = make_candidate(label=spec.LABEL_DISLIKE, watched=1, watch_pct=1.0)
        signal = training.derive_signal(c)
        assert not signal.positive
        assert signal.reason == training.DISLIKED

    def test_saved_beats_everything(self, make_candidate):
        c = make_candidate(saved=1, label=spec.LABEL_DISLIKE, watched=1, watch_pct=0.0)
        signal = training.derive_signal(c)
        assert signal.positive
        assert signal.reason == training.SAVED

    def test_explicit_labels_outweigh_measured_playback(self, make_candidate):
        liked = training.derive_signal(make_candidate(label=spec.LABEL_LIKE))
        watched = training.derive_signal(make_candidate(watched=1, watch_pct=0.95))
        assert liked.weight > watched.weight

    def test_saved_outweighs_a_like(self, make_candidate):
        saved = training.derive_signal(make_candidate(saved=1))
        liked = training.derive_signal(make_candidate(label=spec.LABEL_LIKE))
        assert saved.weight > liked.weight

    def test_history_is_the_weakest_positive(self, make_candidate):
        history = training.derive_signal(make_candidate(label=spec.LABEL_HISTORY))
        liked = training.derive_signal(make_candidate(label=spec.LABEL_LIKE))
        completed = training.derive_signal(make_candidate(watched=1, watch_pct=0.95))
        assert history.positive
        assert history.weight < liked.weight
        assert history.weight < completed.weight


class TestDeriveSignalPlaybackBands:
    @pytest.mark.parametrize("pct", [0.60, 0.75, 0.95, 1.0])
    def test_at_or_above_the_threshold_is_a_watch(self, make_candidate, pct):
        signal = training.derive_signal(make_candidate(watched=1, watch_pct=pct))
        assert signal.positive
        assert signal.reason == training.COMPLETED

    @pytest.mark.parametrize("pct", [0.15, 0.30, 0.45, 0.59])
    def test_the_middle_band_is_a_negative(self, make_candidate, pct):
        signal = training.derive_signal(make_candidate(watched=1, watch_pct=pct))
        assert not signal.positive
        assert signal.reason == training.PARTIAL

    @pytest.mark.parametrize("pct", [0.0, 0.05, 0.14])
    def test_below_the_floor_is_a_quick_exit(self, make_candidate, pct):
        signal = training.derive_signal(make_candidate(watched=1, watch_pct=pct))
        assert not signal.positive
        assert signal.reason == training.QUICK_EXIT

    def test_a_partial_watch_is_a_stronger_negative_than_a_quick_exit(self, make_candidate):
        # The core of the band design: bailing at 35% means it got a fair
        # hearing and lost. Bailing at 3% means you misread the thumbnail.
        partial = training.derive_signal(make_candidate(watched=1, watch_pct=0.35))
        quick = training.derive_signal(make_candidate(watched=1, watch_pct=0.03))
        assert partial.weight > quick.weight
        assert not partial.positive and not quick.positive

    def test_a_partial_watch_is_a_stronger_negative_than_never_opening_it(
        self, make_candidate
    ):
        partial = training.derive_signal(make_candidate(watched=1, watch_pct=0.35))
        ignored = training.derive_signal(
            make_candidate(selected=1, selection="model", downloaded=1, filepath="/x.mp4")
        )
        assert partial.weight > ignored.weight

    def test_never_opened_is_the_weakest_negative(self, make_candidate):
        ignored = training.derive_signal(
            make_candidate(selected=1, selection="model", downloaded=1, filepath="/x.mp4")
        )
        quick = training.derive_signal(make_candidate(watched=1, watch_pct=0.03))
        assert ignored.weight <= quick.weight
        assert ignored.reason == training.IGNORED

    def test_the_thresholds_are_configurable(self, make_candidate):
        from ytaf.config import FeedbackConfig

        cfg = FeedbackConfig(watch_threshold=0.3, partial_floor=0.1)
        assert training.derive_signal(make_candidate(watched=1, watch_pct=0.4), cfg).positive

    def test_a_rewatch_adds_weight(self, make_candidate):
        once = training.derive_signal(make_candidate(watched=1, watch_pct=0.95, play_count=1))
        thrice = training.derive_signal(
            make_candidate(watched=1, watch_pct=0.95, play_count=3)
        )
        assert thrice.weight > once.weight

    def test_an_unlabelled_undownloaded_candidate_has_no_signal(self, make_candidate):
        assert training.derive_signal(make_candidate()) is None

    def test_a_reaped_unwatched_row_is_still_a_negative(self, make_candidate):
        # Retention cleared the file, but "we offered it and you ignored it"
        # is exactly the signal we want to keep.
        c = make_candidate(
            selected=1, selection="model", downloaded=1, filepath=None, offer_count=1
        )
        signal = training.derive_signal(c)
        assert signal is not None
        assert not signal.positive


class TestBuildTrainingSet:
    def test_keeps_only_rows_with_a_signal(self, make_candidate):
        rows = training.build_training_set(
            [
                make_candidate("aaaaaaaaaa1", label=spec.LABEL_LIKE),
                make_candidate("bbbbbbbbbb2"),
                make_candidate("cccccccccc3", watched=1, watch_pct=0.3),
            ]
        )
        assert {r.video_id for r in rows} == {"aaaaaaaaaa1", "cccccccccc3"}

    def test_attaches_the_channel_key(self, make_candidate):
        rows = training.build_training_set(
            [make_candidate("aaaaaaaaaa1", label=spec.LABEL_LIKE, channel_id="UCxyz")]
        )
        assert rows[0].channel_key == "UCxyz"

    def test_handles_an_empty_input(self):
        assert training.build_training_set([]) == []


class TestLabelMix:
    def test_counts_by_reason(self, make_candidate):
        rows = training.build_training_set(
            [
                make_candidate("aaaaaaaaaa1", label=spec.LABEL_LIKE),
                make_candidate("bbbbbbbbbb2", label=spec.LABEL_LIKE),
                make_candidate("cccccccccc3", watched=1, watch_pct=0.3),
            ]
        )
        mix = training.label_mix(rows)
        assert mix[training.LIKED] == 2
        assert mix[training.PARTIAL] == 1

    def test_empty_input_is_an_empty_mix(self):
        assert training.label_mix([]) == {}


class TestChannelHoldoutSplit:
    @pytest.fixture
    def rows(self):
        out = []
        for channel in "ABCDEFGH":
            for i in range(5):
                out.append(row(f"{channel}{i}", channel, 1 if i % 2 else 0))
        return out

    def test_no_channel_appears_on_both_sides(self, rows):
        split = training.channel_holdout_split(rows, test_fraction=0.25, seed=0)
        assert {r.channel_key for r in split.train}.isdisjoint(
            r.channel_key for r in split.test
        )

    def test_every_row_lands_somewhere(self, rows):
        split = training.channel_holdout_split(rows, seed=0)
        assert len(split.train) + len(split.test) == len(rows)

    def test_no_row_is_duplicated(self, rows):
        split = training.channel_holdout_split(rows, seed=0)
        ids = [r.video_id for r in split.train] + [r.video_id for r in split.test]
        assert len(ids) == len(set(ids))

    def test_reports_the_held_out_channels(self, rows):
        split = training.channel_holdout_split(rows, seed=0)
        assert set(split.held_out_channels) == {r.channel_key for r in split.test}

    def test_respects_the_test_fraction_approximately(self, rows):
        split = training.channel_holdout_split(rows, test_fraction=0.25, seed=0)
        assert 0.1 <= len(split.test) / len(rows) <= 0.45

    def test_a_larger_fraction_holds_out_more(self, rows):
        small = training.channel_holdout_split(rows, test_fraction=0.125, seed=0)
        large = training.channel_holdout_split(rows, test_fraction=0.5, seed=0)
        assert len(large.test) > len(small.test)

    def test_is_deterministic_for_a_seed(self, rows):
        a = training.channel_holdout_split(rows, seed=42)
        b = training.channel_holdout_split(rows, seed=42)
        assert [r.video_id for r in a.test] == [r.video_id for r in b.test]

    def test_different_seeds_give_different_splits(self, rows):
        a = training.channel_holdout_split(rows, seed=1)
        b = training.channel_holdout_split(rows, seed=2)
        assert set(a.held_out_channels) != set(b.held_out_channels)

    def test_both_sides_are_non_empty(self, rows):
        for seed in range(10):
            split = training.channel_holdout_split(rows, test_fraction=0.25, seed=seed)
            assert split.train and split.test

    def test_training_side_retains_a_positive(self, rows):
        for seed in range(10):
            assert any(r.label == 1 for r in training.channel_holdout_split(rows, seed=seed).train)

    def test_a_single_channel_raises(self):
        # There is nothing to hold out: any split leaks.
        with pytest.raises(ModelError):
            training.channel_holdout_split([row(f"a{i}", "A", 1) for i in range(10)])

    def test_an_empty_input_raises(self):
        with pytest.raises(ModelError):
            training.channel_holdout_split([])

    def test_two_channels_split_one_each(self):
        rows = [row(f"a{i}", "A", 1) for i in range(5)]
        rows += [row(f"b{i}", "B", 0) for i in range(5)]
        split = training.channel_holdout_split(rows, test_fraction=0.5, seed=0)
        assert len(split.held_out_channels) == 1

    def test_unknown_channel_rows_do_not_merge_into_one_group(self):
        # Rows with no channel would otherwise form one giant pseudo-channel
        # landing entirely on one side of the split.
        rows = [
            LabeledRow(
                candidate=Candidate(video_id=f"x{i}", seen_at="2026-09-14T00:00:00+00:00"),
                signal=Signal(label=i % 2, weight=1.0, reason="test"),
                channel_key=training.UNKNOWN_CHANNEL,
            )
            for i in range(10)
        ]
        rows += [row(f"a{i}", "A", 1) for i in range(5)]
        split = training.channel_holdout_split(rows, seed=0)
        assert split.train and split.test


class TestSampleWeights:
    def test_one_weight_per_row(self):
        assert training.sample_weights([row("a", "A", 1), row("b", "B", 0)]).shape == (2,)

    def test_reflects_the_signal_weights(self):
        rows = [row("a", "A", 1, weight=3.0), row("b", "B", 1, weight=0.5)]
        weights = training.sample_weights(rows)
        assert weights[0] > weights[1]

    def test_weights_are_positive(self):
        assert (training.sample_weights([row("a", "A", 1), row("b", "B", 0)]) > 0).all()

    def test_empty_input(self):
        assert training.sample_weights([]).shape == (0,)


class TestTrainCentroid:
    def test_builds_a_centroid_from_positives_only(self, embedder):
        rows = [row("a", "A", 1), row("b", "B", 1), row("c", "C", 0)]
        assert training.train_centroid(rows, embedder).n_examples == 2

    def test_records_the_embedder_identity(self, embedder):
        from ytaf import models

        rows = [row("a", "A", 1), row("b", "B", 1)]
        assert training.train_centroid(rows, embedder).embedder_id == models.embedder_id(
            embedder
        )

    def test_weighting_shifts_the_centroid(self, embedder):
        heavy = [
            LabeledRow(
                candidate=Candidate(
                    video_id="h",
                    seen_at="2026-09-14T00:00:00+00:00",
                    metadata={"title": "neural network gradient descent", "channel": "A"},
                ),
                signal=Signal(label=1, weight=10.0, reason=training.SAVED),
                channel_key="UCA",
            ),
            LabeledRow(
                candidate=Candidate(
                    video_id="l",
                    seen_at="2026-09-14T00:00:00+00:00",
                    metadata={"title": "sourdough bread baking", "channel": "B"},
                ),
                signal=Signal(label=1, weight=0.1, reason=training.HISTORY),
                channel_key="UCB",
            ),
        ]
        model = training.train_centroid(heavy, embedder, weighted=True)
        on_topic = embedder.encode(["neural network gradient descent explained"])
        off_topic = embedder.encode(["sourdough bread baking tutorial"])
        assert model.score(on_topic)[0] > model.score(off_topic)[0]

    def test_raises_with_no_positives(self, embedder):
        with pytest.raises(ModelError):
            training.train_centroid([row("a", "A", 0), row("b", "B", 0)], embedder)

    def test_raises_on_an_empty_input(self, embedder):
        with pytest.raises(ModelError):
            training.train_centroid([], embedder)


class TestTrainClassifier:
    def test_trains_over_embeddings_and_structured_features(self, embedder):
        pytest.importorskip("sklearn")
        rows = [row(f"a{i}", "A", 1) for i in range(10)]
        rows += [row(f"b{i}", "B", 0) for i in range(10)]
        model = training.train_classifier(rows, embedder)
        assert model.kind == "classifier"
        assert model.n_examples == 20

    def test_raises_with_only_one_class(self, embedder):
        pytest.importorskip("sklearn")
        with pytest.raises(ModelError):
            training.train_classifier([row(f"a{i}", "A", 1) for i in range(10)], embedder)

    def test_raises_clearly_when_sklearn_is_absent(self, embedder, monkeypatch):
        # Training is not something the serving box does; the error should say
        # so rather than surfacing a bare ImportError.
        import builtins

        real_import = builtins.__import__

        def no_sklearn(name, *args, **kwargs):
            if name.startswith("sklearn"):
                raise ImportError("No module named 'sklearn'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_sklearn)
        rows = [row(f"a{i}", "A", 1) for i in range(5)]
        rows += [row(f"b{i}", "B", 0) for i in range(5)]
        with pytest.raises(ModelError):
            training.train_classifier(rows, embedder)


class TestEvaluate:
    @pytest.fixture
    def split(self):
        rows = [row(f"a{i}", "A", 1) for i in range(5)]
        rows += [row(f"b{i}", "B", 1) for i in range(5)]
        rows += [row(f"c{i}", "C", 0) for i in range(5)]
        rows += [row(f"d{i}", "D", 0) for i in range(5)]
        return training.channel_holdout_split(rows, test_fraction=0.5, seed=0)

    def test_reports_precision_on_the_held_out_channels(self, embedder, split):
        model = training.train_centroid(split.train, embedder)
        report = training.evaluate(model, embedder, split, k=3)
        assert report.n_test == len(split.test)
        assert report.k == 3
        assert 0.0 <= report.precision_at_k <= 1.0

    def test_names_the_held_out_channels(self, embedder, split):
        model = training.train_centroid(split.train, embedder)
        assert training.evaluate(model, embedder, split).held_out_channels == (
            split.held_out_channels
        )

    def test_k_larger_than_the_test_set_is_handled(self, embedder, split):
        model = training.train_centroid(split.train, embedder)
        assert training.evaluate(model, embedder, split, k=100).precision_at_k is not None


class TestBaselineMeanWatchPct:
    def test_averages_the_top_k_by_score(self, ledger, make_candidate):
        for i, (score, pct) in enumerate([(0.9, 1.0), (0.8, 0.8), (0.1, 0.0)]):
            ledger.upsert(
                make_candidate(f"vid{i:08d}", score=score, watched=1, watch_pct=pct)
            )
        assert training.baseline_mean_watch_pct(ledger, top_k=2) == pytest.approx(0.9)

    def test_none_when_there_is_no_feedback_yet(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", score=0.9))
        assert training.baseline_mean_watch_pct(ledger) is None

    def test_ignores_unscored_rows(self, ledger, make_candidate):
        ledger.upsert(make_candidate("aaaaaaaaaa1", score=None, watched=1, watch_pct=0.1))
        ledger.upsert(make_candidate("bbbbbbbbbb2", score=0.9, watched=1, watch_pct=0.9))
        assert training.baseline_mean_watch_pct(ledger, top_k=5) == pytest.approx(0.9)


class TestRetrain:
    def test_writes_a_model_and_returns_a_report(self, config, ledger, make_candidate):
        for i in range(8):
            ledger.upsert(
                make_candidate(
                    f"vid{i:08d}",
                    channel=f"Channel {i % 4}",
                    channel_id=f"UC{i % 4}",
                    watched=1,
                    watch_pct=0.9 if i % 2 else 0.3,
                    downloaded=1,
                    filepath=f"/x{i}.mp4",
                )
            )
        report = training.retrain(config, ledger=ledger)
        assert config.paths.centroid_path.exists()
        assert report.n_test >= 0

    def test_reports_the_label_mix(self, config, ledger, make_candidate):
        for i in range(8):
            ledger.upsert(
                make_candidate(
                    f"vid{i:08d}",
                    channel=f"Channel {i % 4}",
                    channel_id=f"UC{i % 4}",
                    watched=1,
                    watch_pct=0.9 if i % 2 else 0.3,
                    downloaded=1,
                    filepath=f"/x{i}.mp4",
                )
            )
        assert training.retrain(config, ledger=ledger).label_mix

    def test_a_failed_retrain_leaves_the_previous_model_in_place(
        self, config, ledger, make_candidate, embedder
    ):
        # Last week's model beats no model. A bad retrain must not take the
        # next scoring run down with it.
        from ytaf.models import CentroidModel

        CentroidModel.fit(np.ones((1, embedder.dim), dtype=np.float32)).save(
            config.paths.centroid_path
        )
        before = config.paths.centroid_path.read_bytes()
        ledger.upsert(make_candidate("aaaaaaaaaa1"))  # no signal at all
        with pytest.raises(ModelError):
            training.retrain(config, ledger=ledger)
        assert config.paths.centroid_path.read_bytes() == before

    def test_raises_when_there_is_nothing_to_learn_from(self, config, ledger):
        with pytest.raises(ModelError):
            training.retrain(config, ledger=ledger)
