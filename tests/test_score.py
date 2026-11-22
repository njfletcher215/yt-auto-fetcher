"""The scoring stage.

Two behaviours matter beyond "it writes numbers": scoring must survive having
no model at all (week one, or a failed retrain) by falling through to pure
exploration, and it must clear stale selections so an undownloaded pick from
last Tuesday does not arrive a week late.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from ytaf import models, score
from ytaf.config import ModelConfig
from ytaf.errors import ModelError
from ytaf.models import CentroidModel


@pytest.fixture
def scorable(ledger, make_candidate):
    for i in range(30):
        ledger.upsert(make_candidate(f"vid{i:08d}", title=f"Video number {i}"))
    return ledger


@pytest.fixture
def trained(config, embedder):
    """A centroid on disk, matching the configured hashing embedder."""
    vectors = embedder.encode(
        [
            "neural network gradient descent backpropagation",
            "calculus derivative integral explained",
        ]
    )
    model = CentroidModel.fit(vectors, embedder_id=models.embedder_id(embedder))
    model.save(config.paths.centroid_path)
    return config.with_(model=ModelConfig(embedder="hashing", embedding_dim=embedder.dim))


class TestLoadScoringModel:
    def test_loads_a_centroid_from_disk(self, trained):
        model = score.load_scoring_model(trained)
        assert model.kind == "centroid"

    def test_cold_starts_from_liked_rows_when_no_pickle_exists(
        self, config, ledger, make_candidate
    ):
        for i in range(3):
            ledger.upsert(make_candidate(f"liked{i:07d}", label=2))
        model = score.load_scoring_model(config.with_(model=ModelConfig(embedding_dim=64)))
        assert model is not None

    def test_returns_none_when_there_is_nothing_to_build_from(self, config):
        # Day one: no model, no likes. Scoring still has to run so the random
        # picks happen.
        assert score.load_scoring_model(config) is None

    def test_refuses_a_model_built_with_a_different_embedder(self, config, embedder):
        wrong = CentroidModel.fit(
            np.ones((2, embedder.dim), dtype=np.float32),
            embedder_id="sentence-transformer:bge-small-en-v1.5",
        )
        wrong.save(config.paths.centroid_path)
        with pytest.raises(ModelError):
            score.load_scoring_model(
                config.with_(model=ModelConfig(embedder="hashing", embedding_dim=embedder.dim))
            )


class TestRun:
    def test_scores_every_unscored_candidate(self, trained, scorable):
        report = score.run(trained, ledger=scorable, rng=random.Random(0))
        assert report.scored == 30
        assert all(c.score is not None for c in scorable.all())

    def test_scores_are_finite(self, trained, scorable):
        score.run(trained, ledger=scorable, rng=random.Random(0))
        assert all(np.isfinite(c.score) for c in scorable.all())

    def test_does_not_rescore_a_scored_candidate(self, trained, scorable, make_candidate):
        scorable.upsert(make_candidate("fixed0000001", score=0.12345))
        score.run(trained, ledger=scorable, rng=random.Random(0))
        assert scorable.get("fixed0000001").score == pytest.approx(0.12345)

    def test_selects_five_and_three_with_no_repeat_pool(self, trained, scorable):
        report = score.run(trained, ledger=scorable, rng=random.Random(0))
        assert len(report.model_picks) == 5
        assert len(report.random_picks) == 3
        assert report.repeat_picks == ()
        assert scorable.count(selection="model") == 5
        assert scorable.count(selection="random") == 3

    def test_fills_the_repeat_arm_when_the_pool_has_candidates(
        self, trained, scorable, make_candidate
    ):
        import datetime as dt

        for i in range(3):
            scorable.upsert(
                make_candidate(
                    f"reap{i:08d}",
                    score=0.5,
                    downloaded=1,
                    filepath=None,
                    offer_count=1,
                    last_offered_at=(
                        dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
                    ).isoformat(),
                )
            )
        report = score.run(trained, ledger=scorable, rng=random.Random(0))
        assert len(report.repeat_picks) == 2
        assert scorable.count(selection="repeat") == 2

    def test_repeat_picks_never_overlap_the_other_arms(
        self, trained, scorable, make_candidate
    ):
        import datetime as dt

        for i in range(3):
            scorable.upsert(
                make_candidate(
                    f"reap{i:08d}",
                    score=0.5,
                    downloaded=1,
                    filepath=None,
                    offer_count=1,
                    last_offered_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc).isoformat(),
                )
            )
        report = score.run(trained, ledger=scorable, rng=random.Random(0))
        arms = [set(report.model_picks), set(report.repeat_picks), set(report.random_picks)]
        assert len(set().union(*arms)) == sum(len(a) for a in arms)

    def test_model_picks_are_the_highest_scoring(self, trained, scorable):
        report = score.run(trained, ledger=scorable, rng=random.Random(0))
        scores = {c.video_id: c.score for c in scorable.all()}
        picked = [scores[v] for v in report.model_picks]
        others = [
            s
            for v, s in scores.items()
            if v not in report.model_picks and s is not None
        ]
        assert min(picked) >= max(others)

    def test_random_picks_never_overlap_model_picks(self, trained, scorable):
        report = score.run(trained, ledger=scorable, rng=random.Random(0))
        assert set(report.model_picks).isdisjoint(report.random_picks)

    def test_reports_the_model_kind(self, trained, scorable):
        report = score.run(trained, ledger=scorable, rng=random.Random(0))
        assert report.model_kind == "centroid"

    def test_is_reproducible_for_a_seed(self, trained, scorable):
        first = score.run(trained, ledger=scorable, rng=random.Random(7))
        scorable.clear_selections()
        second = score.run(trained, ledger=scorable, rng=random.Random(7))
        assert first.random_picks == second.random_picks

    def test_clears_stale_selections_first(self, trained, scorable, make_candidate):
        scorable.upsert(
            make_candidate("stale0000001", score=0.5, selected=1, selection="model")
        )
        report = score.run(trained, ledger=scorable, rng=random.Random(0))
        assert scorable.count(selected=1) == 8
        assert report.selected == 8

    def test_does_not_disturb_rows_we_hold(
        self, trained, scorable, make_candidate
    ):
        scorable.upsert(
            make_candidate(
                "done00000001",
                score=0.99,
                selected=1,
                selection="model",
                downloaded=1,
                filepath="/x.mp4",
            )
        )
        score.run(trained, ledger=scorable, rng=random.Random(0))
        row = scorable.get("done00000001")
        assert row.downloaded == 1
        assert row.selection == "model"

    def test_rows_we_hold_are_not_rescored(
        self, trained, scorable, make_candidate
    ):
        scorable.upsert(
            make_candidate("done00000001", score=None, downloaded=1, filepath="/x.mp4")
        )
        score.run(trained, ledger=scorable, rng=random.Random(0))
        assert scorable.get("done00000001").score is None

    def test_with_no_model_it_still_explores(self, config, scorable):
        # A missing model degrades to pure exploration rather than to nothing
        # happening at all.
        report = score.run(config, ledger=scorable, rng=random.Random(0))
        assert len(report.random_picks) == 3
        assert report.model_picks == ()

    def test_with_no_model_it_notes_why(self, config, scorable):
        report = score.run(config, ledger=scorable, rng=random.Random(0))
        assert report.notes

    def test_an_empty_ledger_selects_nothing(self, trained, ledger):
        report = score.run(trained, ledger=ledger, rng=random.Random(0))
        assert report.selected == 0
        assert report.scored == 0

    def test_a_pool_smaller_than_ten_still_explores(self, trained, ledger, make_candidate):
        for i in range(6):
            ledger.upsert(make_candidate(f"vid{i:08d}"))
        report = score.run(trained, ledger=ledger, rng=random.Random(0))
        assert len(report.random_picks) >= 1

    def test_dry_run_writes_nothing(self, trained, scorable):
        report = score.run(trained, ledger=scorable, rng=random.Random(0), dry_run=True)
        assert scorable.count(selected=1) == 0
        assert all(c.score is None for c in scorable.all())
        assert len(report.model_picks) == 5

    def test_scores_are_committed_incrementally(self, trained, scorable, monkeypatch):
        # Embedding is the expensive part of this stage on a Pi; an
        # interrupted run should not throw away the work it already did.
        real_apply = score.select.apply_plan

        def boom(*args, **kwargs):
            raise KeyboardInterrupt()

        monkeypatch.setattr(score.select, "apply_plan", boom)
        with pytest.raises(KeyboardInterrupt):
            score.run(trained, ledger=scorable, rng=random.Random(0))
        assert scorable.count(score=None) == 0
        assert real_apply is not None

    def test_opens_its_own_ledger_when_not_given_one(self, trained):
        score.run(trained, rng=random.Random(0))
        assert trained.paths.ledger.exists()
