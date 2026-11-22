"""Phases 6 and 7: the centroid baseline, the classifier, and persistence.

The guide's warning drives these tests: the zero-training centroid "will often
be uncomfortably good". So the centroid has to be correct and cheap, and the
pickle has to refuse to load against the wrong embedder -- a mismatched model
produces confident, meaningless scores and nothing downstream would notice.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from ytaf import models
from ytaf.embed import HashingEmbedder
from ytaf.errors import ModelError
from ytaf.models import CentroidModel, STRUCTURED_FEATURES


@pytest.fixture
def embeddings():
    rng = np.random.default_rng(0)
    return rng.normal(size=(10, 16)).astype(np.float32)


class TestCentroidFit:
    def test_is_the_mean_direction_of_the_inputs(self):
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        model = CentroidModel.fit(vectors)
        expected = np.array([0.5, 0.5], dtype=np.float32)
        expected /= np.linalg.norm(expected)
        assert np.allclose(model.centroid, expected, atol=1e-6)

    def test_centroid_is_unit_norm(self, embeddings):
        model = CentroidModel.fit(embeddings)
        assert np.linalg.norm(model.centroid) == pytest.approx(1.0, abs=1e-5)

    def test_records_the_example_count(self, embeddings):
        assert CentroidModel.fit(embeddings).n_examples == 10

    def test_records_the_dimension(self, embeddings):
        assert CentroidModel.fit(embeddings).dim == 16

    def test_records_the_embedder_identity(self, embeddings):
        model = CentroidModel.fit(embeddings, embedder_id="hashing-16-7")
        assert model.embedder_id == "hashing-16-7"

    def test_records_when_it_was_trained(self, embeddings):
        model = CentroidModel.fit(embeddings, trained_at="2026-09-14T05:00:00+00:00")
        assert model.trained_at == "2026-09-14T05:00:00+00:00"

    def test_a_single_example_works(self):
        model = CentroidModel.fit(np.array([[1.0, 0.0]], dtype=np.float32))
        assert np.allclose(model.centroid, [1.0, 0.0])

    def test_empty_input_raises(self):
        # A centroid of nothing scores every candidate identically, which
        # disables discovery while looking like it works.
        with pytest.raises(ModelError):
            CentroidModel.fit(np.zeros((0, 16), dtype=np.float32))

    def test_all_zero_input_raises(self):
        with pytest.raises(ModelError):
            CentroidModel.fit(np.zeros((5, 16), dtype=np.float32))

    def test_weights_shift_the_centroid(self):
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        weighted = CentroidModel.fit(vectors, weights=np.array([10.0, 1.0]))
        assert weighted.centroid[0] > weighted.centroid[1]

    def test_mismatched_weight_length_raises(self, embeddings):
        with pytest.raises(ModelError):
            CentroidModel.fit(embeddings, weights=np.ones(3))

    def test_kind_is_centroid(self, embeddings):
        assert CentroidModel.fit(embeddings).kind == "centroid"


class TestCentroidScore:
    def test_scores_one_value_per_row(self, embeddings):
        model = CentroidModel.fit(embeddings)
        assert model.score(embeddings).shape == (10,)

    def test_a_vector_identical_to_the_centroid_scores_one(self):
        model = CentroidModel.fit(np.array([[1.0, 0.0]], dtype=np.float32))
        score = model.score(np.array([[1.0, 0.0]], dtype=np.float32))
        assert score[0] == pytest.approx(1.0, abs=1e-5)

    def test_ranks_similar_above_dissimilar(self):
        model = CentroidModel.fit(np.array([[1.0, 0.0]], dtype=np.float32))
        scores = model.score(
            np.array([[0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
        )
        assert scores[0] > scores[1] > scores[2]

    def test_scores_are_finite(self, embeddings):
        model = CentroidModel.fit(embeddings)
        scores = model.score(np.vstack([embeddings, np.zeros((1, 16), dtype=np.float32)]))
        assert np.isfinite(scores).all()

    def test_scoring_an_empty_matrix_returns_an_empty_array(self, embeddings):
        model = CentroidModel.fit(embeddings)
        assert model.score(np.zeros((0, 16), dtype=np.float32)).shape == (0,)

    def test_dimension_mismatch_raises(self, embeddings):
        model = CentroidModel.fit(embeddings)
        with pytest.raises(ModelError):
            model.score(np.zeros((3, 8), dtype=np.float32))

    def test_does_not_need_candidates(self, embeddings):
        # The centroid is text-only; the signature takes candidates for
        # interface compatibility with the classifier.
        CentroidModel.fit(embeddings).score(embeddings)


class TestPersistence:
    def test_round_trips(self, config, embeddings):
        model = CentroidModel.fit(embeddings, embedder_id="hashing-16-7")
        model.save(config.paths.centroid_path)
        loaded = CentroidModel.load(config.paths.centroid_path)
        assert np.allclose(loaded.centroid, model.centroid)
        assert loaded.embedder_id == "hashing-16-7"
        assert loaded.n_examples == model.n_examples

    def test_creates_parent_directories(self, tmp_path, embeddings):
        target = tmp_path / "deep" / "model" / "centroid.pkl"
        CentroidModel.fit(embeddings).save(target)
        assert target.exists()

    def test_overwrites_an_existing_model(self, config, embeddings):
        CentroidModel.fit(embeddings).save(config.paths.centroid_path)
        second = CentroidModel.fit(embeddings[:3])
        second.save(config.paths.centroid_path)
        assert CentroidModel.load(config.paths.centroid_path).n_examples == 3

    def test_save_is_atomic(self, config, embeddings, monkeypatch):
        # A crash mid-write must leave the previous model intact; the next
        # scoring run depends on being able to load something.
        CentroidModel.fit(embeddings).save(config.paths.centroid_path)
        original = config.paths.centroid_path.read_bytes()

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(models.os, "replace", boom)
        with pytest.raises(OSError):
            CentroidModel.fit(embeddings[:2]).save(config.paths.centroid_path)
        assert config.paths.centroid_path.read_bytes() == original

    def test_save_leaves_no_temp_files_behind(self, config, embeddings):
        CentroidModel.fit(embeddings).save(config.paths.centroid_path)
        leftovers = [
            p for p in config.paths.model_dir.iterdir() if p.name != "centroid.pkl"
        ]
        assert leftovers == []

    def test_loading_a_missing_file_raises_model_error(self, config):
        with pytest.raises(ModelError):
            CentroidModel.load(config.paths.centroid_path)

    def test_loading_garbage_raises_model_error(self, config):
        config.paths.centroid_path.write_bytes(b"not a pickle")
        with pytest.raises(ModelError):
            CentroidModel.load(config.paths.centroid_path)

    def test_loading_a_truncated_pickle_raises_model_error(self, config, embeddings):
        CentroidModel.fit(embeddings).save(config.paths.centroid_path)
        data = config.paths.centroid_path.read_bytes()
        config.paths.centroid_path.write_bytes(data[: len(data) // 2])
        with pytest.raises(ModelError):
            CentroidModel.load(config.paths.centroid_path)

    def test_rejects_an_unknown_format_version(self, config, embeddings):
        model = CentroidModel.fit(embeddings)
        object.__setattr__(model, "format_version", 99)
        config.paths.model_dir.mkdir(parents=True, exist_ok=True)
        config.paths.centroid_path.write_bytes(pickle.dumps(model))
        with pytest.raises(ModelError):
            CentroidModel.load(config.paths.centroid_path)

    def test_load_model_dispatches_on_kind(self, config, embeddings):
        CentroidModel.fit(embeddings).save(config.paths.centroid_path)
        loaded = models.load_model(config.paths.centroid_path)
        assert loaded.kind == "centroid"


class TestCompatibilityCheck:
    def test_accepts_a_matching_embedder(self, embeddings):
        e = HashingEmbedder(dim=16, seed=7)
        model = CentroidModel.fit(embeddings, embedder_id=models.embedder_id(e))
        models.check_compatible(model, e)

    def test_rejects_a_different_embedder(self, embeddings):
        # Scoring bge vectors against a hashing centroid yields plausible
        # numbers that mean nothing at all.
        model = CentroidModel.fit(embeddings, embedder_id="sentence-transformer:bge-small")
        with pytest.raises(ModelError):
            models.check_compatible(model, HashingEmbedder(dim=16, seed=7))

    def test_rejects_a_dimension_mismatch(self, embeddings):
        e = HashingEmbedder(dim=64, seed=7)
        model = CentroidModel.fit(embeddings, embedder_id=models.embedder_id(e))
        with pytest.raises(ModelError):
            models.check_compatible(model, HashingEmbedder(dim=16, seed=7))

    def test_an_unlabelled_legacy_model_is_allowed_with_matching_dims(self, embeddings):
        model = CentroidModel.fit(embeddings, embedder_id="")
        models.check_compatible(model, HashingEmbedder(dim=16, seed=7))

    def test_embedder_id_is_stable(self):
        a = models.embedder_id(HashingEmbedder(dim=16, seed=7))
        b = models.embedder_id(HashingEmbedder(dim=16, seed=7))
        assert a == b

    def test_embedder_id_distinguishes_configurations(self):
        assert models.embedder_id(HashingEmbedder(dim=16, seed=7)) != models.embedder_id(
            HashingEmbedder(dim=32, seed=7)
        )


class TestStructuredFeatures:
    def test_shape_matches_the_feature_list(self, make_candidate):
        candidates = [make_candidate(f"vid{i:08d}") for i in range(4)]
        out = models.structured_features(candidates, now="2026-09-14")
        assert out.shape == (4, len(STRUCTURED_FEATURES))

    def test_is_float32(self, make_candidate):
        out = models.structured_features([make_candidate()], now="2026-09-14")
        assert out.dtype == np.float32

    def test_missing_values_become_zero_not_nan(self):
        from ytaf.ledger import Candidate

        bare = Candidate(video_id="x", seen_at="2026-09-14T00:00:00+00:00")
        out = models.structured_features([bare], now="2026-09-14")
        assert np.isfinite(out).all()

    def test_counts_are_log_scaled(self, make_candidate):
        # Raw view counts range over seven orders of magnitude and would
        # swamp every embedding dimension in a linear model.
        small = models.structured_features(
            [make_candidate(view_count=100)], now="2026-09-14"
        )
        large = models.structured_features(
            [make_candidate(view_count=100_000_000)], now="2026-09-14"
        )
        idx = STRUCTURED_FEATURES.index("view_count")
        assert large[0, idx] < 10 * small[0, idx]
        assert large[0, idx] > small[0, idx]

    def test_like_view_ratio_is_computed(self, make_candidate):
        out = models.structured_features(
            [make_candidate(view_count=1000, like_count=100)], now="2026-09-14"
        )
        idx = STRUCTURED_FEATURES.index("like_view_ratio")
        assert out[0, idx] == pytest.approx(0.1, abs=1e-3)

    def test_like_view_ratio_handles_zero_views(self, make_candidate):
        out = models.structured_features(
            [make_candidate(view_count=0, like_count=5)], now="2026-09-14"
        )
        assert np.isfinite(out).all()

    def test_recency_is_measured_against_now(self, make_candidate):
        idx = STRUCTURED_FEATURES.index("upload_recency_days")
        recent = models.structured_features(
            [make_candidate(upload_date="20260914")], now="2026-09-14"
        )
        old = models.structured_features(
            [make_candidate(upload_date="20200101")], now="2026-09-14"
        )
        assert old[0, idx] > recent[0, idx]

    def test_recency_is_reproducible_for_a_fixed_now(self, make_candidate):
        c = [make_candidate(upload_date="20260101")]
        a = models.structured_features(c, now="2026-09-14")
        b = models.structured_features(c, now="2026-09-14")
        assert np.array_equal(a, b)

    def test_handles_an_empty_candidate_list(self):
        out = models.structured_features([], now="2026-09-14")
        assert out.shape == (0, len(STRUCTURED_FEATURES))


class TestClassifierModel:
    def test_scores_return_probabilities(self, make_candidate):
        sklearn = pytest.importorskip("sklearn")  # noqa: F841
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(0)
        x = rng.normal(size=(20, 16 + len(STRUCTURED_FEATURES)))
        y = (x[:, 0] > 0).astype(int)
        estimator = LogisticRegression().fit(x, y)
        model = models.ClassifierModel(estimator=estimator, dim=16)
        candidates = [make_candidate(f"vid{i:08d}") for i in range(5)]
        scores = model.score(rng.normal(size=(5, 16)).astype(np.float32), candidates)
        assert scores.shape == (5,)
        assert ((scores >= 0) & (scores <= 1)).all()

    def test_round_trips_through_a_pickle(self, config):
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(0)
        x = rng.normal(size=(20, 4))
        estimator = LogisticRegression().fit(x, (x[:, 0] > 0).astype(int))
        model = models.ClassifierModel(
            estimator=estimator, dim=16, metrics={"precision_at_10": 0.7}
        )
        model.save(config.paths.classifier_path)
        loaded = models.ClassifierModel.load(config.paths.classifier_path)
        assert loaded.metrics["precision_at_10"] == pytest.approx(0.7)

    def test_kind_is_classifier(self):
        assert models.ClassifierModel().kind == "classifier"
