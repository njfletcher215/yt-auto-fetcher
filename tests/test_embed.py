"""Phase 6.1: embedding text construction and the embedders.

`build_embedding_text` gets pinned hard because it defines the vector space.
Change the field order or the separator and every stored centroid silently
becomes meaningless -- scores keep coming out, they just stop corresponding to
taste.
"""

from __future__ import annotations

import numpy as np
import pytest

from ytaf import embed
from ytaf.config import ModelConfig
from ytaf.embed import HashingEmbedder, cosine_similarity, l2_normalise
from ytaf.errors import ConfigError
from ytaf.ledger import Candidate


class TestBuildEmbeddingText:
    def test_includes_title_description_and_channel(self, make_candidate):
        text = embed.build_embedding_text(
            make_candidate(title="T", description="D", channel="C")
        )
        assert "T" in text and "D" in text and "C" in text

    def test_order_is_title_description_channel(self, make_candidate):
        text = embed.build_embedding_text(
            make_candidate(title="AAA", description="BBB", channel="CCC")
        )
        assert text.index("AAA") < text.index("BBB") < text.index("CCC")

    def test_accepts_a_raw_metadata_dict(self):
        text = embed.build_embedding_text(
            {"title": "T", "description": "D", "channel": "C"}
        )
        assert "T" in text and "C" in text

    def test_never_renders_none(self):
        text = embed.build_embedding_text(
            Candidate(video_id="x", seen_at="2026-09-14T00:00:00+00:00")
        )
        assert "None" not in text

    def test_missing_fields_do_not_leave_stray_separators(self):
        text = embed.build_embedding_text({"title": "Only A Title"})
        assert text.strip() == "Only A Title"

    def test_empty_metadata_yields_empty_text(self):
        assert embed.build_embedding_text({}).strip() == ""

    def test_truncates_a_long_description(self, make_candidate):
        text = embed.build_embedding_text(
            make_candidate(description="x" * 10_000), max_description_chars=100
        )
        assert len(text) < 500

    def test_truncation_keeps_the_title_and_channel(self, make_candidate):
        text = embed.build_embedding_text(
            make_candidate(title="KEEPTITLE", description="x" * 10_000, channel="KEEPCHAN"),
            max_description_chars=50,
        )
        assert "KEEPTITLE" in text
        assert "KEEPCHAN" in text

    def test_collapses_whitespace(self, make_candidate):
        text = embed.build_embedding_text(
            make_candidate(title="A  \n\n  B", description="C\t\tD")
        )
        assert "\n\n" not in text
        assert "  " not in text

    def test_is_deterministic(self, make_candidate):
        c = make_candidate(title="T", description="D", channel="C")
        assert embed.build_embedding_text(c) == embed.build_embedding_text(c)

    def test_falls_back_to_uploader_when_channel_is_absent(self):
        text = embed.build_embedding_text({"title": "T", "uploader": "U"})
        assert "U" in text


class TestNormalisation:
    def test_normalises_rows_to_unit_length(self):
        matrix = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
        out = l2_normalise(matrix)
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0)

    def test_leaves_a_zero_row_as_zeros(self):
        # A video with no text at all must not produce NaNs that poison the
        # whole batch's scores.
        out = l2_normalise(np.array([[0.0, 0.0]], dtype=np.float32))
        assert np.all(out == 0)
        assert not np.isnan(out).any()

    def test_handles_a_single_vector(self):
        out = l2_normalise(np.array([3.0, 4.0], dtype=np.float32))
        assert np.allclose(np.linalg.norm(out), 1.0)

    def test_does_not_mutate_the_input(self):
        matrix = np.array([[3.0, 4.0]], dtype=np.float32)
        before = matrix.copy()
        l2_normalise(matrix)
        assert np.array_equal(matrix, before)


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors_score_zero(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_score_minus_one(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, -a) == pytest.approx(-1.0, abs=1e-5)

    def test_is_scale_invariant(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        assert cosine_similarity(a, a * 100) == pytest.approx(1.0, abs=1e-5)

    def test_matrix_against_vector_returns_one_score_per_row(self):
        matrix = np.eye(3, dtype=np.float32)
        out = cosine_similarity(matrix, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        assert out.shape == (3,)
        assert out[0] == pytest.approx(1.0, abs=1e-5)

    def test_zero_vector_scores_zero_not_nan(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        out = cosine_similarity(a, b)
        assert not np.isnan(out).any()
        assert float(out) == pytest.approx(0.0)


class TestHashingEmbedder:
    def test_reports_its_dimension(self):
        assert HashingEmbedder(dim=64).dim == 64

    def test_encodes_to_the_right_shape(self):
        out = HashingEmbedder(dim=64).encode(["a", "b", "c"])
        assert out.shape == (3, 64)

    def test_rows_are_unit_norm(self):
        out = HashingEmbedder(dim=64).encode(["hello world", "goodbye world"])
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)

    def test_is_float32(self):
        assert HashingEmbedder(dim=16).encode(["x"]).dtype == np.float32

    def test_is_deterministic_within_a_process(self):
        e = HashingEmbedder(dim=64, seed=7)
        assert np.array_equal(e.encode(["same text"]), e.encode(["same text"]))

    def test_is_deterministic_across_instances(self):
        # Training happens on natgentoo and inference on the Pi. If these
        # disagree, the pickled centroid is garbage on the other machine.
        a = HashingEmbedder(dim=64, seed=7).encode(["same text"])
        b = HashingEmbedder(dim=64, seed=7).encode(["same text"])
        assert np.array_equal(a, b)

    def test_does_not_depend_on_pythonhashseed(self):
        # Python's str.__hash__ is randomised per process by default; using it
        # here would make the centroid valid only until the next reboot.
        import subprocess
        import sys

        script = (
            "import sys; sys.path.insert(0, 'src');"
            "from ytaf.embed import HashingEmbedder;"
            "print(HashingEmbedder(dim=8, seed=7).encode(['stable text'])[0].tolist())"
        )
        runs = []
        for seed in ("0", "1", "12345"):
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
                cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
            )
            assert result.returncode == 0, result.stderr
            assert result.stdout.strip(), "subprocess produced no vector"
            runs.append(result.stdout)
        assert len(set(runs)) == 1, runs

    def test_different_text_gives_different_vectors(self):
        e = HashingEmbedder(dim=256, seed=7)
        out = e.encode(["neural networks and calculus", "sourdough bread recipes"])
        assert not np.allclose(out[0], out[1])

    def test_similar_text_scores_higher_than_unrelated_text(self):
        e = HashingEmbedder(dim=512, seed=7)
        out = e.encode(
            [
                "neural network backpropagation gradient descent",
                "neural network gradient descent explained",
                "how to bake sourdough bread at home",
            ]
        )
        assert cosine_similarity(out[0], out[1]) > cosine_similarity(out[0], out[2])

    def test_handles_empty_text(self):
        out = HashingEmbedder(dim=64).encode([""])
        assert out.shape == (1, 64)
        assert not np.isnan(out).any()

    def test_handles_an_empty_batch(self):
        out = HashingEmbedder(dim=64).encode([])
        assert out.shape == (0, 64)

    def test_handles_unicode(self):
        out = HashingEmbedder(dim=64).encode(["日本語のテキスト", "Ünïcödé"])
        assert not np.isnan(out).any()

    def test_batching_matches_single_encoding(self):
        e = HashingEmbedder(dim=64, seed=7)
        batched = e.encode(["one", "two"])
        singly = np.vstack([e.encode(["one"]), e.encode(["two"])])
        assert np.allclose(batched, singly)

    def test_is_case_insensitive(self):
        e = HashingEmbedder(dim=64, seed=7)
        assert np.allclose(e.encode(["Hello World"]), e.encode(["hello world"]))

    def test_tokenize_splits_on_non_word_characters(self):
        assert HashingEmbedder.tokenize("Hello, world! 3Blue1Brown") == [
            "hello",
            "world",
            "3blue1brown",
        ]

    def test_tokenize_handles_empty_input(self):
        assert HashingEmbedder.tokenize("") == []


class TestBuildEmbedder:
    def test_builds_the_hashing_embedder(self):
        e = embed.build_embedder(ModelConfig(embedder="hashing", embedding_dim=128))
        assert isinstance(e, HashingEmbedder)
        assert e.dim == 128

    def test_rejects_an_unknown_embedder(self):
        with pytest.raises(ConfigError):
            embed.build_embedder(ModelConfig(embedder="word2vec"))

    def test_sentence_transformer_is_not_imported_eagerly(self):
        # The Pi should be able to import this package without torch present.
        import sys

        assert "torch" not in sys.modules

    @pytest.mark.slow
    def test_sentence_transformer_embedder_round_trips(self):
        pytest.importorskip("sentence_transformers")
        from ytaf.embed import SentenceTransformerEmbedder

        e = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
        out = e.encode(["neural networks", "sourdough bread"])
        assert out.shape == (2, e.dim)
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-3)


class TestEmbedCandidates:
    def test_returns_candidates_aligned_with_rows(self, embedder, make_candidate):
        candidates = [make_candidate(f"vid{i:08d}") for i in range(4)]
        ordered, matrix = embed.embed_candidates(embedder, candidates)
        assert [c.video_id for c in ordered] == [c.video_id for c in candidates]
        assert matrix.shape == (4, embedder.dim)

    def test_handles_an_empty_input(self, embedder):
        ordered, matrix = embed.embed_candidates(embedder, [])
        assert ordered == []
        assert matrix.shape == (0, embedder.dim)

    def test_accepts_a_generator(self, embedder, make_candidate):
        gen = (make_candidate(f"vid{i:08d}") for i in range(3))
        ordered, matrix = embed.embed_candidates(embedder, gen)
        assert len(ordered) == 3
        assert matrix.shape[0] == 3

    def test_embeds_the_built_text_not_the_raw_json(self, embedder, make_candidate):
        c = make_candidate(title="Distinctive Title Here")
        _, matrix = embed.embed_candidates(embedder, [c])
        expected = embedder.encode([embed.build_embedding_text(c)])
        assert np.allclose(matrix, expected)
