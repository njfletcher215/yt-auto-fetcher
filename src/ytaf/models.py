"""The scoring models and their on-disk form.

`CentroidModel` is the zero-training baseline -- the mean of your liked
videos' embeddings, scored by cosine similarity. `ClassifierModel` wraps a
logistic regression / LightGBM over the embedding plus structured features.

Both pickle to `state/model/`. Pickles carry a format version and the
embedder's identity, so loading a centroid built by a different embedder
fails loudly instead of returning plausible nonsense.
"""

from __future__ import annotations

import math
import os
import pickle
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Protocol, Self, runtime_checkable

import numpy as np

from .embed import HashingEmbedder, SentenceTransformerEmbedder, cosine_similarity
from .errors import ModelError
from .ledger import Candidate

MODEL_FORMAT_VERSION = 1

# Structured features the classifier adds on top of the embedding.
STRUCTURED_FEATURES = (
    "duration",
    "view_count",
    "like_view_ratio",
    "channel_follower_count",
    "upload_recency_days",
)


@runtime_checkable
class ScoringModel(Protocol):
    kind: str

    def score(self, embeddings: np.ndarray, candidates: Sequence[Candidate]) -> np.ndarray:
        """One score per row. Higher is better. Finite, never NaN."""
        ...


def structured_features(
    candidates: Sequence[Candidate], *, now: str | None = None
) -> np.ndarray:
    """(n, len(STRUCTURED_FEATURES)) float32 matrix.

    Missing values become 0.0 and counts are log1p-scaled so a 10M-view video
    does not swamp the embedding dimensions. `upload_recency_days` is measured
    against `now` (ISO date) so training is reproducible.
    """
    now_date = date.fromisoformat(now) if now else date.today()
    index = {name: i for i, name in enumerate(STRUCTURED_FEATURES)}
    out = np.zeros((len(candidates), len(STRUCTURED_FEATURES)), dtype=np.float32)

    for row, candidate in enumerate(candidates):
        duration = candidate.duration or 0.0
        view_count = candidate.view_count or 0
        like_count = candidate.metadata.get("like_count") or 0
        follower_count = candidate.metadata.get("channel_follower_count") or 0
        upload_date = candidate.upload_date

        out[row, index["duration"]] = float(duration)
        out[row, index["view_count"]] = math.log1p(max(view_count, 0))
        ratio = (like_count / view_count) if view_count else 0.0
        out[row, index["like_view_ratio"]] = float(ratio)
        out[row, index["channel_follower_count"]] = math.log1p(max(follower_count, 0))

        recency = 0.0
        if upload_date and len(upload_date) >= 8:
            try:
                uploaded = date(
                    int(upload_date[0:4]), int(upload_date[4:6]), int(upload_date[6:8])
                )
                recency = float((now_date - uploaded).days)
            except ValueError:
                recency = 0.0
        out[row, index["upload_recency_days"]] = recency

    return out


def _save_atomic(obj: object, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(obj, handle)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _load_checked(path: str | os.PathLike[str], expected_cls: type) -> Any:
    path = Path(path)
    if not path.exists():
        raise ModelError(f"model not found: {path}")
    try:
        with path.open("rb") as handle:
            obj = pickle.load(handle)
    except Exception as exc:
        raise ModelError(f"could not load model at {path}: {exc}") from exc
    if not isinstance(obj, expected_cls):
        raise ModelError(f"unexpected object type at {path}: {type(obj)!r}")
    if obj.format_version != MODEL_FORMAT_VERSION:
        raise ModelError(
            f"model format version {obj.format_version} is not supported "
            f"(expected {MODEL_FORMAT_VERSION})"
        )
    return obj


@dataclass
class CentroidModel:
    """The zero-training baseline: mean of positive embeddings, scored by
    cosine similarity."""

    centroid: np.ndarray
    kind: str = "centroid"
    embedder_id: str = ""
    dim: int = 0
    n_examples: int = 0
    trained_at: str = ""
    format_version: int = MODEL_FORMAT_VERSION

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        *,
        embedder_id: str = "",
        trained_at: str | None = None,
        weights: np.ndarray | None = None,
    ) -> Self:
        """Average (optionally weighted) and normalise.

        An empty input raises ModelError -- a centroid of nothing would score
        everything identically and quietly disable discovery.
        """
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] == 0:
            raise ModelError("cannot fit a centroid from zero examples")

        if weights is not None:
            weights_arr = np.asarray(weights, dtype=np.float32)
            if weights_arr.shape[0] != arr.shape[0]:
                raise ModelError("weights length does not match embeddings")
            total = float(weights_arr.sum())
            if total == 0:
                raise ModelError("weights sum to zero")
            mean = (arr * weights_arr[:, None]).sum(axis=0) / total
        else:
            mean = arr.mean(axis=0)

        norm = float(np.linalg.norm(mean))
        if norm == 0:
            raise ModelError("centroid of all-zero embeddings has no direction")
        centroid = (mean / norm).astype(np.float32)

        return cls(
            centroid=centroid,
            embedder_id=embedder_id,
            dim=arr.shape[1],
            n_examples=arr.shape[0],
            trained_at=trained_at or "",
        )

    def score(self, embeddings: np.ndarray, candidates: Sequence[Candidate] = ()) -> np.ndarray:
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((0,), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.centroid.shape[0]:
            raise ModelError(
                f"embedding dimension {arr.shape[1]} does not match "
                f"centroid dimension {self.centroid.shape[0]}"
            )
        return np.asarray(cosine_similarity(arr, self.centroid), dtype=np.float32)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Pickle atomically (temp file + rename) so a crash mid-write cannot
        leave a truncated model that the next scoring run would load."""
        _save_atomic(self, path)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Self:
        return _load_checked(path, cls)


@dataclass
class ClassifierModel:
    """Supervised model over embedding + structured features."""

    estimator: Any = None
    kind: str = "classifier"
    embedder_id: str = ""
    dim: int = 0
    n_examples: int = 0
    trained_at: str = ""
    feature_names: tuple[str, ...] = STRUCTURED_FEATURES
    metrics: dict[str, float] = field(default_factory=dict)
    format_version: int = MODEL_FORMAT_VERSION

    def score(self, embeddings: np.ndarray, candidates: Sequence[Candidate]) -> np.ndarray:
        """Positive-class probability per row."""
        candidates = list(candidates)
        arr = np.asarray(embeddings, dtype=np.float32)
        if not candidates:
            return np.zeros((0,), dtype=np.float32)
        features = structured_features(candidates)
        x = np.hstack([arr, features])
        probs = self.estimator.predict_proba(x)[:, 1]
        return np.asarray(probs, dtype=np.float32)

    def save(self, path: str | os.PathLike[str]) -> None:
        _save_atomic(self, path)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Self:
        return _load_checked(path, cls)


def load_model(path: str | os.PathLike[str]) -> ScoringModel:
    """Load whichever model kind is at `path`.

    Raises ModelError for a missing file, an unpicklable file, or a
    format_version this build does not understand.
    """
    return _load_checked(path, (CentroidModel, ClassifierModel))


def embedder_id(embedder: object) -> str:
    """Stable identity string for an embedder, stored inside the pickle.

    Must distinguish anything that changes the vector space -- implementation,
    model name, dimension, seed -- and must be identical for the same
    configuration on natgentoo and on the Pi.
    """
    if isinstance(embedder, HashingEmbedder):
        # Deliberately excludes the seed: `build_embedder` always reconstructs
        # a hashing embedder with the default seed, so a seed-qualified id
        # would never match a freshly loaded model regardless of dimension.
        return f"hashing:{embedder.dim}"
    if isinstance(embedder, SentenceTransformerEmbedder):
        return f"sentence-transformer:{embedder.model_name}"
    return f"{type(embedder).__name__}:{getattr(embedder, 'dim', '?')}"


def check_compatible(model: ScoringModel, embedder: object) -> None:
    """Raise ModelError if the model was trained with a different embedder.

    Scoring a bge-small centroid with hashing vectors produces numbers that
    look fine and mean nothing; this is the guard against that.
    """
    model_dim = getattr(model, "dim", 0)
    embedder_dim = getattr(embedder, "dim", None)
    if model_dim and embedder_dim is not None and model_dim != embedder_dim:
        raise ModelError(
            f"model dimension {model_dim} does not match embedder dimension {embedder_dim}"
        )
    model_id = getattr(model, "embedder_id", "")
    if model_id and model_id != embedder_id(embedder):
        raise ModelError(
            f"model was trained with embedder {model_id!r}, but scoring "
            f"with {embedder_id(embedder)!r}"
        )
