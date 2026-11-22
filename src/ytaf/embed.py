"""Metadata embeddings.

The embedder is an interface with two implementations:

  * `HashingEmbedder` -- deterministic, numpy-only, no model download. Used by
    the test suite and available as a stopgap before the real model is in
    place on the Pi.
  * `SentenceTransformerEmbedder` -- the real thing (`bge-small-en-v1.5` or
    `all-MiniLM-L6-v2`), imported lazily so the package stays importable
    without torch.

Both return L2-normalised float32 vectors, so cosine similarity is a dot
product and the centroid is just a mean.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from .config import ModelConfig
from .errors import ConfigError
from .ledger import Candidate

# Keeps a pathological description from dominating the embedding and blowing
# up the sentence-transformer's token budget.
MAX_DESCRIPTION_CHARS = 2000


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into unit-norm row vectors."""

    @property
    def dim(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """(len(texts), dim) float32, each row L2-normalised."""
        ...


def build_embedding_text(
    candidate: Candidate | dict[str, object],
    *,
    max_description_chars: int = MAX_DESCRIPTION_CHARS,
) -> str:
    """"title + description + channel" for one candidate.

    Newline-joined in that order, missing fields dropped rather than rendered
    as "None", description truncated, whitespace collapsed. Order is fixed:
    changing it invalidates every stored centroid.
    """
    if isinstance(candidate, Candidate):
        title = candidate.title
        description = candidate.description
        channel = candidate.channel
    else:
        title = str(candidate.get("title") or "")
        description = str(candidate.get("description") or "")
        channel = str(candidate.get("channel") or candidate.get("uploader") or "")

    description = description[:max_description_chars]
    parts = [part for part in (title, description, channel) if part.strip()]
    text = "\n".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; zero rows are left as zeros."""
    arr = np.asarray(matrix, dtype=np.float32)
    single = arr.ndim == 1
    arr2d = arr.reshape(1, -1) if single else arr
    norms = np.linalg.norm(arr2d, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    out = (arr2d / safe).astype(np.float32)
    return out.reshape(-1) if single else out


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between row vectors, or a matrix and a vector.

    Does not assume normalised input. A zero vector scores 0.0, not NaN.
    """
    a_norm = l2_normalise(np.asarray(a, dtype=np.float32))
    b_norm = l2_normalise(np.asarray(b, dtype=np.float32))
    if a_norm.ndim == 1 and b_norm.ndim == 1:
        return float(np.dot(a_norm, b_norm))
    if a_norm.ndim == 2 and b_norm.ndim == 1:
        return a_norm @ b_norm
    if a_norm.ndim == 1 and b_norm.ndim == 2:
        return b_norm @ a_norm
    return np.sum(a_norm * b_norm, axis=1)


class HashingEmbedder:
    """Deterministic hashed bag-of-words. No model, no download, no training.

    Same text in, same vector out, across processes and machines -- which is
    what makes it usable in tests and reproducible on two boxes.
    """

    def __init__(self, dim: int = 384, *, seed: int = 0) -> None:
        self._dim = dim
        self._seed = seed

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self.tokenize(text):
                digest = hashlib.blake2b(
                    f"{self._seed}:{token}".encode("utf-8"), digest_size=8
                ).digest()
                index = int.from_bytes(digest, "big") % self._dim
                matrix[row, index] += 1.0
        return l2_normalise(matrix)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Lowercase word tokens. Pinned, since it defines the vector space."""
        return re.findall(r"[a-z0-9]+", text.lower())


class SentenceTransformerEmbedder:
    """Wraps sentence-transformers. Imports torch lazily, on first encode."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        model = self._load()
        # sentence-transformers renamed this method; support both old and new
        # installs since we pin only a floor version (>=2.2).
        if hasattr(model, "get_embedding_dimension"):
            return int(model.get_embedding_dimension())
        return int(model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(list(texts), convert_to_numpy=True)
        return l2_normalise(np.asarray(vectors, dtype=np.float32))


def build_embedder(config: ModelConfig) -> Embedder:
    """Construct the embedder named by config. Unknown name raises ConfigError."""
    if config.embedder == "hashing":
        return HashingEmbedder(dim=config.embedding_dim)
    if config.embedder == "sentence-transformer":
        return SentenceTransformerEmbedder(config.sentence_transformer)
    raise ConfigError(f"unknown embedder: {config.embedder!r}")


def embed_candidates(
    embedder: Embedder, candidates: Iterable[Candidate]
) -> tuple[list[Candidate], np.ndarray]:
    """Embed many candidates in one batch.

    Returns the candidates in the same order as the returned matrix rows, so
    callers can zip scores back onto video ids without a second lookup.
    """
    ordered = list(candidates)
    if not ordered:
        return [], np.zeros((0, embedder.dim), dtype=np.float32)
    texts = [build_embedding_text(c) for c in ordered]
    matrix = np.asarray(embedder.encode(texts), dtype=np.float32)
    return ordered, matrix
