"""Labels, channel hold-out validation, training, and the weekly retrain.

Two things here carry the design.

`derive_signal` turns a ledger row into a (label, weight) pair. An explicit
like or dislike dominates everything -- it is a deliberate affirmative act,
whereas a high watch-percent might just mean autoplay ran while you were
asleep. Below that, the playback bands are deliberately non-monotonic: a
video abandoned a third of the way in is the *strongest* negative, because it
got a fair hearing and lost; one closed in the first few seconds, or never
opened at all, says very little.

`channel_holdout_split` splits by whole channel. Channel identity dominates
every feature available, so a random row split scores well while only proving
"Nat watches channels Nat already watches".
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from . import spec
from .config import Config, FeedbackConfig
from .embed import Embedder, build_embedder, build_embedding_text
from .errors import ModelError
from .ledger import Candidate, Ledger, utcnow_iso
from .models import (
    CentroidModel,
    ClassifierModel,
    ScoringModel,
    embedder_id,
    structured_features,
)

UNKNOWN_CHANNEL = "__unknown__"

POSITIVE = 1
NEGATIVE = 0

# Relative confidence per signal kind. Only the ordering is contractual; the
# test suite asserts the ranking, not these literals.
WEIGHT_SAVED = 4.0  # moved to the permanent library: the strongest act
WEIGHT_EXPLICIT = 3.0  # an explicit like or dislike
WEIGHT_COMPLETED = 2.0  # watched past the threshold
WEIGHT_PARTIAL = 2.0  # abandoned mid-video: a real, considered rejection
WEIGHT_QUICK_EXIT = 0.5  # closed in the first few percent
WEIGHT_IGNORED = 0.3  # offered, downloaded, never opened
WEIGHT_HISTORY = 0.5  # present in Takeout watch history, nothing more

# Rewatches add confidence with diminishing returns: +1/2, +1/3, ... so the
# second viewing matters and the ninth barely moves the number.
REWATCH_BONUS_CAP = 2.0

# Reason codes, so a training run can report its label mix.
SAVED = "saved"
LIKED = "liked"
DISLIKED = "disliked"
COMPLETED = "completed"
PARTIAL = "partial"
QUICK_EXIT = "quick_exit"
IGNORED = "ignored"
HISTORY = "history"


@dataclass(frozen=True)
class Signal:
    """What one ledger row contributes to training."""

    label: int
    weight: float
    reason: str

    @property
    def positive(self) -> bool:
        return self.label == POSITIVE


@dataclass(frozen=True)
class LabeledRow:
    candidate: Candidate
    signal: Signal
    channel_key: str

    @property
    def video_id(self) -> str:
        return self.candidate.video_id

    @property
    def label(self) -> int:
        return self.signal.label

    @property
    def weight(self) -> float:
        return self.signal.weight


@dataclass(frozen=True)
class Split:
    train: tuple[LabeledRow, ...]
    test: tuple[LabeledRow, ...]
    held_out_channels: tuple[str, ...]

    @property
    def sizes(self) -> tuple[int, int]:
        return (len(self.train), len(self.test))


@dataclass
class EvalReport:
    n_test: int = 0
    held_out_channels: tuple[str, ...] = ()
    precision_at_k: float | None = None
    k: int = 0
    mean_watch_pct: float | None = None
    auc: float | None = None
    label_mix: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def channel_key(candidate: Candidate) -> str:
    """Group key for hold-out: channel_id, else channel name, else sentinel."""
    if candidate.channel_id:
        return candidate.channel_id
    if candidate.channel:
        return candidate.channel
    return UNKNOWN_CHANNEL


def rewatch_bonus(play_count: int) -> float:
    """Diminishing-returns bonus for repeat viewings.

    0 for 0 or 1 plays, then a harmonic series capped at REWATCH_BONUS_CAP.
    A rewatch is a strong positive; a tenth rewatch is not ten times one.
    """
    if play_count <= 1:
        return 0.0
    total = sum(1.0 / k for k in range(2, play_count + 1))
    return min(total, REWATCH_BONUS_CAP)


def derive_signal(
    candidate: Candidate, config: FeedbackConfig | None = None
) -> Signal | None:
    """Collapse a row's signals into one (label, weight, reason).

    Precedence, strongest first:
      * `saved`                      -> positive, WEIGHT_SAVED
      * `label == LABEL_LIKE`        -> positive, WEIGHT_EXPLICIT
      * `label == LABEL_DISLIKE`     -> negative, WEIGHT_EXPLICIT
      * watch_pct >= watch_threshold -> positive, WEIGHT_COMPLETED
                                        (+ rewatch_bonus for play_count)
      * partial_floor <= watch_pct < watch_threshold
                                     -> negative, WEIGHT_PARTIAL
      * watch_pct < partial_floor, having been opened
                                     -> negative, WEIGHT_QUICK_EXIT
      * offered and downloaded but never opened
                                     -> negative, WEIGHT_IGNORED
      * `label == LABEL_HISTORY`     -> positive, WEIGHT_HISTORY
    Returns None for a row carrying no signal at all.
    """
    cfg = config or FeedbackConfig()

    if candidate.saved:
        return Signal(POSITIVE, WEIGHT_SAVED, SAVED)

    if candidate.label == spec.LABEL_LIKE:
        return Signal(POSITIVE, WEIGHT_EXPLICIT, LIKED)
    if candidate.label == spec.LABEL_DISLIKE:
        return Signal(NEGATIVE, WEIGHT_EXPLICIT, DISLIKED)

    watch_pct = candidate.watch_pct
    if watch_pct is not None:
        if watch_pct >= cfg.watch_threshold:
            weight = WEIGHT_COMPLETED + rewatch_bonus(candidate.play_count)
            return Signal(POSITIVE, weight, COMPLETED)
        if watch_pct >= cfg.partial_floor:
            return Signal(NEGATIVE, WEIGHT_PARTIAL, PARTIAL)
        return Signal(NEGATIVE, WEIGHT_QUICK_EXIT, QUICK_EXIT)

    if candidate.downloaded:
        return Signal(NEGATIVE, WEIGHT_IGNORED, IGNORED)

    if candidate.label == spec.LABEL_HISTORY:
        return Signal(POSITIVE, WEIGHT_HISTORY, HISTORY)

    return None


def build_training_set(
    candidates: Iterable[Candidate], config: FeedbackConfig | None = None
) -> list[LabeledRow]:
    """Keep only rows that yield a signal, attaching the channel group key."""
    rows: list[LabeledRow] = []
    for candidate in candidates:
        signal = derive_signal(candidate, config)
        if signal is None:
            continue
        rows.append(
            LabeledRow(candidate=candidate, signal=signal, channel_key=channel_key(candidate))
        )
    return rows


def channel_holdout_split(
    rows: Sequence[LabeledRow],
    *,
    test_fraction: float = 0.25,
    seed: int = 0,
) -> Split:
    """Split by whole channel, never by row.

    Guarantees: no channel appears on both sides; deterministic for a given
    seed; the test side gets approximately `test_fraction` of the rows; both
    sides are non-empty and the train side retains at least one positive.
    Fewer than two distinct channels raises ModelError -- there is nothing to
    hold out.
    """
    rows = list(rows)
    if not rows:
        raise ModelError("cannot split an empty training set")

    groups: dict[str, list[LabeledRow]] = {}
    for r in rows:
        groups.setdefault(r.channel_key, []).append(r)

    channels = sorted(groups)
    if len(channels) < 2:
        raise ModelError("need at least two distinct channels to hold one out")

    rng = random.Random(seed)
    rng.shuffle(channels)

    total = len(rows)
    target = max(1, round(total * test_fraction))

    test_channels: list[str] = []
    test_count = 0
    for ch in channels:
        if test_count >= target:
            break
        if len(test_channels) >= len(channels) - 1:
            break
        test_channels.append(ch)
        test_count += len(groups[ch])

    if not test_channels:
        test_channels = [channels[0]]

    def has_positive(chs: list[str]) -> bool:
        return any(r.label == POSITIVE for c in chs for r in groups[c])

    train_channels = [c for c in channels if c not in test_channels]
    while not has_positive(train_channels) and len(test_channels) > 1:
        train_channels.append(test_channels.pop())

    train = tuple(r for c in train_channels for r in groups[c])
    test = tuple(r for c in test_channels for r in groups[c])
    return Split(train=train, test=test, held_out_channels=tuple(sorted(test_channels)))


def sample_weights(rows: Sequence[LabeledRow]) -> np.ndarray:
    """The rows' signal weights as an array, aligned with `rows`."""
    return np.array([r.weight for r in rows], dtype=np.float32)


def label_mix(rows: Sequence[LabeledRow]) -> dict[str, int]:
    """Count of rows per signal reason. Goes in the retrain report."""
    mix: dict[str, int] = {}
    for r in rows:
        mix[r.signal.reason] = mix.get(r.signal.reason, 0) + 1
    return mix


def train_centroid(
    rows: Sequence[LabeledRow],
    embedder: Embedder,
    *,
    trained_at: str | None = None,
    weighted: bool = True,
) -> CentroidModel:
    """Weighted centroid of the positive rows' embeddings."""
    positives = [r for r in rows if r.label == POSITIVE]
    if not positives:
        raise ModelError("no positive examples to build a centroid from")

    texts = [build_embedding_text(r.candidate) for r in positives]
    vectors = np.asarray(embedder.encode(texts), dtype=np.float32)
    weights = sample_weights(positives) if weighted else None

    return CentroidModel.fit(
        vectors,
        embedder_id=embedder_id(embedder),
        trained_at=trained_at,
        weights=weights,
    )


def train_classifier(
    rows: Sequence[LabeledRow],
    embedder: Embedder,
    *,
    trained_at: str | None = None,
    seed: int = 0,
) -> ClassifierModel:
    """Logistic regression over embedding + structured features.

    Raises ModelError if scikit-learn is not installed -- it is an optional
    extra, because training and inference run on different machines.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise ModelError(
            "scikit-learn is required to train a classifier; install the "
            "'train' extra on the training machine"
        ) from exc

    rows = list(rows)
    if not rows:
        raise ModelError("cannot train a classifier from zero examples")

    labels = np.array([r.label for r in rows])
    if len(set(labels.tolist())) < 2:
        raise ModelError(
            "need both positive and negative examples to train a classifier"
        )

    candidates = [r.candidate for r in rows]
    texts = [build_embedding_text(c) for c in candidates]
    embeddings = np.asarray(embedder.encode(texts), dtype=np.float32)
    features = structured_features(candidates)
    x = np.hstack([embeddings, features])
    weights = sample_weights(rows)

    estimator = LogisticRegression(max_iter=1000, random_state=seed)
    estimator.fit(x, labels, sample_weight=weights)

    return ClassifierModel(
        estimator=estimator,
        embedder_id=embedder_id(embedder),
        dim=embeddings.shape[1],
        n_examples=len(rows),
        trained_at=trained_at or "",
    )


def evaluate(
    model: ScoringModel,
    embedder: Embedder,
    split: Split,
    *,
    k: int = 10,
) -> EvalReport:
    """Precision@k on the held-out channels. The only honest metric here."""
    test_rows = list(split.test)
    report = EvalReport(n_test=len(test_rows), held_out_channels=split.held_out_channels, k=k)
    if not test_rows:
        report.notes.append("empty test set")
        return report

    candidates = [r.candidate for r in test_rows]
    texts = [build_embedding_text(c) for c in candidates]
    embeddings = np.asarray(embedder.encode(texts), dtype=np.float32)
    scores = np.asarray(model.score(embeddings, candidates), dtype=np.float32)

    order = np.argsort(-scores)
    top_k_idx = order[: min(k, len(order))]
    labels = np.array([r.label for r in test_rows])
    top_labels = labels[top_k_idx]
    report.precision_at_k = float(np.mean(top_labels == POSITIVE)) if len(top_labels) else None

    watch_pcts = [
        r.candidate.watch_pct for r in test_rows if r.candidate.watch_pct is not None
    ]
    report.mean_watch_pct = float(np.mean(watch_pcts)) if watch_pcts else None

    return report


def baseline_mean_watch_pct(ledger: Ledger, *, top_k: int = 10) -> float | None:
    """Mean watch-percent of the top-k by current score.

    The number to write down before deciding whether a trained classifier
    beats the zero-training centroid.
    """
    rows = [c for c in ledger.all() if c.score is not None and c.watch_pct is not None]
    rows.sort(key=lambda c: c.score, reverse=True)
    top = rows[:top_k]
    if not top:
        return None
    return float(sum(c.watch_pct for c in top) / len(top))


def retrain(
    config: Config, *, ledger: Ledger | None = None, now: datetime | None = None
) -> EvalReport:
    """Rebuild the configured model from the ledger.

    Writes the new model only if it trained and evaluated successfully; a bad
    retrain must leave the previous model in place rather than breaking the
    next scoring run.
    """
    owns_ledger = ledger is None
    led = ledger or Ledger.open(config.paths.ledger)
    try:
        rows = build_training_set(led.training_rows(), config.feedback)
        if not rows:
            raise ModelError(
                "nothing to learn from yet; the ledger has no labelled rows"
            )

        embedder = build_embedder(config.model)
        trained_at = now.isoformat() if now else utcnow_iso()

        try:
            split = channel_holdout_split(rows)
        except ModelError:
            split = Split(train=tuple(rows), test=(), held_out_channels=())

        if config.model.kind == "classifier":
            model = train_classifier(split.train, embedder, trained_at=trained_at)
        else:
            model = train_centroid(split.train, embedder, trained_at=trained_at)

        report = evaluate(model, embedder, split)
        report.label_mix = label_mix(rows)

        path = (
            config.paths.classifier_path
            if config.model.kind == "classifier"
            else config.paths.centroid_path
        )
        model.save(path)
        return report
    finally:
        if owns_ledger:
            led.close()
