"""The daily selection: 5 new + 2 re-offer + 3 random.

Three arms, three jobs:

  * `model` -- the top `n_model_new` brand new candidates by score. The
    system's actual output.
  * `repeat` -- `n_model_repeat` videos that were downloaded before, never
    watched, and have since been reaped. A second chance, gated on how long
    ago we last offered it rather than how many times: three ignored offers
    two years ago says much less than one ignored offer last week.
  * `random` -- `n_random` drawn at random. Mandatory. Without it tomorrow's
    training data is entirely today's model output, and -- more concretely --
    there is no control group to measure the model against.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from . import spec
from .config import SelectionConfig
from .ledger import Candidate, Ledger

MODEL = spec.MODEL
REPEAT = spec.REPEAT
RANDOM = spec.RANDOM


@dataclass(frozen=True)
class Selection:
    video_id: str
    selection: str
    score: float | None = None
    rank: int | None = None


@dataclass(frozen=True)
class SelectionPlan:
    model_picks: tuple[Selection, ...] = ()
    repeat_picks: tuple[Selection, ...] = ()
    random_picks: tuple[Selection, ...] = ()

    @property
    def all(self) -> tuple[Selection, ...]:
        return self.model_picks + self.repeat_picks + self.random_picks

    @property
    def video_ids(self) -> tuple[str, ...]:
        return tuple(s.video_id for s in self.all)


def rank_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Sort by score descending.

    Unscored rows sort last. Ties break on video_id so two runs over the same
    pool produce the same order -- without this, "top 5" is not reproducible
    and neither is any measurement built on it.
    """

    def key(c: Candidate) -> tuple[int, float, str]:
        if c.score is None:
            return (1, 0.0, c.video_id)
        return (0, -c.score, c.video_id)

    return sorted(candidates, key=key)


def _parse_dt(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _staleness_days(candidate: Candidate, now_dt: datetime) -> float:
    if not candidate.last_offered_at:
        return float("inf")
    return (now_dt - _parse_dt(candidate.last_offered_at)).total_seconds() / 86400.0


def rank_repeat_candidates(
    candidates: Sequence[Candidate], *, now: datetime | str | None = None
) -> list[Candidate]:
    """Order re-offer candidates best-first.

    Primary key is staleness -- the longest since `last_offered_at` comes
    first, because recency of the snub is the signal that matters. Score and
    then a lower `offer_count` break ties, so among two equally stale videos
    the better-scoring and less-often-ignored one wins.
    """
    if now is None:
        now_dt = datetime.now(timezone.utc)
    else:
        now_dt = _parse_dt(now)

    def key(c: Candidate) -> tuple[float, float, int, str]:
        age = _staleness_days(c, now_dt)
        score = c.score if c.score is not None else float("-inf")
        return (-age, -score, c.offer_count, c.video_id)

    return sorted(candidates, key=key)


def select_daily(
    new_pool: Sequence[Candidate],
    repeat_pool: Sequence[Candidate] = (),
    config: SelectionConfig | None = None,
    *,
    rng: random.Random | None = None,
    now: datetime | str | None = None,
) -> SelectionPlan:
    """Choose the day's downloads from the two pools.

    Rules:
      * model picks are the top `n_model_new` of `new_pool` by score,
        ignoring unscored rows;
      * repeat picks are the first `n_model_repeat` of `repeat_pool` by
        `rank_repeat_candidates`;
      * random picks are drawn from `new_pool` minus the model picks, and may
        be unscored -- that is the point of exploration;
      * no video appears in two arms;
      * a short pool yields fewer picks rather than an error, but as long as
        anything is left over after the model picks, at least one random pick
        is made;
      * an empty `repeat_pool` does not reduce the other arms. The repeat
        slots simply go unused rather than being backfilled with new
        candidates, so the arms stay comparable week to week.
    """
    cfg = config or SelectionConfig()
    active_rng = rng or random.Random()

    ranked_new = rank_candidates(list(new_pool))
    scored = [c for c in ranked_new if c.score is not None]
    model_candidates = scored[: cfg.n_model_new]
    model_ids = {c.video_id for c in model_candidates}

    model_picks = tuple(
        Selection(video_id=c.video_id, selection=MODEL, score=c.score, rank=i + 1)
        for i, c in enumerate(model_candidates)
    )

    remaining_new = [c for c in ranked_new if c.video_id not in model_ids]
    n_random = min(cfg.n_random, len(remaining_new))
    chosen_random = active_rng.sample(remaining_new, n_random) if n_random > 0 else []
    random_picks = tuple(
        Selection(video_id=c.video_id, selection=RANDOM, score=c.score)
        for c in chosen_random
    )

    ranked_repeat = rank_repeat_candidates(list(repeat_pool), now=now)
    repeat_candidates = ranked_repeat[: cfg.n_model_repeat]
    repeat_picks = tuple(
        Selection(video_id=c.video_id, selection=REPEAT, score=c.score)
        for c in repeat_candidates
    )

    return SelectionPlan(
        model_picks=model_picks, repeat_picks=repeat_picks, random_picks=random_picks
    )


def apply_plan(ledger: Ledger, plan: SelectionPlan) -> int:
    """Write the plan to the ledger. Returns rows marked."""
    count = 0
    for selection in plan.all:
        ledger.mark_selected(selection.video_id, selection.selection)
        count += 1
    return count
