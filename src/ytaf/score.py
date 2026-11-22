"""The scoring stage.

Loads the model and embedder, scores every candidate whose `score IS NULL`,
then marks the day's selections across all three arms. Runs after discovery;
a few hundred candidates takes well under a minute on a CPU.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime

from . import embed, models, select
from .config import Config
from .ledger import Ledger
from .select import SelectionPlan


@dataclass
class ScoreReport:
    model_kind: str = ""
    scored: int = 0
    model_picks: tuple[str, ...] = ()
    repeat_picks: tuple[str, ...] = ()
    random_picks: tuple[str, ...] = ()
    new_pool_size: int = 0
    repeat_pool_size: int = 0
    notes: list[str] = field(default_factory=list)
    plan: SelectionPlan | None = None

    @property
    def selected(self) -> int:
        return len(self.model_picks) + len(self.repeat_picks) + len(self.random_picks)


def load_scoring_model(config: Config):
    """Load the configured model.

    Falls back to a cold-start centroid built from the ledger's liked rows if
    no pickle exists yet, and notes that in the report. If there is nothing to
    build one from either, scoring proceeds with every score left NULL so the
    random picks still happen -- a missing model must not stop the pipeline.
    """
    embedder = embed.build_embedder(config.model)
    path = (
        config.paths.classifier_path
        if config.model.kind == "classifier"
        else config.paths.centroid_path
    )

    if path.exists():
        model = models.load_model(path)
        models.check_compatible(model, embedder)
        return model

    led = Ledger.open(config.paths.ledger)
    try:
        liked = led.liked()
    finally:
        led.close()
    if not liked:
        return None

    texts = [embed.build_embedding_text(c) for c in liked]
    vectors = embedder.encode(texts)
    return models.CentroidModel.fit(vectors, embedder_id=models.embedder_id(embedder))


def run(
    config: Config,
    *,
    ledger: Ledger | None = None,
    rng: random.Random | None = None,
    dry_run: bool = False,
    now: datetime | str | None = None,
) -> ScoreReport:
    """Score unscored candidates, then select from both pools.

    Clears stale selections from previous runs first, so an undownloaded pick
    from last week does not linger. Scores are written per candidate, not in
    one final commit, so an interrupted run does not lose the embedding work
    it already did. `now` drives the repeat arm's cooldown check; it defaults
    to the real clock but callers that need to simulate elapsed time (tests,
    mainly) can override it.
    """
    owns_ledger = ledger is None
    led = ledger or Ledger.open(config.paths.ledger)
    try:
        report = ScoreReport(model_kind=config.model.kind)
        led.clear_selections()

        unscored = led.unscored()

        model = None
        try:
            model = load_scoring_model(config)
        except Exception as exc:  # noqa: BLE001 - a bad model must not stop the pipeline
            report.notes.append(f"could not load scoring model: {exc}")

        score_overrides: dict[str, float] = {}
        if model is None:
            report.notes.append(
                "no scoring model available; exploring only until one is trained"
            )
        else:
            if unscored:
                embedder = embed.build_embedder(config.model)
                ordered, matrix = embed.embed_candidates(embedder, unscored)
                scores = model.score(matrix, ordered)
                for candidate, raw_score in zip(ordered, scores):
                    value = float(raw_score)
                    score_overrides[candidate.video_id] = value
                    if not dry_run:
                        led.set_score(candidate.video_id, value)
            report.scored = len(unscored)

        new_pool = led.new_pool()
        for candidate in new_pool:
            if candidate.video_id in score_overrides:
                candidate.score = score_overrides[candidate.video_id]

        repeat_pool = led.repeat_pool(
            now=now,
            cooldown_days=config.selection.repeat_cooldown_days,
            max_offers=config.selection.max_offers,
        )
        report.new_pool_size = len(new_pool)
        report.repeat_pool_size = len(repeat_pool)

        plan = select.select_daily(new_pool, repeat_pool, config.selection, rng=rng, now=now)
        report.plan = plan
        report.model_picks = tuple(p.video_id for p in plan.model_picks)
        report.repeat_picks = tuple(p.video_id for p in plan.repeat_picks)
        report.random_picks = tuple(p.video_id for p in plan.random_picks)

        if not dry_run:
            select.apply_plan(led, plan)

        return report
    finally:
        if owns_ledger:
            led.close()
