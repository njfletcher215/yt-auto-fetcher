"""The daily 5 + 2 + 3 selection.

Three arms with three different jobs, and the tests that matter most are the
ones keeping them separate: a video counted in two arms would silently
corrupt the model-vs-random measurement, and a repeat arm that backfilled
from the new pool when it ran dry would make the arms incomparable week to
week.
"""

from __future__ import annotations

import datetime as dt
import random

import pytest

from ytaf import select
from ytaf.config import SelectionConfig

NOW = dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc)


def days_ago(n: int) -> str:
    return (NOW - dt.timedelta(days=n)).isoformat()


@pytest.fixture
def new_pool(make_candidate):
    """30 scored, never-downloaded candidates, descending score."""
    return [
        make_candidate(f"new{i:08d}", score=round(0.99 - i * 0.01, 4)) for i in range(30)
    ]


@pytest.fixture
def repeat_pool(make_candidate):
    """Six reaped-unwatched candidates with differing snub recency."""
    return [
        make_candidate(
            f"rep{i:08d}",
            score=0.5 + i * 0.01,
            downloaded=1,
            filepath=None,
            offer_count=1 + (i % 3),
            last_offered_at=days_ago(100 + i * 100),
        )
        for i in range(6)
    ]


class TestRankCandidates:
    def test_sorts_by_score_descending(self, make_candidate):
        pool = [
            make_candidate("low00000001", score=0.1),
            make_candidate("high0000001", score=0.9),
            make_candidate("mid00000001", score=0.5),
        ]
        assert [c.video_id for c in select.rank_candidates(pool)] == [
            "high0000001",
            "mid00000001",
            "low00000001",
        ]

    def test_unscored_rows_sort_last(self, make_candidate):
        pool = [
            make_candidate("none0000001", score=None),
            make_candidate("low00000001", score=0.1),
        ]
        assert select.rank_candidates(pool)[-1].video_id == "none0000001"

    def test_negative_scores_sort_correctly(self, make_candidate):
        # Cosine similarity is legitimately negative for content you'd hate.
        pool = [
            make_candidate("neg00000001", score=-0.5),
            make_candidate("zero0000001", score=0.0),
        ]
        assert select.rank_candidates(pool)[0].video_id == "zero0000001"

    def test_ties_break_deterministically(self, make_candidate):
        pool = [
            make_candidate("bbbbbbbbbb1", score=0.5),
            make_candidate("aaaaaaaaaa1", score=0.5),
        ]
        first = [c.video_id for c in select.rank_candidates(pool)]
        second = [c.video_id for c in select.rank_candidates(list(reversed(pool)))]
        assert first == second

    def test_does_not_mutate_the_input(self, new_pool):
        before = [c.video_id for c in new_pool]
        select.rank_candidates(new_pool)
        assert [c.video_id for c in new_pool] == before

    def test_handles_an_empty_pool(self):
        assert select.rank_candidates([]) == []


class TestRankRepeatCandidates:
    def test_stalest_snub_comes_first(self, make_candidate):
        # Recency of the snub is the signal: a video ignored two years ago is
        # a better bet than one ignored last month.
        pool = [
            make_candidate("recent000001", score=0.9, last_offered_at=days_ago(95)),
            make_candidate("ancient00001", score=0.1, last_offered_at=days_ago(700)),
        ]
        ranked = select.rank_repeat_candidates(pool, now=NOW)
        assert ranked[0].video_id == "ancient00001"

    def test_staleness_outranks_score(self, make_candidate):
        pool = [
            make_candidate("newsnub00001", score=0.99, last_offered_at=days_ago(100)),
            make_candidate("oldsnub00001", score=0.01, last_offered_at=days_ago(900)),
        ]
        assert select.rank_repeat_candidates(pool, now=NOW)[0].video_id == "oldsnub00001"

    def test_score_breaks_a_staleness_tie(self, make_candidate):
        same = days_ago(500)
        pool = [
            make_candidate("lowscore0001", score=0.1, last_offered_at=same),
            make_candidate("highscore001", score=0.9, last_offered_at=same),
        ]
        assert select.rank_repeat_candidates(pool, now=NOW)[0].video_id == "highscore001"

    def test_fewer_offers_breaks_a_remaining_tie(self, make_candidate):
        same = days_ago(500)
        pool = [
            make_candidate("often0000001", score=0.5, last_offered_at=same, offer_count=3),
            make_candidate("once00000001", score=0.5, last_offered_at=same, offer_count=1),
        ]
        assert select.rank_repeat_candidates(pool, now=NOW)[0].video_id == "once00000001"

    def test_a_missing_last_offered_at_sorts_as_maximally_stale(self, make_candidate):
        pool = [
            make_candidate("known0000001", last_offered_at=days_ago(100)),
            make_candidate("unknown00001", last_offered_at=None),
        ]
        assert select.rank_repeat_candidates(pool, now=NOW)[0].video_id == "unknown00001"

    def test_is_deterministic(self, repeat_pool):
        a = [c.video_id for c in select.rank_repeat_candidates(repeat_pool, now=NOW)]
        b = [
            c.video_id
            for c in select.rank_repeat_candidates(list(reversed(repeat_pool)), now=NOW)
        ]
        assert a == b

    def test_handles_an_empty_pool(self):
        assert select.rank_repeat_candidates([], now=NOW) == []


class TestSelectDaily:
    def test_picks_five_two_and_three_by_default(self, new_pool, repeat_pool, rng):
        plan = select.select_daily(new_pool, repeat_pool, rng=rng, now=NOW)
        assert len(plan.model_picks) == 5
        assert len(plan.repeat_picks) == 2
        assert len(plan.random_picks) == 3

    def test_ten_downloads_total(self, new_pool, repeat_pool, rng):
        plan = select.select_daily(new_pool, repeat_pool, rng=rng, now=NOW)
        assert len(plan.all) == 10

    def test_model_picks_are_the_top_scoring_new_candidates(self, new_pool, repeat_pool, rng):
        plan = select.select_daily(new_pool, repeat_pool, rng=rng, now=NOW)
        assert [p.video_id for p in plan.model_picks] == [
            f"new{i:08d}" for i in range(5)
        ]

    def test_each_arm_is_labelled(self, new_pool, repeat_pool, rng):
        plan = select.select_daily(new_pool, repeat_pool, rng=rng, now=NOW)
        assert all(p.selection == select.MODEL for p in plan.model_picks)
        assert all(p.selection == select.REPEAT for p in plan.repeat_picks)
        assert all(p.selection == select.RANDOM for p in plan.random_picks)

    def test_no_video_appears_in_two_arms(self, new_pool, repeat_pool, rng):
        plan = select.select_daily(new_pool, repeat_pool, rng=rng, now=NOW)
        assert len(plan.video_ids) == len(set(plan.video_ids))

    def test_random_picks_come_from_outside_the_model_picks(self, new_pool, repeat_pool, rng):
        plan = select.select_daily(new_pool, repeat_pool, rng=rng, now=NOW)
        model_ids = {p.video_id for p in plan.model_picks}
        assert all(p.video_id not in model_ids for p in plan.random_picks)

    def test_random_picks_come_from_the_new_pool_not_the_repeat_pool(
        self, new_pool, repeat_pool, rng
    ):
        # Exploration means genuinely unseen content; drawing a random pick
        # from the re-offer pool would be exploring something already judged.
        plan = select.select_daily(new_pool, repeat_pool, rng=rng, now=NOW)
        repeat_ids = {c.video_id for c in repeat_pool}
        assert all(p.video_id not in repeat_ids for p in plan.random_picks)

    def test_repeat_picks_are_the_stalest(self, new_pool, repeat_pool, rng):
        plan = select.select_daily(new_pool, repeat_pool, rng=rng, now=NOW)
        ranked = select.rank_repeat_candidates(repeat_pool, now=NOW)
        assert [p.video_id for p in plan.repeat_picks] == [
            c.video_id for c in ranked[:2]
        ]

    def test_model_picks_carry_rank_and_score(self, new_pool, repeat_pool, rng):
        plan = select.select_daily(new_pool, repeat_pool, rng=rng, now=NOW)
        assert [p.rank for p in plan.model_picks] == [1, 2, 3, 4, 5]
        assert plan.model_picks[0].score == pytest.approx(0.99)

    def test_is_reproducible_for_a_given_rng(self, new_pool, repeat_pool):
        a = select.select_daily(new_pool, repeat_pool, rng=random.Random(5), now=NOW)
        b = select.select_daily(new_pool, repeat_pool, rng=random.Random(5), now=NOW)
        assert a.video_ids == b.video_ids

    def test_different_seeds_change_the_random_arm(self, new_pool, repeat_pool):
        a = select.select_daily(new_pool, repeat_pool, rng=random.Random(1), now=NOW)
        b = select.select_daily(new_pool, repeat_pool, rng=random.Random(2), now=NOW)
        assert {p.video_id for p in a.random_picks} != {
            p.video_id for p in b.random_picks
        }

    def test_different_seeds_do_not_change_the_model_arm(self, new_pool, repeat_pool):
        a = select.select_daily(new_pool, repeat_pool, rng=random.Random(1), now=NOW)
        b = select.select_daily(new_pool, repeat_pool, rng=random.Random(2), now=NOW)
        assert a.model_picks == b.model_picks

    def test_random_picks_actually_range_over_the_pool(self, new_pool, repeat_pool):
        seen = set()
        for seed in range(25):
            plan = select.select_daily(
                new_pool, repeat_pool, rng=random.Random(seed), now=NOW
            )
            seen.update(p.video_id for p in plan.random_picks)
        # If "random" were really "ranks 6, 7, 8", the filter bubble would
        # reassemble itself immediately.
        assert len(seen) > 5

    def test_honours_a_custom_ratio(self, new_pool, repeat_pool, rng):
        plan = select.select_daily(
            new_pool,
            repeat_pool,
            SelectionConfig(n_model_new=2, n_model_repeat=1, n_random=4),
            rng=rng,
            now=NOW,
        )
        assert (len(plan.model_picks), len(plan.repeat_picks), len(plan.random_picks)) == (
            2,
            1,
            4,
        )

    def test_repeat_arm_can_be_turned_off(self, new_pool, repeat_pool, rng):
        plan = select.select_daily(
            new_pool,
            repeat_pool,
            SelectionConfig(n_model_new=7, n_model_repeat=0, n_random=3),
            rng=rng,
            now=NOW,
        )
        assert plan.repeat_picks == ()
        assert len(plan.model_picks) == 7


class TestEmptyRepeatPool:
    def test_an_empty_repeat_pool_leaves_the_other_arms_alone(self, new_pool, rng):
        plan = select.select_daily(new_pool, [], rng=rng, now=NOW)
        assert plan.repeat_picks == ()
        assert len(plan.model_picks) == 5
        assert len(plan.random_picks) == 3

    def test_repeat_slots_are_not_backfilled_from_the_new_pool(self, new_pool, rng):
        # Backfilling would make the arms incomparable: "model" would mean 5
        # picks some weeks and 7 others, and the watch-percent means would
        # drift for reasons that have nothing to do with the model.
        plan = select.select_daily(new_pool, [], rng=rng, now=NOW)
        assert len(plan.all) == 8

    def test_a_short_repeat_pool_yields_what_it_has(self, new_pool, make_candidate, rng):
        one = [make_candidate("only0000001", last_offered_at=days_ago(500))]
        plan = select.select_daily(new_pool, one, rng=rng, now=NOW)
        assert len(plan.repeat_picks) == 1


class TestSmallNewPool:
    def test_a_pool_of_exactly_eight_fills_both_new_arms(self, make_candidate, rng):
        pool = [make_candidate(f"v{i:010d}", score=0.5 - i * 0.01) for i in range(8)]
        plan = select.select_daily(pool, [], rng=rng, now=NOW)
        assert len(plan.model_picks) == 5
        assert len(plan.random_picks) == 3
        assert len(set(plan.video_ids)) == 8

    def test_a_pool_of_six_still_explores(self, make_candidate, rng):
        # 5 model picks would consume almost everything; exploration must
        # still get a slot rather than being squeezed to zero.
        pool = [make_candidate(f"v{i:010d}", score=0.5 - i * 0.01) for i in range(6)]
        plan = select.select_daily(pool, [], rng=rng, now=NOW)
        assert len(plan.random_picks) >= 1
        assert len(plan.video_ids) == len(set(plan.video_ids))

    def test_a_pool_of_one_yields_one_pick(self, make_candidate, rng):
        plan = select.select_daily([make_candidate("only0000001", score=0.5)], [], rng=rng, now=NOW)
        assert len(plan.all) == 1

    def test_an_empty_plan_from_empty_pools(self, rng):
        assert select.select_daily([], [], rng=rng, now=NOW).all == ()

    def test_never_returns_more_than_the_pools_hold(self, make_candidate, rng):
        for size in range(0, 15):
            pool = [make_candidate(f"v{i:010d}", score=0.5) for i in range(size)]
            plan = select.select_daily(pool, [], rng=rng, now=NOW)
            assert len(plan.all) <= size


class TestUnscoredCandidates:
    def test_unscored_rows_are_not_model_picks(self, make_candidate, rng):
        pool = [make_candidate(f"v{i:010d}", score=None) for i in range(20)]
        plan = select.select_daily(pool, [], rng=rng, now=NOW)
        assert plan.model_picks == ()

    def test_unscored_rows_can_still_be_random_picks(self, make_candidate, rng):
        # With no model yet, exploration is the entire system.
        pool = [make_candidate(f"v{i:010d}", score=None) for i in range(20)]
        plan = select.select_daily(pool, [], rng=rng, now=NOW)
        assert len(plan.random_picks) == 3

    def test_a_mixed_pool_prefers_scored_rows_for_model_picks(self, make_candidate, rng):
        pool = [make_candidate(f"scored{i:05d}", score=0.5) for i in range(3)]
        pool += [make_candidate(f"unscored{i:03d}", score=None) for i in range(10)]
        plan = select.select_daily(pool, [], rng=rng, now=NOW)
        assert len(plan.model_picks) == 3
        assert all(p.video_id.startswith("scored") for p in plan.model_picks)


class TestApplyPlan:
    @pytest.fixture
    def stocked(self, ledger, make_candidate):
        for i in range(20):
            ledger.upsert(make_candidate(f"new{i:08d}", score=0.9 - i * 0.01))
        for i in range(4):
            ledger.upsert(
                make_candidate(
                    f"rep{i:08d}",
                    score=0.5,
                    downloaded=1,
                    filepath=None,
                    offer_count=1,
                    last_offered_at=days_ago(300),
                )
            )
        return ledger

    def test_marks_every_pick(self, stocked, rng):
        plan = select.select_daily(
            stocked.new_pool(), stocked.repeat_pool(now=NOW), rng=rng, now=NOW
        )
        assert select.apply_plan(stocked, plan) == 10
        assert stocked.count(selected=1) == 10

    def test_records_which_arm_each_pick_came_from(self, stocked, rng):
        plan = select.select_daily(
            stocked.new_pool(), stocked.repeat_pool(now=NOW), rng=rng, now=NOW
        )
        select.apply_plan(stocked, plan)
        assert stocked.count(selection="model") == 5
        assert stocked.count(selection="repeat") == 2
        assert stocked.count(selection="random") == 3

    def test_applying_an_empty_plan_is_a_no_op(self, ledger, rng):
        plan = select.select_daily([], [], rng=rng, now=NOW)
        assert select.apply_plan(ledger, plan) == 0

    def test_is_idempotent(self, stocked, rng):
        plan = select.select_daily(
            stocked.new_pool(), stocked.repeat_pool(now=NOW), rng=rng, now=NOW
        )
        select.apply_plan(stocked, plan)
        select.apply_plan(stocked, plan)
        assert stocked.count(selected=1) == 10
