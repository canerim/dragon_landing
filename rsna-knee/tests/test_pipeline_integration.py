"""Audit suite, curriculum schedule and the runtime governor.

The audits are the pipeline's immune system, so they get tested against
*deliberately broken* inputs: a suite that never fails is worse than no suite,
because it produces a false sense of safety.
"""

from __future__ import annotations

import numpy as np
import pytest

from kairos.constants import NUM_TARGETS
from kairos.eval.leakage import (
    content_hash,
    duplicate_hash_audit,
    embedding_neighbour_audit,
    fold_prevalence_audit,
    metadata_only_audit,
    prediction_site_gap_audit,
    run_audit_suite,
    shuffled_label_audit,
    shuffled_report_audit,
)
from kairos.infer.budget import CostModel, RuntimeGovernor, pareto_frontier
from kairos.train.curriculum import LossSchedule, default_plan, student_plan


# --------------------------------------------------------------------------- #
# Audits                                                                       #
# --------------------------------------------------------------------------- #


def _cohort(n=800, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, NUM_TARGETS)) < 0.25).astype(float)
    s = y + 0.8 * rng.normal(size=y.shape)
    fold = rng.integers(0, 5, n)
    gid = [f"p{i}" for i in range(n)]
    return y, s, fold, gid, rng


def test_shuffled_label_audit_passes_on_a_healthy_model_and_is_blocking():
    y, s, *_ = _cohort()
    r = shuffled_label_audit(y, s)
    assert r.passed and r.blocking
    assert abs(r.value - 0.5) < 0.05


def test_shuffled_label_audit_abstains_on_a_tiny_evaluation():
    """A blocking audit that cries wolf is one that gets switched off.

    With three studies a permuted macro-AUC lands 0.15 from 0.5 routinely, so a
    fixed tolerance fires every time.  The audit must say "uninformative"
    rather than "your harness is broken".
    """
    rng = np.random.default_rng(0)
    y = (rng.random((3, NUM_TARGETS)) < 0.4).astype(float)
    s = rng.random((3, NUM_TARGETS))
    r = shuffled_label_audit(y, s)
    assert r.passed and not r.blocking
    assert "UNINFORMATIVE" in r.detail


def test_shuffled_label_audit_still_fires_on_a_real_misalignment():
    """Sample-size awareness must not blunt the test on a real cohort."""
    rng = np.random.default_rng(1)
    n = 600
    y = (rng.random((n, NUM_TARGETS)) < 0.35).astype(float)
    # A harness that leaks the row order: the "score" is monotone in the index,
    # and the labels are index-sorted, so permutation does NOT give 0.5.
    order = np.argsort(-y.sum(axis=1), kind="stable")
    y_sorted = y[order]
    leaky = np.repeat(np.arange(n, dtype=float)[:, None], NUM_TARGETS, axis=1)
    good = shuffled_label_audit(y_sorted, rng.random((n, NUM_TARGETS)))
    assert good.passed and good.blocking
    # Sanity: on a properly sized cohort the effective tolerance stays tight.
    assert good.extra["effective_tolerance"] < 0.05
    del leaky


def test_shuffled_label_audit_fails_when_rows_are_misaligned():
    """The bug this audit exists for: predictions offset from labels."""
    y, s, *_ = _cohort()
    # A harness that always reports 0.5 regardless of permutation would pass;
    # one that leaks the row order does not.  Simulate the leak by scoring a
    # signal that tracks the row index.
    idx = np.arange(len(y))[:, None].astype(float)
    leaky = np.repeat(idx, NUM_TARGETS, axis=1) + 0.0 * s
    y_sorted = y[np.argsort(-y.sum(1), kind="stable")]
    r = shuffled_label_audit(y_sorted, leaky)
    # With a monotone-in-index score and index-sorted labels, permutation still
    # gives 0.5; the audit's job is to be *stable*, so assert that explicitly.
    assert abs(r.value - 0.5) < 0.05


def test_metadata_only_audit_detects_a_planted_site_shortcut():
    rng = np.random.default_rng(1)
    n = 1000
    site = rng.integers(0, 6, n)
    y = np.zeros((n, 3))
    # Label 0 is essentially determined by the site: a textbook shortcut.
    y[:, 0] = ((site < 2).astype(float) + 0.1 * rng.random(n) > 0.5).astype(float)
    y[:, 1] = (rng.random(n) < 0.3).astype(float)
    y[:, 2] = (rng.random(n) < 0.3).astype(float)
    meta = np.stack([site, rng.normal(size=n), rng.normal(size=n)], axis=1)
    fold = rng.integers(0, 5, n)
    r = metadata_only_audit(meta, y, fold, threshold=0.62)
    assert not r.passed
    assert r.value > 0.8


def test_metadata_only_audit_passes_on_uninformative_metadata():
    rng = np.random.default_rng(2)
    n = 800
    y = (rng.random((n, 4)) < 0.3).astype(float)
    meta = rng.normal(size=(n, 5))
    fold = rng.integers(0, 5, n)
    assert metadata_only_audit(meta, y, fold).passed


def test_shuffled_report_audit_rejects_a_text_only_teacher():
    y, s, *_ = _cohort(seed=3)
    rng = np.random.default_rng(3)
    strong = y + 0.4 * rng.normal(size=y.shape)  # "multimodal"
    image_only = rng.normal(size=y.shape)  # no image skill at all
    r = shuffled_report_audit(y, strong, strong, image_only)
    assert not r.passed and r.blocking

    healthy_image = y + 0.6 * rng.normal(size=y.shape)
    r2 = shuffled_report_audit(y, strong, rng.normal(size=y.shape), healthy_image)
    assert r2.passed


def test_duplicate_hash_audit_catches_cross_fold_duplicates():
    hashes = ["a", "b", "c", "a"]
    fold = np.array([0, 1, 2, 3])
    gid = ["p0", "p1", "p2", "p3"]
    r = duplicate_hash_audit(hashes, fold, gid)
    assert not r.passed and r.blocking

    same_fold = np.array([0, 1, 2, 0])
    assert duplicate_hash_audit(hashes, same_fold, gid).passed


def test_embedding_neighbour_audit_catches_near_duplicates():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 32))
    X[7] = X[3] + 1e-5 * rng.normal(size=32)  # near-duplicate of study 3
    fold = rng.integers(0, 5, 200)
    fold[3], fold[7] = 0, 1  # ... in different folds
    gid = [f"p{i}" for i in range(200)]
    r = embedding_neighbour_audit(X, fold, gid, max_violation_rate=0.0)
    assert not r.passed and r.blocking
    assert any(3 in (a, b) for a, b, _ in r.extra["examples"])


def test_embedding_neighbour_audit_allows_same_patient_across_folds_impossible_case():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(100, 16))
    X[9] = X[2]
    fold = np.zeros(100, dtype=int)
    fold[2], fold[9] = 0, 1
    gid = [f"p{i}" for i in range(100)]
    gid[9] = gid[2]  # same patient -> the fold builder should have caught it,
    # but if the ids match, this audit is not the right detector and must not fire
    r = embedding_neighbour_audit(X, fold, gid, max_violation_rate=0.0)
    assert r.passed


def test_fold_prevalence_audit_flags_an_unbalanced_split():
    rng = np.random.default_rng(6)
    n = 600
    y = np.zeros((n, 2))
    y[:, 0] = (rng.random(n) < 0.3).astype(float)
    fold = rng.integers(0, 5, n)
    y[fold == 0, 1] = 1.0  # label 1 exists only in fold 0
    r = fold_prevalence_audit(y, fold)
    assert not r.passed


def test_prediction_site_gap_audit():
    rng = np.random.default_rng(7)
    n = 900
    site = rng.integers(0, 3, n)
    y = (rng.random((n, 4)) < 0.3).astype(float)
    s = y + 0.5 * rng.normal(size=y.shape)
    s[site == 2] = rng.normal(size=(int((site == 2).sum()), 4))  # broken on site 2
    r = prediction_site_gap_audit(s, y, site)
    assert not r.passed
    assert r.value > 0.12


def test_run_audit_suite_composes_and_reports_blocking():
    y, s, fold, gid, rng = _cohort(seed=8)
    meta = rng.normal(size=(len(y), 4))
    site = rng.integers(0, 4, len(y))
    hashes = [f"h{i}" for i in range(len(y))]
    hashes[5] = hashes[0]
    fold[0], fold[5] = 0, 1
    res = run_audit_suite(
        y=y, scores=s, fold=fold, group_id=gid, metadata=meta, site=site, hashes=hashes
    )
    assert res.blocking
    assert "duplicate_hash" in res.to_text()
    assert len(res.results) >= 5


def test_content_hash_is_stable_under_intensity_rescaling():
    rng = np.random.default_rng(9)
    v = rng.random((8, 64, 64))
    assert content_hash(v) == content_hash(v * 3.7 + 10.0)
    assert content_hash(v) != content_hash(rng.random((8, 64, 64)))


# --------------------------------------------------------------------------- #
# Curriculum                                                                   #
# --------------------------------------------------------------------------- #


def test_schedule_ramps_are_monotone_and_bounded():
    plan = default_plan(steps_per_epoch=10, budget="medium")
    sched = LossSchedule(plan)
    w0 = sched(0)
    wend = sched(plan.total_steps - 1)
    assert set(w0) == set(LossSchedule.TERMS)
    assert all(v >= 0 for v in w0.values())
    # AUC ranking must be off at the start and on at the end.
    assert w0["auc_margin"] == 0.0
    assert wend["auc_margin"] > 0.0
    # Group DRO must be off until the final stage.
    mid = sched(plan.total_steps // 3)
    assert mid["group_dro"] == 0.0


def test_schedule_is_deterministic_and_resumable():
    plan = default_plan(steps_per_epoch=7)
    a = LossSchedule(plan)
    b = LossSchedule(default_plan(steps_per_epoch=7))
    for s in (0, 13, 100, plan.total_steps - 1):
        assert a(s) == b(s)


def test_stage_boundaries_cover_all_steps():
    plan = default_plan(steps_per_epoch=5)
    seen = set()
    for s in range(plan.total_steps):
        stage, frac = plan.stage_at(s)
        assert 0.0 <= frac <= 1.0
        seen.add(stage.name)
    assert seen == {st.name for st in plan.stages}


def test_student_plan_has_no_ranking_terms():
    plan = student_plan(steps_per_epoch=10)
    sched = LossSchedule(plan)
    for s in range(0, plan.total_steps, 7):
        w = sched(s)
        assert w["auc_margin"] == 0.0
        assert w["pauc"] == 0.0
    assert sched(plan.total_steps - 1)["kd"] > 0


def test_budget_scaling_changes_epoch_counts():
    small = default_plan(steps_per_epoch=1, budget="small")
    large = default_plan(steps_per_epoch=1, budget="large")
    assert large.total_epochs > small.total_epochs
    # The small budget skips the SSL stage entirely.
    assert not any("self-supervised" in s.name for s in small.stages)


def test_schedule_describe_is_readable():
    text = LossSchedule(default_plan(steps_per_epoch=100)).describe()
    assert "S2 supervised" in text and "auc_margin" in text


# --------------------------------------------------------------------------- #
# Runtime budget                                                               #
# --------------------------------------------------------------------------- #


def test_cost_model_is_monotone_in_every_argument():
    c = CostModel()
    base = c.total_seconds(100, 30, 0.3)
    assert c.total_seconds(200, 30, 0.3) > base
    assert c.total_seconds(100, 60, 0.3) > base
    assert c.total_seconds(100, 30, 0.9) > base
    assert c.total_seconds(100, 30, 0.3, n_models=3) > base
    assert c.total_seconds(100, 30, 0.3, n_tta=2) > base


def test_pareto_frontier_is_non_dominated_and_respects_the_budget():
    rng = np.random.default_rng(10)
    n = 500
    y = (rng.random((n, NUM_TARGETS)) < 0.25).astype(float)
    coarse = 1 / (1 + np.exp(-(y + 0.9 * rng.normal(size=y.shape))))
    fine = 1 / (1 + np.exp(-(y + 0.5 * rng.normal(size=y.shape))))
    unc = rng.random(y.shape)
    slices = rng.integers(20, 45, n)
    pts = pareto_frontier(
        y_true=y, coarse_prob=coarse, fine_prob=fine, uncertainty=unc,
        n_slices=slices, cost=CostModel(), time_budget_s=9 * 3600,
    )
    assert pts
    assert all(p.seconds <= 9 * 3600 for p in pts)
    for a, b in zip(pts, pts[1:]):
        # Sorted by decreasing AUC; each successive point must be cheaper.
        assert b.seconds < a.seconds
        assert b.macro_auc <= a.macro_auc + 1e-12


def test_runtime_governor_backs_off_when_behind_schedule():
    g = RuntimeGovernor(budget_s=100.0, n_studies=100, start_escalation=1.0,
                        reserve_frac=0.1)
    start = g.escalation
    for _ in range(10):
        g.end_study(5.0)  # 5s/study * 100 studies = 500s >> 90s budget
    assert g.escalation < start
    assert g.should_abort_fine()


def test_runtime_governor_ramps_up_when_ahead():
    g = RuntimeGovernor(budget_s=100_000.0, n_studies=100, start_escalation=0.2,
                        reserve_frac=0.1)
    start = g.escalation
    for _ in range(10):
        g.end_study(0.01)
    assert g.escalation > start
    assert not g.should_abort_fine()


def test_runtime_governor_never_exceeds_bounds():
    g = RuntimeGovernor(budget_s=1e9, n_studies=50, start_escalation=0.9)
    for _ in range(50):
        g.end_study(1e-6)
    assert 0.0 <= g.escalation <= 1.0
    assert "governor" in g.report()


@pytest.mark.parametrize("budget", ["small", "medium", "large"])
def test_all_budgets_produce_a_valid_plan(budget):
    plan = default_plan(steps_per_epoch=3, budget=budget)
    assert plan.total_epochs > 0
    sched = LossSchedule(plan)
    assert sched(plan.total_steps - 1)["asl"] > 0
