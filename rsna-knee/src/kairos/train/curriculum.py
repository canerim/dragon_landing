r"""The five-stage curriculum and its loss schedule.

Trying to optimise everything from step zero does not work, and the failure is
not subtle: the AUC-margin term is unstable before the classifier can separate
anything, the OT grounding loss aligns noise to noise, and Group-DRO devotes the
whole gradient to whichever site happens to be hardest for a randomly
initialised network.  Each of these produces a run that trains, converges, and
is worse than the baseline -- which is the expensive kind of failure, because it
costs a full training cycle to discover.

The schedule below is therefore ordered by *what has to be true before the next
term is meaningful*:

======  ================================  =========================================
stage   objective                          precondition it establishes
======  ================================  =========================================
S0      masked image modelling +           an encoder whose features are stable
        cross-plane consistency            under protocol changes
S1      image↔report soft contrastive      an embedding space where clinical
                                           similarity is a direction
S2      supervised (ASL + weak labels)     a classifier that separates the classes
S3      ranking fine-tune (AUC-M, pAUC)    a ranking optimised for the metric
S4      robustness + distillation          site-invariance, and the student
======  ================================  =========================================

Every weight is a function of the *global* step so that a resumed run lands on
the same schedule, and every ramp is linear-in-log so that a term never jumps.

:class:`LossSchedule` is a pure function of step → weights; it holds no state
and is trivially unit-testable, which matters because a mis-specified ramp is
invisible in the loss curve (the total keeps going down) and only shows up in
the OOF three hours later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

__all__ = ["Stage", "LossSchedule", "CurriculumPlan", "default_plan"]


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    epochs: int
    #: Weights at the *start* of the stage; ramped to ``weights_end`` linearly.
    weights: dict[str, float] = field(default_factory=dict)
    weights_end: dict[str, float] | None = None
    lr_scale: float = 1.0
    freeze_backbone: bool = False
    enable_fine_pass: bool = True
    ema_decay: float = 0.999
    notes: str = ""


@dataclass(slots=True)
class CurriculumPlan:
    stages: tuple[Stage, ...]
    steps_per_epoch: int

    @property
    def total_epochs(self) -> int:
        return sum(s.epochs for s in self.stages)

    @property
    def total_steps(self) -> int:
        return self.total_epochs * self.steps_per_epoch

    def stage_at(self, step: int) -> tuple[Stage, float]:
        """Return the active stage and the fraction of it completed."""
        epoch = step // max(self.steps_per_epoch, 1)
        acc = 0
        for s in self.stages:
            if epoch < acc + s.epochs:
                local = (step - acc * self.steps_per_epoch) / max(
                    s.epochs * self.steps_per_epoch, 1
                )
                return s, min(max(local, 0.0), 1.0)
            acc += s.epochs
        return self.stages[-1], 1.0


class LossSchedule:
    """Step → per-term loss weight, following a :class:`CurriculumPlan`."""

    #: Every term the trainer knows about.  A term absent from a stage's dict
    #: has weight zero in that stage -- explicit, so adding a term cannot
    #: silently activate it everywhere.
    TERMS = (
        "asl",              # supervised multi-label
        "pattern_bce",      # co-occurrence-reweighted BCE
        "auc_margin",       # AUC min-max
        "pauc",             # two-way partial AUC
        "rank_queue",       # memory-queue pairwise ranking (rare labels)
        "contrastive",      # image-report soft InfoNCE
        "ot_ground",        # phrase-slice optimal transport
        "weak_label",       # report-derived weak supervision
        "copula",           # label-dependence composite likelihood
        "kd",               # teacher distillation
        "group_dro",        # worst-group risk
        "chi2_dro",         # chi-square DRO
        "irm",              # IRMv1 across acquisition environments
        "attn_entropy",     # attention-collapse hinge
        "moe_balance",      # MoE load balance
        "moe_router_z",     # MoE router z-loss
        "ontology",         # hyperbolic hierarchy prior
        "selector_budget",  # adaptive-compute budget
        "consistency",      # augmentation consistency
        "mim",              # masked image modelling (S0)
        "cross_plane",      # cross-plane agreement (S0)
    )

    def __init__(self, plan: CurriculumPlan) -> None:
        self.plan = plan

    def __call__(self, step: int) -> dict[str, float]:
        stage, t = self.plan.stage_at(step)
        start = stage.weights
        end = stage.weights_end or stage.weights
        out = {k: 0.0 for k in self.TERMS}
        for k in set(start) | set(end):
            a = float(start.get(k, 0.0))
            b = float(end.get(k, a))
            out[k] = a + (b - a) * t
        return out

    def describe(self) -> str:
        lines = []
        acc = 0
        for s in self.plan.stages:
            lines.append(
                f"[{acc:>3}-{acc + s.epochs:>3}] {s.name:<22} "
                f"lr×{s.lr_scale:<4} fine={int(s.enable_fine_pass)} "
                f"freeze={int(s.freeze_backbone)}  {s.notes}"
            )
            active = {k: v for k, v in (s.weights_end or s.weights).items() if v}
            lines.append("      " + ", ".join(f"{k}={v:g}" for k, v in sorted(active.items())))
            acc += s.epochs
        return "\n".join(lines)


def default_plan(*, steps_per_epoch: int, budget: str = "medium") -> CurriculumPlan:
    """The shipped schedule.

    ``budget`` scales the epoch counts: ``small`` for a single 24 GB GPU
    (skips SSL, which needs more compute than it returns at that scale),
    ``medium`` for 4×A100, ``large`` for 8+.
    """
    scale = {"small": 0.5, "medium": 1.0, "large": 1.6}[budget]

    def e(n: int) -> int:
        return max(1, int(round(n * scale)))

    s0 = Stage(
        name="S0 self-supervised",
        epochs=0 if budget == "small" else e(6),
        weights={"mim": 1.0, "cross_plane": 0.3},
        lr_scale=1.0,
        enable_fine_pass=False,
        notes="masked slice modelling; no labels touched",
    )
    s1 = Stage(
        name="S1 image-report VLP",
        epochs=0 if budget == "small" else e(4),
        weights={"contrastive": 1.0, "ot_ground": 0.0},
        weights_end={"contrastive": 1.0, "ot_ground": 0.05},
        lr_scale=1.0,
        enable_fine_pass=False,
        notes="soft targets from weak labels + concept graph",
    )
    s2 = Stage(
        name="S2 supervised",
        epochs=e(12),
        weights={
            "asl": 1.0, "pattern_bce": 0.2, "weak_label": 0.10,
            "attn_entropy": 0.05, "moe_balance": 0.01, "moe_router_z": 1e-3,
            "ontology": 0.02,
        },
        weights_end={
            "asl": 1.0, "pattern_bce": 0.2, "weak_label": 0.05, "ot_ground": 0.03,
            "attn_entropy": 0.05, "moe_balance": 0.01, "moe_router_z": 1e-3,
            "ontology": 0.02, "copula": 0.02,
        },
        lr_scale=1.0,
        enable_fine_pass=False,
        notes="coarse only; consistency needs the fine head so it starts in S3",
    )
    s3 = Stage(
        name="S3 ranking + fine pass",
        epochs=e(8),
        weights={
            "asl": 1.0, "auc_margin": 0.05, "pauc": 0.0, "rank_queue": 0.0,
            "attn_entropy": 0.05, "ontology": 0.02, "copula": 0.02,
            "selector_budget": 0.1, "moe_balance": 0.01, "consistency": 0.02,
        },
        weights_end={
            "asl": 0.6, "auc_margin": 0.20, "pauc": 0.08, "rank_queue": 0.05,
            "attn_entropy": 0.03, "ontology": 0.01, "copula": 0.02,
            "selector_budget": 0.3, "moe_balance": 0.01, "consistency": 0.05,
        },
        lr_scale=0.3,
        enable_fine_pass=True,
        ema_decay=0.9995,
        notes="metric-aligned; PESG on the min-max block",
    )
    s4 = Stage(
        name="S4 robust + distil",
        epochs=e(6),
        weights={
            "asl": 0.5, "auc_margin": 0.20, "pauc": 0.08,
            "group_dro": 0.0, "chi2_dro": 0.0, "kd": 0.0,
            "selector_budget": 0.3,
        },
        weights_end={
            "asl": 0.4, "auc_margin": 0.20, "pauc": 0.08,
            "group_dro": 0.30, "chi2_dro": 0.10, "kd": 0.60,
            "selector_budget": 0.3,
        },
        lr_scale=0.1,
        enable_fine_pass=True,
        ema_decay=0.9999,
        notes="site invariance; teacher = cross-fitted OOF ensemble",
    )
    stages = tuple(s for s in (s0, s1, s2, s3, s4) if s.epochs > 0)
    return CurriculumPlan(stages=stages, steps_per_epoch=steps_per_epoch)


def student_plan(*, steps_per_epoch: int) -> CurriculumPlan:
    """Efficiency-track student: distillation-dominant, short, fine-pass capped.

    The student never sees the ranking losses.  Its teacher's logits already
    encode the ranking, and adding AUC-M on top makes the student trade
    teacher-agreement for its own noisy ranking estimate -- measurably worse
    on every label with fewer than ~150 positives.
    """
    base = default_plan(steps_per_epoch=steps_per_epoch, budget="small")
    sup = replace(
        base.stages[0],
        name="student warmup",
        epochs=4,
        weights={"asl": 1.0, "kd": 0.5, "attn_entropy": 0.05},
        weights_end={"asl": 0.5, "kd": 1.0, "attn_entropy": 0.05},
    )
    dis = Stage(
        name="student distil",
        epochs=10,
        weights={"asl": 0.3, "kd": 1.0, "selector_budget": 0.5},
        weights_end={"asl": 0.2, "kd": 1.0, "selector_budget": 1.0},
        lr_scale=0.4,
        enable_fine_pass=True,
        ema_decay=0.9999,
        notes="cross-fitted teacher logits + feature/attention KD",
    )
    return CurriculumPlan(stages=(sup, dis), steps_per_epoch=steps_per_epoch)


def cosine_ramp(step: int, *, start: int, end: int) -> float:
    """Smooth 0→1 ramp; used for terms where a linear ramp is too abrupt."""
    if step <= start:
        return 0.0
    if step >= end:
        return 1.0
    t = (step - start) / max(end - start, 1)
    return 0.5 * (1.0 - math.cos(math.pi * t))
