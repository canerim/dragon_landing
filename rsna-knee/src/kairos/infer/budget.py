r"""Runtime governor and the accuracy–runtime Pareto frontier.

The efficiency prize minimises

.. math::
    E = \frac{\max(0,\; A_{\max} - A_{\text{sub}})}{\max(0,\; A_{\max} - A_{\text{base}})}
        \;+\; \frac{t_{\text{sub}}}{t_{\max}}

(up to the exact normalisation the organisers publish), so *runtime enters
linearly while accuracy enters through a normalised deficit*.  The practical
consequence is that trading a small amount of AUC for a large amount of runtime
is almost always correct near the frontier, and the *only* way to know where
"near" is, is to measure the frontier.

Two things live here.

:func:`pareto_frontier`
    Sweep the adaptive-compute knobs -- gate thresholds
    :math:`(\tau_{\text{lo}}, \tau_{\text{hi}}, \tau_U)`, top-:math:`k`, TTA
    count, sequence-skip policy -- on OOF predictions with a *measured* cost
    model, and return the non-dominated set.  The knobs are then read off that
    curve rather than guessed.

:class:`RuntimeGovernor`
    A closed-loop controller for the actual notebook run.  Kaggle gives us nine
    wall-clock hours for an unknown number of test studies; the hidden test set
    may be 3× the public one.  A fixed policy therefore either wastes the budget
    or blows it.  The governor tracks realised throughput and adjusts the
    escalation rate so that the projected finish time lands inside the budget
    with a safety margin:

    .. math::
        \text{escalation}_{t+1} = \operatorname{clip}\Big(
          \text{escalation}_t \cdot \frac{T_{\text{remain}}}
          {\hat c\,(N - n_t)}\,,\; \varepsilon,\; 1\Big),

    where :math:`\hat c` is an EMA of the realised per-study cost.  This is a
    plain proportional controller and that is deliberate: an integral term
    oscillates when the study sizes are heavy-tailed, which they are.

The single most important line in this file is the one that reserves
``reserve_frac`` of the budget: a submission that produces no CSV scores
nothing, so the governor is tuned to *always finish*, degrading to a
coarse-only pass if it must.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

__all__ = ["CostModel", "PolicyPoint", "pareto_frontier", "RuntimeGovernor"]


@dataclass(slots=True)
class CostModel:
    """Measured per-unit costs, in seconds.  Fill these by profiling, not guessing."""

    dicom_decode_per_slice: float = 0.0035
    preprocess_per_slice: float = 0.0012
    coarse_forward_per_slice: float = 0.0021
    fine_forward_per_slice: float = 0.0090
    model_load: float = 12.0
    fixed_overhead: float = 30.0

    def study_seconds(
        self, n_slices: float, fine_fraction: float, *, n_models: int = 1, n_tta: int = 1
    ) -> float:
        decode = n_slices * (self.dicom_decode_per_slice + self.preprocess_per_slice)
        coarse = n_slices * self.coarse_forward_per_slice * n_models * n_tta
        fine = n_slices * fine_fraction * self.fine_forward_per_slice * n_models * n_tta
        return decode + coarse + fine

    def total_seconds(
        self, n_studies: int, mean_slices: float, fine_fraction: float, *,
        n_models: int = 1, n_tta: int = 1
    ) -> float:
        return (
            self.fixed_overhead
            + self.model_load * n_models
            + n_studies * self.study_seconds(
                mean_slices, fine_fraction, n_models=n_models, n_tta=n_tta
            )
        )


@dataclass(slots=True)
class PolicyPoint:
    tau_lo: float
    tau_hi: float
    tau_u: float
    top_k: int
    n_tta: int
    n_models: int
    macro_auc: float
    fine_fraction: float
    seconds: float
    extra: dict = field(default_factory=dict)


def pareto_frontier(
    *,
    y_true: np.ndarray,
    coarse_prob: np.ndarray,
    fine_prob: np.ndarray,
    uncertainty: np.ndarray | None,
    n_slices: np.ndarray,
    cost: CostModel,
    tau_grid: np.ndarray | None = None,
    u_grid: np.ndarray | None = None,
    top_k_grid: tuple[int, ...] = (4, 6, 8),
    tta_grid: tuple[int, ...] = (1, 2),
    model_grid: tuple[int, ...] = (1, 3, 5),
    time_budget_s: float = 9 * 3600,
) -> list[PolicyPoint]:
    r"""Non-dominated (macro-AUC, runtime) policies under the escalation gate.

    ``coarse_prob`` and ``fine_prob`` are the OOF probabilities obtained by
    forcing the fine pass off and on respectively.  The gate then *selects*
    per (study, label) which of the two to use, and the resulting mixed
    prediction is scored -- which is exactly what happens at inference and is
    why this cannot be estimated from a single set of predictions.
    """
    from ..eval.metrics import macro_auc

    tau_grid = tau_grid if tau_grid is not None else np.linspace(0.02, 0.45, 10)
    u_grid = u_grid if u_grid is not None else np.array([np.inf, 0.75, 0.5, 0.25])
    mean_slices = float(np.mean(n_slices))

    points: list[PolicyPoint] = []
    for lo in tau_grid:
        for hi_off in (0.25, 0.45, 0.70):
            hi = min(1.0 - 1e-3, lo + hi_off)
            for tu in u_grid:
                gate = (coarse_prob > lo) & (coarse_prob < hi)
                if uncertainty is not None and np.isfinite(tu):
                    gate = gate | (uncertainty > tu)
                mixed = np.where(gate, fine_prob, coarse_prob)
                auc = macro_auc(y_true, mixed)
                frac = float(gate.any(axis=1).mean())
                for k in top_k_grid:
                    kfrac = min(1.0, frac * k / 6.0)
                    for tta in tta_grid:
                        for nm in model_grid:
                            secs = cost.total_seconds(
                                len(y_true), mean_slices, kfrac, n_models=nm, n_tta=tta
                            )
                            if secs > time_budget_s:
                                continue
                            points.append(
                                PolicyPoint(
                                    float(lo), float(hi), float(tu), k, tta, nm,
                                    float(auc), kfrac, float(secs),
                                )
                            )

    points.sort(key=lambda p: (-p.macro_auc, p.seconds))
    frontier: list[PolicyPoint] = []
    best_time = np.inf
    for p in points:
        if p.seconds < best_time:
            frontier.append(p)
            best_time = p.seconds
    return frontier


class RuntimeGovernor:
    """Closed-loop escalation controller for the inference notebook."""

    def __init__(
        self,
        *,
        budget_s: float = 9 * 3600,
        reserve_frac: float = 0.12,
        n_studies: int,
        ema: float = 0.9,
        min_escalation: float = 0.0,
        max_escalation: float = 1.0,
        start_escalation: float = 0.35,
    ) -> None:
        self.budget = budget_s * (1.0 - reserve_frac)
        self.n_studies = int(n_studies)
        self.ema = ema
        self.min_e = min_escalation
        self.max_e = max_escalation
        self.escalation = float(start_escalation)
        self.t0 = time.monotonic()
        self.done = 0
        self.cost_ema = float("nan")
        self.history: list[tuple[int, float, float]] = []

    def start_study(self) -> float:
        """Return the escalation fraction to use for the next study."""
        return self.escalation

    def end_study(self, elapsed_s: float, *, n_studies_done: int = 1) -> None:
        self.done += n_studies_done
        per = elapsed_s / max(n_studies_done, 1)
        self.cost_ema = per if np.isnan(self.cost_ema) else (
            self.ema * self.cost_ema + (1 - self.ema) * per
        )
        remaining_studies = max(self.n_studies - self.done, 0)
        remaining_time = self.budget - (time.monotonic() - self.t0)
        if remaining_studies == 0:
            return
        projected = self.cost_ema * remaining_studies
        if projected <= 1e-6:
            return
        ratio = remaining_time / projected
        # Proportional control with asymmetric gains: back off fast, ramp slowly.
        gain = 0.6 if ratio < 1.0 else 0.15
        self.escalation = float(
            np.clip(self.escalation * (1.0 + gain * (ratio - 1.0)), self.min_e, self.max_e)
        )
        self.history.append((self.done, remaining_time, self.escalation))

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    @property
    def projected_total(self) -> float:
        if np.isnan(self.cost_ema):
            return float("nan")
        return self.elapsed + self.cost_ema * max(self.n_studies - self.done, 0)

    def should_abort_fine(self) -> bool:
        """Hard stop on escalation when the projection breaches the budget."""
        pt = self.projected_total
        return bool(np.isfinite(pt) and pt > self.budget)

    def report(self) -> str:
        return (
            f"[governor] {self.done}/{self.n_studies} studies  "
            f"elapsed {self.elapsed / 60:.1f} min  "
            f"cost/study {self.cost_ema:.3f}s  "
            f"projected {self.projected_total / 60:.1f} min  "
            f"budget {self.budget / 60:.1f} min  "
            f"escalation {self.escalation:.3f}"
        )
