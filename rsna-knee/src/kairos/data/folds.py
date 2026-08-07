"""Patient-safe, multi-objective cross-validation splits.

The competition metric is a macro-AUC over twelve labels whose prevalences
differ by more than an order of magnitude, computed on studies drawn from 16
sites, five continents and ~12 report languages.  A split that is merely
"stratified" on the marginal label frequencies is not good enough: it leaves
the *rare* labels (fracture, synovitis) with fold-to-fold prevalence swings
large enough that an AUC difference of 0.004 -- the size of a real
improvement -- is indistinguishable from split noise.

This module builds folds by explicitly minimising a scalarised objective

.. math::

    J(\\pi) \\;=\\; \\lambda_{\\text{mar}} \\underbrace{\\sum_{k}\\sum_{l}
        \\frac{w_l\\,(p_{kl} - p_l)^2}{p_l(1-p_l) + \\varepsilon}}_{\\text{marginal prevalence}}
      \\;+\\; \\lambda_{\\text{co}} \\underbrace{\\sum_k \\lVert C_k - C \\rVert_F^2}_{\\text{co-occurrence}}
      \\;+\\; \\lambda_{\\text{cov}} \\underbrace{\\sum_k \\sum_c
        \\mathrm{KL}\\!\\left(q_{kc} \\,\\|\\, q_c\\right)}_{\\text{site / language / scanner}}
      \\;+\\; \\lambda_{\\text{sz}} \\underbrace{\\sum_k (n_k - \\bar n)^2}_{\\text{fold size}}

over assignments :math:`\\pi` of **patient groups** (never studies) to folds,
subject to the hard constraint that a group is indivisible.  Rare labels get a
larger :math:`w_l` because a one-study swing matters more there; the
:math:`p_l(1-p_l)` denominator is the variance of a Bernoulli draw, so the
first term is a chi-square-like statistic rather than a raw squared error and
is therefore comparable across labels of very different prevalence.

Optimisation is a two-phase procedure:

1. **Seed** with group-level iterative stratification (Sechidis, Tsoumakas &
   Vlahavas, 2011), which greedily places the group carrying the rarest
   still-unbalanced label into the fold with the largest deficit for that
   label.  This gets the marginals close and is O(G · L).
2. **Refine** with simulated annealing over *move* and *swap* proposals,
   accepting with the Metropolis criterion.  Every proposal's objective delta
   is computed incrementally in O(L² + C) rather than by recomputing J, which
   is what makes 10⁵ proposals affordable.

The result is written once and treated as an immutable artefact for the rest of
the competition: `fold_hash` is stored alongside every checkpoint so an OOF
matrix can never be silently mixed across split versions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "FoldSpec",
    "FoldResult",
    "make_folds",
    "fold_report",
    "assignment_hash",
]

_EPS = 1e-9


@dataclass(slots=True)
class FoldSpec:
    n_folds: int = 5
    seed: int = 20261022
    lambda_marginal: float = 1.0
    lambda_cooccurrence: float = 0.35
    lambda_covariate: float = 0.50
    lambda_size: float = 0.15
    #: Extra weight applied to label ``l`` as ``(p_max / p_l) ** rare_power``.
    #: 0 → all labels equal, 1 → weight inversely proportional to prevalence.
    #: 0.5 is the geometric compromise and is what we ship.
    rare_power: float = 0.5
    n_anneal_steps: int = 60_000
    t_start: float = 1.0
    t_end: float = 1e-3
    swap_fraction: float = 0.5


@dataclass(slots=True)
class FoldResult:
    group_ids: list[str]
    group_fold: np.ndarray  # (G,) int
    row_fold: np.ndarray  # (N,) int, broadcast back to studies
    objective: float
    objective_trace: np.ndarray
    diagnostics: dict[str, object] = field(default_factory=dict)
    fold_hash: str = ""


# --------------------------------------------------------------------------- #
# Objective                                                                    #
# --------------------------------------------------------------------------- #


class _Objective:
    """Incrementally-maintained scalarised split objective.

    State per fold ``k``:
      ``S[k, l]``   = number of positives of label l in fold k
      ``Co[k, l, m]`` = number of studies in fold k positive for both l and m
      ``Cov[c][k, v]`` = number of studies in fold k with covariate c taking value v
      ``n[k]``      = number of studies in fold k
    """

    def __init__(
        self,
        y: np.ndarray,  # (G, L) positive counts per group
        sizes: np.ndarray,  # (G,) studies per group
        covariates: Sequence[np.ndarray],  # each (G, V_c) counts per group
        spec: FoldSpec,
    ) -> None:
        self.y = y.astype(np.float64)
        self.sizes = sizes.astype(np.float64)
        self.cov = [c.astype(np.float64) for c in covariates]
        self.spec = spec
        self.K = spec.n_folds
        self.G, self.L = y.shape

        total_n = float(self.sizes.sum())
        total_pos = self.y.sum(axis=0)
        p = total_pos / max(total_n, _EPS)
        self.p = p
        # chi-square style denominator; floored so a label with zero or full
        # prevalence cannot produce an infinite weight.
        self.denom = np.maximum(p * (1.0 - p), 1.0 / max(total_n, 1.0))
        pmax = max(float(p.max()), _EPS)
        with np.errstate(divide="ignore"):
            w = (pmax / np.maximum(p, _EPS)) ** spec.rare_power
        self.w = np.where(p > 0, w, 0.0)

        # Group-level co-occurrence contribution: for a group with label counts
        # y_g we use the outer product y_g y_g^T as the group's contribution to
        # the fold co-occurrence matrix.  For single-study groups this is exact.
        self.y_outer = np.einsum("gl,gm->glm", self.y, self.y)
        self.C_target = self.y_outer.sum(axis=0) / max(float(self.K), 1.0)

        self.cov_target = [c.sum(axis=0) / float(self.K) for c in self.cov]
        self.n_target = total_n / float(self.K)

        self.S = np.zeros((self.K, self.L))
        self.Co = np.zeros((self.K, self.L, self.L))
        self.Cn = [np.zeros((self.K, c.shape[1])) for c in self.cov]
        self.n = np.zeros(self.K)

    # -- full recompute (used for validation / final reporting) ------------- #

    def value(self) -> float:
        spec = self.spec
        n_safe = np.maximum(self.n, _EPS)
        p_k = self.S / n_safe[:, None]
        marg = float(
            np.sum(self.w[None, :] * (p_k - self.p[None, :]) ** 2 / self.denom[None, :])
        )
        co = float(np.sum((self.Co - self.C_target[None]) ** 2))
        co /= max(float(self.C_target.sum()) ** 2, _EPS) / self.L
        cov = 0.0
        for Cn, tgt in zip(self.Cn, self.cov_target):
            cov += float(np.sum((Cn - tgt[None]) ** 2)) / max(float(tgt.sum()) ** 2, _EPS)
        size = float(np.sum((self.n - self.n_target) ** 2)) / max(self.n_target**2, _EPS)
        return (
            spec.lambda_marginal * marg
            + spec.lambda_cooccurrence * co
            + spec.lambda_covariate * cov
            + spec.lambda_size * size
        )

    # -- incremental mutation ---------------------------------------------- #

    def add(self, g: int, k: int) -> None:
        self.S[k] += self.y[g]
        self.Co[k] += self.y_outer[g]
        for Cn, c in zip(self.Cn, self.cov):
            Cn[k] += c[g]
        self.n[k] += self.sizes[g]

    def remove(self, g: int, k: int) -> None:
        self.S[k] -= self.y[g]
        self.Co[k] -= self.y_outer[g]
        for Cn, c in zip(self.Cn, self.cov):
            Cn[k] -= c[g]
        self.n[k] -= self.sizes[g]

    def delta_move(self, g: int, src: int, dst: int) -> float:
        """Objective change from moving group ``g`` from ``src`` to ``dst``.

        Evaluated by mutate → value → un-mutate.  The mutation touches only two
        rows of each state array, so this is O(L² + ΣV_c) even though ``value``
        is written as a full reduction -- for L = 12 and a handful of
        covariates that is a few hundred flops, and it keeps the incremental
        path provably consistent with the batch path (see
        ``tests/test_folds.py::test_incremental_matches_batch``).
        """
        before = self.value()
        self.remove(g, src)
        self.add(g, dst)
        after = self.value()
        self.remove(g, dst)
        self.add(g, src)
        return after - before


# --------------------------------------------------------------------------- #
# Phase 1: group-level iterative stratification                                #
# --------------------------------------------------------------------------- #


def _iterative_stratification(
    y: np.ndarray, sizes: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Sechidis et al. iterative stratification, lifted to indivisible groups."""
    G, L = y.shape
    desired_per_label = y.sum(axis=0)[None, :] / k  # (1, L) -> broadcast
    desired_per_label = np.repeat(desired_per_label, k, axis=0)  # (k, L)
    desired_size = np.full(k, sizes.sum() / k)

    remaining = np.ones(G, dtype=bool)
    assign = np.full(G, -1, dtype=np.int64)

    while remaining.any():
        rem_counts = y[remaining].sum(axis=0)
        # Rarest label that still has any positive left to place.
        positive = np.where(rem_counts > 0)[0]
        if positive.size == 0:
            # Only all-negative groups left: distribute by size deficit.
            idx = np.where(remaining)[0]
            order = idx[np.argsort(-sizes[idx])]
            for g in order:
                kbest = int(np.argmax(desired_size))
                assign[g] = kbest
                desired_size[kbest] -= sizes[g]
                remaining[g] = False
            break

        l = int(positive[np.argmin(rem_counts[positive])])
        candidates = np.where(remaining & (y[:, l] > 0))[0]
        # Largest contributors first -- placing a heavy group late leaves no
        # room to compensate.
        candidates = candidates[np.argsort(-y[candidates, l], kind="stable")]

        for g in candidates:
            col = desired_per_label[:, l]
            best = np.flatnonzero(col == col.max())
            if best.size > 1:
                sz = desired_size[best]
                best = best[np.flatnonzero(sz == sz.max())]
            kbest = int(best[0]) if best.size == 1 else int(rng.choice(best))
            assign[g] = kbest
            desired_per_label[kbest] -= y[g]
            desired_size[kbest] -= sizes[g]
            remaining[g] = False

    return assign


# --------------------------------------------------------------------------- #
# Phase 2: simulated annealing refinement                                      #
# --------------------------------------------------------------------------- #


def _anneal(
    obj: _Objective, assign: np.ndarray, spec: FoldSpec, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    G = assign.shape[0]
    K = spec.n_folds
    best = assign.copy()
    best_val = obj.value()
    cur_val = best_val
    trace = np.empty(spec.n_anneal_steps // 200 + 1, dtype=np.float64)
    trace[0] = cur_val
    ti = 1

    log_t0, log_t1 = np.log(spec.t_start), np.log(spec.t_end)

    for step in range(spec.n_anneal_steps):
        t = float(np.exp(log_t0 + (log_t1 - log_t0) * step / max(spec.n_anneal_steps - 1, 1)))

        if rng.random() < spec.swap_fraction:
            g1, g2 = rng.integers(0, G, size=2)
            k1, k2 = int(assign[g1]), int(assign[g2])
            if k1 == k2 or g1 == g2:
                continue
            # Composite delta: apply both moves, measure, revert.
            before = cur_val
            obj.remove(g1, k1); obj.remove(g2, k2)
            obj.add(g1, k2); obj.add(g2, k1)
            after = obj.value()
            d = after - before
            if d <= 0 or rng.random() < np.exp(-d / max(t, _EPS)):
                assign[g1], assign[g2] = k2, k1
                cur_val = after
            else:
                obj.remove(g1, k2); obj.remove(g2, k1)
                obj.add(g1, k1); obj.add(g2, k2)
        else:
            g = int(rng.integers(0, G))
            src = int(assign[g])
            dst = int(rng.integers(0, K))
            if dst == src:
                continue
            before = cur_val
            obj.remove(g, src); obj.add(g, dst)
            after = obj.value()
            d = after - before
            if d <= 0 or rng.random() < np.exp(-d / max(t, _EPS)):
                assign[g] = dst
                cur_val = after
            else:
                obj.remove(g, dst); obj.add(g, src)

        if cur_val < best_val:
            best_val = cur_val
            best[:] = assign
        if (step + 1) % 200 == 0 and ti < trace.size:
            trace[ti] = cur_val
            ti += 1

    return best, trace[:ti]


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #


def _one_hot_counts(values: Sequence[object], group_index: np.ndarray, n_groups: int) -> np.ndarray:
    """Aggregate a categorical covariate into (G, V) per-group counts."""
    cats = sorted({("__NA__" if v is None else str(v)) for v in values})
    lookup = {c: i for i, c in enumerate(cats)}
    out = np.zeros((n_groups, len(cats)), dtype=np.float64)
    for row, v in enumerate(values):
        out[group_index[row], lookup["__NA__" if v is None else str(v)]] += 1.0
    return out


def make_folds(
    *,
    group_id: Sequence[str],
    labels: np.ndarray,
    covariates: Mapping[str, Sequence[object]] | None = None,
    spec: FoldSpec | None = None,
) -> FoldResult:
    """Build patient-safe folds.

    Parameters
    ----------
    group_id
        Per-study patient/group identifier.  Studies sharing an id are
        guaranteed to land in the same fold.  When the competition provides no
        patient id, pass a surrogate built from a de-duplication hash of the
        study (see :func:`kairos.data.dedup.surrogate_patient_id`) -- an
        imperfect surrogate is strictly better than pretending studies are
        independent.
    labels
        ``(N, 12)`` binary label matrix in :data:`kairos.constants.TARGETS`
        order.  ``NaN`` is treated as negative for balancing purposes only.
    covariates
        Optional categorical nuisance variables to balance across folds
        (``site``, ``report_language``, ``manufacturer``, ``field_strength``,
        ``n_sequences`` bucketed, ...).
    """
    spec = spec or FoldSpec()
    rng = np.random.default_rng(spec.seed)

    gid = [str(g) for g in group_id]
    labels = np.asarray(labels, dtype=np.float64)
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2-D, got shape {labels.shape}")
    if len(gid) != labels.shape[0]:
        raise ValueError("group_id and labels have different lengths")

    uniq = sorted(set(gid))
    gpos = {g: i for i, g in enumerate(uniq)}
    group_index = np.array([gpos[g] for g in gid], dtype=np.int64)
    G = len(uniq)

    y = np.zeros((G, labels.shape[1]), dtype=np.float64)
    np.add.at(y, group_index, np.nan_to_num(labels, nan=0.0))
    sizes = np.zeros(G, dtype=np.float64)
    np.add.at(sizes, group_index, 1.0)

    cov_mats: list[np.ndarray] = []
    cov_names: list[str] = []
    for name, values in (covariates or {}).items():
        if len(values) != len(gid):
            raise ValueError(f"covariate {name!r} has wrong length")
        cov_mats.append(_one_hot_counts(list(values), group_index, G))
        cov_names.append(name)

    obj = _Objective(y, sizes, cov_mats, spec)
    assign = _iterative_stratification(y, sizes, spec.n_folds, rng)
    for g in range(G):
        obj.add(g, int(assign[g]))
    seed_value = obj.value()

    assign, trace = _anneal(obj, assign, spec, rng)

    # Recompute state for the *best* assignment so diagnostics are exact.
    obj_final = _Objective(y, sizes, cov_mats, spec)
    for g in range(G):
        obj_final.add(g, int(assign[g]))
    final_value = obj_final.value()

    row_fold = assign[group_index]
    diagnostics = {
        "n_groups": G,
        "n_rows": int(labels.shape[0]),
        "seed_objective": float(seed_value),
        "final_objective": float(final_value),
        "improvement": float(seed_value - final_value),
        "covariates": cov_names,
        "per_fold_size": obj_final.n.tolist(),
        "per_fold_prevalence": (obj_final.S / np.maximum(obj_final.n, _EPS)[:, None]).tolist(),
        "global_prevalence": obj_final.p.tolist(),
        "max_abs_prevalence_deviation": float(
            np.max(
                np.abs(
                    obj_final.S / np.maximum(obj_final.n, _EPS)[:, None]
                    - obj_final.p[None, :]
                )
            )
        ),
    }

    return FoldResult(
        group_ids=uniq,
        group_fold=assign.astype(np.int64),
        row_fold=row_fold.astype(np.int64),
        objective=float(final_value),
        objective_trace=trace,
        diagnostics=diagnostics,
        fold_hash=assignment_hash(uniq, assign, spec),
    )


def assignment_hash(group_ids: Sequence[str], assign: np.ndarray, spec: FoldSpec) -> str:
    """Stable content hash of a split, stored next to every checkpoint."""
    payload = {
        "spec": {
            "n_folds": spec.n_folds,
            "seed": spec.seed,
            "lambda_marginal": spec.lambda_marginal,
            "lambda_cooccurrence": spec.lambda_cooccurrence,
            "lambda_covariate": spec.lambda_covariate,
            "lambda_size": spec.lambda_size,
            "rare_power": spec.rare_power,
        },
        "assignment": {g: int(k) for g, k in zip(group_ids, assign)},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def fold_report(result: FoldResult, target_names: Iterable[str]) -> str:
    """Human-readable per-fold prevalence table."""
    names = list(target_names)
    prev = np.asarray(result.diagnostics["per_fold_prevalence"], dtype=float)
    glob = np.asarray(result.diagnostics["global_prevalence"], dtype=float)
    sizes = np.asarray(result.diagnostics["per_fold_size"], dtype=float)

    width = max(len(n) for n in names) + 2
    header = "label".ljust(width) + "".join(f"  fold{k}" for k in range(prev.shape[0]))
    header += "    all      maxdev"
    lines = [header, "-" * len(header)]
    for l, name in enumerate(names):
        row = name.ljust(width)
        row += "".join(f"  {prev[k, l]:.4f}" for k in range(prev.shape[0]))
        row += f"  {glob[l]:.4f}"
        row += f"  {np.max(np.abs(prev[:, l] - glob[l])):.4f}"
        lines.append(row)
    lines.append("-" * len(header))
    lines.append("n".ljust(width) + "".join(f"  {int(s):6d}" for s in sizes))
    lines.append(f"objective = {result.objective:.6f}   hash = {result.fold_hash[:16]}")
    return "\n".join(lines)
