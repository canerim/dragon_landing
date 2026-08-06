r"""Evaluation: the competition metric, and the statistics needed to trust it.

Macro-AUC on ~5 000 studies with twelve labels of prevalence 1–40 % is a noisy
number.  A 0.002 macro-AUC difference between two models is, for most label
mixes, well inside the sampling noise -- and picking between models on that
difference is how a leaderboard-chasing pipeline ends up 30 places down on the
private split.  So this module ships not just ``macro_auc`` but the machinery to
say *whether a difference is real*:

``delong_auc_variance`` / ``delong_test``
    DeLong's non-parametric estimator of the (co)variance of one or more
    empirical AUCs, using the fast :math:`O(n\log n)` midrank formulation of
    Sun & Xu (2014).  Because the AUC is a two-sample U-statistic, its variance
    has the closed form

    .. math::
        \mathrm{Var}(\hat A) = \frac{S_{10}}{m} + \frac{S_{01}}{n},

    with :math:`S_{10}, S_{01}` the empirical variances of the structural
    components :math:`V^{(10)}_i = \frac1n\sum_j \psi(x_i, y_j)` and
    :math:`V^{(01)}_j = \frac1m\sum_i \psi(x_i,y_j)`.  For *paired* models the
    same components give the covariance, so the test for "is model A better
    than model B on this label" is a z-test with an exact variance rather than
    a bootstrap.  It is orders of magnitude cheaper than bootstrapping and it
    is the right tool for per-label comparisons.

``patient_bootstrap``
    Cluster bootstrap at the **patient** level (not study, not slice).  Studies
    of the same patient are correlated; resampling studies independently
    understates the variance, sometimes by a factor of two.  Also supports
    stratification by site, so the CI reflects the sampling design.

``macro_auc_ci``
    Bootstrap CI for the macro-average itself, which is *not* the average of
    the per-label CIs -- per-label errors are correlated through the shared
    patient sample, and ignoring that correlation gives an interval that is too
    wide by roughly √L.

Also here: expected calibration error with a debiased estimator, Brier
decomposition, and the "worst-label AUC" model-selection criterion that the
competition's macro average makes decisive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "roc_auc",
    "macro_auc",
    "per_label_auc",
    "delong_auc_variance",
    "delong_test",
    "patient_bootstrap",
    "macro_auc_ci",
    "expected_calibration_error",
    "brier_decomposition",
    "EvaluationReport",
    "evaluate",
]

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# AUC                                                                          #
# --------------------------------------------------------------------------- #


def _midrank(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, in O(n log n)."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and xs[j + 1] == xs[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Empirical ROC-AUC (Mann–Whitney U), ties counted as 1/2."""
    y = np.asarray(y_true).astype(np.float64).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    finite = np.isfinite(y) & np.isfinite(s)
    y, s = y[finite], s[finite]
    pos, neg = y > 0.5, y <= 0.5
    m, n = int(pos.sum()), int(neg.sum())
    if m == 0 or n == 0:
        return float("nan")
    r = _midrank(s)
    return float((r[pos].sum() - m * (m + 1) / 2.0) / (m * n))


def per_label_auc(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    return np.array(
        [roc_auc(y_true[:, l], y_score[:, l]) for l in range(y_true.shape[1])],
        dtype=np.float64,
    )


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """The competition metric: unweighted mean of per-label AUCs."""
    a = per_label_auc(y_true, y_score)
    finite = np.isfinite(a)
    return float(np.mean(a[finite])) if finite.any() else float("nan")


# --------------------------------------------------------------------------- #
# DeLong                                                                       #
# --------------------------------------------------------------------------- #


def _delong_components(
    scores: np.ndarray, pos: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Structural components for K score vectors on one label.

    ``scores``: ``(K, N)``; ``pos``: ``(N,)`` boolean.
    Returns ``(auc (K,), V10 (K,m), V01 (K,n))``.
    """
    K = scores.shape[0]
    x = scores[:, pos]  # positives
    y = scores[:, ~pos]  # negatives
    m, n = x.shape[1], y.shape[1]

    tx = np.empty((K, m))
    ty = np.empty((K, n))
    tz = np.empty((K, m + n))
    for k in range(K):
        tx[k] = _midrank(x[k])
        ty[k] = _midrank(y[k])
        tz[k] = _midrank(np.concatenate([x[k], y[k]]))

    auc = (tz[:, :m].sum(axis=1) / (m * n)) - (m + 1) / (2.0 * n)
    v10 = (tz[:, :m] - tx) / n
    v01 = 1.0 - (tz[:, m:] - ty) / m
    return auc, v10, v01


def delong_auc_variance(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """DeLong AUCs and their covariance matrix for K paired score vectors.

    ``scores``: ``(K, N)``.  Returns ``(auc (K,), cov (K, K))``.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    scores = np.atleast_2d(np.asarray(scores, dtype=np.float64))
    pos = y > 0.5
    m, n = int(pos.sum()), int((~pos).sum())
    if m == 0 or n == 0:
        K = scores.shape[0]
        return np.full(K, np.nan), np.full((K, K), np.nan)

    auc, v10, v01 = _delong_components(scores, pos)
    s10 = np.cov(v10, ddof=1) if v10.shape[0] > 1 else np.array([[np.var(v10[0], ddof=1)]])
    s01 = np.cov(v01, ddof=1) if v01.shape[0] > 1 else np.array([[np.var(v01[0], ddof=1)]])
    s10 = np.atleast_2d(s10)
    s01 = np.atleast_2d(s01)
    cov = s10 / m + s01 / n
    return auc, cov


def delong_test(
    y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray
) -> dict[str, float]:
    r"""Paired DeLong test for :math:`H_0: A_a = A_b` on one label.

    Returns the AUCs, their difference, the standard error of the difference
    :math:`\sqrt{\sigma_a^2 + \sigma_b^2 - 2\sigma_{ab}}`, the z statistic and
    the two-sided p-value.  The covariance term is the whole point: two models
    scored on the same studies are strongly positively correlated, so the naive
    unpaired SE overstates the uncertainty by a large factor and hides real
    improvements.
    """
    auc, cov = delong_auc_variance(y_true, np.stack([score_a.ravel(), score_b.ravel()]))
    if not np.all(np.isfinite(auc)):
        return {"auc_a": float("nan"), "auc_b": float("nan"), "diff": float("nan"),
                "se": float("nan"), "z": float("nan"), "p_value": float("nan")}
    var = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    se = float(np.sqrt(max(var, _EPS)))
    diff = float(auc[0] - auc[1])
    z = diff / se if se > 0 else 0.0
    from math import erfc, sqrt

    p = float(erfc(abs(z) / sqrt(2.0)))
    return {"auc_a": float(auc[0]), "auc_b": float(auc[1]), "diff": diff,
            "se": se, "z": float(z), "p_value": p}


# --------------------------------------------------------------------------- #
# Bootstrap                                                                    #
# --------------------------------------------------------------------------- #


def patient_bootstrap(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_id: np.ndarray,
    *,
    statistic=macro_auc,
    n_boot: int = 2000,
    strata: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Cluster bootstrap over patients, optionally stratified by site.

    Resamples *clusters* with replacement, keeping every study of a resampled
    patient.  Stratification keeps the site composition of each replicate equal
    to the observed one, which is the correct design when the site mix is fixed
    by the challenge organisers rather than sampled.
    """
    rng = np.random.default_rng(seed)
    gids, inv = np.unique(np.asarray(group_id), return_inverse=True)
    idx_by_group = [np.flatnonzero(inv == g) for g in range(len(gids))]

    if strata is None:
        strata_of_group = np.zeros(len(gids), dtype=np.int64)
    else:
        strata = np.asarray(strata)
        strata_of_group = np.array(
            [strata[idx_by_group[g]][0] for g in range(len(gids))]
        )
        _, strata_of_group = np.unique(strata_of_group, return_inverse=True)

    groups_by_stratum = [
        np.flatnonzero(strata_of_group == s) for s in range(strata_of_group.max() + 1)
    ]

    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        picks = []
        for gs in groups_by_stratum:
            if gs.size == 0:
                continue
            chosen = rng.integers(0, gs.size, size=gs.size)
            picks.extend(gs[chosen])
        rows = np.concatenate([idx_by_group[g] for g in picks]) if picks else np.array([], int)
        out[b] = statistic(y_true[rows], y_score[rows]) if rows.size else np.nan
    return out


def macro_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_id: np.ndarray,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    strata: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap CI for the macro-AUC."""
    point = macro_auc(y_true, y_score)
    draws = patient_bootstrap(
        y_true, y_score, group_id, n_boot=n_boot, strata=strata, seed=seed
    )
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Calibration                                                                  #
# --------------------------------------------------------------------------- #


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 15, adaptive: bool = True
) -> float:
    r"""ECE with equal-mass (adaptive) bins and a debiasing correction.

    Equal-width bins are the usual choice and are badly biased when the
    predictions concentrate near 0 -- which is exactly the regime for a 1 %-
    prevalence label, where 14 of 15 equal-width bins are empty.  Equal-mass
    bins fix that.  The per-bin debiasing subtracts the sampling variance
    :math:`\bar p(1-\bar p)/n_b` that would produce a nonzero ECE even for a
    perfectly calibrated model.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if y.size == 0:
        return float("nan")

    if adaptive:
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        edges = np.unique(edges)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    total = 0.0
    for i in range(len(edges) - 1):
        m = (p >= edges[i]) & (p < edges[i + 1]) if i < len(edges) - 2 else (p >= edges[i])
        nb = int(m.sum())
        if nb == 0:
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        gap = (conf - acc) ** 2 - conf * (1 - conf) / nb
        total += nb / y.size * np.sqrt(max(gap, 0.0))
    return float(total)


def brier_decomposition(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 15
                        ) -> dict[str, float]:
    r"""Murphy decomposition: Brier = reliability − resolution + uncertainty."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if y.size == 0:
        return {"brier": float("nan"), "reliability": float("nan"),
                "resolution": float("nan"), "uncertainty": float("nan")}
    base = float(y.mean())
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    rel = res = 0.0
    for i in range(len(edges) - 1):
        m = (p >= edges[i]) & (p < edges[i + 1]) if i < len(edges) - 2 else (p >= edges[i])
        nb = int(m.sum())
        if nb == 0:
            continue
        pk, ok_ = float(p[m].mean()), float(y[m].mean())
        rel += nb / y.size * (pk - ok_) ** 2
        res += nb / y.size * (ok_ - base) ** 2
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(base * (1 - base)),
    }


# --------------------------------------------------------------------------- #
# Report                                                                       #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class EvaluationReport:
    macro_auc: float
    macro_ci: tuple[float, float]
    per_label_auc: np.ndarray
    per_label_se: np.ndarray
    worst_label_auc: float
    worst_label_name: str
    per_fold_macro: np.ndarray
    fold_std: float
    ece: np.ndarray
    brier: np.ndarray
    n_studies: int
    n_patients: int

    def to_text(self, target_names) -> str:
        names = list(target_names)
        w = max(len(n) for n in names) + 2
        lines = [
            f"macro AUC  {self.macro_auc:.5f}   "
            f"95% CI [{self.macro_ci[0]:.5f}, {self.macro_ci[1]:.5f}]",
            f"worst label  {self.worst_label_name}  {self.worst_label_auc:.5f}",
            f"fold macro  {np.array2string(self.per_fold_macro, precision=5)}  "
            f"(sd {self.fold_std:.5f})",
            f"n = {self.n_studies} studies / {self.n_patients} patients",
            "",
            "label".ljust(w) + "   AUC      ±SE      ECE      Brier",
            "-" * (w + 38),
        ]
        for l, nm in enumerate(names):
            lines.append(
                nm.ljust(w)
                + f" {self.per_label_auc[l]:.5f}  {self.per_label_se[l]:.5f}"
                + f"  {self.ece[l]:.5f}  {self.brier[l]:.5f}"
            )
        return "\n".join(lines)


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    group_id: np.ndarray,
    fold: np.ndarray | None = None,
    target_names=None,
    site: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> EvaluationReport:
    from ..constants import TARGETS

    names = list(target_names or TARGETS)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    L = y_true.shape[1]

    auc = per_label_auc(y_true, y_score)
    se = np.empty(L)
    for l in range(L):
        _, cov = delong_auc_variance(y_true[:, l], y_score[None, :, l])
        se[l] = float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else np.nan

    point, lo, hi = macro_auc_ci(
        y_true, y_score, group_id, n_boot=n_boot, strata=site, seed=seed
    )

    if fold is not None:
        folds = np.unique(fold)
        per_fold = np.array([macro_auc(y_true[fold == f], y_score[fold == f]) for f in folds])
    else:
        per_fold = np.array([point])

    ece = np.array([expected_calibration_error(y_true[:, l], y_score[:, l]) for l in range(L)])
    brier = np.array(
        [brier_decomposition(y_true[:, l], y_score[:, l])["brier"] for l in range(L)]
    )

    finite = np.isfinite(auc)
    worst_idx = int(np.argmin(np.where(finite, auc, np.inf)))
    return EvaluationReport(
        macro_auc=point,
        macro_ci=(lo, hi),
        per_label_auc=auc,
        per_label_se=se,
        worst_label_auc=float(auc[worst_idx]),
        worst_label_name=names[worst_idx],
        per_fold_macro=per_fold,
        fold_std=float(np.nanstd(per_fold)),
        ece=ece,
        brier=brier,
        n_studies=int(y_true.shape[0]),
        n_patients=int(len(np.unique(group_id))),
    )
