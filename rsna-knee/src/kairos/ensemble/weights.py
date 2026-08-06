r"""Per-label ensemble weighting that does not overfit the OOF matrix.

Twelve labels × :math:`M` models is :math:`12(M-1)` free parameters fitted on
an OOF matrix whose *effective* sample size for the rarest label is the number
of positives -- perhaps 40.  Unconstrained per-label weight search on that is a
reliable way to gain 0.004 on OOF and lose 0.003 on the private split.  This is
the single most common late-stage self-inflicted wound in medical-imaging
competitions.

Three mechanisms make per-label weighting safe here:

**1. Mirror descent on the simplex with an entropic anchor.**  We maximise a
smooth AUC surrogate subject to :math:`w_l \in \Delta^{M-1}` using exponentiated
gradient (mirror descent with the KL Bregman divergence), whose iterate

.. math::
    w^{(t+1)}_{lm} \;\propto\; w^{(t)}_{lm}\,
      \exp\!\big(\eta\,\partial_m \mathcal A_l - \eta\lambda\,
      \log \tfrac{w^{(t)}_{lm}}{u_m}\big)

stays feasible by construction and is *anchored* to the uniform weights
:math:`u`.  The anchor strength :math:`\lambda` is the regularisation dial: at
:math:`\lambda\to\infty` we recover the simple average.

**2. Hierarchical shrinkage along the ontology.**  A rare label's weights are
shrunk towards its *mechanism group's* weights rather than towards uniform:

.. math::
    \tilde w_l = (1-\kappa_l)\,w_l + \kappa_l\,\bar w_{g(l)},
    \qquad \kappa_l = \frac{\sigma^2_l}{\sigma^2_l + n^+_l\,\tau^2},

which is the James–Stein / empirical-Bayes shrinkage factor with
:math:`n^+_l` the positive count.  Fracture borrows Contusion's weights, which
is a far better prior than uniform.

**3. Nested leave-one-fold-out selection.**  Weights for fold :math:`k` are
fitted on folds :math:`\ne k` only, and the reported gain is measured on
:math:`k`.  If the "gain" evaporates under nesting -- and for aggressive
weighting it does -- you learn that *before* the submission deadline instead of
after.

:class:`RankAverageEnsemble` and :func:`greedy_selection` complete the toolkit.
Rank averaging is scale-free and is the right default when models disagree in
calibration; it is, however, useless for distillation (ranks are not
probabilities), so the distillation teacher always uses probability averaging.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..eval.metrics import macro_auc, per_label_auc, roc_auc

__all__ = [
    "EnsembleWeights",
    "fit_label_weights",
    "nested_evaluate",
    "rank_transform",
    "greedy_selection",
    "bayesian_model_average",
]

_EPS = 1e-12


def rank_transform(scores: np.ndarray) -> np.ndarray:
    """Map each column to :math:`(\\mathrm{rank}-1)/(N-1) \\in [0,1]`.

    Applied per model per label before averaging.  AUC is invariant to this
    transform for a *single* model but not for a *mixture*, and the mixture of
    ranks is robust to one model's logits living on a different scale -- which
    happens automatically when one member is trained with AUC-margin and
    another with ASL.
    """
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(s, axis=0, kind="mergesort")
    ranks = np.empty_like(s)
    n = s.shape[0]
    idx = np.arange(n, dtype=np.float64)[:, None]
    np.put_along_axis(ranks, order, np.broadcast_to(idx, s.shape), axis=0)
    return ranks / max(n - 1, 1)


def _smooth_auc(y: np.ndarray, s: np.ndarray, *, tau: float = 0.05
                ) -> tuple[float, np.ndarray]:
    r"""Sigmoid-smoothed AUC and its gradient w.r.t. the scores.

    .. math::
        \mathcal A_\tau = \frac{1}{mn}\sum_{i\in P}\sum_{j\in N}
          \sigma\!\Big(\frac{s_i - s_j}{\tau}\Big).

    Smoothing is what makes gradient-based weight search possible at all: the
    exact AUC is piecewise constant in :math:`w`, so its gradient is zero
    almost everywhere.
    """
    pos = np.flatnonzero(y > 0.5)
    neg = np.flatnonzero(y <= 0.5)
    if pos.size == 0 or neg.size == 0:
        return float("nan"), np.zeros_like(s)
    d = (s[pos][:, None] - s[neg][None, :]) / tau
    sig = 1.0 / (1.0 + np.exp(-np.clip(d, -60, 60)))
    val = float(sig.mean())
    g = sig * (1.0 - sig) / (tau * pos.size * neg.size)
    grad = np.zeros_like(s)
    np.add.at(grad, pos, g.sum(axis=1))
    np.add.at(grad, neg, -g.sum(axis=0))
    return val, grad


@dataclass(slots=True)
class EnsembleWeights:
    weights: np.ndarray  # (L, M) rows on the simplex
    model_names: list[str]
    anchor_lambda: float
    shrinkage: np.ndarray = field(default_factory=lambda: np.zeros(0))
    oof_macro_uniform: float = float("nan")
    oof_macro_weighted: float = float("nan")
    nested_macro_uniform: float = float("nan")
    nested_macro_weighted: float = float("nan")

    def apply(self, preds: np.ndarray) -> np.ndarray:
        """``preds``: ``(M, N, L)`` → ``(N, L)``."""
        return np.einsum("lm,mnl->nl", self.weights, preds)

    def summary(self, target_names) -> str:
        names = list(target_names)
        w = max(len(n) for n in names) + 2
        head = "label".ljust(w) + "".join(f"{m[:10]:>12}" for m in self.model_names)
        lines = [head, "-" * len(head)]
        for l, nm in enumerate(names):
            lines.append(nm.ljust(w) + "".join(f"{v:12.3f}" for v in self.weights[l]))
        lines += [
            "-" * len(head),
            f"OOF   uniform {self.oof_macro_uniform:.5f} → weighted "
            f"{self.oof_macro_weighted:.5f}",
            f"NESTED uniform {self.nested_macro_uniform:.5f} → weighted "
            f"{self.nested_macro_weighted:.5f}   "
            f"(trust this one)",
        ]
        return "\n".join(lines)


def fit_label_weights(
    y_true: np.ndarray,  # (N, L)
    preds: np.ndarray,  # (M, N, L)
    *,
    model_names: list[str] | None = None,
    anchor_lambda: float = 0.15,
    lr: float = 0.5,
    n_steps: int = 300,
    tau: float = 0.05,
    group_of_label: np.ndarray | None = None,  # (L,) mechanism group index
    shrink_tau2: float = 0.02,
    use_ranks: bool = True,
    rows: np.ndarray | None = None,
) -> EnsembleWeights:
    """Fit per-label simplex weights by anchored exponentiated gradient."""
    y_true = np.asarray(y_true, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    M, N, L = preds.shape
    names = model_names or [f"model_{i}" for i in range(M)]

    if rows is not None:
        y_fit, p_fit = y_true[rows], preds[:, rows]
    else:
        y_fit, p_fit = y_true, preds

    if use_ranks:
        p_fit = np.stack([rank_transform(p_fit[m]) for m in range(M)])

    u = np.full(M, 1.0 / M)
    W = np.tile(u, (L, 1))

    for _ in range(n_steps):
        grad = np.zeros_like(W)
        for l in range(L):
            s = p_fit[:, :, l].T @ W[l]  # (N,)
            _, gs = _smooth_auc(y_fit[:, l], s, tau=tau)
            if not np.all(np.isfinite(gs)):
                continue
            grad[l] = p_fit[:, :, l] @ gs  # (M,)
        # Entropic anchor pulls the log-weights towards log u.
        anchor = anchor_lambda * (np.log(np.maximum(W, _EPS)) - np.log(u)[None, :])
        logits = np.log(np.maximum(W, _EPS)) + lr * (grad - anchor)
        logits -= logits.max(axis=1, keepdims=True)
        W = np.exp(logits)
        W /= W.sum(axis=1, keepdims=True)

    shrink = np.zeros(L)
    if group_of_label is not None:
        g = np.asarray(group_of_label)
        n_pos = np.nansum(y_fit > 0.5, axis=0).astype(np.float64)
        var_w = W.var(axis=1) + _EPS
        shrink = var_w / (var_w + np.maximum(n_pos, 1.0) * shrink_tau2)
        for gid in np.unique(g):
            sel = np.flatnonzero(g == gid)
            bar = W[sel].mean(axis=0)
            W[sel] = (1 - shrink[sel])[:, None] * W[sel] + shrink[sel][:, None] * bar[None, :]
        W /= W.sum(axis=1, keepdims=True)

    ew = EnsembleWeights(
        weights=W, model_names=names, anchor_lambda=anchor_lambda, shrinkage=shrink
    )
    p_all = np.stack([rank_transform(preds[m]) for m in range(M)]) if use_ranks else preds
    ew.oof_macro_uniform = macro_auc(y_true, p_all.mean(axis=0))
    ew.oof_macro_weighted = macro_auc(y_true, ew.apply(p_all))
    return ew


def nested_evaluate(
    y_true: np.ndarray,
    preds: np.ndarray,
    fold: np.ndarray,
    *,
    model_names: list[str] | None = None,
    use_ranks: bool = True,
    **fit_kwargs,
) -> EnsembleWeights:
    """Leave-one-fold-out weight fitting; the only honest estimate of the gain.

    Weights for the *final submission* are then refit on all folds -- using the
    nested number only to decide **whether to use weighting at all**.  If the
    nested weighted macro does not beat the nested uniform macro by more than
    one bootstrap standard error, ship the uniform average.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    fold = np.asarray(fold)
    M, N, L = preds.shape

    held = np.zeros((N, L), dtype=np.float64)
    p_ranked = np.stack([rank_transform(preds[m]) for m in range(M)]) if use_ranks else preds

    for k in np.unique(fold):
        tr = np.flatnonzero(fold != k)
        te = np.flatnonzero(fold == k)
        w = fit_label_weights(
            y_true, preds, model_names=model_names, rows=tr, use_ranks=use_ranks, **fit_kwargs
        )
        held[te] = np.einsum("lm,mnl->nl", w.weights, p_ranked[:, te])

    final = fit_label_weights(
        y_true, preds, model_names=model_names, use_ranks=use_ranks, **fit_kwargs
    )
    final.nested_macro_uniform = macro_auc(y_true, p_ranked.mean(axis=0))
    final.nested_macro_weighted = macro_auc(y_true, held)
    return final


def greedy_selection(
    y_true: np.ndarray,
    preds: np.ndarray,
    *,
    model_names: list[str] | None = None,
    max_size: int = 12,
    with_replacement: bool = True,
    use_ranks: bool = True,
    tol: float = 1e-5,
) -> tuple[list[int], float]:
    """Caruana-style forward selection with replacement, on macro-AUC.

    With replacement matters: it lets a strong model be selected several times,
    which is how greedy selection expresses a non-uniform weight without ever
    leaving the space of averages -- and averages of a bag are far harder to
    overfit than free weights.
    """
    preds = np.asarray(preds, dtype=np.float64)
    M = preds.shape[0]
    p = np.stack([rank_transform(preds[m]) for m in range(M)]) if use_ranks else preds
    del model_names

    chosen: list[int] = []
    best_score = -np.inf
    acc = np.zeros_like(p[0])
    for _ in range(max_size):
        cand_best, cand_idx = -np.inf, -1
        for m in range(M):
            if not with_replacement and m in chosen:
                continue
            blended = (acc * len(chosen) + p[m]) / (len(chosen) + 1)
            sc = macro_auc(y_true, blended)
            if sc > cand_best:
                cand_best, cand_idx = sc, m
        if cand_idx < 0 or cand_best <= best_score + tol:
            break
        acc = (acc * len(chosen) + p[cand_idx]) / (len(chosen) + 1)
        chosen.append(cand_idx)
        best_score = cand_best
    return chosen, float(best_score)


def bayesian_model_average(
    y_true: np.ndarray,
    preds: np.ndarray,
    *,
    n_draws: int = 400,
    prior_strength: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Bayesian bootstrap over per-label model weights.

    Draw :math:`\pi \sim \mathrm{Dir}(\mathbf 1)` over the OOF studies, compute
    the per-label AUC of each model under the reweighted sample, and turn the
    resulting *win frequencies* into weights.  This propagates the finite-sample
    uncertainty in "which model is better for Fracture" into the weights
    themselves, instead of committing to the point estimate.

    Returns ``(weights (L, M), win_probability (L, M))``.
    """
    rng = np.random.default_rng(seed)
    preds = np.asarray(preds, dtype=np.float64)
    M, N, L = preds.shape
    wins = np.zeros((L, M))

    for _ in range(n_draws):
        w = rng.dirichlet(np.full(N, prior_strength))
        idx = rng.choice(N, size=N, replace=True, p=w)
        a = np.stack([per_label_auc(y_true[idx], preds[m][idx]) for m in range(M)])  # (M, L)
        best = np.nanargmax(np.where(np.isfinite(a), a, -np.inf), axis=0)
        wins[np.arange(L), best] += 1.0

    prob = wins / max(n_draws, 1)
    # Smooth towards uniform by the Dirichlet posterior mean with a flat prior.
    weights = (wins + 1.0) / (n_draws + M)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights, prob
