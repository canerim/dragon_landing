r"""Adaptive computation: spend FLOPs where the pathology is.

A knee study is 3–7 series of 20–50 slices.  Processing every slice at 384 px
costs ~8× what processing every slice at 256 px costs, and the extra resolution
matters for perhaps 10 % of slices -- the ones containing a meniscal root, a
cortical break, or a 2 mm cartilage defect.  Uniform compute is therefore
wasting roughly 7/8 of the fine-resolution budget, which is precisely the budget
the nine-hour Kaggle limit is made of, and the efficiency prize is scored on.

Three components:

:class:`GumbelTopKSelector`
    Differentiable selection of :math:`k` slices per (label, series).  Hard
    top-:math:`k` is not differentiable, and straight-through estimators on
    top-:math:`k` are badly biased.  We use *sampling without replacement* via
    the Gumbel top-:math:`k` trick (Kool et al., 2019): perturbing scores with
    Gumbel noise and taking the top :math:`k` is an exact sample from the
    Plackett–Luce distribution over ordered :math:`k`-subsets.  Gradients come
    from a SoftSort-style relaxation of the same quantity, so the estimator is
    consistent as the temperature anneals.

    The exploration mixture is not optional.  A selector trained purely on its
    own scores has an obvious degenerate optimum: always pick the same slice
    indices, get a decent loss, and never discover that the pathology is
    elsewhere.  We therefore sample ``exploration_frac`` of the budget
    uniformly at random during training, and additionally force inclusion of
    the neighbours of every selected slice, so the fine encoder always sees a
    contiguous window rather than isolated planes.

:class:`ConfidenceGate`
    Decides *whether* the fine pass runs at all, per (study, label), from the
    coarse prediction's distance to the decision region **and** its epistemic
    uncertainty:

    .. math::
        \texttt{run\_fine}_l = \mathbb 1\big[\,U_l > \tau_U \ \lor\
          p_l \in (\tau_{\text{lo}}, \tau_{\text{hi}})\,\big].

    Thresholds are not hand-picked; they are read off the *runtime–AUC Pareto
    frontier* computed on OOF predictions by :mod:`kairos.infer.budget`.

:class:`PonderHalting`
    PonderNet-style learned halting for the iterative refinement loop, with the
    geometric prior that regularises the expected number of steps.  Used by the
    main-leaderboard model where a third refinement pass occasionally helps; the
    efficiency student caps at two.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GumbelTopKSelector", "ConfidenceGate", "PonderHalting", "expand_windows"]


def _gumbel_like(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    u = torch.rand_like(x).clamp(eps, 1.0 - eps)
    return -torch.log(-torch.log(u))


def expand_windows(
    selected: torch.Tensor, radius: int = 1, *, valid: torch.Tensor | None = None
) -> torch.Tensor:
    r"""Dilate a boolean slice-selection mask by ``radius`` along the slice axis.

    ``selected``: ``(..., S)`` bool.  A tear visible on slice :math:`i` is
    almost always visible on :math:`i\pm1`, and the fine encoder's 2.5D stem
    needs those neighbours anyway; dilating here means the scheduler knows the
    true decode cost instead of discovering it later.
    """
    x = selected.to(torch.float32)
    k = 2 * radius + 1
    shape = x.shape
    x = x.reshape(-1, 1, shape[-1])
    ker = torch.ones(1, 1, k, device=x.device, dtype=x.dtype)
    out = F.conv1d(x, ker, padding=radius) > 0.5
    out = out.reshape(shape)
    if valid is not None:
        out = out & valid
    return out


class GumbelTopKSelector(nn.Module):
    """Differentiable top-k slice selection with exploration and dilation."""

    def __init__(
        self,
        *,
        k: int = 6,
        temperature: float = 1.0,
        temperature_min: float = 0.1,
        anneal_steps: int = 20_000,
        exploration_frac: float = 0.25,
        window_radius: int = 1,
        hard: bool = True,
    ) -> None:
        super().__init__()
        self.k = k
        self.t0 = temperature
        self.t_min = temperature_min
        self.anneal_steps = anneal_steps
        self.exploration_frac = exploration_frac
        self.window_radius = window_radius
        self.hard = hard
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long))

    @property
    def temperature(self) -> float:
        s = float(self._step.item())
        r = min(s / max(self.anneal_steps, 1), 1.0)
        return self.t0 + (self.t_min - self.t0) * r

    def forward(
        self,
        scores: torch.Tensor,  # (B, L, S) relevance of slice s for label l
        *,
        valid: torch.Tensor,  # (B, S) bool
        k: int | None = None,
        training: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        train = self.training if training is None else training
        k = k or self.k
        B, L, S = scores.shape
        k = min(k, int(valid.sum(dim=1).min().item()) if bool(valid.any()) else k)
        k = max(k, 1)

        neg = torch.finfo(scores.dtype).min / 4
        s = scores.masked_fill(~valid[:, None, :], neg)

        if train:
            self._step += 1
            perturbed = s + self.temperature * _gumbel_like(s)
        else:
            perturbed = s

        topv, topi = perturbed.topk(k, dim=-1)
        hard_mask = torch.zeros_like(s, dtype=torch.bool).scatter_(-1, topi, True)

        if train and self.exploration_frac > 0:
            n_explore = max(1, int(round(self.exploration_frac * k)))
            rand = torch.rand(B, L, S, device=s.device).masked_fill(~valid[:, None, :], -1.0)
            _, ri = rand.topk(n_explore, dim=-1)
            hard_mask = hard_mask.scatter(-1, ri, True)

        if self.window_radius > 0:
            hard_mask = expand_windows(hard_mask, self.window_radius, valid=valid[:, None, :])

        # Soft weights for the gradient path: a temperature-annealed softmax
        # restricted to the selected set reproduces the relaxed Plackett-Luce
        # marginals and keeps the estimator unbiased as t -> 0.
        soft = torch.softmax(perturbed / max(self.temperature, 1e-3), dim=-1)
        soft = soft * hard_mask.to(soft.dtype)
        soft = soft / soft.sum(-1, keepdim=True).clamp_min(1e-8)

        weights = hard_mask.to(soft.dtype) + (soft - soft.detach()) if self.hard else soft

        return {
            "mask": hard_mask,
            "weights": weights,
            "union": hard_mask.any(dim=1),  # (B, S) slices any label wants
            "n_selected": hard_mask.any(dim=1).sum(dim=1).float(),
            "temperature": torch.tensor(self.temperature, device=s.device),
        }

    def budget_loss(self, out: dict[str, torch.Tensor], target_fraction: float,
                    valid: torch.Tensor) -> torch.Tensor:
        r"""Penalty pulling the *actual* selected fraction to a target.

        Selection cost is what the efficiency score integrates, so it must be in
        the objective rather than enforced by a hard cap at inference -- a hard
        cap applied to a model trained without one silently truncates exactly
        the slices the model relies on.
        """
        n_valid = valid.sum(dim=1).clamp_min(1).float()
        frac = out["n_selected"] / n_valid
        return F.relu(frac - target_fraction).mean() ** 2


class ConfidenceGate(nn.Module):
    r"""Per-(study, label) decision to escalate to the high-resolution pass."""

    def __init__(
        self,
        num_labels: int,
        *,
        p_low: float = 0.05,
        p_high: float = 0.85,
        u_threshold: float = 0.5,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        init = torch.stack(
            [
                torch.logit(torch.full((num_labels,), p_low)),
                torch.logit(torch.full((num_labels,), p_high)),
            ]
        )
        self.thresholds = nn.Parameter(init, requires_grad=learnable)
        self.u_threshold = nn.Parameter(
            torch.full((num_labels,), float(u_threshold)), requires_grad=learnable
        )

    def forward(
        self, coarse_logits: torch.Tensor, uncertainty: torch.Tensor | None = None
    ) -> torch.Tensor:
        lo = torch.sigmoid(self.thresholds[0]).to(coarse_logits.dtype)[None]
        hi = torch.sigmoid(self.thresholds[1]).to(coarse_logits.dtype)[None]
        p = torch.sigmoid(coarse_logits)
        ambiguous = (p > lo) & (p < hi)
        if uncertainty is not None:
            ambiguous = ambiguous | (uncertainty > self.u_threshold.to(p.dtype)[None])
        return ambiguous

    @torch.no_grad()
    def set_from_pareto(self, lo: torch.Tensor, hi: torch.Tensor, u: torch.Tensor) -> None:
        """Install thresholds selected on OOF by :mod:`kairos.infer.budget`."""
        self.thresholds[0].copy_(torch.logit(lo.clamp(1e-4, 1 - 1e-4)))
        self.thresholds[1].copy_(torch.logit(hi.clamp(1e-4, 1 - 1e-4)))
        self.u_threshold.copy_(u)


class PonderHalting(nn.Module):
    r"""PonderNet halting with a geometric prior (Banino et al., 2021).

    At step :math:`n` the model emits a halting probability :math:`\lambda_n`;
    the distribution over halting steps is
    :math:`p_n = \lambda_n\prod_{j<n}(1-\lambda_j)`.  The prediction is
    :math:`\sum_n p_n \hat y_n` and the loss is
    :math:`\sum_n p_n \mathcal L(\hat y_n, y) + \beta\,
    \mathrm{KL}(p \,\|\, \mathrm{Geom}(\lambda_p))`.

    The KL term is what makes this work: without it the model always halts
    immediately (cheapest) or never (most accurate), and the prior's
    :math:`\lambda_p` is the knob that trades the two.  We set
    :math:`\lambda_p = 0.5` for the main model (expected 2 steps) and 0.8 for
    the efficiency student (expected 1.25).
    """

    def __init__(self, dim: int, *, max_steps: int = 3, lambda_prior: float = 0.5) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.lambda_prior = lambda_prior
        self.gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def halting_probs(self, lambdas: torch.Tensor) -> torch.Tensor:
        """``lambdas``: ``(B, N)`` → ``(B, N)`` halting distribution."""
        cont = torch.cumprod(
            torch.cat([torch.ones_like(lambdas[:, :1]), 1.0 - lambdas[:, :-1]], dim=1), dim=1
        )
        p = lambdas * cont
        # Force the last step to absorb the remaining mass so p sums to 1.
        p = torch.cat([p[:, :-1], (1.0 - p[:, :-1].sum(dim=1, keepdim=True)).clamp_min(0.0)], dim=1)
        return p

    def prior(self, n: int, device, dtype) -> torch.Tensor:
        lp = self.lambda_prior
        k = torch.arange(n, device=device, dtype=dtype)
        p = lp * (1.0 - lp) ** k
        return p / p.sum()

    def kl_to_prior(self, p: torch.Tensor) -> torch.Tensor:
        q = self.prior(p.shape[1], p.device, p.dtype)[None]
        p = p.clamp_min(1e-8)
        return (p * (p.log() - q.clamp_min(1e-8).log())).sum(dim=1).mean()

    def forward(self, states: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        lam = torch.sigmoid(torch.cat([self.gate(s) for s in states], dim=1))
        return lam, self.halting_probs(lam)
