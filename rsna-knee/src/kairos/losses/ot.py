r"""Entropic optimal transport for phrase→slice grounding.

The dataset pairs every study with its original radiology report, but with **no
bounding boxes**.  A report sentence such as *"full-thickness tear of the
anterior cruciate ligament"* asserts that some slice, somewhere in the sagittal
stack, shows an ACL tear -- it does not say which.  This is a weakly-supervised
grounding problem, and the natural mathematical object is a *transport plan*
between the set of report phrases and the set of image slices.

Let :math:`\{u_i\}_{i=1}^{n}` be phrase embeddings and :math:`\{v_j\}_{j=1}^{m}`
slice embeddings, both L2-normalised, and let the ground cost be
:math:`C_{ij} = 1 - \langle u_i, v_j\rangle`.  Entropic OT solves

.. math::
    \min_{T \in \Pi(\mu,\nu)} \langle T, C\rangle - \varepsilon H(T),
    \qquad H(T) = -\sum_{ij} T_{ij}(\log T_{ij} - 1),

whose solution has the form :math:`T = \operatorname{diag}(f)\,K\,
\operatorname{diag}(g)` with :math:`K = e^{-C/\varepsilon}`, found by Sinkhorn
iteration.  The value :math:`\langle T^\star, C\rangle` is the alignment loss.

Two departures from textbook Sinkhorn are essential here and both are
implemented:

**Unbalanced OT.**  Not every phrase has a visual correlate (*"clinical
history: pain"*), and not every slice has a described finding.  Forcing
:math:`T\mathbf 1 = \mu` therefore *creates* spurious alignments.  We relax
both marginals with a KL penalty of strength :math:`\tau`
(Chizat et al., 2018), which turns the Sinkhorn update into a damped one,

.. math::
    f \leftarrow \Big(\frac{\mu}{Kg}\Big)^{\tfrac{\tau}{\tau+\varepsilon}},
    \qquad
    g \leftarrow \Big(\frac{\nu}{K^\top f}\Big)^{\tfrac{\tau}{\tau+\varepsilon}},

and lets mass be destroyed where nothing matches.

**Log-domain stabilisation.**  With :math:`\varepsilon = 0.05` and a cosine
cost, :math:`e^{-C/\varepsilon}` underflows fp16 immediately.  All iterations
run on potentials in log space, so the loss is safe under AMP.

Gradients use the *envelope theorem*: at the optimum the derivative of the OT
value with respect to :math:`C` is :math:`T^\star` itself, so we detach the
plan and back-propagate only through :math:`\langle T^\star_{\text{detached}},
C\rangle`.  This is both cheaper and far more stable than unrolling the
iterations, and is exact up to the entropic bias.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "sinkhorn_log",
    "unbalanced_sinkhorn_log",
    "PhraseSliceOT",
    "sinkhorn_divergence",
]

_LOG_EPS = -1e9


def _logsumexp_masked(x: torch.Tensor, dim: int, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is not None:
        x = x.masked_fill(~mask, _LOG_EPS)
    return torch.logsumexp(x, dim=dim)


def sinkhorn_log(
    cost: torch.Tensor,
    *,
    epsilon: float = 0.05,
    n_iter: int = 60,
    mu: torch.Tensor | None = None,
    nu: torch.Tensor | None = None,
    row_mask: torch.Tensor | None = None,
    col_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Balanced entropic OT in the log domain.

    Parameters
    ----------
    cost
        ``(B, n, m)`` ground cost.
    mu, nu
        ``(B, n)`` / ``(B, m)`` marginals.  Default uniform over the unmasked
        entries.

    Returns
    -------
    plan
        ``(B, n, m)`` transport plan (rows sum to ``mu`` up to convergence).
    value
        ``(B,)`` :math:`\langle T, C\rangle`.
    """
    B, n, m = cost.shape
    dev, dt = cost.device, cost.dtype

    if row_mask is None:
        row_mask = torch.ones(B, n, dtype=torch.bool, device=dev)
    if col_mask is None:
        col_mask = torch.ones(B, m, dtype=torch.bool, device=dev)

    if mu is None:
        mu = row_mask.to(dt) / row_mask.sum(1, keepdim=True).clamp_min(1).to(dt)
    if nu is None:
        nu = col_mask.to(dt) / col_mask.sum(1, keepdim=True).clamp_min(1).to(dt)

    log_mu = torch.log(mu.clamp_min(1e-30)).masked_fill(~row_mask, _LOG_EPS)
    log_nu = torch.log(nu.clamp_min(1e-30)).masked_fill(~col_mask, _LOG_EPS)

    M = (-cost / epsilon).to(torch.float32)
    pair_mask = row_mask[:, :, None] & col_mask[:, None, :]
    f = torch.zeros(B, n, device=dev, dtype=torch.float32)
    g = torch.zeros(B, m, device=dev, dtype=torch.float32)

    with torch.no_grad():
        for _ in range(n_iter):
            f = log_mu.float() - _logsumexp_masked(
                M + g[:, None, :], dim=2, mask=pair_mask
            )
            f = torch.nan_to_num(f, neginf=_LOG_EPS)
            g = log_nu.float() - _logsumexp_masked(
                M + f[:, :, None], dim=1, mask=pair_mask
            )
            g = torch.nan_to_num(g, neginf=_LOG_EPS)

    log_T = (M + f[:, :, None] + g[:, None, :]).masked_fill(~pair_mask, _LOG_EPS)
    plan = torch.exp(log_T).to(dt)
    plan = plan * pair_mask.to(dt)
    value = (plan.detach() * cost).sum(dim=(1, 2))
    return plan.detach(), value


def unbalanced_sinkhorn_log(
    cost: torch.Tensor,
    *,
    epsilon: float = 0.05,
    tau: float = 0.5,
    n_iter: int = 80,
    mu: torch.Tensor | None = None,
    nu: torch.Tensor | None = None,
    row_mask: torch.Tensor | None = None,
    col_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Unbalanced entropic OT with KL marginal relaxation of strength ``tau``.

    ``tau -> inf`` recovers balanced OT; ``tau -> 0`` decouples the problem
    entirely.  ``tau = 0.5`` with ``epsilon = 0.05`` empirically lets roughly a
    third of report phrases go untransported, which matches the fraction of
    sentences in a knee report that carry no localisable finding (history,
    technique, comparison).
    """
    B, n, m = cost.shape
    dev, dt = cost.device, cost.dtype
    if row_mask is None:
        row_mask = torch.ones(B, n, dtype=torch.bool, device=dev)
    if col_mask is None:
        col_mask = torch.ones(B, m, dtype=torch.bool, device=dev)
    if mu is None:
        mu = row_mask.to(dt) / row_mask.sum(1, keepdim=True).clamp_min(1).to(dt)
    if nu is None:
        nu = col_mask.to(dt) / col_mask.sum(1, keepdim=True).clamp_min(1).to(dt)

    log_mu = torch.log(mu.clamp_min(1e-30)).masked_fill(~row_mask, _LOG_EPS).float()
    log_nu = torch.log(nu.clamp_min(1e-30)).masked_fill(~col_mask, _LOG_EPS).float()

    M = (-cost / epsilon).to(torch.float32)
    pair_mask = row_mask[:, :, None] & col_mask[:, None, :]
    f = torch.zeros(B, n, device=dev, dtype=torch.float32)
    g = torch.zeros(B, m, device=dev, dtype=torch.float32)
    damp = tau / (tau + epsilon)

    with torch.no_grad():
        for _ in range(n_iter):
            f = damp * (
                log_mu - _logsumexp_masked(M + g[:, None, :], dim=2, mask=pair_mask)
            )
            f = torch.nan_to_num(f, neginf=_LOG_EPS)
            g = damp * (
                log_nu - _logsumexp_masked(M + f[:, :, None], dim=1, mask=pair_mask)
            )
            g = torch.nan_to_num(g, neginf=_LOG_EPS)

    log_T = (M + f[:, :, None] + g[:, None, :]).masked_fill(~pair_mask, _LOG_EPS)
    plan = (torch.exp(log_T).to(dt) * pair_mask.to(dt)).detach()
    value = (plan * cost).sum(dim=(1, 2))
    return plan, value


def sinkhorn_divergence(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epsilon: float = 0.05,
    n_iter: int = 60,
) -> torch.Tensor:
    r"""Debiased Sinkhorn divergence
    :math:`S_\varepsilon(\alpha,\beta) = \mathrm{OT}_\varepsilon(\alpha,\beta)
    - \tfrac12\mathrm{OT}_\varepsilon(\alpha,\alpha)
    - \tfrac12\mathrm{OT}_\varepsilon(\beta,\beta)`.

    The self-transport terms remove the entropic bias, which is what makes the
    divergence vanish iff the two distributions coincide.  Used as the
    representation-level domain-alignment penalty between sites in
    ``train.curriculum`` -- and *only* between sites, never between labels.
    """
    def _cost(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return 1.0 - torch.einsum("bnd,bmd->bnm", a, b)

    _, v_xy = sinkhorn_log(_cost(x, y), epsilon=epsilon, n_iter=n_iter)
    _, v_xx = sinkhorn_log(_cost(x, x), epsilon=epsilon, n_iter=n_iter)
    _, v_yy = sinkhorn_log(_cost(y, y), epsilon=epsilon, n_iter=n_iter)
    return (v_xy - 0.5 * v_xx - 0.5 * v_yy).clamp_min(0.0)


class PhraseSliceOT(nn.Module):
    r"""Weakly-supervised phrase→slice grounding loss.

    Beyond the plain OT value we add two terms that encode knowledge the plan
    alone cannot express:

    ``anatomical prior``
        The cost is *increased* for (phrase, slice) pairs whose sequence family
        is a priori uninformative for the phrase's label (a PF-OA phrase should
        not ground onto a coronal T1 slice).  This is a soft cost bump of
        ``prior_penalty``, not a mask -- atypical protocols exist.

    ``attention consistency``
        The model's own label-specific slice attention :math:`a_{l,s}` should
        agree with the marginal of the transport plan restricted to phrases
        carrying label :math:`l`.  We penalise the Jensen–Shannon divergence
        between the two, which is bounded (unlike KL) and therefore cannot
        blow up when the plan is nearly degenerate early in training.

    The whole term is gated on ``confidence``: phrases whose assertion status
    was extracted with low confidence contribute proportionally less.  Ungated,
    this loss will happily ground a *negated* finding ("no meniscal tear") onto
    a slice, which is worse than no supervision at all.
    """

    def __init__(
        self,
        *,
        epsilon: float = 0.05,
        tau: float = 0.5,
        n_iter: int = 60,
        prior_penalty: float = 0.25,
        consistency_weight: float = 0.5,
        min_confidence: float = 0.6,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.tau = tau
        self.n_iter = n_iter
        self.prior_penalty = prior_penalty
        self.consistency_weight = consistency_weight
        self.min_confidence = min_confidence

    def forward(
        self,
        phrase_emb: torch.Tensor,  # (B, P, D)
        slice_emb: torch.Tensor,  # (B, S, D)
        *,
        phrase_mask: torch.Tensor,  # (B, P) bool
        slice_mask: torch.Tensor,  # (B, S) bool
        phrase_conf: torch.Tensor | None = None,  # (B, P) in [0,1]
        prior_incompatible: torch.Tensor | None = None,  # (B, P, S) bool
        phrase_label: torch.Tensor | None = None,  # (B, P) long, -1 = none
        label_attention: torch.Tensor | None = None,  # (B, L, S)
    ) -> dict[str, torch.Tensor]:
        u = F.normalize(phrase_emb, dim=-1)
        v = F.normalize(slice_emb, dim=-1)
        cost = 1.0 - torch.einsum("bpd,bsd->bps", u, v)

        if prior_incompatible is not None:
            cost = cost + self.prior_penalty * prior_incompatible.to(cost.dtype)

        mu = None
        if phrase_conf is not None:
            w = torch.where(
                phrase_conf >= self.min_confidence, phrase_conf, torch.zeros_like(phrase_conf)
            )
            w = w * phrase_mask.to(w.dtype)
            mu = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)

        plan, value = unbalanced_sinkhorn_log(
            cost,
            epsilon=self.epsilon,
            tau=self.tau,
            n_iter=self.n_iter,
            mu=mu,
            row_mask=phrase_mask,
            col_mask=slice_mask,
        )
        # Envelope theorem: plan is detached, so grad flows only via `cost`.
        ot_loss = (plan * cost).sum(dim=(1, 2)).mean()

        out = {"ot": ot_loss, "plan_mass": plan.sum(dim=(1, 2)).mean().detach()}

        if (
            label_attention is not None
            and phrase_label is not None
            and self.consistency_weight > 0
        ):
            B, L, S = label_attention.shape
            # Marginal of the plan over slices, grouped by the phrase's label.
            col = plan.sum(dim=1)  # (B, S) total, unused but kept for clarity
            del col
            tgt = torch.zeros(B, L, S, device=plan.device, dtype=plan.dtype)
            lab = phrase_label.clamp_min(0)
            valid = (phrase_label >= 0) & phrase_mask
            src = plan * valid[:, :, None].to(plan.dtype)
            tgt.scatter_add_(1, lab[:, :, None].expand(-1, -1, S), src)
            mass = tgt.sum(dim=2, keepdim=True)
            has = (mass.squeeze(-1) > 1e-8)
            tgt = tgt / mass.clamp_min(1e-8)

            a = label_attention.clamp_min(1e-8)
            a = a / a.sum(dim=2, keepdim=True).clamp_min(1e-8)
            m = 0.5 * (a + tgt)
            js = 0.5 * (
                (a * (a.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(-1)
                + (tgt * (tgt.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(-1)
            )
            js = (js * has.to(js.dtype)).sum() / has.sum().clamp_min(1).to(js.dtype)
            out["attention_consistency"] = js
            out["total"] = ot_loss + self.consistency_weight * js
        else:
            out["total"] = ot_loss
        return out
