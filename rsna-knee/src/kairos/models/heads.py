r"""Uncertainty-aware output heads.

Why not just a linear layer: at inference we need three distinct quantities and
a linear head gives us only the first.

1. **A score** for the leaderboard (rank only -- calibration is irrelevant to
   AUC).
2. **A calibrated probability** for the report-gating and distillation logic.
3. **A distance-aware uncertainty** that separates *"the model is unsure
   because this case is genuinely borderline"* from *"the model is unsure
   because this study looks like nothing in the training set"*.  Only the
   second justifies escalating to the expensive fine-resolution pass, and only
   the second predicts silent failure on an unseen site.

:class:`SNGPHead` provides all three.  It replaces the final linear layer with
a Gaussian-process approximated by random Fourier features (Liu et al., 2020):

.. math::
    \Phi(h) = \sqrt{\tfrac{2}{M}}\,\cos(W h + b), \qquad
    W_{ij}\sim\mathcal N(0,1),\; b_i \sim \mathcal U[0, 2\pi],

so :math:`\Phi(h)^\top\Phi(h')\approx k_{\mathrm{RBF}}(h,h')`.  The output is
:math:`f = \beta^\top \Phi(h)` and the posterior variance is
:math:`\Phi^\top \Sigma \Phi` where :math:`\Sigma` is the Laplace
approximation of the precision, accumulated in closed form over the training
set:

.. math::
    \Sigma^{-1} = I + \sum_i p_i(1-p_i)\,\Phi_i\Phi_i^\top .

The distance-awareness only holds if the feature extractor is bi-Lipschitz,
which is what :class:`SpectralNormLinear` enforces on the layers feeding the
head.  Without spectral normalisation SNGP degrades to an ordinary head with a
decorative variance -- this is the single most common way of getting SNGP
"wrong" and it produces uncertainties that look plausible and mean nothing.

:class:`EvidentialHead` is the cheaper alternative used by the efficiency-track
student: a Beta-distribution head trained with the evidential regression loss,
one forward pass, no covariance state.  It is less faithful than SNGP under
severe shift but costs nothing.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SNGPHead", "EvidentialHead", "SpectralNormLinear", "mean_field_logits"]


class SpectralNormLinear(nn.Module):
    r"""Linear layer with a *soft* spectral-norm bound.

    Hard :math:`\sigma_{\max} = 1` normalisation (as in SN-GAN) is too
    aggressive for a classifier trunk -- it throttles the effective learning
    rate of every layer.  We instead rescale only when the estimated top
    singular value exceeds ``c``:

    .. math::
        W \leftarrow \frac{c}{\max(c, \sigma_{\max})}\,W,

    which leaves the layer untouched in the common case and enforces the
    Lipschitz bound exactly when it is violated.  :math:`\sigma_{\max}` is
    tracked by one power-iteration step per forward, which is free.
    """

    def __init__(self, in_features: int, out_features: int, *, c: float = 6.0,
                 bias: bool = True, n_power_iterations: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.c = c
        self.n_power = n_power_iterations
        self.register_buffer("u", F.normalize(torch.randn(out_features), dim=0))
        self.register_buffer("v", F.normalize(torch.randn(in_features), dim=0))

    def _sigma(self) -> torch.Tensor:
        W = self.linear.weight
        u, v = self.u, self.v
        with torch.no_grad():
            for _ in range(self.n_power):
                v = F.normalize(W.t() @ u, dim=0, eps=1e-12)
                u = F.normalize(W @ v, dim=0, eps=1e-12)
            self.u.copy_(u)
            self.v.copy_(v)
        return torch.dot(u, W @ v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self._sigma()
        scale = self.c / torch.clamp(sigma, min=self.c)
        w = self.linear.weight * scale
        return F.linear(x, w, self.linear.bias)


def mean_field_logits(
    logits: torch.Tensor, variance: torch.Tensor, *, lam: float = math.pi / 8.0
) -> torch.Tensor:
    r"""Mean-field correction of a GP logit to an approximate posterior mean.

    :math:`\mathbb E[\sigma(f)] \approx \sigma\!\big(f/\sqrt{1+\lambda v}\big)`
    with :math:`\lambda = \pi/8` from the probit approximation to the logistic.
    This is the step that actually turns the SNGP variance into *better
    probabilities*; without it the variance is only useful as an OOD score.
    """
    return logits / torch.sqrt(1.0 + lam * variance.clamp_min(0.0))


class SNGPHead(nn.Module):
    """Random-feature GP head with a Laplace precision, one per label."""

    def __init__(
        self,
        in_features: int,
        num_labels: int,
        *,
        num_random_features: int = 1024,
        gp_kernel_scale: float = 2.0,
        ridge: float = 1.0,
        momentum: float = 0.999,
        mean_field: bool = True,
        input_norm: bool = True,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.M = num_random_features
        self.mean_field = mean_field

        self.norm = nn.LayerNorm(in_features) if input_norm else nn.Identity()
        # Fixed random projection -- NOT trained; that is what makes Phi a
        # bona fide RBF feature map.
        W = torch.randn(num_random_features, in_features) / math.sqrt(gp_kernel_scale)
        b = torch.rand(num_random_features) * 2.0 * math.pi
        self.register_buffer("rff_W", W)
        self.register_buffer("rff_b", b)

        self.beta = nn.Parameter(torch.zeros(num_labels, num_random_features))
        nn.init.normal_(self.beta, std=num_random_features**-0.5)

        # Per-label precision, accumulated with no_grad during training.
        self.register_buffer(
            "precision", torch.eye(num_random_features).repeat(num_labels, 1, 1) * ridge
        )
        self.register_buffer("covariance", torch.zeros(num_labels, num_random_features,
                                                       num_random_features))
        self.register_buffer("cov_valid", torch.zeros(1))
        self.ridge = ridge
        self.momentum = momentum

    def features(self, h: torch.Tensor) -> torch.Tensor:
        h = self.norm(h)
        return math.sqrt(2.0 / self.M) * torch.cos(
            F.linear(h, self.rff_W.to(h.dtype), self.rff_b.to(h.dtype))
        )

    def forward(
        self,
        h: torch.Tensor,  # (B, L, D) per-label features
        *,
        update_precision: bool = False,
        return_variance: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        phi = self.features(h)  # (B, L, M)
        logits = (phi * self.beta.to(phi.dtype)[None]).sum(-1)  # (B, L)

        if update_precision and self.training:
            with torch.no_grad():
                p = torch.sigmoid(logits.float())
                w = (p * (1 - p)).clamp_min(1e-4)  # (B, L)
                pf = phi.float()
                upd = torch.einsum("blm,bl,bln->lmn", pf, w, pf)
                self.precision.mul_(self.momentum).add_(upd, alpha=1.0 - self.momentum)
                self.cov_valid.zero_()

        var = None
        if return_variance:
            if self.cov_valid.item() < 0.5:
                self._refresh_covariance()
            # ``.clone()`` is required, not defensive: the head is called twice
            # per forward in the coarse→fine path, and the second call may
            # refresh the covariance buffer in place.  Autograd saves the
            # buffer for the einsum backward, so an in-place refresh between
            # the two calls invalidates the first call's graph with a version-
            # counter error that surfaces at ``loss.backward()``.
            cov = self.covariance.detach().clone().to(phi.dtype)
            var = torch.einsum("blm,lmn,bln->bl", phi, cov, phi).clamp_min(0.0)
            if self.mean_field:
                logits = mean_field_logits(logits, var)
        return logits, var

    @torch.no_grad()
    def _refresh_covariance(self) -> None:
        """Invert the accumulated precision.  Cheap enough to do once per epoch."""
        P = self.precision.float()
        eye = torch.eye(self.M, device=P.device).expand_as(P)
        try:
            self.covariance.copy_(torch.linalg.solve(P + 1e-4 * eye, eye))
        except Exception:  # singular -- fall back to a diagonal approximation
            d = torch.diagonal(P, dim1=-2, dim2=-1).clamp_min(1e-4)
            self.covariance.zero_()
            self.covariance.diagonal(dim1=-2, dim2=-1).copy_(1.0 / d)
        self.cov_valid.fill_(1.0)

    @torch.no_grad()
    def reset_precision(self) -> None:
        self.precision.zero_()
        self.precision.diagonal(dim1=-2, dim2=-1).fill_(self.ridge)
        self.cov_valid.zero_()


class EvidentialHead(nn.Module):
    r"""Beta-evidential head: predicts :math:`(\alpha, \beta)` per label.

    With :math:`\alpha = 1 + e^+,\ \beta = 1 + e^-` (evidence from a softplus),
    the predictive mean is :math:`\alpha/(\alpha+\beta)` and the epistemic
    uncertainty is :math:`2/(\alpha+\beta)` -- large exactly when the model has
    accumulated little evidence either way.

    The training objective is the type-II maximum likelihood (marginal
    likelihood of the Bernoulli under the Beta prior) plus an evidence
    regulariser on wrong predictions:

    .. math::
        \mathcal L = \big[\psi(\alpha+\beta) - \psi(\alpha)\big] y
                   + \big[\psi(\alpha+\beta) - \psi(\beta)\big](1-y)
                   + \lambda\,\mathrm{KL}\big(\mathrm{Beta}(\tilde\alpha,\tilde\beta)
                     \,\|\,\mathrm{Beta}(1,1)\big),

    with :math:`\tilde\cdot` the evidence remaining after removing the correct
    class.  ``lambda`` must be annealed from 0 -- a constant KL weight makes the
    head output zero evidence for everything, which looks like a dead head.
    """

    def __init__(self, in_features: int, num_labels: int, *, hidden: int | None = None) -> None:
        super().__init__()
        d = hidden or in_features
        self.net = nn.Sequential(
            nn.LayerNorm(in_features), nn.Linear(in_features, d), nn.GELU(), nn.Linear(d, 2)
        )
        self.num_labels = num_labels

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        e = F.softplus(self.net(h)).clamp(0.0, 50.0)  # (B, L, 2)
        alpha = 1.0 + e[..., 0]
        beta = 1.0 + e[..., 1]
        s = alpha + beta
        p = alpha / s
        return {
            "prob": p,
            "logit": torch.log(p.clamp(1e-6, 1 - 1e-6)) - torch.log((1 - p).clamp(1e-6, 1 - 1e-6)),
            "alpha": alpha,
            "beta": beta,
            "epistemic": 2.0 / s,
            "aleatoric": alpha * beta / (s * s * (s + 1.0)),
        }

    @staticmethod
    def loss(
        out: dict[str, torch.Tensor],
        targets: torch.Tensor,
        *,
        kl_weight: float = 0.0,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        alpha, beta = out["alpha"], out["beta"]
        y = torch.nan_to_num(targets, nan=0.0)
        valid = torch.isfinite(targets).to(alpha.dtype)
        if mask is not None:
            valid = valid * mask.to(alpha.dtype)

        s = alpha + beta
        nll = y * (torch.digamma(s) - torch.digamma(alpha)) + (1 - y) * (
            torch.digamma(s) - torch.digamma(beta)
        )
        loss = (nll * valid).sum() / valid.sum().clamp_min(1.0)

        if kl_weight > 0:
            a_t = y * 1.0 + (1 - y) * alpha
            b_t = (1 - y) * 1.0 + y * beta
            kl = (
                torch.lgamma(a_t + b_t)
                - torch.lgamma(a_t)
                - torch.lgamma(b_t)
                - torch.lgamma(torch.tensor(2.0, device=alpha.device))
                + (a_t - 1) * (torch.digamma(a_t) - torch.digamma(a_t + b_t))
                + (b_t - 1) * (torch.digamma(b_t) - torch.digamma(a_t + b_t))
            )
            loss = loss + kl_weight * (kl * valid).sum() / valid.sum().clamp_min(1.0)
        return loss
