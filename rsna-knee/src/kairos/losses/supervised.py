r"""Supervised multi-label objectives, and a copula head for label dependence.

``AsymmetricLoss``
    Ridnik et al. (ICCV 2021).  Decouples the focusing exponent for positives
    and negatives and hard-thresholds easy negatives out of the gradient
    entirely via probability shifting.  With :math:`p_m = \max(p - m, 0)`,

    .. math::
        \mathcal L = -y (1-p)^{\gamma^+}\log p
                     - (1-y)\, p_m^{\gamma^-}\log(1 - p_m).

    For a label at 1 % prevalence, :math:`\gamma^- = 4, m = 0.05` removes
    ~95 % of negatives from the gradient after a few epochs, which is the
    difference between a usable and a dead rare-label head.

``SoftJaccardBCE``
    A masked BCE whose per-example weight is derived from the *label set*
    rather than from single labels.  Studies whose label vector is rare as a
    **combination** (e.g. Fracture without Contusion) are up-weighted, which
    targets exactly the co-occurrence structure a macro-AUC ensemble gets wrong.

``GaussianCopulaNLL``
    The twelve labels are strongly dependent: ACL tears co-occur with lateral
    contusions (pivot-shift injury), medial meniscus tears with medial OA.
    Twelve independent sigmoids cannot represent that; they can only represent
    the marginals.  We model the joint with a Gaussian copula: latent
    :math:`\mathbf z \sim \mathcal N(0, \Sigma)` with unit diagonal, and
    :math:`y_l = \mathbb 1[z_l > \Phi^{-1}(1 - \pi_l)]`.  The correlation
    matrix :math:`\Sigma` is parameterised by a low-rank plus diagonal
    factorisation :math:`\Sigma = \mathrm{diag}(d) + VV^\top`, normalised to
    unit diagonal, which keeps it positive definite by construction.

    The NLL requires an orthant probability, which has no closed form for
    :math:`L = 12`.  We use the standard *pairwise composite likelihood*
    (Varin & Vidoni, 2005): sum the exact bivariate NLLs over all 66 pairs.
    Composite likelihood is a consistent estimator of :math:`\Sigma`, is cheap,
    and -- critically for us -- **does not change the marginals**, so it can be
    added to an AUC-optimised system without disturbing the per-label ranking
    that the leaderboard actually scores.  Its role is as a regulariser and as
    a source of a *joint* posterior at inference for the report-gating logic.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "AsymmetricLoss",
    "SoftJaccardBCE",
    "GaussianCopulaNLL",
    "bivariate_normal_cdf",
]

_EPS = 1e-8


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for multi-label classification with NaN masking."""

    def __init__(
        self,
        *,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        pos_weight: torch.Tensor | None = None,
        eps: float = 1e-8,
        disable_grad_focal: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.disable_grad_focal = disable_grad_focal
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None  # type: ignore[assignment]

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        reduction: str | None = None,
    ) -> torch.Tensor:
        reduction = reduction or self.reduction
        y = torch.nan_to_num(targets, nan=0.0)
        valid = torch.isfinite(targets).to(logits.dtype)
        if mask is not None:
            valid = valid * mask.to(logits.dtype)

        p = torch.sigmoid(logits)
        p_neg = p
        if self.clip > 0:
            p_neg = (p - self.clip).clamp_min(0.0)

        loss_pos = y * torch.log(p.clamp_min(self.eps))
        loss_neg = (1 - y) * torch.log((1 - p_neg).clamp_min(self.eps))

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            ctx = torch.no_grad() if self.disable_grad_focal else _NullCtx()
            with ctx:
                pt = p * y + (1 - p_neg) * (1 - y)
                gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                w = torch.pow(1 - pt, gamma)
            loss_pos = loss_pos * w
            loss_neg = loss_neg * w

        if self.pos_weight is not None:
            loss_pos = loss_pos * self.pos_weight.to(logits.dtype)[None, :]

        loss = -(loss_pos + loss_neg) * valid
        if sample_weight is not None:
            sw = sample_weight if sample_weight.dim() == 2 else sample_weight[:, None]
            loss = loss * sw

        if reduction == "none":
            return loss
        if reduction == "per_example":
            return loss.sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        if reduction == "per_label":
            return loss.sum(dim=0) / valid.sum(dim=0).clamp_min(1.0)
        if reduction == "sum":
            return loss.sum()
        return loss.sum() / valid.sum().clamp_min(1.0)


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class SoftJaccardBCE(nn.Module):
    r"""Masked BCE reweighted by the rarity of the *label combination*.

    The weight of study :math:`i` is

    .. math::
        w_i = \Big(\frac{1}{\hat P(\mathbf y_i)}\Big)^{\kappa},\qquad
        \hat P(\mathbf y) = \prod_l \pi_l^{y_l}(1-\pi_l)^{1-y_l},

    clipped to ``[1/clip, clip]``.  Under the independence factorisation this
    is *exactly* the inverse propensity of the pattern; because the true joint
    is more concentrated than the product, the estimate over-weights genuinely
    unusual combinations, which is the intended behaviour: those are the
    studies where a 12-headed independent model is most wrong.
    """

    def __init__(
        self,
        prevalence: torch.Tensor,
        *,
        kappa: float = 0.25,
        clip: float = 4.0,
    ) -> None:
        super().__init__()
        self.register_buffer("log_pi", torch.log(prevalence.clamp(1e-4, 1 - 1e-4)))
        self.register_buffer("log_1mpi", torch.log1p(-prevalence.clamp(1e-4, 1 - 1e-4)))
        self.kappa = kappa
        self.clip = clip

    def pattern_weight(self, targets: torch.Tensor) -> torch.Tensor:
        y = torch.nan_to_num(targets, nan=0.0)
        valid = torch.isfinite(targets).to(y.dtype)
        log_p = ((y * self.log_pi + (1 - y) * self.log_1mpi) * valid).sum(dim=1)
        w = torch.exp(-self.kappa * log_p)
        w = w / w.mean().clamp_min(_EPS)
        return w.clamp(1.0 / self.clip, self.clip)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w = self.pattern_weight(targets)
        y = torch.nan_to_num(targets, nan=0.0)
        valid = torch.isfinite(targets).to(logits.dtype)
        if mask is not None:
            valid = valid * mask.to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        return (bce * valid * w[:, None]).sum() / valid.sum().clamp_min(1.0)


# --------------------------------------------------------------------------- #
# Gaussian copula                                                              #
# --------------------------------------------------------------------------- #


def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def bivariate_normal_cdf(
    h: torch.Tensor, k: torch.Tensor, rho: torch.Tensor, *, n_quad: int = 24
) -> torch.Tensor:
    r""":math:`\Phi_2(h, k; \rho)` by Gauss–Legendre quadrature on Drezner's
    integral form

    .. math::
        \Phi_2(h,k;\rho) = \Phi(h)\Phi(k)
          + \frac{1}{2\pi}\int_0^\rho \frac{1}{\sqrt{1-t^2}}
            \exp\!\Big(-\frac{h^2 - 2thk + k^2}{2(1-t^2)}\Big)\,dt.

    24 nodes give ~1e-10 absolute accuracy for :math:`|\rho| \le 0.95`, which is
    far below anything that matters here, and the whole thing is a handful of
    tensor ops so it is differentiable in ``rho``, ``h`` and ``k``.
    """
    dev, dt = h.device, h.dtype
    nodes, weights = _gauss_legendre(n_quad, dev, torch.float64)

    h64, k64, r64 = h.double(), k.double(), rho.double().clamp(-0.999, 0.999)
    t = 0.5 * r64[..., None] * (nodes + 1.0)  # map [-1,1] -> [0, rho]
    jac = 0.5 * r64[..., None]
    one_m = (1.0 - t * t).clamp_min(1e-12)
    expo = -(h64[..., None] ** 2 - 2.0 * t * h64[..., None] * k64[..., None] + k64[..., None] ** 2) / (
        2.0 * one_m
    )
    integrand = torch.exp(expo.clamp_min(-60.0)) / torch.sqrt(one_m)
    integral = (integrand * weights * jac).sum(dim=-1)
    out = _norm_cdf(h64) * _norm_cdf(k64) + integral / (2.0 * math.pi)
    return out.clamp(1e-9, 1.0 - 1e-9).to(dt)


_GL_CACHE: dict[tuple[int, str, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _gauss_legendre(n: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    key = (n, str(device), str(dtype))
    if key not in _GL_CACHE:
        # Golub–Welsch: nodes/weights are the eigen-decomposition of the
        # Jacobi matrix of the Legendre three-term recurrence.
        i = torch.arange(1, n, dtype=torch.float64)
        beta = i / torch.sqrt(4.0 * i * i - 1.0)
        J = torch.diag(beta, 1) + torch.diag(beta, -1)
        vals, vecs = torch.linalg.eigh(J)
        w = 2.0 * vecs[0, :] ** 2
        _GL_CACHE[key] = (vals.to(device=device, dtype=dtype), w.to(device=device, dtype=dtype))
    return _GL_CACHE[key]


class GaussianCopulaNLL(nn.Module):
    """Pairwise composite negative log-likelihood of a Gaussian copula.

    The marginals come from the network's own logits (so the copula never
    fights the AUC objective); only the correlation structure is learnt here.
    """

    def __init__(self, num_labels: int, *, rank: int = 4, init_scale: float = 0.1) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.V = nn.Parameter(torch.randn(num_labels, rank) * init_scale)
        self.log_d = nn.Parameter(torch.zeros(num_labels))

    def correlation(self) -> torch.Tensor:
        d = F.softplus(self.log_d) + 1e-3
        S = torch.diag(d) + self.V @ self.V.T
        s = torch.sqrt(torch.diagonal(S).clamp_min(1e-6))
        R = S / s[:, None] / s[None, :]
        # Clamp the *off-diagonal* only.  Clamping the whole matrix would set
        # the diagonal to 0.98, which is not a correlation matrix and makes the
        # bivariate CDF inconsistent with the marginals it is built from.
        eye = torch.eye(R.shape[0], device=R.device, dtype=R.dtype)
        return eye + (1.0 - eye) * R.clamp(-0.98, 0.98)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        detach_marginals: bool = True,
    ) -> torch.Tensor:
        p = torch.sigmoid(logits.detach() if detach_marginals else logits)
        p = p.clamp(1e-6, 1 - 1e-6)
        # Threshold in latent space: P(z > thr) = p  =>  thr = Phi^{-1}(1 - p)
        thr = -_ndtri(p)

        R = self.correlation().to(logits.dtype)
        y = torch.nan_to_num(targets, nan=0.0)
        valid = torch.isfinite(targets)
        if mask is not None:
            valid = valid & mask.bool()

        L = self.num_labels
        iu, ju = torch.triu_indices(L, L, offset=1)
        hi, hj = thr[:, iu], thr[:, ju]
        rho = R[iu, ju][None, :].expand_as(hi)

        # P(z_i <= h_i, z_j <= h_j) etc. from the bivariate CDF.
        p00 = bivariate_normal_cdf(hi, hj, rho)
        pi_ = _norm_cdf(hi)
        pj_ = _norm_cdf(hj)
        p01 = (pi_ - p00).clamp_min(1e-9)  # i negative, j positive
        p10 = (pj_ - p00).clamp_min(1e-9)
        p11 = (1.0 - pi_ - pj_ + p00).clamp_min(1e-9)

        yi, yj = y[:, iu], y[:, ju]
        ll = (
            (1 - yi) * (1 - yj) * torch.log(p00.clamp_min(1e-9))
            + (1 - yi) * yj * torch.log(p01)
            + yi * (1 - yj) * torch.log(p10)
            + yi * yj * torch.log(p11)
        )
        pair_valid = (valid[:, iu] & valid[:, ju]).to(ll.dtype)
        return -(ll * pair_valid).sum() / pair_valid.sum().clamp_min(1.0)


def _ndtri(p: torch.Tensor) -> torch.Tensor:
    """Inverse standard-normal CDF, :math:`\\Phi^{-1}`."""
    return math.sqrt(2.0) * torch.erfinv((2.0 * p - 1.0).clamp(-1 + 1e-7, 1 - 1e-7))
