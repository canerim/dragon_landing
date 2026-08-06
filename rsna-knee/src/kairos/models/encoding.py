r"""Geometry-aware conditioning.

A knee MRI series is not a video and it is not a bag.  Every slice has a
*physical* position, a *physical* thickness, and belongs to a series with a
*physical* in-plane resolution -- and those quantities vary by a factor of three
across the 16 acquisition sites.  A model that encodes slice index ``i / N``
learns a site-specific mapping from index to anatomy and then fails on the next
site's protocol.

Two mechanisms here, both cheap and both load-bearing:

:class:`PhysicalPositionalEncoding`
    Random Fourier features of the *metric* slice coordinate.  For a coordinate
    :math:`z` in millimetres,

    .. math::
        \phi(z) = \big[\cos(2\pi \omega_k z),\ \sin(2\pi \omega_k z)\big]_{k=1}^{K/2},
        \qquad \omega_k = \omega_{\min}\Big(\tfrac{\omega_{\max}}{\omega_{\min}}\Big)^{\frac{k-1}{K/2-1}},

    with the band chosen so that the lowest frequency has a period of ~200 mm
    (the whole knee) and the highest ~2 mm (a meniscal root).  This is the
    same construction as NeRF/Fourier-feature networks, and the reason it works
    here is the reason it works there: an MLP on a raw scalar coordinate has a
    strong low-frequency spectral bias and simply cannot represent "slice 14 is
    different from slice 15", which is exactly the resolution at which knee
    pathology lives.

    Crucially the encoding is a function of *millimetres*, so the same anatomy
    receives the same code at a site scanning 3 mm slices and a site scanning
    0.8 mm slices.

:class:`AcquisitionFiLM`
    Feature-wise linear modulation of the slice features by the acquisition
    context vector (in-plane spacing, slice thickness, field strength,
    echo/repetition time, fat-sat flag, sequence family embedding).  This lets
    the trunk *undo* protocol-induced appearance shifts rather than having to
    be invariant to them by brute force -- the same trick that makes
    conditional normalisation work for style transfer, applied to what is
    really a physics-parameter shift.

    We condition on log-spacing rather than spacing: a 0.3→0.6 mm change is the
    same perceptual step as 0.6→1.2 mm, and the log makes the network's job
    linear in that step.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PhysicalPositionalEncoding",
    "AcquisitionFiLM",
    "AcquisitionContext",
    "build_context_vector",
]


class PhysicalPositionalEncoding(nn.Module):
    """Log-spaced Fourier features of a physical coordinate in millimetres."""

    def __init__(
        self,
        dim: int,
        *,
        min_period_mm: float = 2.0,
        max_period_mm: float = 240.0,
        learnable_scale: bool = True,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even")
        k = dim // 2
        w_max = 1.0 / min_period_mm
        w_min = 1.0 / max_period_mm
        if k == 1:
            omegas = torch.tensor([math.sqrt(w_min * w_max)])
        else:
            omegas = torch.exp(
                torch.linspace(math.log(w_min), math.log(w_max), k)
            )
        self.register_buffer("omegas", omegas)
        self.scale = nn.Parameter(torch.ones(1)) if learnable_scale else None

    def forward(self, z_mm: torch.Tensor) -> torch.Tensor:
        """``z_mm``: ``(..., )`` physical coordinate → ``(..., dim)``."""
        w = self.omegas.to(z_mm.dtype)
        if self.scale is not None:
            w = w * self.scale.to(z_mm.dtype).abs().clamp(0.25, 4.0)
        ang = 2.0 * math.pi * z_mm[..., None] * w
        return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)


class AcquisitionContext(nn.Module):
    """Embed the DICOM acquisition parameters into a dense context vector."""

    #: Continuous fields, in the order produced by :func:`build_context_vector`.
    CONTINUOUS = (
        "log_in_plane_mm",
        "log_slice_mm",
        "log_extent_mm",
        "field_strength_t",
        "log_te_ms",
        "log_tr_ms",
        "n_slices_norm",
        "spacing_cv",
    )

    def __init__(
        self,
        dim: int,
        *,
        num_sequence_families: int,
        num_manufacturers: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_cont = len(self.CONTINUOUS)
        self.family = nn.Embedding(num_sequence_families, dim)
        self.manufacturer = nn.Embedding(num_manufacturers + 1, dim)
        self.fat_sat = nn.Embedding(3, dim)  # no / yes / unknown
        self.mlp = nn.Sequential(
            nn.Linear(self.n_cont, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        continuous: torch.Tensor,  # (..., n_cont)
        family_id: torch.Tensor,  # (...,)
        manufacturer_id: torch.Tensor,  # (...,)
        fat_sat_id: torch.Tensor,  # (...,)
    ) -> torch.Tensor:
        c = self.mlp(torch.nan_to_num(continuous, nan=0.0))
        e = self.family(family_id) + self.manufacturer(manufacturer_id) + self.fat_sat(fat_sat_id)
        return self.drop(self.norm(c + e))


class AcquisitionFiLM(nn.Module):
    r"""FiLM: :math:`h \leftarrow (1 + \gamma(c)) \odot h + \beta(c)`.

    The ``1 +`` matters: at initialisation the modulation is the identity, so
    adding FiLM to a pretrained backbone does not destroy its features on step
    zero.  ``gamma`` is additionally passed through a ``tanh`` scaled by
    ``max_gain`` so a pathological context vector cannot blow up activations --
    without that bound, a study with a missing ``EchoTime`` (imputed as an
    outlier) can produce NaNs three blocks later, and the traceback will point
    at the wrong module.
    """

    def __init__(self, feature_dim: int, context_dim: int, *, max_gain: float = 1.0) -> None:
        super().__init__()
        self.to_gamma = nn.Linear(context_dim, feature_dim)
        self.to_beta = nn.Linear(context_dim, feature_dim)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)
        self.max_gain = max_gain

    def forward(self, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gamma = self.max_gain * torch.tanh(self.to_gamma(context))
        beta = self.to_beta(context)
        while gamma.dim() < h.dim():
            gamma = gamma.unsqueeze(-2)
            beta = beta.unsqueeze(-2)
        return (1.0 + gamma) * h + beta


def build_context_vector(
    *,
    in_plane_mm: torch.Tensor,
    slice_mm: torch.Tensor,
    extent_mm: torch.Tensor,
    field_strength_t: torch.Tensor,
    te_ms: torch.Tensor,
    tr_ms: torch.Tensor,
    n_slices: torch.Tensor,
    spacing_cv: torch.Tensor,
) -> torch.Tensor:
    """Assemble and robustly normalise the continuous acquisition features.

    All the log transforms are ``log1p`` on a clamped positive value so a
    missing tag (imputed as 0) maps to 0 rather than ``-inf``.  ``n_slices`` is
    divided by 32 -- roughly the median -- so it lands near 1 and the linear
    layer sees an O(1) input like everything else.
    """

    def _log(x: torch.Tensor, lo: float = 1e-3) -> torch.Tensor:
        return torch.log1p(torch.nan_to_num(x, nan=0.0).clamp_min(lo))

    return torch.stack(
        [
            _log(in_plane_mm),
            _log(slice_mm),
            _log(extent_mm) / 5.0,
            torch.nan_to_num(field_strength_t, nan=0.0) / 3.0,
            _log(te_ms) / 5.0,
            _log(tr_ms) / 8.0,
            torch.nan_to_num(n_slices, nan=0.0) / 32.0,
            torch.nan_to_num(spacing_cv, nan=0.0).clamp(0.0, 1.0),
        ],
        dim=-1,
    )


class RotaryScalarEmbedding(nn.Module):
    r"""Rotary embedding driven by a physical scalar rather than a token index.

    Standard RoPE rotates query/key pairs by an angle proportional to the token
    *index*.  Slices are not equally spaced -- a series can have a 6 mm jump in
    the middle -- so index-driven RoPE encodes a lie.  Substituting the metric
    coordinate makes the relative-position structure of attention correspond to
    *actual millimetres between slices*, which is what the anatomy cares about.

    Attention then becomes translation-equivariant in physical space: two
    slices 4 mm apart attend to each other the same way regardless of where
    they sit in the stack or how the stack was sampled.
    """

    def __init__(self, head_dim: int, *, base_period_mm: float = 128.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        half = head_dim // 2
        inv = 1.0 / (
            base_period_mm ** (torch.arange(0, half, dtype=torch.float32) / max(half, 1))
        )
        self.register_buffer("inv_period", inv)

    def forward(self, x: torch.Tensor, z_mm: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, H, S, D)``, ``z_mm``: ``(B, S)``."""
        ang = z_mm[:, None, :, None] * self.inv_period.to(x.dtype)  # (B,1,S,D/2)
        cos, sin = torch.cos(ang), torch.sin(ang)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return out.flatten(-2)
