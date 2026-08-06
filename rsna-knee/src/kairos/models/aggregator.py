r"""Within-series slice aggregation.

Given per-slice tokens :math:`\{h_{s,i}\}` for series :math:`s`, produce a
contextualised sequence that respects (a) the physical, possibly irregular
slice spacing, (b) missing/dropped slices, and (c) the fact that a knee series
can be 18 or 60 slices long depending on the site.

Two interchangeable aggregators are provided and both are used in the final
ensemble because their inductive biases genuinely differ:

:class:`SliceTransformer`
    Pre-norm transformer with physical-coordinate rotary attention and an
    explicit *distance bias*

    .. math::
        A_{ij} \mathrel{+}= -\lambda \, |z_i - z_j| \;+\; b_{\text{bucket}(|z_i-z_j|)},

    a learned monotone-initialised bucket bias over metric distance.  This is
    ALiBi generalised from token counts to millimetres.  It gives the model a
    locality prior it can override -- important because effusion is a global
    finding while a meniscal root tear is confined to 2–3 contiguous slices.

:class:`SelectiveScanAggregator`
    A diagonal, input-selective state-space layer (the Mamba/S6
    parameterisation) with the discretisation step :math:`\Delta` driven by the
    **actual slice gap in millimetres**:

    .. math::
        \bar A_i = \exp(\Delta_i A), \qquad
        \Delta_i = \mathrm{softplus}(w^\top h_i + b) \cdot \frac{\delta z_i}{\bar\delta z}.

    This is the one place where an SSM is strictly more natural than attention:
    the continuous-time formulation of an SSM *already is* a model of a signal
    sampled at irregular intervals, so a variable slice gap is handled exactly
    rather than approximated.  Cost is linear in the number of slices, which
    also makes it the aggregator of choice for the efficiency-track student.

Both return per-slice contextual tokens; pooling into a study representation is
the job of :mod:`kairos.models.label_queries`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import RotaryScalarEmbedding

__all__ = ["SliceTransformer", "SelectiveScanAggregator", "MetricDistanceBias"]


class MetricDistanceBias(nn.Module):
    r"""Learned attention bias over |Δz| in millimetres (metric ALiBi).

    Distances are bucketed logarithmically so the resolution is fine where it
    matters (0–10 mm) and coarse where it does not (>60 mm).  Initialised
    monotonically decreasing so that at step 0 the layer behaves like a soft
    locality prior, then free to learn otherwise.
    """

    def __init__(self, n_heads: int, *, n_buckets: int = 20, max_mm: float = 200.0) -> None:
        super().__init__()
        self.n_buckets = n_buckets
        self.max_mm = max_mm
        edges = torch.expm1(
            torch.linspace(0.0, math.log1p(max_mm), n_buckets)
        )
        self.register_buffer("edges", edges)
        init = -torch.linspace(0.0, 2.0, n_buckets)[None, :].repeat(n_heads, 1)
        self.bias = nn.Parameter(init)

    def forward(self, z_mm: torch.Tensor) -> torch.Tensor:
        """``z_mm``: ``(B, S)`` → ``(B, H, S, S)``."""
        d = (z_mm[:, :, None] - z_mm[:, None, :]).abs()
        idx = torch.bucketize(d, self.edges.to(d.dtype)).clamp(0, self.n_buckets - 1)
        b = self.bias.to(d.dtype)  # (H, K)
        return b[:, idx]  # (H, B, S, S)  -> permute below

    def as_attn_bias(self, z_mm: torch.Tensor) -> torch.Tensor:
        return self.forward(z_mm).permute(1, 0, 2, 3).contiguous()


class _MHA(nn.Module):
    """Multi-head attention with rotary physical positions and additive bias."""

    def __init__(self, dim: int, n_heads: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % n_heads:
            raise ValueError("dim must be divisible by n_heads")
        self.h = n_heads
        self.dk = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.rope = RotaryScalarEmbedding(self.dk)
        self.drop = dropout

    def forward(
        self,
        x: torch.Tensor,  # (B, S, D)
        z_mm: torch.Tensor,  # (B, S)
        key_padding_mask: torch.Tensor | None,  # (B, S) True = valid
        attn_bias: torch.Tensor | None,  # (B, H, S, S)
    ) -> torch.Tensor:
        B, S, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, S, self.h, self.dk).transpose(1, 2)
        k = k.view(B, S, self.h, self.dk).transpose(1, 2)
        v = v.view(B, S, self.h, self.dk).transpose(1, 2)

        q = self.rope(q, z_mm)
        k = self.rope(k, z_mm)

        mask = attn_bias
        if key_padding_mask is not None:
            pad = (~key_padding_mask)[:, None, None, :]
            neg = torch.finfo(q.dtype).min / 4
            add = torch.zeros(B, 1, 1, S, device=x.device, dtype=q.dtype).masked_fill(pad, neg)
            mask = add if mask is None else mask + add

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.drop if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.proj(out)


class _Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, *, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 drop_path: float = 0.0) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = _MHA(dim, n_heads, dropout=dropout)
        self.n2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )
        self.drop_path = drop_path
        # LayerScale: small init lets deep aggregators be added on top of a
        # pretrained 2D backbone without disturbing it early in training.
        self.ls1 = nn.Parameter(1e-4 * torch.ones(dim))
        self.ls2 = nn.Parameter(1e-4 * torch.ones(dim))

    def _dp(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_path <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_path
        m = torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype) < keep
        return x * m / keep

    def forward(self, x, z_mm, kpm, bias):
        x = x + self._dp(self.ls1 * self.attn(self.n1(x), z_mm, kpm, bias))
        x = x + self._dp(self.ls2 * self.mlp(self.n2(x)))
        return x


class SliceTransformer(nn.Module):
    """Stack of pre-norm blocks with metric rotary attention."""

    def __init__(
        self,
        dim: int,
        *,
        depth: int = 2,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        use_distance_bias: bool = True,
    ) -> None:
        super().__init__()
        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(
            [_Block(dim, n_heads, mlp_ratio=mlp_ratio, dropout=dropout, drop_path=float(p))
             for p in dpr]
        )
        self.bias = MetricDistanceBias(n_heads) if use_distance_bias else None
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,  # (B, S, D)
        z_mm: torch.Tensor,  # (B, S)
        slice_mask: torch.Tensor | None = None,  # (B, S) True = valid
    ) -> torch.Tensor:
        bias = self.bias.as_attn_bias(z_mm) if self.bias is not None else None
        for blk in self.blocks:
            x = blk(x, z_mm, slice_mask, bias)
        return self.norm(x)


# --------------------------------------------------------------------------- #
# Selective state-space aggregator                                             #
# --------------------------------------------------------------------------- #


class SelectiveScanAggregator(nn.Module):
    r"""Bidirectional diagonal SSM with physically-scaled discretisation.

    State recursion (per channel :math:`c`, state dim :math:`n`):

    .. math::
        h_{i} = \bar A_i \odot h_{i-1} + \bar B_i x_i, \qquad
        y_i = C_i^\top h_i + D x_i,

    with :math:`\bar A_i = \exp(\Delta_i A)`,
    :math:`\bar B_i = \Delta_i B_i` (Euler / zero-order-hold), and
    :math:`B_i, C_i, \Delta_i` produced from the input (the *selective*
    mechanism).  :math:`A` is initialised to the S4D-Lin spectrum
    :math:`A_n = -\tfrac12 - i\pi n`, real part only here, which gives a
    well-conditioned set of decay timescales spanning three orders of
    magnitude.

    The scan is written as an explicit loop over slices.  With
    :math:`S \le 48` that costs nothing and it keeps the code readable and
    exactly checkpointable; if you profile this as a bottleneck the correct fix
    is a chunked parallel scan, not a rewrite.

    Bidirectionality matters: pathology at slice :math:`i` is evidenced by
    context on *both* sides (a meniscal tear is confirmed by the adjacent
    normal slices), and a causal scan cannot see forward.
    """

    def __init__(
        self,
        dim: int,
        *,
        state_dim: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.d_inner = dim * expand
        self.n = state_dim
        self.bidirectional = bidirectional

        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)
        self.conv = nn.Conv1d(
            self.d_inner, self.d_inner, conv_kernel,
            groups=self.d_inner, padding=conv_kernel - 1, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.n * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        a = torch.arange(1, self.n + 1, dtype=torch.float32)[None, :].repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner * (2 if bidirectional else 1), dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

        nn.init.uniform_(self.dt_proj.bias, math.log(1e-3), math.log(1e-1))

    def _scan(
        self,
        u: torch.Tensor,  # (B, S, d_inner)
        dt: torch.Tensor,  # (B, S, d_inner)
        B_: torch.Tensor,  # (B, S, n)
        C_: torch.Tensor,  # (B, S, n)
        mask: torch.Tensor,  # (B, S) float 1/0
        reverse: bool,
    ) -> torch.Tensor:
        A = -torch.exp(self.A_log.to(u.dtype))  # (d_inner, n), negative real
        Bsz, S, _ = u.shape
        h = torch.zeros(Bsz, self.d_inner, self.n, device=u.device, dtype=u.dtype)
        outs = []
        idx = range(S - 1, -1, -1) if reverse else range(S)
        for i in idx:
            dti = dt[:, i, :]  # (B, d_inner)
            aBar = torch.exp(dti[..., None] * A[None])  # (B, d_inner, n)
            bBar = dti[..., None] * B_[:, i, None, :]  # (B, d_inner, n)
            m = mask[:, i, None, None]
            h = m * (aBar * h + bBar * u[:, i, :, None]) + (1 - m) * h
            outs.append(torch.einsum("bdn,bn->bd", h, C_[:, i, :]))
        if reverse:
            outs = outs[::-1]
        y = torch.stack(outs, dim=1)
        return y + u * self.D.to(u.dtype)[None, None, :]

    def forward(
        self,
        x: torch.Tensor,  # (B, S, D)
        z_mm: torch.Tensor,  # (B, S) physical coordinate
        slice_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        Bsz, S, _ = x.shape
        mask = (
            slice_mask.to(x.dtype)
            if slice_mask is not None
            else torch.ones(Bsz, S, device=x.device, dtype=x.dtype)
        )
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        u, gate = xz.chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., :S].transpose(1, 2)
        u = F.silu(u)

        proj = self.x_proj(u)
        B_, C_, dt_raw = proj[..., : self.n], proj[..., self.n : 2 * self.n], proj[..., -1:]

        # Physical slice gap, normalised by the series median, scales Δ.
        dz = torch.zeros_like(z_mm)
        dz[:, 1:] = (z_mm[:, 1:] - z_mm[:, :-1]).abs()
        valid = mask > 0
        med = torch.where(valid, dz, torch.full_like(dz, float("nan")))
        med = torch.nan_to_num(med, nan=0.0)
        denom = (med.sum(dim=1) / valid.sum(dim=1).clamp_min(1)).clamp_min(1e-3)
        dz_rel = (dz / denom[:, None]).clamp(0.1, 5.0)
        dz_rel[:, 0] = 1.0

        dt = F.softplus(self.dt_proj(dt_raw)) * dz_rel[..., None]

        y = self._scan(u, dt, B_, C_, mask, reverse=False)
        if self.bidirectional:
            y_rev = self._scan(u, dt, B_, C_, mask, reverse=True)
            y = torch.cat([y, y_rev], dim=-1)
            gate = torch.cat([gate, gate], dim=-1)

        y = y * F.silu(gate)
        return residual + self.drop(self.out_proj(y))
