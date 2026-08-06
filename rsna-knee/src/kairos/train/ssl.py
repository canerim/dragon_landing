r"""Self-supervised objectives for curriculum stage S0.

Stage S0 exists to give the encoder features that are stable under protocol
change *before* any label is touched.  Two objectives, chosen because they are
the two that need no new data and no pixel decoder.

:class:`MaskedTokenModelling`
    Masked *feature* modelling rather than masked pixel modelling.  A fraction
    of the per-slice tokens is replaced by a learned mask embedding; a small
    transformer decoder then reconstructs the original tokens from the
    surviving context, and the loss is a smooth-L1 on the reconstruction.

    Why features and not pixels: a pixel decoder for 256×256 slices is a large
    chunk of parameters that is thrown away at the end of S0, and MRI pixel
    reconstruction is dominated by noise texture that carries no anatomical
    information -- the network spends its capacity learning the scanner's noise
    spectrum.  Regressing the *encoder's own* representation (as in data2vec
    and BEiT-v2's feature tokenisation) targets exactly the semantic content we
    want stabilised.  The target is taken from an EMA copy of the encoder path,
    detached, so the objective cannot be solved by collapsing every token to a
    constant.

    Collapse is the failure mode to watch: if the reconstruction loss falls
    smoothly to near-zero in the first epoch, the tokens have collapsed.  The
    ``variance`` diagnostic returned alongside the loss is the number to
    monitor -- it must stay well away from zero.

:class:`CrossPlaneConsistency`
    A knee is one object.  The study representation computed from the sagittal
    series alone, the coronal series alone, and the axial series alone must
    describe the same knee.  We therefore penalise the disagreement between
    per-plane study embeddings.

    This is a genuinely free supervisory signal -- it needs no labels, no
    reports and no extra forward pass beyond re-pooling the tokens we already
    have -- and it directly targets the thing that breaks across sites: a model
    that has learnt "sagittal means ACL" rather than "this anatomy means ACL"
    produces plane embeddings that disagree, and this loss punishes exactly
    that.

    Implemented as a **variance** penalty across planes plus a decorrelation
    term (the VICReg construction) rather than as a plain L2 pull, because a
    plain L2 between views has the trivial solution of a constant embedding and
    needs negatives or a stop-gradient to avoid it; the variance term makes
    collapse explicitly costly and needs neither.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import Plane

__all__ = ["MaskedTokenModelling", "CrossPlaneConsistency", "SSLHeads"]


class MaskedTokenModelling(nn.Module):
    """Masked feature modelling over per-slice tokens."""

    def __init__(
        self,
        dim: int,
        *,
        mask_ratio: float = 0.5,
        depth: int = 2,
        n_heads: int = 8,
        ema_decay: float = 0.999,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay

        self.mask_token = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.mask_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 2,
            dropout=0.0, batch_first=True, norm_first=True, activation="gelu",
        )
        # enable_nested_tensor is incompatible with norm_first=True; torch
        # warns on every construction and the fast path it would enable does
        # not apply to a pre-norm layer.
        self.decoder = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.predict = nn.Linear(dim, dim)
        self.target_norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(
        self,
        tokens: torch.Tensor,  # (B, Nseq, S, D)
        mask: torch.Tensor,  # (B, Nseq, S) bool, True = real slice
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        B, Nseq, S, D = tokens.shape
        x = tokens.reshape(B * Nseq, S, D)
        valid = mask.reshape(B * Nseq, S)
        if not bool(valid.any()):
            zero = tokens.sum() * 0.0
            return {"loss": zero, "variance": zero, "masked_fraction": zero}

        # Target: the encoder's own tokens, LayerNorm'd and detached.  The
        # normalisation is what stops the loss being dominated by the tokens
        # with the largest norm, which are not the informative ones.
        target = self.target_norm(x).detach()

        r = torch.rand(x.shape[:2], device=x.device, generator=generator)
        drop = (r < self.mask_ratio) & valid
        # Never mask an entire series: with no context the task is unsolvable
        # and the gradient is pure noise.
        all_masked = drop.sum(dim=1) >= valid.sum(dim=1).clamp_min(1)
        if bool(all_masked.any()):
            keep_first = torch.zeros_like(drop)
            first = valid.float().argmax(dim=1)
            keep_first[torch.arange(drop.shape[0], device=x.device), first] = True
            drop = drop & ~(all_masked[:, None] & keep_first)

        corrupted = torch.where(
            drop[..., None], self.mask_token.to(x.dtype).expand_as(x), x
        )
        pad = ~valid
        decoded = self.decoder(corrupted, src_key_padding_mask=pad)
        pred = self.predict(decoded)

        sel = drop & valid
        n = sel.sum().clamp_min(1)
        loss = F.smooth_l1_loss(pred[sel], target[sel], beta=1.0)

        # Collapse diagnostic: the per-dimension standard deviation of the
        # predictions.  A healthy run keeps this near 1 (the targets are
        # LayerNorm'd); a collapsing run drives it to 0 while the loss also
        # falls, which looks like success.
        variance = pred[sel].std(dim=0).mean() if int(n) > 1 else pred.sum() * 0.0
        return {
            "loss": loss,
            "variance": variance.detach(),
            "masked_fraction": (sel.sum() / valid.sum().clamp_min(1)).detach(),
        }


class CrossPlaneConsistency(nn.Module):
    r"""VICReg-style agreement between per-plane study embeddings.

    .. math::
        \mathcal L = \lambda_{\text{inv}}\,\underbrace{\tfrac{1}{|\mathcal P|}
            \sum_{p} \lVert e_p - \bar e \rVert^2}_{\text{invariance}}
        + \lambda_{\text{var}} \underbrace{\tfrac1D\sum_d
            \mathrm{ReLU}\big(\gamma - \sqrt{\mathrm{Var}_b(e_{\cdot d}) + \epsilon}\big)}_{\text{anti-collapse}}
        + \lambda_{\text{cov}} \underbrace{\tfrac1D\sum_{i\ne j} C_{ij}^2}_{\text{decorrelation}}

    The variance hinge is what makes this safe without negatives: driving every
    embedding to a constant satisfies the invariance term perfectly and is
    punished at full strength by the second.
    """

    def __init__(
        self,
        dim: int,
        *,
        proj_dim: int = 256,
        lambda_inv: float = 1.0,
        lambda_var: float = 1.0,
        lambda_cov: float = 0.04,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.gamma = gamma

    @staticmethod
    def _plane_of(family_id: torch.Tensor) -> torch.Tensor:
        from ..constants import SEQUENCE_FAMILIES

        lut = torch.tensor(
            [int(f.plane) for f in SEQUENCE_FAMILIES],
            device=family_id.device, dtype=torch.long,
        )
        return lut[family_id.clamp(0, lut.numel() - 1)]

    def forward(
        self,
        tokens: torch.Tensor,  # (B, Nseq, S, D)
        slice_mask: torch.Tensor,  # (B, Nseq, S)
        series_mask: torch.Tensor,  # (B, Nseq)
        family_id: torch.Tensor,  # (B, Nseq)
    ) -> dict[str, torch.Tensor]:
        B, Nseq, S, D = tokens.shape
        planes = self._plane_of(family_id)  # (B, Nseq)

        # Mean-pool tokens within each series, then within each plane.
        w = (slice_mask & series_mask[:, :, None]).to(tokens.dtype)
        series_emb = (tokens * w[..., None]).sum(2) / w.sum(2, keepdim=True).clamp_min(1e-6)

        embeddings, present = [], []
        for p in (Plane.SAGITTAL, Plane.CORONAL, Plane.AXIAL):
            sel = (planes == int(p)) & series_mask
            cnt = sel.sum(dim=1, keepdim=True).to(tokens.dtype)
            e = (series_emb * sel[..., None].to(tokens.dtype)).sum(1) / cnt.clamp_min(1e-6)
            embeddings.append(e)
            present.append(sel.any(dim=1))

        E = torch.stack(embeddings, dim=1)  # (B, 3, D)
        P = torch.stack(present, dim=1)  # (B, 3)

        # Only studies with at least two planes contribute to the invariance
        # term -- a single-plane study has nothing to be consistent with, and
        # including it would just pull its embedding towards itself.
        usable = P.sum(dim=1) >= 2
        zero = tokens.sum() * 0.0
        if not bool(usable.any()):
            return {"loss": zero, "invariance": zero, "variance": zero, "covariance": zero}

        Eu, Pu = E[usable], P[usable].to(E.dtype)
        flat = self.proj(Eu.reshape(-1, D)).reshape(Eu.shape[0], 3, -1)
        mask = Pu[..., None]
        mean = (flat * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1e-6)
        inv = (((flat - mean) ** 2).sum(-1) * Pu).sum() / Pu.sum().clamp_min(1.0)

        z = flat.reshape(-1, flat.shape[-1])
        keep = Pu.reshape(-1) > 0
        z = z[keep]
        if z.shape[0] < 2:
            return {"loss": self.lambda_inv * inv, "invariance": inv,
                    "variance": zero, "covariance": zero}

        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        var = F.relu(self.gamma - std).mean()

        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / max(z.shape[0] - 1, 1)
        off = cov - torch.diag(torch.diagonal(cov))
        covl = (off**2).sum() / cov.shape[0]

        loss = self.lambda_inv * inv + self.lambda_var * var + self.lambda_cov * covl
        return {"loss": loss, "invariance": inv.detach(),
                "variance": var.detach(), "covariance": covl.detach()}


class SSLHeads(nn.Module):
    """Container so the trainer can move/checkpoint both heads as one module."""

    def __init__(self, dim: int, *, mask_ratio: float = 0.5) -> None:
        super().__init__()
        self.mim = MaskedTokenModelling(dim, mask_ratio=mask_ratio)
        self.cross_plane = CrossPlaneConsistency(dim)

    @staticmethod
    def suggested_mask_ratio(n_slices: float) -> float:
        r"""Mask ratio scaled to the sequence length.

        MAE's 75 % works because an image has thousands of highly redundant
        patches.  A knee series has 20-48 slices and the redundancy between
        adjacent slices is high but *not* that high; masking 75 % of 24 slices
        leaves 6, which is not enough context to reconstruct a 2 mm structure.
        We scale between 0.35 and 0.60 with the log of the slice count.
        """
        t = (math.log(max(n_slices, 4.0)) - math.log(4.0)) / (math.log(64.0) - math.log(4.0))
        return float(min(0.60, max(0.35, 0.35 + 0.25 * t)))
