r"""Anatomy-safe augmentation for knee MRI.

Knee anatomy is rigid and the pathology is small.  The augmentation menu that
works for natural images actively destroys signal here, and the failure is not
visible in the training loss -- it shows up as a rare-label AUC that never
improves.

Three rules the ranges below encode:

**Geometry stays plausible.**  ±10° of rotation is already at the edge of what
a real patient position produces; ±30° produces a knee that does not exist, and
the network spends capacity learning that such knees are still knees.  The same
transform is applied to every slice of a series -- augmenting slices
independently breaks the 2.5D stack's premise that adjacent channels are
adjacent anatomy.

**Intensity augmentation simulates scanners, not art.**  Bias field, gamma,
noise and blur are the four things that genuinely differ between a 1.5 T
Siemens and a 3 T GE, and they are the four we apply.  A bias field is a smooth
multiplicative low-order polynomial, which is what B1 inhomogeneity actually
is; a random contrast jitter is not.

**No raw CutMix.**  Pasting a rectangle from another study creates anatomically
impossible compositions -- an ACL that stops mid-notch, a femur with two
condyles.  The label says "ACL tear" and the pixels say something that cannot
occur.  MixUp is applied at the *feature* level instead
(:func:`feature_mixup`), where the interpolation is between representations
rather than between anatomies.

**Horizontal flip is not free.**  Laterality is canonicalised upstream
(`data/dataset.py`), so a left-right flip re-introduces exactly the nuisance
that canonicalisation removed -- and worse, it swaps ``Medial OA`` and
``Lateral OA`` without swapping the labels.  It is therefore **off by default**
and, when enabled, comes with the label permutation applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import TARGETS

__all__ = [
    "AugmentConfig",
    "SeriesAugmenter",
    "MEDIAL_LATERAL_SWAP",
    "bias_field",
    "feature_mixup",
]


#: Index permutation applied when a series is mirrored: medial ↔ lateral.
#: Deriving this by hand once, here, is much safer than doing it at each call
#: site -- a mirror without this permutation silently corrupts four labels.
def _build_swap() -> tuple[int, ...]:
    pairs = {
        "Medial Meniscus": "Lateral Meniscus",
        "Lateral Meniscus": "Medial Meniscus",
        "Medial OA": "Lateral OA",
        "Lateral OA": "Medial OA",
    }
    idx = {t: i for i, t in enumerate(TARGETS)}
    return tuple(idx[pairs.get(t, t)] for t in TARGETS)


MEDIAL_LATERAL_SWAP: tuple[int, ...] = _build_swap()


@dataclass(slots=True)
class AugmentConfig:
    enabled: bool = True
    rotation_deg: float = 8.0
    scale: tuple[float, float] = (0.92, 1.08)
    translate_frac: float = 0.06
    crop_jitter_frac: float = 0.08
    gamma: tuple[float, float] = (0.85, 1.18)
    bias_field_strength: float = 0.20
    noise_sigma: float = 0.08
    blur_prob: float = 0.15
    slice_dropout: float = 0.10
    p_geometric: float = 0.8
    p_intensity: float = 0.7
    #: Left-right mirror.  Requires permuting the labels; see the module note.
    horizontal_flip_prob: float = 0.0
    seed: int | None = None
    _rng: np.random.Generator | None = field(default=None, repr=False, compare=False)


def bias_field(
    shape: tuple[int, int], strength: float, rng: np.random.Generator, *, order: int = 3
) -> np.ndarray:
    r"""Smooth multiplicative field simulating B1 inhomogeneity.

    A random low-order 2-D polynomial exponentiated:
    :math:`b(x,y) = \exp\big(s \sum_{i+j \le n} c_{ij} x^i y^j\big)` with
    :math:`c_{ij}\sim\mathcal N(0,1)` and coordinates on :math:`[-1,1]`.  The
    exponential keeps it strictly positive (a bias field cannot invert
    contrast) and the low order keeps it smooth on the scale of the whole
    image, which is what the physics produces -- a high-frequency multiplicative
    field is just noise wearing a different name.
    """
    h, w = shape
    y = np.linspace(-1.0, 1.0, h)[:, None]
    x = np.linspace(-1.0, 1.0, w)[None, :]
    acc = np.zeros((h, w), dtype=np.float32)
    for i in range(order + 1):
        for j in range(order + 1 - i):
            if i == 0 and j == 0:
                continue
            acc += float(rng.normal()) * (y**i) * (x**j)
    acc /= max(np.abs(acc).max(), 1e-6)
    return np.exp(strength * acc).astype(np.float32)


def _affine_grid(h: int, w: int, angle_deg: float, scale: float,
                 tx: float, ty: float) -> tuple[np.ndarray, np.ndarray]:
    """Source coordinates for an inverse-warped affine transform."""
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32) - (h - 1) / 2.0,
        np.arange(w, dtype=np.float32) - (w - 1) / 2.0,
        indexing="ij",
    )
    src_x = (ca * xx + sa * yy) / scale + (w - 1) / 2.0 + tx
    src_y = (-sa * xx + ca * yy) / scale + (h - 1) / 2.0 + ty
    return src_y, src_x


def _sample_bilinear(img: np.ndarray, sy: np.ndarray, sx: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x0 = np.clip(np.floor(sx).astype(np.int32), 0, w - 1)
    y0 = np.clip(np.floor(sy).astype(np.int32), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = np.clip(sx - x0, 0.0, 1.0).astype(np.float32)
    wy = np.clip(sy - y0, 0.0, 1.0).astype(np.float32)
    top = img[y0, x0] * (1 - wx) + img[y0, x1] * wx
    bot = img[y1, x0] * (1 - wx) + img[y1, x1] * wx
    out = top * (1 - wy) + bot * wy
    # Anything sampled from outside the original support is background.
    inside = (sx >= -0.5) & (sx <= w - 0.5) & (sy >= -0.5) & (sy <= h - 0.5)
    return np.where(inside, out, 0.0).astype(np.float32)


class SeriesAugmenter:
    """Applies one sampled transform consistently across a whole series."""

    def __init__(self, cfg: AugmentConfig | None = None) -> None:
        self.cfg = cfg or AugmentConfig()
        self.rng = self.cfg._rng or np.random.default_rng(self.cfg.seed)

    def sample_flip(self) -> bool:
        """Draw the mirror decision once, for a whole study.

        Exposed separately because the flip is the one augmentation that is a
        property of the *study* rather than of a series: it permutes four
        labels, and a per-series draw makes pixels and labels disagree.
        """
        p = self.cfg.horizontal_flip_prob
        return bool(p > 0 and self.rng.random() < p)

    def __call__(
        self, volume: np.ndarray, *, labels: np.ndarray | None = None,
        force_flip: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """``volume``: ``(S, H, W)``.  Returns the augmented volume and labels.

        Labels are returned because a horizontal flip permutes four of them.
        Callers must use the returned array, not the one they passed in.
        """
        cfg = self.cfg
        if not cfg.enabled:
            return volume, labels

        v = np.asarray(volume, dtype=np.float32)
        S, H, W = v.shape
        rng = self.rng
        out_labels = labels

        if rng.random() < cfg.p_geometric:
            angle = float(rng.uniform(-cfg.rotation_deg, cfg.rotation_deg))
            scale = float(rng.uniform(*cfg.scale))
            tx = float(rng.uniform(-cfg.translate_frac, cfg.translate_frac) * W)
            ty = float(rng.uniform(-cfg.translate_frac, cfg.translate_frac) * H)
            # ONE grid for the whole series: adjacent 2.5D channels must remain
            # adjacent anatomy.
            sy, sx = _affine_grid(H, W, angle, scale, tx, ty)
            v = np.stack([_sample_bilinear(v[s], sy, sx) for s in range(S)])

        do_flip = (
            self.sample_flip() if force_flip is None else bool(force_flip)
        )
        if do_flip:
            v = v[:, :, ::-1].copy()
            # Permute only when labels were supplied: the caller passes them on
            # the first series of a study and None thereafter, so the swap is
            # applied exactly once however many series the study has.
            if out_labels is not None:
                out_labels = np.asarray(out_labels)[list(MEDIAL_LATERAL_SWAP)]

        if rng.random() < cfg.p_intensity:
            if cfg.bias_field_strength > 0:
                v = v * bias_field((H, W), cfg.bias_field_strength, rng)[None]
            if cfg.gamma is not None:
                g = float(rng.uniform(*cfg.gamma))
                # Gamma needs a non-negative argument; the volume is z-scored,
                # so shift into [0, 1] and back rather than clipping (clipping
                # would delete the dark meniscus, which is the signal).
                lo, hi = float(v.min()), float(v.max())
                if hi > lo:
                    u = (v - lo) / (hi - lo)
                    v = np.power(u, g) * (hi - lo) + lo
            if cfg.noise_sigma > 0:
                v = v + rng.normal(0.0, cfg.noise_sigma, size=v.shape).astype(np.float32)
            if rng.random() < cfg.blur_prob:
                v = _blur3(v)

        if cfg.slice_dropout > 0 and S > 4:
            keep = rng.random(S) >= cfg.slice_dropout
            if keep.sum() >= max(3, S // 2):
                v[~keep] = 0.0

        return v.astype(np.float32), out_labels


def _blur3(v: np.ndarray) -> np.ndarray:
    """Separable 3-tap [1,2,1]/4 blur, in-plane only."""
    k = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    out = v.copy()
    pad = np.pad(out, ((0, 0), (1, 1), (0, 0)), mode="edge")
    out = k[0] * pad[:, :-2] + k[1] * pad[:, 1:-1] + k[2] * pad[:, 2:]
    pad = np.pad(out, ((0, 0), (0, 0), (1, 1)), mode="edge")
    out = k[0] * pad[:, :, :-2] + k[1] * pad[:, :, 1:-1] + k[2] * pad[:, :, 2:]
    return out.astype(np.float32)


def feature_mixup(features, targets, *, alpha: float = 0.2, rng=None):
    r"""MixUp applied to representations, not to pixels.

    :math:`\tilde h = \lambda h_i + (1-\lambda) h_j`,
    :math:`\tilde y = \lambda y_i + (1-\lambda) y_j`,
    :math:`\lambda \sim \mathrm{Beta}(\alpha,\alpha)`.

    Doing this at the feature level rather than the pixel level is what makes
    it safe here: interpolating two knees in pixel space produces an image of
    no knee, whereas interpolating their representations produces a point on
    the segment between two valid representations, which is exactly the
    smoothness assumption MixUp is trying to enforce.

    NaN targets (unobserved labels) are preserved as NaN whenever *either*
    parent is NaN -- mixing a known label with an unknown one produces an
    unknown, not a half-known.
    """
    import torch

    if alpha <= 0:
        return features, targets, 1.0
    lam = float(np.random.default_rng(rng).beta(alpha, alpha)) if rng is not None else \
        float(np.random.beta(alpha, alpha))
    perm = torch.randperm(features.shape[0], device=features.device)
    mixed_f = lam * features + (1 - lam) * features[perm]
    t, tp = targets, targets[perm]
    mixed_t = lam * torch.nan_to_num(t) + (1 - lam) * torch.nan_to_num(tp)
    unknown = torch.isnan(t) | torch.isnan(tp)
    mixed_t = torch.where(unknown, torch.full_like(mixed_t, float("nan")), mixed_t)
    return mixed_f, mixed_t, lam
