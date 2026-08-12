r"""Slice encoders: pretrained 2D backbones adapted to 2.5D MRI input.

Design decision, and the evidence behind it.  CVPR 2026's *Revisiting 2D
Foundation Models for Scalable 3D Medical Image Classification* benchmarks 12
volumetric classification tasks and reports three findings that shape this
module: (i) properly-adapted 2D foundation models **beat** native 3D
architectures on 3D classification; (ii) general-purpose SSL backbones match
medical-specific ones once the adaptation is right; (iii) the adaptation
mechanism matters more than the backbone choice.  That is consistent with every
recent RSNA-Kaggle result -- 2022 cervical spine, 2023 abdominal trauma and
2024 lumbar spine were all won by 2.5D CNN/ViT + sequence-head designs, not by
3D nets.

So: a strong pretrained 2D encoder per slice, a sequence model across slices,
and a *volumetric branch kept only for ensemble diversity*.

Adaptation mechanisms implemented here:

``inflate_stem``
    Turn an RGB stem into an :math:`N`-channel 2.5D stem by replicating and
    rescaling the pretrained filters so that the *response to a constant input
    is preserved*: :math:`W' = \tfrac{3}{N}\,\mathrm{tile}(W)`.  Naive
    replication multiplies the pre-activation by :math:`N/3` and silently
    shifts every downstream BatchNorm statistic.

``LoRAAdapter``
    Low-rank residual adapters on the attention projections, so a frozen
    DINOv3/MedSigLIP backbone can be specialised with ~1 % of its parameters.
    This is the "lightweight plugin on a frozen backbone" recipe from the CVPR
    paper, and it is what makes running four different foundation backbones in
    one ensemble affordable.

``registry``
    Names resolve through ``timm`` when available; the wrapper falls back to a
    small built-in ConvNeXt-ish encoder so the pipeline is runnable and
    testable in an environment without ``timm`` (CI, and the first hour of a
    fresh Kaggle notebook).

Candidate checkpoints, all offline-packageable as Kaggle datasets:

===============================  ==========================================
checkpoint                       role
===============================  ==========================================
``convnext_small.fb_in22k_ft1k`` main CNN baseline / student teacher
``swin_small_patch4_window7``    hierarchical-attention diversity branch
``vit_base_patch14_dinov2``      SSL features, strong linear-probe transfer
``vit_base_patch16_dinov3``      newer SSL family; Gram-anchored patch feats
``medsiglip_400m``               medically-tuned SigLIP vision tower (448 px)
``resnet50.a1_in1k`` (RadImageNet-init)  radiology-domain 2D init
===============================  ==========================================

Every one of them must be ablated against ImageNet init on the *same* folds and
the same budget before it earns a slot -- domain-specific pretraining is a
hypothesis, not a guarantee, and chest-pretrained weights in particular have a
long history of not transferring to musculoskeletal MRI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BackboneSpec",
    "build_backbone",
    "inflate_stem",
    "LoRAAdapter",
    "apply_lora",
    "FallbackEncoder",
]


@dataclass(slots=True)
class BackboneSpec:
    #: ``fb_in22k_ft_in1k``, not ``fb_in22k_ft1k`` -- the latter is not a tag
    #: timm has ever published, and it silently produced a random-init encoder.
    name: str = "convnext_small.fb_in22k_ft_in1k"
    pretrained: bool = True
    in_chans: int = 5
    drop_path_rate: float = 0.1
    freeze: bool = False
    lora_rank: int = 0
    grad_checkpointing: bool = False
    out_indices: tuple[int, ...] | None = None
    #: Permit the randomly-initialised :class:`FallbackEncoder` when the named
    #: backbone cannot be built.  True keeps the unit tests and the smoke run
    #: executable without timm; ``04_train.py`` sets it False, because a real
    #: training run must never silently swap its encoder.
    allow_fallback: bool = True


def inflate_stem(weight: torch.Tensor, in_chans: int) -> torch.Tensor:
    r"""Adapt a ``(out, 3, kh, kw)`` conv kernel to ``in_chans`` channels.

    Preserves the expected pre-activation magnitude for a constant input:

    .. math::
        \sum_{c=1}^{N} W'_{:,c} = \sum_{c=1}^{3} W_{:,c}
        \;\Longrightarrow\; W' = \frac{3}{N}\,\mathrm{tile}_N(W).

    The centre channel additionally keeps a larger share (``centre_boost``) so
    the middle slice of the 2.5D stack -- the one the label is nominally about
    -- starts out dominating, matching how a radiologist reads the stack.
    """
    out_c, in_c, kh, kw = weight.shape
    if in_c == in_chans:
        return weight.clone()
    reps = math.ceil(in_chans / in_c)
    w = weight.repeat(1, reps, 1, 1)[:, :in_chans]
    w = w * (in_c / in_chans)
    centre = in_chans // 2
    boost = torch.ones(in_chans, device=w.device, dtype=w.dtype)
    boost[centre] = 1.5
    boost = boost / boost.sum() * in_chans
    return w * boost[None, :, None, None]


class LoRAAdapter(nn.Module):
    r"""Low-rank residual adapter: :math:`h \mapsto h + \tfrac{\alpha}{r} B A h`.

    ``A`` is Gaussian-initialised and ``B`` zero-initialised so the adapter is
    the identity at step 0 -- a frozen foundation backbone therefore starts
    exactly where its pretraining left it.
    """

    def __init__(self, base: nn.Linear, *, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(rank, base.in_features) * (base.in_features**-0.5))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * F.linear(F.linear(self.drop(x), self.A), self.B)


def apply_lora(
    module: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    match: Callable[[str], bool] | None = None,
) -> int:
    """Wrap matching ``nn.Linear`` children in :class:`LoRAAdapter`. Returns count."""
    if match is None:
        def match(name: str) -> bool:  # noqa: E306
            return any(k in name for k in ("qkv", "attn.proj", "q_proj", "v_proj", "fc1", "fc2"))

    n = 0
    for name, child in list(module.named_children()):
        full = name
        if isinstance(child, nn.Linear) and match(full):
            setattr(module, name, LoRAAdapter(child, rank=rank, alpha=alpha))
            n += 1
        else:
            n += apply_lora(child, rank=rank, alpha=alpha, match=match)
    return n


# --------------------------------------------------------------------------- #
# Fallback encoder (no timm)                                                   #
# --------------------------------------------------------------------------- #


class _ConvBlock(nn.Module):
    """Inverted-bottleneck block in the ConvNeXt style."""

    def __init__(self, dim: int, *, mult: int = 4, drop_path: float = 0.0) -> None:
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pw1 = nn.Conv2d(dim, dim * mult, 1)
        self.pw2 = nn.Conv2d(dim * mult, dim, 1)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim, 1, 1))
        self.drop_path = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pw2(F.gelu(self.pw1(self.norm(self.dw(x)))))
        h = self.gamma * h
        if self.drop_path > 0 and self.training:
            keep = 1.0 - self.drop_path
            m = (torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype) < keep).to(x.dtype)
            h = h * m / keep
        return x + h


class FallbackEncoder(nn.Module):
    """Small hierarchical CNN used when ``timm`` is unavailable.

    Not a competitive backbone.  Its job is to keep every unit test, shape
    check and end-to-end smoke run executable in a bare environment, so a
    missing dependency surfaces as "you are running the fallback" rather than
    as an import error five modules deep.
    """

    def __init__(self, in_chans: int = 5, *, dims: tuple[int, ...] = (48, 96, 192, 384),
                 depths: tuple[int, ...] = (2, 2, 4, 2), drop_path_rate: float = 0.0) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], 4, stride=4), nn.GroupNorm(1, dims[0])
        )
        stages, dp, total = [], 0.0, sum(depths)
        for i, (d, n) in enumerate(zip(dims, depths)):
            blocks = []
            if i > 0:
                blocks.append(nn.Sequential(nn.GroupNorm(1, dims[i - 1]),
                                            nn.Conv2d(dims[i - 1], d, 2, stride=2)))
            for _ in range(n):
                blocks.append(_ConvBlock(d, drop_path=drop_path_rate * dp / max(total - 1, 1)))
                dp += 1
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.num_features = dims[-1]
        self.norm = nn.GroupNorm(1, dims[-1])

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for s in self.stages:
            x = s(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x).mean(dim=(2, 3))


class BackboneWrapper(nn.Module):
    """Uniform interface: ``(B, C, H, W) -> (B, D)`` plus a spatial map."""

    def __init__(self, module: nn.Module, num_features: int, *, is_timm: bool) -> None:
        super().__init__()
        self.module = module
        self.num_features = num_features
        self.is_timm = is_timm

    def forward_map(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_timm:
            f = self.module.forward_features(x)
            if f.dim() == 3:  # ViT: (B, N, D) -> drop cls, reshape to a square
                n = f.shape[1]
                side = int(round(math.sqrt(n)))
                if side * side != n and n > 1:
                    f = f[:, 1:]
                    n = f.shape[1]
                    side = int(round(math.sqrt(n)))
                f = f.transpose(1, 2).reshape(f.shape[0], -1, side, side)
            return f
        return self.module.forward_features(x)

    def forward(self, x: torch.Tensor, *, return_map: bool = False):
        f = self.forward_map(x)
        pooled = f.mean(dim=(2, 3))
        return (pooled, f) if return_map else pooled


def build_backbone(spec: BackboneSpec) -> BackboneWrapper:
    """Instantiate a backbone, adapting the stem and optionally adding LoRA.

    On failure this falls back to :class:`FallbackEncoder` only when
    ``spec.allow_fallback`` is set, and says so loudly either way.  It used to
    fall back silently from a bare ``except Exception``, which meant a wrong
    pretrained tag -- ``convnext_small.fb_in22k_ft1k`` does not exist in timm
    1.0.x; the tag is ``fb_in22k_ft_in1k`` -- produced a randomly-initialised
    4.2M-parameter CNN while every log line still named the backbone that had
    been *requested*.  The failure was invisible until someone noticed the
    parameter count was 13M instead of 60M, and a training run that reaches
    the leaderboard on a random-init encoder costs a competition.
    """
    try:
        import timm  # type: ignore

        model = timm.create_model(
            spec.name,
            pretrained=spec.pretrained,
            in_chans=spec.in_chans,
            num_classes=0,
            drop_path_rate=spec.drop_path_rate,
        )
        num_features = int(getattr(model, "num_features", 0)) or _infer_dim(model, spec.in_chans)
        if spec.grad_checkpointing and hasattr(model, "set_grad_checkpointing"):
            model.set_grad_checkpointing(True)
        if spec.freeze:
            for p in model.parameters():
                p.requires_grad_(False)
        if spec.lora_rank > 0:
            apply_lora(model, rank=spec.lora_rank)
        return BackboneWrapper(model, num_features, is_timm=True)
    except Exception as exc:  # timm missing, bad tag, or no offline weights
        detail = f"{type(exc).__name__}: {exc}"
        if not spec.allow_fallback:
            raise RuntimeError(
                f"could not build backbone {spec.name!r} ({detail}).\n"
                "Refusing to substitute the randomly-initialised FallbackEncoder: "
                "training would run to completion and report healthy losses on an "
                "encoder that has learned nothing.\n"
                "Check the pretrained tag with `timm.list_pretrained('<name>*')`, "
                "or pass allow_fallback=True / --allow-fallback-backbone if a "
                "throwaway encoder is genuinely what you want."
            ) from exc
        import warnings

        msg = (f"!! backbone {spec.name!r} could not be built ({detail}); "
               "falling back to the randomly-initialised FallbackEncoder. "
               "This is NOT a competitive backbone.")
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        print(msg, flush=True)
        model = FallbackEncoder(in_chans=spec.in_chans, drop_path_rate=spec.drop_path_rate)
        return BackboneWrapper(model, model.num_features, is_timm=False)


@torch.no_grad()
def _infer_dim(model: nn.Module, in_chans: int) -> int:
    was = model.training
    model.eval()
    out = model(torch.zeros(1, in_chans, 64, 64))
    model.train(was)
    return int(out.shape[-1])
