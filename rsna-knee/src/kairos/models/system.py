r"""KAIROS: the assembled coarse-to-fine, label-query, multi-sequence model.

Forward pass, end to end:

.. code-block:: text

    study
      └─ for each series s (variable count, missing families allowed)
           ├─ 2.5D slice stacks           (B·S, C, H, W) at COARSE resolution
           ├─ pretrained 2D backbone       → per-slice tokens h_{s,i}
           ├─ AcquisitionFiLM(context_s)   → protocol-conditioned tokens
           ├─ SliceTransformer / SSM       → contextual tokens (metric ΔZ aware)
           └─ LabelQueryPool               → v_{l,s}, attention a_{l,s,·}
      ├─ CrossSequenceFusion               → z_l (missing-sequence masked)
      ├─ LabelExpertRouter (top-2 MoE)     → z_l refined
      ├─ SNGPHead                          → coarse logits + epistemic variance
      │
      ├─ ConfidenceGate(coarse, variance)  → which labels need the fine pass
      ├─ GumbelTopKSelector(a_{l,s,·})     → which slices, dilated to windows
      ├─ fine backbone on selected windows at FINE resolution
      └─ gated fusion(coarse z_l, fine z_l) → final logits

Everything that can be skipped at inference is skipped *per label*, so a study
where eleven labels are confidently negative and one is borderline costs almost
exactly one coarse pass plus one small fine pass.

The class is deliberately configuration-driven rather than subclass-driven:
the ensemble's diversity comes from instantiating this same class with
different backbones, aggregators and resolutions, which keeps the OOF matrices
directly comparable and the inference engine single-path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import (
    LABEL_COMPARTMENT,
    NUM_SEQUENCE_FAMILIES,
    NUM_TARGETS,
    SEQUENCE_LABEL_PRIOR,
    TARGETS,
    GROUP_OF_LABEL,
    LABEL_GROUPS,
)
from .adaptive import ConfidenceGate, GumbelTopKSelector
from .aggregator import SelectiveScanAggregator, SliceTransformer
from .backbones import BackboneSpec, build_backbone
from .encoding import AcquisitionContext, AcquisitionFiLM, PhysicalPositionalEncoding
from .heads import SNGPHead
from .hyperbolic import OntologyEmbedding
from .label_queries import CrossSequenceFusion, LabelExpertRouter, LabelHeadBank, LabelQueryPool

__all__ = ["KairosConfig", "KairosModel", "StudyBatch"]


@dataclass(slots=True)
class KairosConfig:
    backbone: BackboneSpec = field(default_factory=BackboneSpec)
    fine_backbone: BackboneSpec | None = None  # None -> share the coarse backbone
    dim: int = 384
    num_labels: int = NUM_TARGETS
    aggregator: str = "transformer"  # "transformer" | "ssm" | "both"
    agg_depth: int = 2
    agg_heads: int = 8
    n_query_tokens: int = 2
    fusion_mode: str = "attention"  # "attention" | "transformer"
    use_moe: bool = True
    n_experts: int = 6
    moe_top_k: int = 2
    use_sngp: bool = True
    sngp_features: int = 1024
    use_ontology: bool = True
    ontology_dim: int = 16
    dropout: float = 0.1
    # Adaptive compute
    enable_fine_pass: bool = True
    fine_top_k: int = 6
    fine_window_radius: int = 1
    fine_budget_fraction: float = 0.30
    # Regularisation knobs surfaced to the trainer
    attn_entropy_floor: float = 1.0
    sequence_dropout: float = 0.15
    slice_dropout: float = 0.10


@dataclass(slots=True)
class StudyBatch:
    """One batch of studies.

    Ragged dimensions are padded and accompanied by boolean masks; nothing in
    the model may branch on the *number* of series or slices, because the
    inference engine batches studies of different shapes together.
    """

    pixels: torch.Tensor  # (B, Nseq, S, C, H, W) coarse stacks
    slice_mask: torch.Tensor  # (B, Nseq, S) bool
    series_mask: torch.Tensor  # (B, Nseq) bool
    z_mm: torch.Tensor  # (B, Nseq, S) physical coordinate
    family_id: torch.Tensor  # (B, Nseq) long
    manufacturer_id: torch.Tensor  # (B, Nseq) long
    fat_sat_id: torch.Tensor  # (B, Nseq) long
    context: torch.Tensor  # (B, Nseq, n_cont)
    targets: torch.Tensor | None = None  # (B, L)
    group_id: torch.Tensor | None = None  # (B,) site×language×scanner bucket
    study_uid: list[str] | None = None
    fine_pixels: torch.Tensor | None = None  # (B, Nseq, S, C, Hf, Wf) if precomputed

    # ---- optional supervision channels -------------------------------- #
    # All default to None so a purely image-only batch is fully valid.  The
    # objective registry checks for presence and the schedule validator refuses
    # to start a run that *schedules* a term whose inputs are absent -- silently
    # skipping a scheduled loss is how a training run comes back looking fine
    # and being worthless.
    weak_labels: torch.Tensor | None = None  # (B, L) soft targets from reports
    weak_confidence: torch.Tensor | None = None  # (B, L) in [0, 1]
    text_embedding: torch.Tensor | None = None  # (B, D_text) pooled report
    graph_embedding: torch.Tensor | None = None  # (B, D_graph) concept graph
    phrase_embedding: torch.Tensor | None = None  # (B, P, D_text)
    phrase_mask: torch.Tensor | None = None  # (B, P) bool
    phrase_label: torch.Tensor | None = None  # (B, P) long, -1 = no label
    phrase_confidence: torch.Tensor | None = None  # (B, P)
    has_report: torch.Tensor | None = None  # (B,) bool
    teacher_logits: torch.Tensor | None = None  # (B, L) cross-fitted OOF teacher
    teacher_attention: torch.Tensor | None = None  # (B, L, S_flat)
    env_id: torch.Tensor | None = None  # (B,) environment for IRM

    def to(self, device) -> "StudyBatch":
        """Move every tensor field to ``device``, leaving ``None`` fields alone."""
        for f in self.__slots__:
            v = getattr(self, f, None)
            if torch.is_tensor(v):
                setattr(self, f, v.to(device, non_blocking=True))
        return self


class KairosModel(nn.Module):
    def __init__(self, cfg: KairosConfig) -> None:
        super().__init__()
        self.cfg = cfg
        L = cfg.num_labels

        self.backbone = build_backbone(cfg.backbone)
        self.fine_backbone = (
            build_backbone(cfg.fine_backbone) if cfg.fine_backbone is not None else None
        )
        d_bb = self.backbone.num_features

        self.slice_proj = nn.Sequential(
            nn.Linear(d_bb, cfg.dim), nn.LayerNorm(cfg.dim), nn.GELU()
        )
        self.pos = PhysicalPositionalEncoding(cfg.dim)
        self.ctx = AcquisitionContext(cfg.dim, num_sequence_families=NUM_SEQUENCE_FAMILIES)
        self.film = AcquisitionFiLM(cfg.dim, cfg.dim)

        if cfg.aggregator in ("transformer", "both"):
            self.agg_tf = SliceTransformer(
                cfg.dim, depth=cfg.agg_depth, n_heads=cfg.agg_heads, dropout=cfg.dropout
            )
        else:
            self.agg_tf = None
        if cfg.aggregator in ("ssm", "both"):
            self.agg_ssm = SelectiveScanAggregator(cfg.dim, dropout=cfg.dropout)
        else:
            self.agg_ssm = None
        if cfg.aggregator == "both":
            self.agg_merge = nn.Linear(2 * cfg.dim, cfg.dim)
        else:
            self.agg_merge = None

        self.pool = LabelQueryPool(
            cfg.dim,
            L,
            n_query_tokens=cfg.n_query_tokens,
            dropout=cfg.dropout,
            attn_entropy_floor=cfg.attn_entropy_floor,
        )
        prior = torch.tensor(SEQUENCE_LABEL_PRIOR, dtype=torch.float32)
        self.fusion = CrossSequenceFusion(
            cfg.dim, L, NUM_SEQUENCE_FAMILIES, mode=cfg.fusion_mode,
            n_heads=cfg.agg_heads, prior=prior, dropout=cfg.dropout,
        )

        if cfg.use_moe:
            group_names = list(LABEL_GROUPS.keys())
            hint = torch.tensor(
                [group_names.index(GROUP_OF_LABEL[t]) for t in TARGETS], dtype=torch.long
            )
            self.router = LabelExpertRouter(
                cfg.dim, L, n_experts=cfg.n_experts, top_k=cfg.moe_top_k,
                label_prior=hint, dropout=cfg.dropout,
            )
        else:
            self.router = None

        if cfg.use_ontology:
            group_names = list(LABEL_GROUPS.keys())
            parent = torch.tensor(
                [group_names.index(GROUP_OF_LABEL[t]) for t in TARGETS], dtype=torch.long
            )
            self.ontology = OntologyEmbedding(
                L, len(group_names), cfg.ontology_dim, parent_of_label=parent
            )
            self.ontology_lift = nn.Linear(cfg.ontology_dim, cfg.dim, bias=False)
        else:
            self.ontology = None
            self.ontology_lift = None

        if cfg.use_sngp:
            self.head = SNGPHead(cfg.dim, L, num_random_features=cfg.sngp_features)
            self.linear_head = None
        else:
            self.head = None
            self.linear_head = LabelHeadBank(cfg.dim, L, dropout=cfg.dropout)

        # Compartment prior: a small learned bias added to slice-attention
        # logits, initialised from the anatomical compartment of each label.
        self.compartment = nn.Embedding(len(set(LABEL_COMPARTMENT.values())) + 1, 1)
        nn.init.zeros_(self.compartment.weight)

        # --- adaptive compute -------------------------------------------- #
        self.selector = GumbelTopKSelector(
            k=cfg.fine_top_k, window_radius=cfg.fine_window_radius
        )
        self.gate = ConfidenceGate(L)
        self.fine_merge = nn.Sequential(
            nn.LayerNorm(2 * cfg.dim), nn.Linear(2 * cfg.dim, cfg.dim), nn.GELU()
        )
        self.fine_gate = nn.Linear(2 * cfg.dim, 1)
        nn.init.zeros_(self.fine_gate.weight)
        nn.init.constant_(self.fine_gate.bias, -2.0)  # start mostly-coarse

    # ------------------------------------------------------------------ #
    # Series encoding                                                     #
    # ------------------------------------------------------------------ #

    def encode_series(
        self,
        pixels: torch.Tensor,  # (B*Nseq, S, C, H, W)
        z_mm: torch.Tensor,  # (B*Nseq, S)
        slice_mask: torch.Tensor,  # (B*Nseq, S)
        context: torch.Tensor,  # (B*Nseq, D)
        *,
        backbone=None,
        subset: torch.Tensor | None = None,  # (B*Nseq, S) bool: encode only these
    ) -> torch.Tensor:
        bb = backbone or self.backbone
        N, S, C, H, W = pixels.shape
        flat = pixels.reshape(N * S, C, H, W)
        keep = (slice_mask if subset is None else (slice_mask & subset)).reshape(-1)

        feats = torch.zeros(N * S, bb.num_features, device=pixels.device, dtype=pixels.dtype)
        if bool(keep.any()):
            idx = keep.nonzero(as_tuple=True)[0]
            # Chunk to bound peak memory independently of the batch shape.
            chunk = 256
            outs = []
            for i in range(0, idx.numel(), chunk):
                outs.append(bb(flat[idx[i : i + chunk]]))
            feats[idx] = torch.cat(outs, dim=0).to(feats.dtype)

        h = self.slice_proj(feats.reshape(N, S, -1))
        h = h + self.pos(z_mm)
        h = self.film(h, context[:, None, :].expand(-1, S, -1))

        outs = []
        if self.agg_tf is not None:
            outs.append(self.agg_tf(h, z_mm, slice_mask))
        if self.agg_ssm is not None:
            outs.append(self.agg_ssm(h, z_mm, slice_mask))
        if len(outs) == 2:
            h = self.agg_merge(torch.cat(outs, dim=-1))
        else:
            h = outs[0]
        return h

    # ------------------------------------------------------------------ #

    def _attention_bias(self, batch_shape, device, dtype) -> torch.Tensor:
        comp = torch.tensor(
            [int(LABEL_COMPARTMENT[t]) for t in TARGETS], device=device, dtype=torch.long
        )
        b = self.compartment(comp).squeeze(-1)  # (L,)
        return b.to(dtype)

    def _apply_modality_dropout(self, series_mask: torch.Tensor) -> torch.Tensor:
        if not self.training or self.cfg.sequence_dropout <= 0:
            return series_mask
        drop = torch.rand_like(series_mask, dtype=torch.float32) < self.cfg.sequence_dropout
        out = series_mask & ~drop
        # Never drop every series of a study.
        empty = ~out.any(dim=1)
        if bool(empty.any()):
            first = series_mask.float().argmax(dim=1)
            out[empty, first[empty]] = True
        return out

    def forward(
        self,
        batch: StudyBatch,
        *,
        run_fine: bool | None = None,
        update_precision: bool = False,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        B, Nseq, S = batch.slice_mask.shape
        L = cfg.num_labels
        dev = batch.pixels.device

        series_mask = self._apply_modality_dropout(batch.series_mask)
        slice_mask = batch.slice_mask & series_mask[:, :, None]
        if self.training and cfg.slice_dropout > 0:
            keep = torch.rand(B, Nseq, S, device=dev) >= cfg.slice_dropout
            candidate = slice_mask & keep
            slice_mask = torch.where(candidate.any(dim=2, keepdim=True), candidate, slice_mask)

        ctx = self.ctx(
            batch.context.reshape(B * Nseq, -1),
            batch.family_id.reshape(-1),
            batch.manufacturer_id.reshape(-1),
            batch.fat_sat_id.reshape(-1),
        )

        h = self.encode_series(
            batch.pixels.reshape(B * Nseq, S, *batch.pixels.shape[3:]),
            batch.z_mm.reshape(B * Nseq, S),
            slice_mask.reshape(B * Nseq, S),
            ctx,
        )

        bias = self._attention_bias((B, Nseq, S), dev, h.dtype)
        logit_bias = bias[None, :, None].expand(B * Nseq, L, S)
        pooled, attn, entropy = self.pool(
            h, slice_mask=slice_mask.reshape(B * Nseq, S), logit_bias=logit_bias
        )

        v = pooled.reshape(B, Nseq, L, -1)
        z, seq_attn = self.fusion(
            v, series_mask=series_mask, family_id=batch.family_id
        )

        aux: dict[str, torch.Tensor] = {
            "attn_entropy": entropy.reshape(B, Nseq, L).mean(),
            "attn_entropy_penalty": self.pool.entropy_penalty(entropy),
        }

        if self.ontology is not None:
            tangent = self.ontology.ball.logmap0(self.ontology.label_points())
            q = self.ontology_lift(tangent)
            z = z + q.to(z.dtype)[None]
            aux["ontology_hierarchy"] = self.ontology.hierarchy_loss()

        if self.router is not None:
            z, router_aux = self.router(z)
            aux.update({f"moe_{k}": v_ for k, v_ in router_aux.items()})

        if self.head is not None:
            coarse_logits, variance = self.head(
                z, update_precision=update_precision, return_variance=True
            )
        else:
            coarse_logits = self.linear_head(z)
            variance = None

        out: dict[str, torch.Tensor] = {
            "logits": coarse_logits,
            "coarse_logits": coarse_logits,
            "z": z,
            # Per-slice contextual tokens.  Exposed because the self-supervised
            # stage regresses masked tokens against them and the phrase->slice
            # OT loss transports onto them; recomputing either would double the
            # backbone cost of those stages.
            "slice_tokens": h.reshape(B, Nseq, S, -1),
            "study_embedding": z.mean(dim=1),
            "slice_attention": attn.reshape(B, Nseq, L, S),
            "sequence_attention": seq_attn,
            **aux,
        }
        if variance is not None:
            out["variance"] = variance

        do_fine = cfg.enable_fine_pass if run_fine is None else run_fine
        if not do_fine:
            return out

        # ---------------- fine pass ---------------------------------------- #
        need = self.gate(coarse_logits, variance)  # (B, L)
        out["gate"] = need
        if not bool(need.any()) and not self.training:
            return out

        rel = attn.reshape(B, Nseq, L, S).permute(0, 2, 1, 3).reshape(B, L, Nseq * S)
        valid_flat = slice_mask.reshape(B, Nseq * S)
        sel = self.selector(rel, valid=valid_flat)
        # A label that does not need the fine pass contributes no slices.
        want = sel["mask"] & need[:, :, None]
        union = want.any(dim=1) & valid_flat
        out["fine_fraction"] = union.sum(dim=1).float() / valid_flat.sum(dim=1).clamp_min(1).float()
        out["selector_budget"] = self.selector.budget_loss(
            {"n_selected": union.sum(dim=1).float()}, cfg.fine_budget_fraction, valid_flat
        )

        fine_pixels = batch.fine_pixels
        if fine_pixels is None:
            # Upsample the coarse crops as a stand-in.  In production the
            # dataloader supplies genuinely higher-resolution crops; this path
            # keeps the module runnable and shape-correct without them.
            p = batch.pixels
            fine_pixels = F.interpolate(
                p.reshape(-1, *p.shape[3:]), scale_factor=1.5, mode="bilinear",
                align_corners=False,
            ).reshape(*p.shape[:3], p.shape[3], int(p.shape[4] * 1.5), int(p.shape[5] * 1.5))

        subset = union.reshape(B, Nseq, S).reshape(B * Nseq, S)
        hf = self.encode_series(
            fine_pixels.reshape(B * Nseq, S, *fine_pixels.shape[3:]),
            batch.z_mm.reshape(B * Nseq, S),
            slice_mask.reshape(B * Nseq, S),
            ctx,
            backbone=self.fine_backbone or self.backbone,
            subset=subset,
        )
        pooled_f, attn_f, _ = self.pool(
            hf,
            slice_mask=(slice_mask.reshape(B * Nseq, S) & subset),
            logit_bias=logit_bias,
        )
        vf = pooled_f.reshape(B, Nseq, L, -1)
        zf, _ = self.fusion(vf, series_mask=series_mask, family_id=batch.family_id)

        cat = torch.cat([z, zf], dim=-1)
        g = torch.sigmoid(self.fine_gate(cat))  # (B, L, 1)
        g = g * need[:, :, None].to(g.dtype)
        z_final = z + g * (self.fine_merge(cat) - z)

        if self.head is not None:
            final_logits, var_f = self.head(
                z_final, update_precision=update_precision, return_variance=True
            )
            out["variance"] = var_f
        else:
            final_logits = self.linear_head(z_final)

        out["logits"] = torch.where(need, final_logits, coarse_logits)
        out["fine_logits"] = final_logits
        out["fine_gate_mean"] = g.mean()
        out["fine_slice_attention"] = attn_f.reshape(B, Nseq, L, S)
        return out

    # ------------------------------------------------------------------ #

    def parameter_groups(self, *, backbone_lr: float, head_lr: float,
                         weight_decay: float = 0.02) -> list[dict]:
        """Discriminative learning rates, with no decay on norms/biases/queries.

        Decaying the label queries pulls them towards each other, which is the
        exact opposite of what the architecture is for; decaying LayerNorm gains
        is a well-known way to lose half a point of accuracy for no reason.
        """
        bb, head, no_decay = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(".query") or "tangent_" in name or "ls1" in name or "ls2" in name:
                no_decay.append(p)
            elif name.startswith("backbone.") or name.startswith("fine_backbone."):
                bb.append(p)
            else:
                head.append(p)
        return [
            {"params": bb, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": head, "lr": head_lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": head_lr, "weight_decay": 0.0},
        ]
