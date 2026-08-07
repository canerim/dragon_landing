r"""The objective registry: curriculum term name → a callable that computes it.

:class:`~kairos.train.loop.Trainer` multiplies each registered term by its
scheduled weight and sums.  This module is what fills that registry, and it
carries one non-negotiable safety property:

    **A term that the curriculum schedules with a non-zero weight, but which
    the registry cannot compute, is a hard error at construction time.**

The alternative -- skipping it silently, which is what a plain ``dict.get``
does -- is the single most expensive bug class in a multi-term training system.
The run completes, the loss curve looks healthy (it is just a different, smaller
objective), and the only symptom is an OOF score that is inexplicably worse
than the ablation predicted, three GPU-days later.  :func:`validate_schedule`
turns that into a message before the first step.

Terms fall into three groups:

*Always available* -- computable from ``(model outputs, batch.targets)`` alone:
``asl``, ``pattern_bce``, ``auc_margin``, ``pauc``, ``rank_queue``, ``copula``,
``attn_entropy``, ``moe_balance``, ``moe_router_z``, ``ontology``,
``selector_budget``, ``consistency``, ``chi2_dro``.

*Batch-conditional* -- need a field the dataloader may or may not populate:
``group_dro`` (``group_id``), ``irm`` (``env_id``), ``weak_label``
(``weak_labels``), ``contrastive`` (``text_embedding``),
``ot_ground`` (``phrase_embedding``), ``kd`` (``teacher_logits``).

*Head-conditional* -- need an auxiliary module: ``mim`` and ``cross_plane``
require :class:`~kairos.train.ssl.SSLHeads`.

Every term object exposes ``.module`` (an ``nn.Module`` or ``None``) so the
trainer can collect its parameters for PESG and call ``.project()`` on the
min-max auxiliaries after each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import NUM_TARGETS
from ..losses.auc import AUCMarginLoss, PairwiseRankQueue, PartialAUCLoss
from ..losses.multimodal import (
    ReportDistillation,
    ReportShortcutRegulariser,
    SoftContrastive,
)
from ..losses.ot import PhraseSliceOT
from ..losses.robust import ChiSquareDRO, GroupDRO, IRMPenalty
from ..losses.supervised import AsymmetricLoss, GaussianCopulaNLL, SoftJaccardBCE
from .curriculum import LossSchedule
from .ssl import SSLHeads

__all__ = [
    "ObjectiveConfig",
    "ObjectiveRegistry",
    "build_objectives",
    "validate_schedule",
    "MissingObjectiveError",
]


class MissingObjectiveError(RuntimeError):
    """A scheduled term has no implementation, or its inputs are absent."""


@dataclass(slots=True)
class ObjectiveConfig:
    num_labels: int = NUM_TARGETS
    prevalence: Sequence[float] | None = None
    # supervised
    gamma_neg: float = 4.0
    gamma_pos: float = 0.0
    asl_clip: float = 0.05
    pattern_kappa: float = 0.25
    # ranking
    auc_margin: float = 1.0
    pauc_fpr_max: float = 0.30
    pauc_tpr_min: float = 0.50
    rank_queue_capacity: int = 512
    # robustness
    num_groups: int = 32
    dro_eta_q: float = 0.02
    dro_shrinkage: float = 1.0
    chi2_rho: float = 1.0
    irm_weight: float = 1.0
    # multimodal
    contrastive_temperature: float = 0.07
    ot_epsilon: float = 0.05
    ot_tau: float = 0.5
    kd_temperature: float = 2.0
    weak_min_confidence: float = 0.6
    # copula / structure
    copula_rank: int = 4
    # ssl
    ssl_dim: int | None = None
    ssl_mask_ratio: float = 0.5


# --------------------------------------------------------------------------- #
# Term wrappers                                                                #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Term:
    """One registry entry.

    ``fn`` returns a scalar tensor, or ``None`` when the term is genuinely
    inapplicable to *this batch* (for example ``kd`` on a batch with no teacher
    logits).  ``requires`` names the ``StudyBatch`` fields it needs; the
    validator uses it to explain exactly what is missing rather than saying
    "term unavailable".
    """

    name: str
    fn: Callable[[dict, Any, dict], torch.Tensor | None]
    module: nn.Module | None = None
    requires: tuple[str, ...] = ()
    note: str = ""
    #: Keys the *model output dict* must contain.  ``requires`` alone covers
    #: only the batch, which left a blind spot the validator was written to
    #: close: ``shortcut`` needs three logit variants that the shipped model
    #: never emits, carried a ``module`` (so the "inert note" heuristic skipped
    #: it), and was scheduled at weight 0.5 for the whole of S1 while returning
    #: ``None`` every single step.  Declaring output preconditions makes that a
    #: startup error instead of a silent one.
    requires_outputs: tuple[str, ...] = ()
    #: Optional exact additive decomposition of ``fn`` over the twelve labels:
    #: a ``(L,)`` tensor whose sum equals ``fn(...)``.  Gradient surgery needs
    #: per-task losses, and it needs them to sum to *exactly* the term it is
    #: replacing -- see :meth:`kairos.optim.pesg.GradientSurgery.correct_`.
    #: A term without one simply does not take part in surgery.
    per_label: Callable[[dict, Any, dict], torch.Tensor | None] | None = None

    def __call__(self, outputs, batch, state):
        return self.fn(outputs, batch, state)


def _targets(batch) -> torch.Tensor | None:
    return getattr(batch, "targets", None)


def _has(batch, field: str) -> bool:
    v = getattr(batch, field, None)
    return v is not None


# --------------------------------------------------------------------------- #


class ObjectiveRegistry(dict):
    """``dict`` of name → :class:`Term`, plus the modules the trainer must own."""

    def __init__(self, terms: Mapping[str, Term]) -> None:
        super().__init__(terms)
        self.modules = nn.ModuleDict(
            {k: t.module for k, t in terms.items() if t.module is not None}
        )

    def parameters(self):
        return self.modules.parameters()

    def project(self) -> None:
        """Project every min-max auxiliary back into its feasible set."""
        for m in self.modules.values():
            if hasattr(m, "project"):
                m.project()

    def to(self, device):
        self.modules.to(device)
        return self

    def describe(self) -> str:
        lines = []
        for name in sorted(self):
            t = self[name]
            req = f"  requires {', '.join(t.requires)}" if t.requires else ""
            mod = "  [module]" if t.module is not None else ""
            lines.append(f"  {name:<18}{mod}{req}")
        return "\n".join(lines)


def build_objectives(
    cfg: ObjectiveConfig | None = None,
    *,
    model: nn.Module | None = None,
    device: str | torch.device = "cpu",
) -> ObjectiveRegistry:
    """Construct every objective term.  See the module docstring for groups."""
    cfg = cfg or ObjectiveConfig()
    L = cfg.num_labels
    prev = (
        torch.tensor(list(cfg.prevalence), dtype=torch.float32)
        if cfg.prevalence is not None
        else None
    )

    asl = AsymmetricLoss(
        gamma_pos=cfg.gamma_pos, gamma_neg=cfg.gamma_neg, clip=cfg.asl_clip
    )
    aucm = AUCMarginLoss(L, margin=cfg.auc_margin, prevalence=prev)
    pauc = PartialAUCLoss(fpr_max=cfg.pauc_fpr_max, tpr_min=cfg.pauc_tpr_min)
    queue = PairwiseRankQueue(L, capacity=cfg.rank_queue_capacity)
    copula = GaussianCopulaNLL(L, rank=cfg.copula_rank)
    dro = GroupDRO(
        cfg.num_groups, eta_q=cfg.dro_eta_q, shrinkage=cfg.dro_shrinkage,
        standardise_labels=True, num_labels=L,
    )
    chi2 = ChiSquareDRO(rho=cfg.chi2_rho)
    irm = IRMPenalty(weight=cfg.irm_weight)
    contrastive = SoftContrastive(temperature=cfg.contrastive_temperature)
    ot = PhraseSliceOT(epsilon=cfg.ot_epsilon, tau=cfg.ot_tau)
    shortcut = ReportShortcutRegulariser()
    kd = ReportDistillation(temperature=cfg.kd_temperature)
    pattern = (
        SoftJaccardBCE(prev, kappa=cfg.pattern_kappa) if prev is not None else None
    )

    ssl_dim = cfg.ssl_dim or (getattr(getattr(model, "cfg", None), "dim", None) or 384)
    ssl = SSLHeads(ssl_dim, mask_ratio=cfg.ssl_mask_ratio)

    terms: dict[str, Term] = {}

    def reg(name, fn, *, module=None, requires=(), note="", per_label=None,
            requires_outputs=()):
        terms[name] = Term(
            name=name, fn=fn, module=module, requires=tuple(requires), note=note,
            requires_outputs=tuple(requires_outputs), per_label=per_label,
        )

    # -- always available ------------------------------------------------ #

    reg("asl", lambda o, b, s: (
        None if _targets(b) is None else asl(o["logits"], b.targets)
    ), module=asl, requires=("targets",), per_label=lambda o, b, s: (
        None if _targets(b) is None
        else asl(o["logits"], b.targets, reduction="label_contrib")
    ))

    if pattern is not None:
        reg("pattern_bce", lambda o, b, s: (
            None if _targets(b) is None else pattern(o["logits"], b.targets)
        ), module=pattern, requires=("targets",))
    else:
        reg("pattern_bce", lambda o, b, s: None,
            note="needs ObjectiveConfig.prevalence; pass the training-set rates")

    reg("auc_margin", lambda o, b, s: (
        None if _targets(b) is None else aucm(o["logits"], b.targets)
    ), module=aucm, requires=("targets",), per_label=lambda o, b, s: (
        None if _targets(b) is None
        else aucm(o["logits"], b.targets, reduction="label_contrib")
    ))

    reg("pauc", lambda o, b, s: (
        None if _targets(b) is None else pauc(o["logits"], b.targets)
    ), module=pauc, requires=("targets",))

    reg("rank_queue", lambda o, b, s: (
        None if _targets(b) is None else queue(o["logits"], b.targets)
    ), module=queue, requires=("targets",))

    reg("copula", lambda o, b, s: (
        None if _targets(b) is None else copula(o["logits"], b.targets)
    ), module=copula, requires=("targets",))

    reg("attn_entropy", lambda o, b, s: o.get("attn_entropy_penalty"))
    reg("moe_balance", lambda o, b, s: o.get("moe_balance"))
    reg("moe_router_z", lambda o, b, s: o.get("moe_router_z"))
    reg("ontology", lambda o, b, s: o.get("ontology_hierarchy"))
    reg("selector_budget", lambda o, b, s: o.get("selector_budget"))

    def _consistency(o, b, s):
        """Coarse and fine heads must not disagree wildly.

        The fine pass sees a *subset* of the slices at higher resolution.  If
        its logits diverge sharply from the coarse ones, either the selector is
        discarding the evidence or the two encoders have drifted apart -- both
        are failures, and both are cheap to penalise here.  Symmetric KL over
        the Bernoulli pair, so neither head is privileged.
        """
        fine = o.get("fine_logits")
        if fine is None:
            return None
        p = torch.sigmoid(o["coarse_logits"]).clamp(1e-6, 1 - 1e-6)
        q = torch.sigmoid(fine).clamp(1e-6, 1 - 1e-6)
        kl = lambda a, c: (  # noqa: E731
            a * (a.log() - c.log()) + (1 - a) * ((1 - a).log() - (1 - c).log())
        )
        return 0.5 * (kl(p, q) + kl(q, p)).mean()

    reg("consistency", _consistency)

    reg("chi2_dro", lambda o, b, s: (
        None if _targets(b) is None
        else chi2(asl(o["logits"], b.targets, reduction="per_example"))
    ), module=chi2, requires=("targets",))

    # -- batch-conditional ----------------------------------------------- #

    def _group_dro(o, b, s):
        if _targets(b) is None or not _has(b, "group_id"):
            return None
        per = asl(o["logits"], b.targets, reduction="none")
        return dro(per, b.group_id)

    reg("group_dro", _group_dro, module=dro, requires=("targets", "group_id"))

    reg("irm", lambda o, b, s: (
        None if _targets(b) is None or not _has(b, "env_id")
        else irm(o["logits"], b.targets, b.env_id)
    ), module=irm, requires=("targets", "env_id"))

    def _weak(o, b, s):
        """Report-derived soft labels, gated on extraction confidence.

        Weak labels are *masked out* wherever the gold label exists -- they are
        supervision for the studies and labels the gold does not cover, not a
        second opinion on the ones it does.  Letting them argue with gold is how
        a report parser's systematic errors get baked into the classifier.
        """
        if not _has(b, "weak_labels"):
            return None
        w = b.weak_labels
        conf = b.weak_confidence if _has(b, "weak_confidence") else torch.ones_like(w)
        keep = torch.isfinite(w) & (conf >= cfg.weak_min_confidence)
        if _targets(b) is not None:
            keep = keep & ~torch.isfinite(b.targets)
        if not bool(keep.any()):
            return None
        bce = F.binary_cross_entropy_with_logits(
            o["logits"], torch.nan_to_num(w), reduction="none"
        )
        wgt = keep.to(bce.dtype) * conf.clamp(0, 1)
        return (bce * wgt).sum() / wgt.sum().clamp_min(1.0)

    reg("weak_label", _weak, requires=("weak_labels",))

    def _contrastive(o, b, s):
        if not _has(b, "text_embedding"):
            return None
        return contrastive(
            o["study_embedding"],
            b.text_embedding,
            weak_labels=getattr(b, "weak_labels", None),
            graph_emb=getattr(b, "graph_embedding", None),
            valid=getattr(b, "has_report", None),
        )

    reg("contrastive", _contrastive, module=contrastive,
        requires=("text_embedding",))

    def _ot(o, b, s):
        if not _has(b, "phrase_embedding") or not _has(b, "phrase_mask"):
            return None
        tok = o["slice_tokens"]  # (B, Nseq, S, D)
        B, Nseq, S, D = tok.shape
        flat = tok.reshape(B, Nseq * S, D)
        smask = b.slice_mask.reshape(B, Nseq * S)
        attn = o["slice_attention"].permute(0, 2, 1, 3).reshape(B, -1, Nseq * S)
        return ot(
            b.phrase_embedding, flat,
            phrase_mask=b.phrase_mask,
            slice_mask=smask,
            phrase_conf=getattr(b, "phrase_confidence", None),
            phrase_label=getattr(b, "phrase_label", None),
            label_attention=attn,
        )["total"]

    reg("ot_ground", _ot, module=ot, requires=("phrase_embedding", "phrase_mask"))

    _SHORTCUT_OUTPUTS = (
        "logits_report", "logits_report_shuffled", "logits_image_only",
    )

    def _shortcut(o, b, s):
        """Anti-shortcut margin for a *report-conditioned* classifier.

        :class:`~kairos.models.system.KairosModel` does not build one, on
        purpose: the test set has no reports, so a report-conditioned branch
        can only pay off through the representation (which ``contrastive`` and
        ``ot_ground`` already give us) or as a KD teacher -- and a teacher that
        reads the finding out of the report produces logits the student cannot
        reproduce from pixels, so distilling them is just label smoothing.

        The term stays registered because the regulariser is correct and tested
        and a report-conditioned variant may want it; it declares its output
        preconditions so that scheduling it against a model that cannot feed it
        is a startup error rather than a term that quietly returns ``None``
        forever.  It is *not* in the shipped curriculum.
        """
        if not all(k in o for k in _SHORTCUT_OUTPUTS):
            return None
        return shortcut(*(o[k] for k in _SHORTCUT_OUTPUTS))["total"]

    reg("shortcut", _shortcut, module=shortcut,
        requires_outputs=_SHORTCUT_OUTPUTS,
        note="needs a report-conditioned branch; the shipped model has none")

    def _kd(o, b, s):
        if not _has(b, "teacher_logits"):
            return None
        # Attention KD is the component that transfers *where the teacher
        # looks*, and it is the most valuable one for the rare osseous labels.
        # Hard-coding student_attn=None disabled it even when the batch carried
        # teacher attention, silently reducing KD to logit matching.
        st_attn = o.get("slice_attention")
        if st_attn is not None and st_attn.dim() == 4:
            B, Nseq, L, S = st_attn.shape
            st_attn = st_attn.permute(0, 2, 1, 3).reshape(B, L, Nseq * S)
        return kd(
            o["logits"], b.teacher_logits,
            student_attn=st_attn,
            teacher_attn=getattr(b, "teacher_attention", None),
        )["total"]

    reg("kd", _kd, module=kd, requires=("teacher_logits",))

    # -- head-conditional (SSL) ------------------------------------------ #

    def _mim(o, b, s):
        tok = o.get("slice_tokens")
        if tok is None:
            return None
        return ssl.mim(tok, b.slice_mask & b.series_mask[:, :, None])["loss"]

    reg("mim", _mim, module=ssl.mim)

    def _cross_plane(o, b, s):
        tok = o.get("slice_tokens")
        if tok is None:
            return None
        return ssl.cross_plane(tok, b.slice_mask, b.series_mask, b.family_id)["loss"]

    reg("cross_plane", _cross_plane, module=ssl.cross_plane)

    registry = ObjectiveRegistry(terms)
    registry.to(device)
    return registry


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #


def validate_schedule(
    registry: Mapping[str, Term],
    schedule: LossSchedule,
    *,
    sample_batch: Any | None = None,
    sample_outputs: Mapping[str, Any] | None = None,
    strict: bool = True,
) -> list[str]:
    """Check that every term the schedule activates can actually be computed.

    Three levels:

    1. **Registry coverage.**  Every term with a non-zero weight at *any* step
       must have an entry.  A missing entry is always fatal: the schedule is
       asking for an objective that does not exist.

    2. **Batch coverage.**  If ``sample_batch`` is given, every scheduled
       term's ``requires`` fields must be present on it.  This catches the
       common and expensive case of scheduling ``kd`` or ``ot_ground`` against
       a dataloader that does not emit teacher logits or phrase embeddings --
       the run would otherwise train a strictly smaller objective and say
       nothing.

    3. **Output coverage.**  If ``sample_outputs`` is given (the trainer runs
       one no-grad forward pass to get it), every scheduled term's
       ``requires_outputs`` keys must be present.  This is the level that was
       missing: a term can be fully registered, own a module, need nothing from
       the batch, and still be dead because the *model* does not emit what it
       reads.

    Returns the list of problems; raises :class:`MissingObjectiveError` when
    ``strict`` (the default) and the list is non-empty.
    """
    active: dict[str, float] = {}
    step = 0
    total = max(schedule.plan.total_steps, 1)
    # Sample the schedule densely enough to catch a term that is only active
    # inside one stage's ramp.
    while step < total:
        for name, w in schedule(step).items():
            if w > 0:
                active[name] = max(active.get(name, 0.0), w)
        step += max(1, total // 400)

    problems: list[str] = []
    for name, w in sorted(active.items()):
        term = registry.get(name)
        if term is None:
            problems.append(
                f"'{name}' is scheduled (max weight {w:g}) but has no registered "
                f"objective. Either implement it or set its weight to 0 in the "
                f"curriculum."
            )
            continue
        # ``term.module is None`` used to be part of this test, which made it
        # unfirable for exactly the terms that need it most: owning parameters
        # says the trainer must collect them, not that the term is computable.
        # ``shortcut`` owns a module, needs nothing from the batch, and can
        # never be computed by the shipped model -- and slipped through.
        if getattr(term, "note", "") and not term.requires and not term.requires_outputs:
            problems.append(f"'{name}' is scheduled but inert: {term.note}")
        if sample_batch is not None:
            missing = [f for f in term.requires if getattr(sample_batch, f, None) is None]
            if missing:
                problems.append(
                    f"'{name}' is scheduled (max weight {w:g}) but the batch is "
                    f"missing: {', '.join(missing)}. The term would be silently "
                    f"skipped every step."
                )
        if sample_outputs is not None and term.requires_outputs:
            missing_out = [
                k for k in term.requires_outputs if sample_outputs.get(k) is None
            ]
            if missing_out:
                problems.append(
                    f"'{name}' is scheduled (max weight {w:g}) but the model does "
                    f"not emit: {', '.join(missing_out)}. The term would be "
                    f"silently skipped every step."
                    + (f" Note: {term.note}" if term.note else "")
                )

    if problems and strict:
        raise MissingObjectiveError(
            "the curriculum schedules objectives that cannot be computed:\n  - "
            + "\n  - ".join(problems)
        )
    return problems


def disable_unavailable(
    schedule: LossSchedule, registry: Mapping[str, Term], sample_batch: Any
) -> set[str]:
    """Report which scheduled terms this batch cannot serve, without raising.

    Intended for the *explicit* case where you know reports are absent and want
    an image-only run: call this, log the result, and pass the names to the
    trainer's ``disabled`` set so the omission is recorded in the run manifest
    rather than being invisible.
    """
    return {
        p.split("'")[1]
        for p in validate_schedule(registry, schedule, sample_batch=sample_batch,
                                   strict=False)
    }
