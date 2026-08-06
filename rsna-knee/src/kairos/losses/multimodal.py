r"""Image–report alignment and privileged-information distillation.

The competition's distinguishing asset is that every study ships with its
original report.  Two things must be true of how we use it:

1. **The report is a training signal, not an inference shortcut.**  The system
   must reach full accuracy from images alone, because we cannot assume reports
   exist at test time.  Reports enter through pretraining, weak labels and
   distillation -- all of which leave no trace in the inference graph.

2. **Contrastive learning on reports is wrong out of the box.**  In a batch of
   32 knee studies, several will describe genuinely similar pathology.  Vanilla
   InfoNCE labels all of them mutual negatives, which teaches the encoder to
   *separate clinically identical cases*.  That is precisely backwards.

:class:`SoftContrastive` fixes (2) with a soft target built from clinical
agreement rather than identity:

.. math::
    s^\ast_{ij} = \alpha\,\mathbb 1[i = j]
                + \beta\,J(\mathbf y_i, \mathbf y_j)
                + \gamma\,\cos(\mathbf g_i, \mathbf g_j),

where :math:`J` is the Jaccard index of the weak label sets and
:math:`\mathbf g` is the pooled clinical-concept-graph embedding.  The loss is
then a cross-entropy against the row-normalised :math:`s^\ast` -- i.e. a
knowledge-distillation objective in which the *teacher* is clinical similarity.
This is the same fix MedCLIP applies with semantic targets, generalised to a
graph term.

:class:`ReportDistillation` implements the three-teacher scheme (text-only,
multimodal, image-ensemble) with decoupled KD, feature alignment through a
learned projector, and attention alignment via a symmetric JS divergence.

:class:`ReportShortcutRegulariser` is the guard rail: it explicitly penalises
the multimodal teacher for being *more* confident when the report is shuffled
to a different study than when it is correct.  Without it, the "multimodal
teacher" converges to a report-only classifier that is useless as a teacher for
an image-only student, and the failure is invisible in the teacher's own
validation AUC.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SoftContrastive",
    "ReportDistillation",
    "ReportShortcutRegulariser",
    "jaccard_similarity",
]

_EPS = 1e-8


def jaccard_similarity(y: torch.Tensor, *, mask: torch.Tensor | None = None) -> torch.Tensor:
    r"""Pairwise Jaccard index of binary label vectors, NaN-safe.

    ``y`` is ``(B, L)`` and may contain NaN for unobserved labels; those entries
    are excluded from both intersection and union of every pair that involves
    them, which is the correct treatment (an unknown label is evidence of
    nothing).  Pairs with an empty union get similarity 0 rather than 1 -- two
    completely normal knees are similar in outcome but carry no *positive*
    clinical content to align on, and treating them as identical collapses the
    normal cases into a single point.
    """
    v = torch.isfinite(y).to(y.dtype)
    yy = torch.nan_to_num(y, nan=0.0) * v
    if mask is not None:
        m = mask.to(y.dtype)
        yy, v = yy * m, v * m
    pair_valid = v[:, None, :] * v[None, :, :]
    inter = (yy[:, None, :] * yy[None, :, :] * pair_valid).sum(-1)
    union = (
        ((yy[:, None, :] + yy[None, :, :]) > 0).to(y.dtype) * pair_valid
    ).sum(-1)
    return inter / union.clamp_min(1.0)


class SoftContrastive(nn.Module):
    """Bidirectional image↔text InfoNCE against a clinical soft target."""

    def __init__(
        self,
        *,
        temperature: float = 0.07,
        learn_temperature: bool = True,
        alpha_identity: float = 0.6,
        beta_jaccard: float = 0.3,
        gamma_graph: float = 0.1,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        t = torch.tensor(float(temperature)).log()
        self.log_t = nn.Parameter(t) if learn_temperature else nn.Parameter(t, requires_grad=False)
        self.alpha = alpha_identity
        self.beta = beta_jaccard
        self.gamma = gamma_graph
        self.label_smoothing = label_smoothing

    def soft_target(
        self,
        weak_labels: torch.Tensor | None,
        graph_emb: torch.Tensor | None,
        batch: int,
        device,
        dtype,
    ) -> torch.Tensor:
        eye = torch.eye(batch, device=device, dtype=dtype)
        s = self.alpha * eye
        if weak_labels is not None and self.beta > 0:
            s = s + self.beta * jaccard_similarity(weak_labels.to(dtype))
        if graph_emb is not None and self.gamma > 0:
            g = F.normalize(graph_emb.to(dtype), dim=-1)
            s = s + self.gamma * (g @ g.T).clamp_min(0.0)
        if self.label_smoothing > 0:
            s = (1 - self.label_smoothing) * s + self.label_smoothing / batch
        return s / s.sum(dim=1, keepdim=True).clamp_min(_EPS)

    def forward(
        self,
        image_emb: torch.Tensor,  # (B, D)
        text_emb: torch.Tensor,  # (B, D)
        *,
        weak_labels: torch.Tensor | None = None,
        graph_emb: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,  # (B,) studies that actually have a report
    ) -> torch.Tensor:
        v = F.normalize(image_emb, dim=-1)
        t = F.normalize(text_emb, dim=-1)
        logits = (v @ t.T) / self.log_t.exp().clamp_min(1e-3)

        B = v.shape[0]
        target = self.soft_target(weak_labels, graph_emb, B, v.device, v.dtype)

        if valid is not None:
            keep = valid.bool()
            if keep.sum() < 2:
                return logits.sum() * 0.0
            logits = logits[keep][:, keep]
            target = target[keep][:, keep]
            target = target / target.sum(dim=1, keepdim=True).clamp_min(_EPS)

        li = -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        tgt_t = target.T / target.T.sum(dim=1, keepdim=True).clamp_min(_EPS)
        lt = -(tgt_t * F.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
        return 0.5 * (li + lt)


class ReportDistillation(nn.Module):
    r"""Decoupled KD from a multimodal / ensemble teacher to an image student.

    Standard logit KD mixes two very different signals: how confident the
    teacher is about the *target* class, and how it distributes the remaining
    mass.  Decoupled KD (Zhao et al., CVPR 2022) separates them so the second
    (which carries the transferable "dark knowledge") can be weighted up
    independently.  For binary-per-label heads the decomposition is

    .. math::
        \mathrm{KD} = \underbrace{\mathrm{KL}(b^T \| b^S)}_{\text{target}}
                    + (1 - p^T_{\text{tgt}})\,
                      \underbrace{\mathrm{KL}(\hat p^T \| \hat p^S)}_{\text{non-target}},

    which for a 2-class problem reduces to a reweighting of the two Bernoulli
    branches; we implement it directly on the Bernoulli pair so it stays exact
    rather than approximating a softmax.

    Additional terms:

    * ``feature`` -- L2 between a linear projection of the student's per-label
      query features and the teacher's, which transfers *what the teacher
      looks at* rather than just its answer.
    * ``attention`` -- symmetric JS between teacher and student slice-attention
      distributions, which is what actually transfers localisation and is the
      single most valuable KD component for the rare osseous labels.
    """

    def __init__(
        self,
        *,
        temperature: float = 2.0,
        alpha_target: float = 1.0,
        beta_nontarget: float = 2.0,
        feature_weight: float = 0.1,
        attention_weight: float = 0.05,
        student_dim: int | None = None,
        teacher_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.T = temperature
        self.alpha = alpha_target
        self.beta = beta_nontarget
        self.feature_weight = feature_weight
        self.attention_weight = attention_weight
        self.proj = (
            nn.Linear(student_dim, teacher_dim)
            if student_dim and teacher_dim
            else None
        )

    @staticmethod
    def _bernoulli_kl(pt: torch.Tensor, ps: torch.Tensor) -> torch.Tensor:
        pt = pt.clamp(1e-6, 1 - 1e-6)
        ps = ps.clamp(1e-6, 1 - 1e-6)
        return pt * (pt.log() - ps.log()) + (1 - pt) * ((1 - pt).log() - (1 - ps).log())

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        *,
        student_feat: torch.Tensor | None = None,
        teacher_feat: torch.Tensor | None = None,
        student_attn: torch.Tensor | None = None,
        teacher_attn: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,  # (B,) per-study teacher reliability
    ) -> dict[str, torch.Tensor]:
        T = self.T
        pt = torch.sigmoid(teacher_logits / T)
        ps = torch.sigmoid(student_logits / T)

        # Target branch: agreement on the teacher's own decision.
        hard_t = (pt > 0.5).to(pt.dtype)
        p_tgt_t = hard_t * pt + (1 - hard_t) * (1 - pt)
        p_tgt_s = hard_t * ps + (1 - hard_t) * (1 - ps)
        tckd = self._bernoulli_kl(p_tgt_t, p_tgt_s)
        # Non-target branch: the residual, renormalised.
        nckd = self._bernoulli_kl(pt, ps) - tckd
        kd = self.alpha * tckd + self.beta * (1 - p_tgt_t.detach()) * nckd.clamp_min(0.0)
        kd = kd * (T**2)

        if weight is not None:
            kd = kd * weight[:, None]
        out = {"kd": kd.mean()}
        total = out["kd"]

        if student_feat is not None and teacher_feat is not None and self.feature_weight > 0:
            sf = self.proj(student_feat) if self.proj is not None else student_feat
            sf = F.normalize(sf, dim=-1)
            tf = F.normalize(teacher_feat.detach(), dim=-1)
            fl = ((sf - tf) ** 2).sum(-1).mean()
            out["feature"] = fl
            total = total + self.feature_weight * fl

        if student_attn is not None and teacher_attn is not None and self.attention_weight > 0:
            a = student_attn.clamp_min(1e-8)
            b = teacher_attn.detach().clamp_min(1e-8)
            a = a / a.sum(-1, keepdim=True)
            b = b / b.sum(-1, keepdim=True)
            m = 0.5 * (a + b)
            js = 0.5 * (
                (a * (a.log() - m.log())).sum(-1) + (b * (b.log() - m.log())).sum(-1)
            )
            out["attention"] = js.mean()
            total = total + self.attention_weight * js.mean()

        out["total"] = total
        return out


class ReportShortcutRegulariser(nn.Module):
    r"""Penalise a multimodal model for reading the report instead of the image.

    Given the model's logits under (a) the correct report and (b) a report
    randomly permuted across the batch, we require

    .. math::
        \mathcal L_{\text{shortcut}}
          = \mathbb E\big[\mathrm{ReLU}\big(
              \mathcal C(z^{\text{shuf}}) - \mathcal C(z^{\text{true}}) + \delta
            \big)\big]
          + \eta\;\mathbb E\big[\mathrm{KL}\big(
              \sigma(z^{\text{shuf}}) \,\|\, \sigma(z^{\text{img}})\big)\big],

    where :math:`\mathcal C` is a confidence functional (negative predictive
    entropy).  The first term says a shuffled report must not make the model
    *more* certain; the second says that under a shuffled report the prediction
    should fall back to the image-only prediction.

    This is a diagnostic *and* a loss.  Reported as a scalar each epoch, it is
    the single number that tells you whether your multimodal teacher is real.
    """

    def __init__(self, *, delta: float = 0.0, eta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta
        self.eta = eta

    @staticmethod
    def _neg_entropy(logits: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        h = -(p * p.log() + (1 - p) * (1 - p).log())
        return -h.mean(dim=1)

    def forward(
        self,
        logits_true: torch.Tensor,
        logits_shuffled: torch.Tensor,
        logits_image_only: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        c_true = self._neg_entropy(logits_true)
        c_shuf = self._neg_entropy(logits_shuffled)
        margin = F.relu(c_shuf - c_true + self.delta).mean()

        ps = torch.sigmoid(logits_shuffled).clamp(1e-6, 1 - 1e-6)
        pi = torch.sigmoid(logits_image_only.detach()).clamp(1e-6, 1 - 1e-6)
        kl = (ps * (ps.log() - pi.log()) + (1 - ps) * ((1 - ps).log() - (1 - pi).log())).mean()

        return {
            "margin": margin,
            "fallback_kl": kl,
            "total": margin + self.eta * kl,
            # Diagnostic: >0 means the model is genuinely using the image.
            "report_reliance": (c_true - c_shuf).mean().detach(),
        }
