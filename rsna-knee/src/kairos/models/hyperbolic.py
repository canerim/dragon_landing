r"""Poincaré-ball embedding of the knee pathology ontology.

The twelve targets are not a flat set.  They sit under a shallow tree
(mechanism → pathology) and additionally under a compartment partition
(medial / lateral / patellofemoral / global).  Euclidean space is a poor host
for trees: the number of nodes at tree-distance :math:`d` grows exponentially
in :math:`d`, while the volume of a Euclidean ball grows polynomially, so any
Euclidean embedding of a tree has distortion that grows with depth.  Hyperbolic
space has exponential volume growth and therefore embeds trees with arbitrarily
low distortion in as few as two dimensions (Sarkar, 2011; Nickel & Kiela, 2017).

What we get for that:

* **A structured prior on the label queries.**  Each label query is
  parameterised as a point on the Poincaré ball; the queries of two labels in
  the same mechanism group are close, and generic concepts (the group node) sit
  nearer the origin than specific ones.  This is a genuine inductive bias:
  gradient signal from the common ``Effusion`` label propagates to the rare
  ``Synovitis`` label along the tree, which is the correct sharing structure,
  and does *not* propagate to ``Fracture``, which it should not.

* **A hierarchy-consistent regulariser.**  Entailment cones (Ganea et al.,
  2018) let us state "``Medial OA`` implies ``degenerative``" as a geometric
  constraint and penalise its violation, rather than hoping the classifier
  discovers it.

The exponential/log maps and Möbius addition below are the standard
:math:`\kappa = -1` Poincaré ball operations, written with the numerical guards
that matter in fp16: every ``artanh`` argument is clamped strictly inside
:math:`(-1, 1)` and every point is projected back inside the ball after an
update, because a single point escaping the ball produces NaNs that surface
three modules downstream.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = [
    "PoincareBall",
    "OntologyEmbedding",
    "entailment_cone_loss",
]

_MAX_NORM = 1.0 - 1e-5


class PoincareBall(nn.Module):
    """Operations on the Poincaré ball of curvature :math:`-c`."""

    def __init__(self, c: float = 1.0, *, learnable_curvature: bool = False) -> None:
        super().__init__()
        log_c = torch.tensor(float(math.log(c)))
        self.log_c = nn.Parameter(log_c) if learnable_curvature else nn.Parameter(
            log_c, requires_grad=False
        )

    @property
    def c(self) -> torch.Tensor:
        return self.log_c.exp()

    def project(self, x: torch.Tensor) -> torch.Tensor:
        c = self.c.to(x.dtype)
        norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        maxnorm = _MAX_NORM / c.sqrt()
        return torch.where(norm > maxnorm, x / norm * maxnorm, x)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r""":math:`x \oplus_c y`."""
        c = self.c.to(x.dtype)
        x2 = (x * x).sum(-1, keepdim=True)
        y2 = (y * y).sum(-1, keepdim=True)
        xy = (x * y).sum(-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = 1 + 2 * c * xy + c * c * x2 * y2
        return self.project(num / den.clamp_min(1e-12))

    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        r""":math:`\exp_0(v) = \tanh(\sqrt c\lVert v\rVert)\,v/(\sqrt c \lVert v\rVert)`."""
        c = self.c.to(v.dtype)
        sqrt_c = c.sqrt()
        n = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return self.project(torch.tanh(sqrt_c * n) * v / (sqrt_c * n))

    def logmap0(self, x: torch.Tensor) -> torch.Tensor:
        c = self.c.to(x.dtype)
        sqrt_c = c.sqrt()
        n = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.atanh((sqrt_c * n).clamp(-_MAX_NORM, _MAX_NORM)) * x / (sqrt_c * n)

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r""":math:`d_c(x,y) = \tfrac{2}{\sqrt c}\operatorname{artanh}
        (\sqrt c \lVert -x \oplus_c y\rVert)`."""
        c = self.c.to(x.dtype)
        sqrt_c = c.sqrt()
        diff = self.mobius_add(-x, y)
        n = diff.norm(dim=-1).clamp_max(_MAX_NORM / sqrt_c.item() if sqrt_c.numel() == 1 else 1.0)
        return 2.0 / sqrt_c * torch.atanh((sqrt_c * n).clamp(-_MAX_NORM, _MAX_NORM))


class OntologyEmbedding(nn.Module):
    """Label + group nodes embedded jointly on the ball.

    Parameters are stored in the tangent space at the origin and mapped through
    ``expmap0`` on every access.  That is the standard trick which lets an
    ordinary Euclidean optimiser (AdamW) train hyperbolic parameters without a
    Riemannian optimiser, at the cost of a slightly different metric on the
    updates -- acceptable here because the embedding is a *prior*, not the model.
    """

    def __init__(
        self,
        num_labels: int,
        num_groups: int,
        dim: int = 16,
        *,
        c: float = 1.0,
        parent_of_label: torch.Tensor | None = None,  # (L,) long -> group index
        init_scale: float = 1e-2,
    ) -> None:
        super().__init__()
        self.ball = PoincareBall(c)
        self.num_labels = num_labels
        self.num_groups = num_groups
        self.tangent_label = nn.Parameter(torch.randn(num_labels, dim) * init_scale)
        # Groups initialised nearer the origin: generality ↔ small radius.
        self.tangent_group = nn.Parameter(torch.randn(num_groups, dim) * init_scale * 0.5)
        self.register_buffer("root", torch.zeros(dim))
        if parent_of_label is not None:
            self.register_buffer("parent", parent_of_label.long())
        else:
            self.register_buffer("parent", torch.zeros(num_labels, dtype=torch.long))

    def label_points(self) -> torch.Tensor:
        return self.ball.expmap0(self.tangent_label)

    def group_points(self) -> torch.Tensor:
        return self.ball.expmap0(self.tangent_group)

    def label_similarity(self, *, tau: float = 1.0) -> torch.Tensor:
        r"""Similarity kernel :math:`\exp(-d_c(l, l')/\tau)` used to
        (a) build the soft contrastive target and (b) smooth the per-label
        ensemble weights so that a rare label borrows the weighting of its
        mechanism siblings instead of overfitting its own 40 positives."""
        p = self.label_points()
        d = self.ball.distance(p[:, None, :], p[None, :, :])
        return torch.exp(-d / max(tau, 1e-3))

    def hierarchy_loss(self, *, margin: float = 0.1) -> torch.Tensor:
        r"""Push each label outside its parent group, and each group outside
        the root, by at least ``margin`` in hyperbolic radius.

        This encodes "specific concepts live further from the origin", which is
        what makes the radius interpretable as generality and what makes the
        entailment cone below well-defined.
        """
        lp = self.label_points()
        gp = self.group_points()
        r_l = lp.norm(dim=-1)
        r_g = gp.norm(dim=-1)
        parent_r = r_g[self.parent]
        depth = torch.relu(parent_r + margin - r_l).mean()
        root_gap = torch.relu(margin - r_g).mean()
        attach = self.ball.distance(lp, gp[self.parent]).mean()
        return depth + root_gap + 0.1 * attach

    def query_prior(self, dim_out: int) -> nn.Module:
        """A frozen linear lift from the ball's tangent space to query space."""
        lift = nn.Linear(self.tangent_label.shape[1], dim_out, bias=False)
        nn.init.orthogonal_(lift.weight)
        for p in lift.parameters():
            p.requires_grad_(False)
        return lift


def entailment_cone_loss(
    ball: PoincareBall,
    child: torch.Tensor,
    parent: torch.Tensor,
    *,
    k: float = 0.1,
) -> torch.Tensor:
    r"""Ganea et al. entailment cones: ``child`` must lie inside ``parent``'s cone.

    The half-aperture of the cone at :math:`x` is
    :math:`\psi(x) = \arcsin\!\big(k(1-\lVert x\rVert^2)/\lVert x\rVert\big)`,
    and the angle at :math:`x` towards :math:`y` is

    .. math::
        \Xi(x,y) = \arccos\!\left(
          \frac{\langle x, y-x\rangle\,(1+\lVert x\rVert^2)
                - \lVert x\rVert^2(1+\lVert y\rVert^2)}
               {\lVert x\rVert\,\lVert y-x\rVert
                \sqrt{1 + \lVert x\rVert^2\lVert y\rVert^2 - 2\langle x,y\rangle}}
        \right).

    The penalty is :math:`\max(0, \Xi - \psi)`, zero exactly when the child is
    inside.  This is the geometric statement of "Medial OA ⇒ degenerative".
    """
    del ball  # curvature-1 formulas below
    eps = 1e-6
    xn = parent.norm(dim=-1).clamp(eps, _MAX_NORM)
    yn = child.norm(dim=-1).clamp(eps, _MAX_NORM)
    xy = (parent * child).sum(-1)
    diff = child - parent
    dn = diff.norm(dim=-1).clamp_min(eps)

    num = xy * (1 + xn**2) - xn**2 * (1 + yn**2)
    den = xn * dn * torch.sqrt((1 + xn**2 * yn**2 - 2 * xy).clamp_min(eps))
    xi = torch.arccos((num / den.clamp_min(eps)).clamp(-1 + eps, 1 - eps))

    psi = torch.arcsin((k * (1 - xn**2) / xn).clamp(-1 + eps, 1 - eps))
    return torch.relu(xi - psi).mean()
