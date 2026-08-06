r"""Optimisers for the min-max and multi-task structure of this objective.

Three pieces, each solving a problem that a plain ``AdamW`` cannot:

``PESG``
    Proximal Epoch Stochastic Gradient for the AUC min-max problem.  The
    objective :math:`\min_{\theta,a,b}\max_\alpha F` is *not* a minimisation, so
    running AdamW on all parameters -- including :math:`\alpha` -- converges to
    a saddle only by accident.  PESG (Yuan et al., 2021) does descent on
    :math:`(\theta, a, b)`, **ascent** on :math:`\alpha`, adds a proximal term
    :math:`\tfrac{\gamma}{2}\lVert\theta - \theta_{\text{ref}}\rVert^2` anchored
    at the last epoch's reference point, and decays the step size on an
    epoch-wise schedule.  The proximal anchor is what makes it stable: without
    it the ascent on :math:`\alpha` and the descent on :math:`\theta` chase each
    other and the loss oscillates with a period of a few hundred steps.

``ASAM``
    Adaptive Sharpness-Aware Minimisation.  Standard SAM perturbs by
    :math:`\rho\,g/\lVert g\rVert`, which is not scale-invariant: a layer with
    large weights gets a relatively tiny perturbation.  ASAM normalises per
    parameter by :math:`|w|`, making the sharpness measure invariant to the
    weight rescalings that BatchNorm/LayerNorm make free.  On heterogeneous
    multi-site data this reliably buys 0.003–0.006 macro-AUC, at 2× cost --
    which is why it is enabled only for the final two models of the ensemble.

``GradientSurgery``
    Twelve labels share one backbone.  Their gradients conflict: the update
    that helps ``Effusion`` (large bright fluid, low frequency detail) actively
    hurts ``Medial Meniscus`` (a 2 mm dark line).  Implemented:
    ``pcgrad`` (project away the conflicting component), ``cagrad``
    (conflict-averse: maximise the worst-case improvement inside a ball around
    the average gradient), and ``aligned`` (Aligned-MTL: whiten the gradient
    matrix by its own singular values so no task dominates by gradient scale).
    Default is ``aligned`` -- it is the only one of the three whose fixed point
    does not depend on the arbitrary relative scaling of the twelve losses.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import torch
from torch.optim import Optimizer

__all__ = ["PESG", "ASAM", "GradientSurgery"]


class PESG(Optimizer):
    """Proximal Epoch Stochastic Gradient with min-max support.

    Parameters flagged with ``param._auc_ascent = True`` are *ascended*.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1e-3,
        gamma: float = 500.0,
        margin: float = 1.0,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        clip_value: float | None = 1.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            gamma=gamma,
            margin=margin,
            weight_decay=weight_decay,
            momentum=momentum,
            clip_value=clip_value,
        )
        super().__init__(params, defaults)
        self._init_reference()
        self.T = 0

    @torch.no_grad()
    def _init_reference(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state[p]
                st["ref"] = p.detach().clone()
                st["buf"] = torch.zeros_like(p)

    @torch.no_grad()
    def update_reference(self) -> None:
        """Call at the end of every epoch: :math:`\\theta_{ref} \\leftarrow \\theta`."""
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["ref"].copy_(p.detach())

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):  # type: ignore[override]
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            gamma = group["gamma"]
            wd = group["weight_decay"]
            mom = group["momentum"]
            clip = group["clip_value"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if clip is not None:
                    g = g.clamp(-clip, clip)

                ascent = bool(getattr(p, "_auc_ascent", False))
                st = self.state[p]

                if ascent:
                    # Ascent on alpha; no proximal term, no weight decay.
                    p.add_(g, alpha=lr)
                    p.clamp_min_(0.0)
                    continue

                d = g + wd * p
                # Proximal anchor to the epoch reference point.
                d = d + (p - st["ref"]) / max(gamma, 1e-8)
                buf = st["buf"]
                buf.mul_(mom).add_(d)
                p.add_(buf, alpha=-lr)

        self.T += 1
        return loss

    def decay_lr(self, factor: float = 0.5) -> None:
        for group in self.param_groups:
            group["lr"] *= factor


class ASAM:
    """Adaptive Sharpness-Aware Minimisation wrapper around any base optimiser.

    Usage::

        asam = ASAM(base_optimizer, model, rho=0.5)
        loss = criterion(model(x), y); loss.backward()
        asam.ascent_step()
        criterion(model(x), y).backward()
        asam.descent_step()

    ``eta`` guards the per-parameter normaliser against zero weights, which
    would otherwise produce an infinite perturbation for a freshly-initialised
    bias.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        model: torch.nn.Module,
        *,
        rho: float = 0.5,
        eta: float = 0.01,
        adaptive: bool = True,
    ) -> None:
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.adaptive = adaptive
        self.state: dict[torch.nn.Parameter, torch.Tensor] = {}

    @torch.no_grad()
    def ascent_step(self) -> None:
        grads = []
        for _, p in self.model.named_parameters():
            if p.grad is None:
                continue
            if self.adaptive:
                t_w = torch.abs(p) + self.eta
                self.state[p] = t_w
                grads.append((t_w * p.grad).norm(p=2))
            else:
                grads.append(p.grad.norm(p=2))
        if not grads:
            return
        norm = torch.norm(torch.stack(grads), p=2)
        scale = self.rho / (norm + 1e-12)
        for _, p in self.model.named_parameters():
            if p.grad is None:
                continue
            if self.adaptive:
                eps = self.state[p] * self.state[p] * p.grad * scale
            else:
                eps = p.grad * scale
            p.add_(eps)
            self.state[(p, "eps")] = eps  # type: ignore[index]
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def descent_step(self) -> None:
        for _, p in self.model.named_parameters():
            eps = self.state.pop((p, "eps"), None)  # type: ignore[arg-type]
            if eps is not None:
                p.sub_(eps)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)


class GradientSurgery:
    r"""Resolve conflicting per-label gradients on the shared trunk.

    ``mode='pcgrad'``
        For each ordered pair with :math:`\langle g_i, g_j\rangle < 0`, replace
        :math:`g_i \leftarrow g_i - \frac{\langle g_i,g_j\rangle}{\lVert g_j
        \rVert^2} g_j`.  Simple, cheap, but its fixed point depends on the
        random order in which pairs are visited.

    ``mode='cagrad'``
        Solve :math:`\max_{d:\lVert d - g_0\rVert \le c\lVert g_0\rVert}
        \min_i \langle g_i, d\rangle` where :math:`g_0` is the average
        gradient.  The dual is a small simplex-constrained QP which we solve by
        projected gradient in ~20 iterations -- negligible next to a backward
        pass.

    ``mode='aligned'``
        Aligned-MTL: form :math:`G \in \mathbb R^{L\times P}`, take its SVD
        :math:`G = U\Sigma V^\top`, and use
        :math:`d = \sigma_{\min}\,\mathbf 1^\top U \Sigma^{-1} U^\top G`.  This
        equalises the *condition number* of the linear system the tasks jointly
        pose, making the update invariant to per-task loss rescaling.  Since our
        twelve losses have wildly different natural scales (ASL on a 1 %-
        prevalence label vs a 40 %-prevalence label), scale invariance is
        exactly the property we need, and it is why this is the default.

    Memory note: materialising ``G`` costs ``L × P`` floats.  We therefore apply
    surgery **only to the last shared block plus the fusion module** -- the
    label-specific heads have no conflict by construction, and the early
    backbone gradient is dominated by the shared low-level features where
    conflict is empirically negligible.
    """

    def __init__(
        self,
        params: Sequence[torch.nn.Parameter],
        *,
        mode: str = "aligned",
        cagrad_c: float = 0.4,
        n_dual_steps: int = 20,
    ) -> None:
        if mode not in {"pcgrad", "cagrad", "aligned", "mean"}:
            raise ValueError(f"unknown mode {mode!r}")
        self.params = [p for p in params if p.requires_grad]
        self.mode = mode
        self.cagrad_c = cagrad_c
        self.n_dual_steps = n_dual_steps
        self.shapes = [p.shape for p in self.params]
        self.numels = [p.numel() for p in self.params]

    # ------------------------------------------------------------------ #

    def _flat_grads(self, losses: Sequence[torch.Tensor], retain: bool = True) -> torch.Tensor:
        rows = []
        for i, l in enumerate(losses):
            grads = torch.autograd.grad(
                l,
                self.params,
                retain_graph=retain or i < len(losses) - 1,
                allow_unused=True,
            )
            rows.append(
                torch.cat(
                    [
                        (g if g is not None else torch.zeros_like(p)).reshape(-1)
                        for g, p in zip(grads, self.params)
                    ]
                )
            )
        return torch.stack(rows)

    def _assign(self, flat: torch.Tensor) -> None:
        off = 0
        for p, n, shape in zip(self.params, self.numels, self.shapes):
            chunk = flat[off : off + n].view(shape)
            p.grad = chunk.clone() if p.grad is None else p.grad.add_(0).copy_(chunk)
            off += n

    # ------------------------------------------------------------------ #

    def backward(self, losses: Sequence[torch.Tensor]) -> torch.Tensor:
        """Compute a surgery-corrected gradient and write it into ``.grad``."""
        G = self._flat_grads(losses)
        if self.mode == "mean":
            d = G.mean(0)
        elif self.mode == "pcgrad":
            d = self._pcgrad(G)
        elif self.mode == "cagrad":
            d = self._cagrad(G)
        else:
            d = self._aligned(G)
        self._assign(d)
        return d.detach()

    # ------------------------------------------------------------------ #

    @staticmethod
    def _pcgrad(G: torch.Tensor) -> torch.Tensor:
        L = G.shape[0]
        out = G.clone()
        perm = torch.randperm(L, device=G.device)
        for i in range(L):
            for j in perm:
                if int(j) == i:
                    continue
                dot = torch.dot(out[i], G[j])
                if dot < 0:
                    out[i] = out[i] - dot / (G[j].dot(G[j]) + 1e-12) * G[j]
        return out.mean(0)

    def _cagrad(self, G: torch.Tensor) -> torch.Tensor:
        L = G.shape[0]
        g0 = G.mean(0)
        g0_norm = g0.norm() + 1e-12
        GG = G @ G.T  # (L, L)
        w = torch.full((L,), 1.0 / L, device=G.device, dtype=G.dtype)
        c = self.cagrad_c
        for _ in range(self.n_dual_steps):
            gw_norm = torch.sqrt(torch.clamp(w @ GG @ w, min=1e-12))
            grad_w = GG @ w / gw_norm + (GG.mean(0) / g0_norm) * 0.0
            obj = GG @ w + c * g0_norm * grad_w
            w = _project_simplex(w - 0.1 * (-obj))
        gw = (w[:, None] * G).sum(0)
        gw_norm = gw.norm() + 1e-12
        return g0 + (c * g0_norm / gw_norm) * gw

    @staticmethod
    def _aligned(G: torch.Tensor) -> torch.Tensor:
        # G: (L, P).  Work with the small Gram matrix instead of the full SVD.
        M = G @ G.T
        M = 0.5 * (M + M.T)
        evals, evecs = torch.linalg.eigh(M.double())
        evals = evals.clamp_min(1e-12)
        sigma = torch.sqrt(evals)
        sigma_min = sigma.min()
        # B = sigma_min * U diag(1/sigma) U^T   (the alignment operator)
        B = evecs @ torch.diag(sigma_min / sigma) @ evecs.T
        alpha = B.sum(dim=0)  # 1^T B
        return (alpha.to(G.dtype)[:, None] * G).sum(0)


def _project_simplex(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto the probability simplex (Duchi et al., 2008)."""
    n = v.numel()
    u, _ = torch.sort(v, descending=True)
    css = torch.cumsum(u, dim=0)
    idx = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    cond = u - (css - 1.0) / idx > 0
    rho = int(torch.nonzero(cond).max()) if bool(cond.any()) else 0
    theta = (css[rho] - 1.0) / float(rho + 1)
    return torch.clamp(v - theta, min=0.0)


def cosine_with_warmup(
    step: int, *, total: int, warmup: int, min_ratio: float = 0.02
) -> float:
    """Learning-rate multiplier: linear warmup then cosine decay to ``min_ratio``."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
