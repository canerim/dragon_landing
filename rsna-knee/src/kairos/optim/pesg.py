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

    Ascent parameters are named **explicitly** via ``ascent_params``.  The
    legacy ``param._auc_ascent = True`` marker is still honoured, but it must
    not be the only channel: a plain Python attribute on a Parameter does not
    survive ``module.to(device)`` (``nn.Module._apply`` rebuilds the Parameter
    whenever the shallow-copy check fails, which it does for any device
    change), so a run that tagged on CPU and then moved to CUDA would silently
    *descend* on the variable it is supposed to ascend.  See
    :meth:`kairos.losses.auc.AUCMarginLoss._tag_minimax`.
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
        ascent_params: Iterable[torch.nn.Parameter] = (),
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
        self._ascent_ids = {id(p) for p in ascent_params}
        self._init_reference()
        self.T = 0

    def _is_ascent(self, p: torch.Tensor) -> bool:
        return id(p) in self._ascent_ids or bool(getattr(p, "_auc_ascent", False))

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

                ascent = self._is_ascent(p)
                st = self.state[p]

                if ascent:
                    # Ascent on alpha; no proximal term, no weight decay.
                    p.add_(g, alpha=lr)
                    p.clamp_min_(0.0)
                    continue

                d = g + wd * p
                # Proximal anchor to the epoch reference point.  The reference
                # is cloned at construction, so a model moved to CUDA
                # afterwards would leave it on the CPU; re-home it lazily
                # rather than crashing on the first step of a real run.
                ref = st["ref"]
                if ref.device != p.device or ref.dtype != p.dtype:
                    ref = st["ref"] = ref.to(device=p.device, dtype=p.dtype)
                d = d + (p - ref) / max(gamma, 1e-8)
                buf = st["buf"]
                if buf.device != p.device or buf.dtype != p.dtype:
                    buf = st["buf"] = buf.to(device=p.device, dtype=p.dtype)
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
    def restore(self) -> None:
        """Undo the ascent perturbation *without* stepping.

        Needed when the perturbed gradient comes back non-finite: skipping the
        step is right, but leaving theta at :math:`\\theta+\\epsilon` is not --
        the model would silently keep a random sharpness perturbation baked in
        for the rest of the run.
        """
        for _, p in self.model.named_parameters():
            eps = self.state.pop((p, "eps"), None)  # type: ignore[arg-type]
            if eps is not None:
                p.sub_(eps)

    @torch.no_grad()
    def descent_step(self) -> None:
        self.restore()
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

    def _add_(self, flat: torch.Tensor) -> None:
        """Accumulate ``flat`` into ``.grad`` for the surgery parameters."""
        off = 0
        for p, n, shape in zip(self.params, self.numels, self.shapes):
            chunk = flat[off : off + n].view(shape)
            if p.grad is None:
                p.grad = chunk.detach().clone()
            else:
                p.grad.add_(chunk.detach())
            off += n

    def _combine(self, G: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean":
            return G.mean(0)
        if self.mode == "pcgrad":
            return self._pcgrad(G)
        if self.mode == "cagrad":
            return self._cagrad(G)
        return self._aligned(G)

    # ------------------------------------------------------------------ #

    def backward(self, losses: Sequence[torch.Tensor]) -> torch.Tensor:
        """Compute a surgery-corrected gradient and *overwrite* ``.grad``.

        Use only when the surgery losses are the *whole* objective for these
        parameters; otherwise use :meth:`correct_`, which composes.
        """
        d = self._combine(self._flat_grads(losses))
        off = 0
        for p, n, shape in zip(self.params, self.numels, self.shapes):
            p.grad = d[off : off + n].view(shape).detach().clone()
            off += n
        return d.detach()

    def prepare(
        self, losses: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        r"""Measure the per-task gradients and return the correction to apply.

        Called **before** the main ``backward()``, while the graph is alive but
        no ``.grad`` has been written yet.  That ordering is not cosmetic: it is
        what keeps the memory cost of surgery near zero.  Since ``inputs`` is
        restricted to the fusion/head block, autograd only traverses the *tail*
        of the graph, so the ``retain_graph=True`` here pins a handful of small
        activations.  Doing it the other way round -- ``loss.backward(
        retain_graph=True)`` first -- would pin the entire backbone activation
        stack for the duration, roughly doubling peak memory for no gain.

        Returns ``(delta, logs)``.  ``delta`` is ``None`` when fewer than two
        tasks had a usable gradient on this batch -- a rare label with no
        positive in the minibatch contributes an exactly-zero row, and
        combining a zero row with the rest is not conflict resolution, it is
        division by noise.
        """
        G = self._flat_grads(losses, retain=True)
        with torch.no_grad():
            norms = G.norm(dim=1)
            active = norms > 1e-8 * norms.max().clamp_min(1e-12)
            n_active = int(active.sum())
            if n_active < 2:
                return None, {"surgery/skipped": 1.0,
                              "surgery/active_tasks": float(n_active)}

            combined = self._combine(G[active])
            # ``plain`` sums *all* rows, including the dropped ones: they are
            # what the caller's backward() will put into .grad, so they are what
            # has to be cancelled.  Their norm is ~0, so nothing is lost either
            # way -- but subtracting only the kept rows leaves a silent residual.
            plain = G.sum(0)

            Ga = G[active]
            unit = Ga / Ga.norm(dim=1, keepdim=True).clamp_min(1e-12)
            cos = unit @ unit.T
            off_diag = cos[~torch.eye(n_active, dtype=torch.bool, device=cos.device)]
            logs = {
                "surgery/skipped": 0.0,
                "surgery/active_tasks": float(n_active),
                "surgery/mean_pairwise_cos": float(off_diag.mean()),
                "surgery/conflict_frac": float((off_diag < 0).float().mean()),
                "surgery/grad_norm_ratio": float(
                    combined.norm() / plain.norm().clamp_min(1e-12)
                ),
            }
            return (combined - plain), logs

    def apply_(self, delta: torch.Tensor | None) -> None:
        """Add a :meth:`prepare` correction into ``.grad``, after ``backward``."""
        if delta is not None:
            self._add_(delta)

    def correct_(self, losses: Sequence[torch.Tensor]) -> dict[str, float]:
        r"""Replace the *plain-sum* contribution of ``losses`` with the
        surgery-combined one, leaving every other term's gradient intact.

        The caller has already run a normal ``backward()`` on the full
        objective, so each surgery parameter holds

        .. math:: g = g_{\text{other}} + \textstyle\sum_l g_l .

        We recompute the per-task matrix :math:`G` for these parameters and add

        .. math:: \Delta = \mathcal S(G) - \textstyle\sum_l g_l ,

        which leaves :math:`g_{\text{other}} + \mathcal S(G)` -- exactly the
        intended update, and correct no matter what else is in the objective.
        Parameters *outside* the surgery set keep the ordinary summed gradient,
        which is what we want: conflict resolution is worth its cost near the
        heads and is empirically negligible in the early backbone.

        Requires the graph to still be alive, so the caller must have used
        ``retain_graph=True``.  The training loop uses the cheaper
        :meth:`prepare` / :meth:`apply_` split instead; this one-shot form is
        kept for callers that already hold a retained graph.
        """
        delta, logs = self.prepare(losses)
        self.apply_(delta)
        return logs

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
    def _aligned(G: torch.Tensor, rcond: float = 1e-6) -> torch.Tensor:
        r"""Aligned-MTL direction :math:`d = G^\top\,\sigma_{\min}(GG^\top)^{-1/2}\mathbf 1`.

        Two numerical points that decide whether this works or destroys the run:

        **The spectrum must be truncated, not clamped.**  If two labels produce
        (near-)collinear gradients -- which happens constantly here, ``Medial
        OA`` and ``Lateral OA`` are strongly correlated -- then :math:`GG^\top`
        is rank deficient and its smallest singular value is numerically zero.
        The prefactor :math:`\sigma_{\min}` would then scale the *entire*
        update to zero: training stops, the loss plateaus, and nothing in the
        logs says why.  We therefore discard directions below
        ``rcond * sigma_max`` and take :math:`\sigma_{\min}` over the retained
        spectrum, i.e. the pseudo-inverse on the row space.  Discarded
        directions contribute nothing to any :math:`g_l` by construction, so
        nothing is lost.

        **Degenerate input falls back to the plain sum.**  An all-zero ``G``
        (every task inactive on this batch) has no meaningful alignment; the
        honest answer is the ordinary summed gradient, not ``0``.

        The resulting map is positively homogeneous of degree one
        (:math:`G \mapsto cG` gives :math:`d \mapsto cd`) and, for orthogonal
        rows, exactly invariant to rescaling any non-minimal task -- which is
        the property that makes it the default here.

        **On the magnitude.**  The :math:`\sigma_{\min}` prefactor makes
        :math:`\lVert d\rVert` track the *weakest* task, so it is routinely much
        smaller than :math:`\lVert\sum_l g_l\rVert` -- ``surgery/grad_norm_ratio``
        around 0.07 on twelve knee labels is normal, not a bug.  We deliberately
        do **not** renormalise back.  Surgery is applied to a sub-block while
        the backbone keeps the plain summed gradient, which sounds like it would
        unbalance their effective learning rates; it does not, because AdamW
        divides each parameter by its own second-moment estimate, so a uniformly
        rescaled gradient produces very nearly the same step.  What survives the
        rescale -- and what we actually want -- is the *direction*.
        """
        # G: (L, P).  Work with the small Gram matrix instead of the full SVD.
        M = G @ G.T
        M = 0.5 * (M + M.T)
        evals, evecs = torch.linalg.eigh(M.double())
        sigma = torch.sqrt(evals.clamp_min(0.0))
        sigma_max = sigma.max()
        if not torch.isfinite(sigma_max) or sigma_max <= 1e-20:
            return G.sum(0)
        keep = sigma > rcond * sigma_max
        sigma_min = sigma[keep].min()
        # B = sigma_min * U diag(1/sigma) U^T on the retained subspace.
        inv = torch.where(keep, sigma_min / sigma.clamp_min(1e-30),
                          torch.zeros_like(sigma))
        B = (evecs * inv) @ evecs.T
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
