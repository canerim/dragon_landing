r"""Direct optimisation of the competition metric.

The leaderboard is a macro-average ROC-AUC.  Cross-entropy is a *surrogate for
a surrogate*: it optimises calibrated likelihood, from which a good ranking is
hoped to follow.  For labels whose prevalence is below ~2 % that hope is thin,
because almost every gradient step is spent on easy negatives that already sit
far from the decision region and contribute nothing to the ranking.

Three objectives are implemented here, in increasing order of aggression, and
they are meant to be *composed on a schedule* rather than chosen once:

``AUCMarginLoss``
    The min-max margin reformulation of the squared-hinge AUC surrogate
    (Ying, Wen & Lyu 2016; Yuan et al., *Large-scale Robust Deep AUC
    Maximization*, ICCV 2021).  Its key property is that the pairwise
    :math:`O(n^+ n^-)` sum collapses to a **per-example** expression through
    auxiliary variables :math:`(a, b, \alpha)`, so a minibatch estimate is
    unbiased and cheap.

``PartialAUCLoss``
    A two-way partial AUC that only counts pairs in the operating region
    actually used clinically (high TPR / low FPR).  Formulated as a
    distributionally-robust inner maximisation over the pair distribution and
    solved in closed form by soft top-k, which is differentiable everywhere
    unlike a hard top-k truncation.

``PairwiseRankQueue``
    An exact pairwise squared-hinge over a momentum memory queue of positive
    and negative embeddings-logits.  This is the fallback for the three rarest
    labels, where a minibatch of 32 studies frequently contains *zero*
    positives and both estimators above degenerate.

All three are multi-label aware: every label carries its own auxiliary
variables and its own margin, and the reduction over labels is a plain mean so
that a rare label counts exactly as much as a common one -- which is precisely
what macro-averaging means.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "AUCMarginLoss",
    "PartialAUCLoss",
    "PairwiseRankQueue",
    "soft_topk_weights",
]

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# 1. Min-max margin AUC                                                        #
# --------------------------------------------------------------------------- #


class AUCMarginLoss(nn.Module):
    r"""Deep AUC maximisation via the min-max margin surrogate.

    Let :math:`h(x) \in [0,1]` be the sigmoid score, :math:`p` the label
    prevalence, and :math:`y \in \{0,1\}`.  The squared-hinge AUC surrogate

    .. math::
        \min_\theta \; \mathbb E\big[(m - h(x^+) + h(x^-))^2\big]

    has the equivalent min-max form

    .. math::
        \min_{\theta, a, b} \max_{\alpha \ge 0} \;
          \underbrace{(1-p)\,\mathbb E\big[(h(x) - a)^2 \mid y = 1\big]}_{A_1}
        + \underbrace{p\,\mathbb E\big[(h(x) - b)^2 \mid y = 0\big]}_{A_2}
        + \underbrace{2\alpha\big(m + p\,\mathbb E[h \mid y{=}0]
                                  - (1-p)\,\mathbb E[h \mid y{=}1]\big)
          - p(1-p)\alpha^2}_{A_3}.

    The inner maximisation is concave in :math:`\alpha` with the closed-form
    optimum :math:`\alpha^\star = \mathbb E[h \mid y{=}0] - \mathbb E[h \mid
    y{=}1] + m`, which is exactly what a gradient-*ascent* step on
    :math:`\alpha` converges to; we keep :math:`\alpha` as a parameter and let
    the optimiser do ascent (see :class:`kairos.optim.pesg.PESG`) because the
    closed form is only optimal for the *population* moments and is badly noisy
    on a minibatch.

    The reason this matters practically: :math:`A_1` and :math:`A_2` are
    variance terms.  They pull the positive scores together and the negative
    scores together *without* pushing either towards 0 or 1, which is why AUC-M
    keeps improving ranking long after BCE has saturated.

    Parameters
    ----------
    num_labels
        Number of independent binary tasks (12 here).
    margin
        :math:`m`.  0.6–1.0 works; larger margins behave like a harder ranking
        constraint and need a smaller learning rate on ``alpha``.
    prevalence
        Optional fixed :math:`p_l` per label.  When ``None`` the batch estimate
        is used, which is unbiased but high-variance for rare labels -- pass the
        training-set prevalence instead, it is strictly better.
    ema_prevalence
        If given and ``prevalence`` is ``None``, maintain an EMA of the batch
        prevalence with this decay instead of using the raw batch value.
    """

    def __init__(
        self,
        num_labels: int,
        *,
        margin: float = 1.0,
        prevalence: torch.Tensor | None = None,
        ema_prevalence: float | None = 0.99,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.margin = float(margin)
        self.reduction = reduction

        # Auxiliary variables.  a, b are minimised; alpha is maximised.
        self.a = nn.Parameter(torch.zeros(num_labels))
        self.b = nn.Parameter(torch.zeros(num_labels))
        self.alpha = nn.Parameter(torch.zeros(num_labels))
        self._tag_minimax()

        if prevalence is not None:
            self.register_buffer("_p_fixed", prevalence.float().clone())
        else:
            self.register_buffer("_p_fixed", torch.empty(0))
        self.ema_prevalence = ema_prevalence
        self.register_buffer("_p_ema", torch.full((num_labels,), float("nan")))

    # ------------------------------------------------------------------ #

    def _tag_minimax(self) -> None:
        """(Re-)mark the saddle-point parameters.

        ``_auc_ascent`` means "ascend, then project onto :math:`\\alpha\\ge 0`";
        ``_minimax`` means "hand this to PESG, not to AdamW".

        This has to be re-applied after every ``_apply`` because a plain Python
        attribute on a Parameter **does not survive a device move**.
        ``nn.Module._apply`` keeps the object only when
        ``torch._has_compatible_shallow_copy_type`` holds, which it does for a
        dtype cast but *not* for cpu->cuda: there it constructs
        ``Parameter(param_applied, requires_grad)`` and every user attribute is
        dropped.  The consequence was invisible on CPU and severe on GPU --
        ``build_objectives(..., device='cuda')`` untagged ``alpha``, so PESG
        took the *descent* branch on an objective that is concave in
        :math:`\\alpha`, ``project()`` pinned it at 0, and the whole
        :math:`A_3` block -- the margin, the entire ranking pressure -- was
        identically zero while the reported loss went *down*.

        :meth:`minimax_parameters` is the authoritative accessor; the tags are
        kept for anything that still reads them.
        """
        self.alpha._auc_ascent = True  # type: ignore[attr-defined]
        for p in (self.a, self.b, self.alpha):
            p._minimax = True  # type: ignore[attr-defined]

    def _apply(self, *args, **kwargs):  # noqa: D102 - see _tag_minimax
        out = super()._apply(*args, **kwargs)
        out._tag_minimax()
        return out

    def minimax_parameters(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """``(descent, ascent)`` -- the saddle-point block, by identity.

        Resolved through attribute access, so it always returns the parameters
        the module *currently* owns rather than whatever objects existed when
        some earlier caller looked.
        """
        return [self.a, self.b], [self.alpha]

    def _prevalence(self, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if self._p_fixed.numel() == self.num_labels:
            return self._p_fixed.to(y.device)
        n_valid = valid.sum(dim=0).clamp_min(1.0)
        p_batch = (y * valid).sum(dim=0) / n_valid
        if self.ema_prevalence is None:
            return p_batch
        if torch.isnan(self._p_ema).any():
            self._p_ema.copy_(p_batch.detach())
        else:
            d = self.ema_prevalence
            self._p_ema.mul_(d).add_(p_batch.detach(), alpha=1.0 - d)
        return self._p_ema.clamp(1e-4, 1.0 - 1e-4)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        reduction: str | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits, targets
            ``(B, L)``.  ``targets`` may contain NaN for unobserved labels.
        mask
            ``(B, L)`` boolean; ``False`` excludes an entry.  NaN targets are
            masked automatically.
        sample_weight
            ``(B,)`` or ``(B, L)`` non-negative weights (e.g. from Group DRO).
        reduction
            Overrides ``self.reduction`` for this call.  ``'label_contrib'``
            returns the exact additive decomposition of ``'mean'`` over labels
            (it sums to ``'mean'``), which is what gradient surgery consumes.
        """
        h = torch.sigmoid(logits)
        y = torch.nan_to_num(targets, nan=0.0)
        valid = torch.isfinite(targets).float()
        if mask is not None:
            valid = valid * mask.float()
        if sample_weight is not None:
            w = sample_weight if sample_weight.dim() == 2 else sample_weight[:, None]
            valid = valid * w

        p = self._prevalence(y, valid).to(h.dtype)
        a, b, alpha = self.a.to(h.dtype), self.b.to(h.dtype), self.alpha.to(h.dtype)

        pos_w = valid * y
        neg_w = valid * (1.0 - y)
        n_pos = pos_w.sum(dim=0)
        n_neg = neg_w.sum(dim=0)
        n_valid = valid.sum(dim=0).clamp_min(_EPS)
        # A label with no positives *and* no negatives in the batch contributes
        # nothing; a label with only one class contributes only its variance
        # term, which is the correct degenerate behaviour (it still shapes the
        # score distribution) -- but we damp it so a batch of all-negatives for
        # a rare label cannot dominate.
        have_both = ((n_pos > 0) & (n_neg > 0)).to(h.dtype)

        # NOTE: these are *unconditional* (whole-batch) means with class
        # indicators, not class-conditional means.  That is not a stylistic
        # choice -- it is what makes the identity
        #     f(a*, b*, alpha*) = p(1-p) * E[(m - h(x+) + h(x-))^2]
        # hold exactly.  Dividing by n_pos / n_neg instead weights the two
        # variance terms by (1-p) and p rather than both by p(1-p), and the
        # min-max form then no longer equals the pairwise surrogate.
        A1 = (1.0 - p) * ((pos_w * (h - a) ** 2).sum(dim=0) / n_valid)
        A2 = p * ((neg_w * (h - b) ** 2).sum(dim=0) / n_valid)
        cross = (p * neg_w * h - (1.0 - p) * pos_w * h).sum(dim=0) / n_valid
        A3 = 2.0 * alpha * (p * (1.0 - p) * self.margin + cross)
        A3 = A3 - p * (1.0 - p) * alpha**2

        per_label = (A1 + A2 + A3) * have_both
        denom = have_both.sum().clamp_min(1.0)

        red = reduction or self.reduction
        if red == "none":
            return per_label
        if red == "sum":
            return per_label.sum()
        if red == "label_contrib":
            return per_label / denom
        return per_label.sum() / denom

    @torch.no_grad()
    def set_optimal_auxiliaries(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        r"""Snap :math:`(a, b, \alpha)` to their closed-form optima on a sample.

        :math:`a^\star = \mathbb E[h\mid y{=}1]`,
        :math:`b^\star = \mathbb E[h\mid y{=}0]`,
        :math:`\alpha^\star = m + \mathbb E[h\mid y{=}0] - \mathbb E[h\mid y{=}1]`.

        Used to warm-start the auxiliaries after the supervised stage (which
        saves several hundred steps of the ascent chasing a moving target) and
        by the tests that verify the min-max identity.  It is deliberately not
        called every step: the closed form is optimal only for the population
        moments and is very noisy on a minibatch, which is the whole reason
        PESG exists.
        """
        h = torch.sigmoid(logits)
        y = torch.nan_to_num(targets, nan=0.0)
        valid = torch.isfinite(targets).float()
        pos, neg = valid * y, valid * (1.0 - y)
        mean_pos = (pos * h).sum(0) / pos.sum(0).clamp_min(_EPS)
        mean_neg = (neg * h).sum(0) / neg.sum(0).clamp_min(_EPS)
        self.a.copy_(mean_pos)
        self.b.copy_(mean_neg)
        self.alpha.copy_((self.margin + mean_neg - mean_pos).clamp_min(0.0))

    @torch.no_grad()
    def project(self, bound: float = 1.0) -> None:
        """Project the auxiliary variables back into their feasible set.

        :math:`a, b \\in [0,1]` because they track sigmoid means, and
        :math:`\\alpha \\ge 0`.  Called by the optimiser after every step;
        without it the ascent on :math:`\\alpha` is unbounded.
        """
        self.a.clamp_(0.0, bound)
        self.b.clamp_(0.0, bound)
        self.alpha.clamp_(0.0, 2.0 * bound + self.margin)


# --------------------------------------------------------------------------- #
# 2. Two-way partial AUC                                                       #
# --------------------------------------------------------------------------- #


def soft_topk_weights(
    scores: torch.Tensor,
    k: float,
    *,
    dim: int = 0,
    temperature: float = 0.1,
    n_iter: int = 60,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Differentiable soft top-:math:`k` selection weights in :math:`[0,1]`.

    Solves the entropy-regularised linear program

    .. math::
        \max_{w \in [0,1]^n,\; \mathbf 1^\top w = k}
          \langle w, s \rangle - \tau \sum_i \big[w_i \log w_i + (1-w_i)\log(1-w_i)\big]

    whose solution is :math:`w_i = \sigma((s_i - \nu)/\tau)` with :math:`\nu`
    the unique root of :math:`\sum_i \sigma((s_i - \nu)/\tau) = k`.  We find
    :math:`\nu` by bisection (monotone, so bisection is globally convergent and
    needs no initialisation), and differentiate through the *closed form* with
    :math:`\nu` treated by the implicit function theorem:

    .. math::
        \frac{\partial \nu}{\partial s_j}
        = \frac{w_j(1-w_j)}{\sum_i w_i(1-w_i)}.

    That is materially better than back-propagating through the bisection
    iterations: it is exact, uses O(1) memory, and does not depend on
    ``n_iter``.
    """
    if mask is not None:
        neg_inf = torch.finfo(scores.dtype).min / 4
        scores = scores.masked_fill(~mask, neg_inf)

    with torch.no_grad():
        lo = scores.min(dim=dim, keepdim=True).values - 10.0 * temperature - 1.0
        hi = scores.max(dim=dim, keepdim=True).values + 10.0 * temperature + 1.0
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            w = torch.sigmoid((scores - mid) / temperature)
            if mask is not None:
                w = w * mask.to(w.dtype)
            s = w.sum(dim=dim, keepdim=True)
            too_many = (s > k).to(scores.dtype)
            lo = too_many * mid + (1.0 - too_many) * lo
            hi = too_many * hi + (1.0 - too_many) * mid
        nu = 0.5 * (lo + hi)

    # Implicit differentiation of nu w.r.t. scores.
    w0 = torch.sigmoid((scores - nu) / temperature)
    if mask is not None:
        w0 = w0 * mask.to(w0.dtype)
    g = (w0 * (1.0 - w0)).detach()
    gsum = g.sum(dim=dim, keepdim=True).clamp_min(_EPS)
    nu_diff = (g * scores).sum(dim=dim, keepdim=True) / gsum
    nu_eff = nu.detach() + (nu_diff - nu_diff.detach())

    w = torch.sigmoid((scores - nu_eff) / temperature)
    if mask is not None:
        w = w * mask.to(w.dtype)
    return w


class PartialAUCLoss(nn.Module):
    r"""Two-way partial AUC over the clinically relevant operating region.

    Full AUC integrates the ROC over :math:`\mathrm{FPR} \in [0,1]`.  Nobody
    triages a knee MRI list at 70 % false-positive rate.  The two-way partial
    AUC restricts to :math:`\mathrm{FPR} \le \beta` and
    :math:`\mathrm{TPR} \ge \alpha`, i.e. it counts only pairs formed from the
    **hardest negatives** (highest-scoring) and the **hardest positives**
    (lowest-scoring):

    .. math::
        \mathrm{pAUC} \;\propto\; \frac{1}{|\mathcal P_\alpha||\mathcal N_\beta|}
          \sum_{i \in \mathcal P_\alpha} \sum_{j \in \mathcal N_\beta}
          \ell\big(s_i - s_j\big).

    Hard truncation makes the objective piecewise constant in the selection, so
    we use :func:`soft_topk_weights` for both sets.  The resulting objective is
    the inner maximum of a DRO problem over pair weights constrained to a
    scaled simplex -- entropic regularisation is exactly the ``temperature``.

    Why ship it: the leaderboard metric is full AUC, but full AUC and pAUC
    disagree most for the *rare* labels, and the top of the ranking is where a
    macro-average is won.  We use pAUC as a **secondary** term with small weight
    (0.05–0.15), never alone.
    """

    def __init__(
        self,
        *,
        fpr_max: float = 0.30,
        tpr_min: float = 0.50,
        margin: float = 1.0,
        temperature: float = 0.1,
        min_selected: int = 2,
    ) -> None:
        super().__init__()
        self.fpr_max = float(fpr_max)
        self.tpr_min = float(tpr_min)
        self.margin = float(margin)
        self.temperature = float(temperature)
        self.min_selected = int(min_selected)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        s = logits
        y = torch.nan_to_num(targets, nan=0.0)
        valid = torch.isfinite(targets)
        if mask is not None:
            valid = valid & mask.bool()

        losses = []
        for l in range(s.shape[1]):
            v = valid[:, l]
            pos = v & (y[:, l] > 0.5)
            neg = v & (y[:, l] <= 0.5)
            n_pos, n_neg = int(pos.sum()), int(neg.sum())
            if n_pos == 0 or n_neg == 0:
                continue

            # Hardest negatives: the top beta fraction by score.
            k_neg = max(self.min_selected, int(round(self.fpr_max * n_neg)))
            k_neg = min(k_neg, n_neg)
            w_neg = soft_topk_weights(
                s[:, l], float(k_neg), temperature=self.temperature, mask=neg
            )
            # Hardest positives: the bottom (1 - alpha) fraction, i.e. top-k of
            # the *negated* score.
            k_pos = max(self.min_selected, int(round((1.0 - self.tpr_min) * n_pos)))
            k_pos = min(k_pos, n_pos)
            w_pos = soft_topk_weights(
                -s[:, l], float(k_pos), temperature=self.temperature, mask=pos
            )

            diff = s[pos, l][:, None] - s[neg, l][None, :]
            hinge = F.relu(self.margin - diff) ** 2
            wp = w_pos[pos][:, None]
            wn = w_neg[neg][None, :]
            weight = wp * wn
            losses.append((weight * hinge).sum() / weight.sum().clamp_min(_EPS))

        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


# --------------------------------------------------------------------------- #
# 3. Memory-queue pairwise ranking for ultra-rare labels                       #
# --------------------------------------------------------------------------- #


@dataclass
class _Queue:
    buf: torch.Tensor
    ptr: int = 0
    filled: int = 0


class PairwiseRankQueue(nn.Module):
    r"""Exact pairwise squared hinge against a momentum memory of past scores.

    For a label with prevalence 0.8 %, a batch of 32 studies contains a
    positive roughly one time in four.  On the other three batches the AUC
    surrogate produces *no gradient at all* for that label.  Cranking the
    sampler until every batch has a positive distorts the joint label
    distribution (positives for Fracture co-occur with Contusion, so
    oversampling Fracture silently oversamples Contusion too).

    The fix is a MoCo-style queue: we keep the last :math:`Q` scores per label
    per class, detached, and form pairs between the current batch and the
    queue.  Gradients flow only through the current batch, so the queue costs
    nothing but a buffer, and the effective number of pairs per step goes from
    :math:`O(1)` to :math:`O(Q)`.

    Staleness is the price.  It is bounded by using a short queue
    (``capacity`` ≈ 512) and by only enabling this term after the backbone has
    stopped moving fast (typically epoch ≥ 5), which is exactly when the
    surrogate matters.
    """

    def __init__(
        self,
        num_labels: int,
        *,
        capacity: int = 512,
        margin: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.capacity = int(capacity)
        self.margin = float(margin)
        self.register_buffer("pos_buf", torch.zeros(num_labels, capacity))
        self.register_buffer("neg_buf", torch.zeros(num_labels, capacity))
        self.register_buffer("pos_ptr", torch.zeros(num_labels, dtype=torch.long))
        self.register_buffer("neg_ptr", torch.zeros(num_labels, dtype=torch.long))
        self.register_buffer("pos_n", torch.zeros(num_labels, dtype=torch.long))
        self.register_buffer("neg_n", torch.zeros(num_labels, dtype=torch.long))

    @torch.no_grad()
    def _push(self, buf, ptr, count, l: int, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        v = values.detach().flatten().to(buf.dtype)
        n = min(v.numel(), self.capacity)
        v = v[-n:]
        p = int(ptr[l])
        idx = (torch.arange(n, device=buf.device) + p) % self.capacity
        buf[l, idx] = v
        ptr[l] = (p + n) % self.capacity
        count[l] = min(int(count[l]) + n, self.capacity)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        update: bool = True,
    ) -> torch.Tensor:
        y = torch.nan_to_num(targets, nan=0.0)
        valid = torch.isfinite(targets)
        if mask is not None:
            valid = valid & mask.bool()

        losses = []
        for l in range(self.num_labels):
            v = valid[:, l]
            pos_s = logits[v & (y[:, l] > 0.5), l]
            neg_s = logits[v & (y[:, l] <= 0.5), l]

            terms = []
            if pos_s.numel() and int(self.neg_n[l]) > 0:
                q = self.neg_buf[l, : int(self.neg_n[l])].to(logits.dtype)
                terms.append(F.relu(self.margin - (pos_s[:, None] - q[None, :])) ** 2)
            if neg_s.numel() and int(self.pos_n[l]) > 0:
                q = self.pos_buf[l, : int(self.pos_n[l])].to(logits.dtype)
                terms.append(F.relu(self.margin - (q[None, :] - neg_s[:, None])) ** 2)
            if pos_s.numel() and neg_s.numel():
                terms.append(
                    F.relu(self.margin - (pos_s[:, None] - neg_s[None, :])) ** 2
                )
            if terms:
                losses.append(torch.cat([t.reshape(-1) for t in terms]).mean())

            if update:
                self._push(self.pos_buf, self.pos_ptr, self.pos_n, l, pos_s)
                self._push(self.neg_buf, self.neg_ptr, self.neg_n, l, neg_s)

        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()
