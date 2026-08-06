r"""Label-specific queries, expert routing, and cross-sequence fusion.

The central architectural claim of this system: **twelve pathologies should not
share one pooled representation.**  An ACL tear is a signal-intensity change in
a 15 mm structure in the intercondylar notch on 3–5 sagittal slices; an
effusion is a large bright region that is essentially a global property of the
study; patellofemoral OA lives in the axial plane and nowhere else.  A single
attention-pooled study vector must encode all of that in one place, and the
twelve linear heads then have to disentangle it -- a task the network solves by
spending capacity on the frequent labels and giving up on the rare ones.

Instead each pathology owns a learned query :math:`q_l` and pools *its own*
evidence:

.. math::
    a_{l,s,i} = \operatorname{softmax}_i\big(q_l^\top W_s h_{s,i} + \pi_{l,s}\big),
    \qquad v_{l,s} = \sum_i a_{l,s,i}\, h_{s,i},

then attends across sequences with a missing-sequence mask:

.. math::
    \alpha_{l,s} = \operatorname{softmax}_s\big(q_l^\top U v_{l,s}
      + \rho_{l,\mathrm{fam}(s)} + m_s\big),
    \qquad z_l = \sum_s \alpha_{l,s}\, v_{l,s},

with :math:`m_s = -\infty` for absent sequences and :math:`\rho` the
prior from :data:`kairos.constants.SEQUENCE_LABEL_PRIOR` (a *learned* bias
initialised from the prior, not a hard gate).

On top of that, :class:`LabelExpertRouter` adds a sparse mixture of experts
over the query→logit map.  The motivation is that the twelve labels cluster
into five mechanisms (ligament / meniscus / degenerative / inflammatory /
osseous) that want genuinely different features, but forcing a hard partition
throws away the strong cross-talk (ACL tears co-occur with lateral contusions).
A top-2 soft router learns the partition it wants, with a load-balancing loss
to stop it collapsing onto one expert.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LabelQueryPool", "CrossSequenceFusion", "LabelExpertRouter", "LabelHeadBank"]


class LabelQueryPool(nn.Module):
    r"""Per-label attention pooling over the slices of a single series.

    ``n_query_tokens > 1`` gives each label several query slots whose pooled
    results are concatenated then projected -- multi-head pooling.  For labels
    that are genuinely multi-focal (contusion can appear in two compartments)
    this measurably helps; for ACL it does nothing and the extra slots learn to
    duplicate.  Two slots is the sweet spot and the default.
    """

    def __init__(
        self,
        dim: int,
        num_labels: int,
        *,
        n_query_tokens: int = 2,
        query_dim: int | None = None,
        temperature: float = 1.0,
        dropout: float = 0.0,
        attn_entropy_floor: float = 0.0,
    ) -> None:
        super().__init__()
        qd = query_dim or dim
        self.num_labels = num_labels
        self.n_tokens = n_query_tokens
        self.query = nn.Parameter(torch.randn(num_labels, n_query_tokens, qd) * (qd**-0.5))
        self.key = nn.Linear(dim, qd, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.merge = (
            nn.Linear(dim * n_query_tokens, dim, bias=False) if n_query_tokens > 1 else nn.Identity()
        )
        self.temperature = temperature
        self.drop = nn.Dropout(dropout)
        self.attn_entropy_floor = attn_entropy_floor
        self.scale = qd**-0.5

    def forward(
        self,
        h: torch.Tensor,  # (B, S, D) slice tokens of one series
        *,
        slice_mask: torch.Tensor | None = None,  # (B, S) True = valid
        logit_bias: torch.Tensor | None = None,  # (B, L, S) anatomical prior
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(pooled (B,L,D), attention (B,L,S), entropy (B,L))``."""
        B, S, D = h.shape
        k = self.key(h)  # (B, S, Q)
        v = self.value(h)  # (B, S, D)

        logits = torch.einsum("ltq,bsq->blts", self.query.to(h.dtype), k) * self.scale
        logits = logits / max(self.temperature, 1e-3)
        if logit_bias is not None:
            logits = logits + logit_bias[:, :, None, :]
        if slice_mask is not None:
            neg = torch.finfo(logits.dtype).min / 4
            logits = logits.masked_fill(~slice_mask[:, None, None, :], neg)

        attn = torch.softmax(logits, dim=-1)  # (B, L, T, S)
        attn = self.drop(attn)
        pooled = torch.einsum("blts,bsd->bltd", attn, v)
        pooled = pooled.reshape(B, self.num_labels, self.n_tokens * D)
        pooled = self.merge(pooled)

        attn_mean = attn.mean(dim=2)  # (B, L, S)
        ent = -(attn_mean.clamp_min(1e-8) * attn_mean.clamp_min(1e-8).log()).sum(-1)
        return pooled, attn_mean, ent

    def entropy_penalty(self, entropy: torch.Tensor) -> torch.Tensor:
        r"""Hinge that keeps attention from collapsing onto a single slice.

        A degenerate one-hot attention is a classic failure of query pooling on
        small datasets: the query latches onto one slice index, gets a good
        training loss, and generalises terribly because the anatomy is not at
        that index in the next study.  We penalise
        :math:`\mathrm{ReLU}(H_{\min} - H)` with :math:`H_{\min}` around
        :math:`\log 3` (i.e. "spread over at least ~3 slices").
        """
        if self.attn_entropy_floor <= 0:
            return entropy.sum() * 0.0
        return F.relu(self.attn_entropy_floor - entropy).mean()


class CrossSequenceFusion(nn.Module):
    r"""Per-label attention over series, with an explicit missing mask.

    Sequence availability is not missing-at-random: sites that skip the axial
    series also tend to be the sites that never dictate "synovitis".  Learning
    from the *presence pattern* is therefore a shortcut with real predictive
    power on the training set and no validity on a different site mix.  Two
    countermeasures:

    * the presence pattern is never fed to the classifier as a feature, only as
      an attention mask;
    * ``sequence_dropout`` during training randomly hides available series so
      the model must produce a sensible answer from any subset -- which is also
      what makes the coarse-to-fine sequence-skipping at inference safe.
    """

    def __init__(
        self,
        dim: int,
        num_labels: int,
        num_families: int,
        *,
        mode: str = "attention",  # "attention" | "transformer"
        n_heads: int = 8,
        prior: torch.Tensor | None = None,  # (L, F) bool
        prior_scale: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.num_labels = num_labels
        self.query = nn.Parameter(torch.randn(num_labels, dim) * (dim**-0.5))
        self.proj = nn.Linear(dim, dim, bias=False)
        self.family_bias = nn.Parameter(torch.zeros(num_labels, num_families))
        if prior is not None:
            with torch.no_grad():
                self.family_bias.copy_(prior_scale * (prior.float() * 2.0 - 1.0))
        self.scale = dim**-0.5

        if mode == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        else:
            self.encoder = None

        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        v: torch.Tensor,  # (B, S_seq, L, D) per-series per-label vectors
        *,
        series_mask: torch.Tensor,  # (B, S_seq) True = present
        family_id: torch.Tensor,  # (B, S_seq) long
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, Ns, L, D = v.shape

        if self.encoder is not None:
            # Contextualise the *series* summaries with each other before
            # pooling, so "no fluid on any sequence" can be represented.
            flat = v.mean(dim=2)  # (B, Ns, D)
            ctx = self.encoder(flat, src_key_padding_mask=~series_mask)
            v = v + ctx[:, :, None, :]

        q = self.query.to(v.dtype)
        logits = torch.einsum("ld,bsld->bls", q, self.proj(v)) * self.scale
        fam_bias = self.family_bias.to(v.dtype)[:, family_id]  # (L, B, Ns)
        logits = logits + fam_bias.permute(1, 0, 2)
        neg = torch.finfo(logits.dtype).min / 4
        logits = logits.masked_fill(~series_mask[:, None, :], neg)

        alpha = torch.softmax(logits, dim=-1)  # (B, L, Ns)
        z = torch.einsum("bls,bsld->bld", alpha, v)
        return self.drop(self.norm(z)), alpha


class LabelExpertRouter(nn.Module):
    r"""Top-:math:`k` sparse mixture of experts over the per-label features.

    Router logits :math:`r = W_g z_l`; the top-:math:`k` experts are used with
    renormalised softmax gates.  Two auxiliary terms:

    * **load balance** (Switch-Transformer):
      :math:`\mathcal L_{\text{bal}} = E\sum_e f_e P_e` where :math:`f_e` is the
      fraction of tokens routed to :math:`e` and :math:`P_e` the mean gate mass.
    * **router z-loss**: :math:`\mathbb E[(\log\sum_e e^{r_e})^2]`, which keeps
      the router logits small and prevents the numerical drift that shows up as
      a sudden expert collapse thousands of steps into training.

    ``label_prior`` biases the router towards the mechanism group of each label
    at initialisation, which cuts the warm-up period roughly in half and avoids
    the run-to-run variance of a randomly-initialised router.
    """

    def __init__(
        self,
        dim: int,
        num_labels: int,
        *,
        n_experts: int = 6,
        top_k: int = 2,
        expert_hidden: int | None = None,
        label_prior: torch.Tensor | None = None,  # (L,) long expert hint
        dropout: float = 0.1,
        capacity_factor: float = 1.5,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        hidden = expert_hidden or dim * 2
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
                )
                for _ in range(n_experts)
            ]
        )
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.label_bias = nn.Parameter(torch.zeros(num_labels, n_experts))
        if label_prior is not None:
            with torch.no_grad():
                self.label_bias.scatter_(1, label_prior.long()[:, None].clamp(0, n_experts - 1), 1.0)
        self.norm = nn.LayerNorm(dim)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``z``: ``(B, L, D)`` → ``(B, L, D)`` plus auxiliary losses."""
        B, L, D = z.shape
        h = self.norm(z)
        logits = self.gate(h) + self.label_bias.to(h.dtype)[None]  # (B, L, E)

        topv, topi = logits.topk(self.top_k, dim=-1)
        gates = torch.softmax(topv, dim=-1)

        out = torch.zeros_like(z)
        flat_h = h.reshape(-1, D)
        flat_i = topi.reshape(-1, self.top_k)
        flat_g = gates.reshape(-1, self.top_k)
        acc = torch.zeros_like(flat_h)
        for e, expert in enumerate(self.experts):
            sel = flat_i == e  # (N, k)
            if not bool(sel.any()):
                continue
            rows = sel.any(dim=1).nonzero(as_tuple=True)[0]
            g = (flat_g * sel.to(flat_g.dtype)).sum(dim=1)[rows]
            acc[rows] += g[:, None] * expert(flat_h[rows])
        out = acc.view(B, L, D)

        probs = torch.softmax(logits, dim=-1)
        with torch.no_grad():
            one_hot = torch.zeros_like(probs).scatter_(-1, topi, 1.0)
            f = one_hot.mean(dim=(0, 1))
        P = probs.mean(dim=(0, 1))
        balance = self.n_experts * (f * P).sum()
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()

        return z + out, {"balance": balance, "router_z": z_loss}


class LabelHeadBank(nn.Module):
    r"""Twelve independent affine heads with per-label temperature.

    Deliberately *not* a shared ``Linear(D, 12)``: a shared weight matrix makes
    the twelve logits linear functions of the same vector, which re-couples what
    the label queries just decoupled.  Each head sees only its own
    :math:`z_l`, so the only cross-label coupling left is what the router and
    the copula explicitly model.

    Per-label log-temperature is learnt but **frozen for the leaderboard
    submission**: AUC is invariant to a positive monotone rescale, so a learnt
    temperature cannot change the score -- it exists so that the distillation
    and the report-gating logic downstream see calibrated probabilities.
    """

    def __init__(self, dim: int, num_labels: int, *, hidden: int | None = None,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.num_labels = num_labels
        if hidden:
            self.pre = nn.ModuleList(
                [
                    nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
                                  nn.Dropout(dropout))
                    for _ in range(num_labels)
                ]
            )
            d_out = hidden
        else:
            self.pre = None
            d_out = dim
        self.w = nn.Parameter(torch.zeros(num_labels, d_out))
        self.b = nn.Parameter(torch.zeros(num_labels))
        self.log_temp = nn.Parameter(torch.zeros(num_labels))
        nn.init.normal_(self.w, std=d_out**-0.5)

    def forward(self, z: torch.Tensor, *, apply_temperature: bool = False) -> torch.Tensor:
        if self.pre is not None:
            z = torch.stack([m(z[:, l]) for l, m in enumerate(self.pre)], dim=1)
        logits = (z * self.w.to(z.dtype)[None]).sum(-1) + self.b.to(z.dtype)[None]
        if apply_temperature:
            logits = logits / self.log_temp.exp().clamp(0.25, 4.0).to(z.dtype)[None]
        return logits
