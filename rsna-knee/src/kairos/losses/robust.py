r"""Distributional robustness across sites, scanners and languages.

The training set comes from 16 centres on five continents.  The private test
set is drawn from the same pool but *not necessarily in the same proportions*,
and the label prevalences almost certainly differ.  Empirical risk minimisation
optimises the average over the training mixture, which is the wrong quantity if
the test mixture is different: a model can be excellent on the two sites that
supply 40 % of the data and mediocre everywhere else while showing a perfectly
healthy training curve.

Three complementary robustifiers are implemented.

``GroupDRO``
    Minimises :math:`\max_{k} \mathcal R_k` over a set of *known* groups (site
    × language × scanner), solved by online exponentiated gradient ascent on
    the group weights (Sagawa et al., 2020):

    .. math::
        q^{(t+1)}_k \propto q^{(t)}_k \exp(\eta_q \widehat{\mathcal R}_k^{(t)}).

    Two things make the vanilla version fail on this dataset and both are
    fixed here: (i) a group with 30 studies has a risk estimate so noisy that
    it captures all the weight, so we shrink each group's risk towards the mean
    by its own standard error; (ii) worst-group risk is dominated by *label
    difficulty*, not by domain shift, so risks are standardised per label
    before the max is taken.

``CVaRLoss``
    Distribution-free alternative when groups are unknown or too fine: the
    conditional value-at-risk at level :math:`\alpha`, i.e. the mean of the
    worst :math:`\alpha`-fraction of per-example losses.  Computed by the dual
    form :math:`\min_\lambda \; \lambda + \tfrac1\alpha \mathbb E[(\ell -
    \lambda)_+]`, which is convex in :math:`\lambda` and needs no sorting.

``ChiSquareDRO``
    Smooth interpolation between ERM and CVaR: worst-case risk over an
    :math:`f`-divergence ball, with the closed-form dual
    :math:`\min_\eta\; \sqrt{1+2\rho}\,\lVert(\ell-\eta)_+\rVert_2 + \eta`.
    Less brittle than CVaR because the implied weights decay smoothly rather
    than being a hard indicator, so a single mislabelled study cannot own the
    gradient.

Empirically the right recipe is ERM for the first third of training (you cannot
robustify a model that has not yet learnt the task), then a linear ramp to
``0.3 · GroupDRO + 0.1 · ChiSquareDRO``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GroupDRO", "CVaRLoss", "ChiSquareDRO", "IRMPenalty"]

_EPS = 1e-12


class GroupDRO(nn.Module):
    r"""Online group distributionally-robust optimisation.

    Parameters
    ----------
    num_groups
        Size of the group vocabulary (site × language × scanner bucket).
    eta_q
        Step size of the exponentiated-gradient ascent on the group weights.
        0.01–0.05.  Larger values chase noise.
    shrinkage
        Coefficient :math:`c` in the standard-error shrinkage
        :math:`\tilde{\mathcal R}_k = \mathcal R_k -
        c\,\hat\sigma_k/\sqrt{n_k}`.  Setting ``c = 0`` recovers vanilla
        GroupDRO; ``c = 1`` is a one-sigma penalty on small groups and is a
        sane default.  This is the same correction as in group-adjusted DRO
        and it is not optional at this data scale.
    standardise_labels
        Divide each label's risk by its running mean before aggregating, so
        that "worst group" means *relatively* worst rather than "the group that
        happens to contain the hard labels".
    """

    def __init__(
        self,
        num_groups: int,
        *,
        eta_q: float = 0.02,
        shrinkage: float = 1.0,
        standardise_labels: bool = True,
        num_labels: int | None = None,
        ema: float = 0.98,
    ) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.eta_q = eta_q
        self.shrinkage = shrinkage
        self.standardise_labels = standardise_labels
        self.ema = ema
        self.register_buffer("log_q", torch.zeros(num_groups))
        self.register_buffer("group_count", torch.zeros(num_groups))
        if standardise_labels:
            if num_labels is None:
                raise ValueError("num_labels required when standardise_labels=True")
            self.register_buffer("label_scale", torch.ones(num_labels))

    @property
    def q(self) -> torch.Tensor:
        return torch.softmax(self.log_q, dim=0)

    def forward(
        self,
        per_example_loss: torch.Tensor,  # (B,) or (B, L)
        group_id: torch.Tensor,  # (B,) long
        *,
        update: bool = True,
    ) -> torch.Tensor:
        loss = per_example_loss
        if loss.dim() == 2:
            if self.standardise_labels:
                if update:
                    with torch.no_grad():
                        m = loss.detach().mean(dim=0).clamp_min(1e-6)
                        self.label_scale.mul_(self.ema).add_(m, alpha=1 - self.ema)
                loss = loss / self.label_scale.clamp_min(1e-6).to(loss.dtype)
            loss = loss.mean(dim=1)

        G = self.num_groups
        idx = group_id.long().clamp(0, G - 1)
        ones = torch.ones_like(loss)

        n_k = torch.zeros(G, device=loss.device, dtype=loss.dtype).index_add_(0, idx, ones)
        s_k = torch.zeros(G, device=loss.device, dtype=loss.dtype).index_add_(0, idx, loss)
        sq_k = torch.zeros(G, device=loss.device, dtype=loss.dtype).index_add_(
            0, idx, loss.detach() ** 2
        )
        present = n_k > 0
        mean_k = s_k / n_k.clamp_min(_EPS)

        if update:
            with torch.no_grad():
                var_k = (sq_k / n_k.clamp_min(_EPS) - mean_k.detach() ** 2).clamp_min(0.0)
                self.group_count.mul_(self.ema).add_(n_k.detach(), alpha=1 - self.ema)
                n_eff = self.group_count.clamp_min(1.0)
                adj = mean_k.detach() - self.shrinkage * torch.sqrt(var_k / n_eff)
                adj = torch.where(present, adj, torch.zeros_like(adj))
                self.log_q.add_(self.eta_q * adj)
                self.log_q.sub_(self.log_q.max())

        q = torch.softmax(self.log_q, dim=0).to(loss.dtype)
        q = q * present.to(loss.dtype)
        q = q / q.sum().clamp_min(_EPS)
        return (q * mean_k).sum()


class CVaRLoss(nn.Module):
    r"""Conditional value-at-risk of the per-example loss, dual form.

    .. math::
        \mathrm{CVaR}_\alpha(\ell) = \min_\lambda \Big\{
            \lambda + \tfrac{1}{\alpha}\,\mathbb E\big[(\ell - \lambda)_+\big]
        \Big\}.

    :math:`\lambda` is a learnable scalar optimised jointly with the network,
    which is valid because the objective is jointly convex in
    :math:`(\lambda, \ell)` for fixed :math:`\theta` and the envelope theorem
    gives the correct gradient w.r.t. :math:`\theta`.
    """

    def __init__(self, alpha: float = 0.2, *, init_lambda: float = 0.0) -> None:
        super().__init__()
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.lam = nn.Parameter(torch.tensor(float(init_lambda)))

    def forward(self, per_example_loss: torch.Tensor) -> torch.Tensor:
        loss = per_example_loss.flatten()
        return self.lam + F.relu(loss - self.lam).mean() / self.alpha


class ChiSquareDRO(nn.Module):
    r"""Worst-case risk over a :math:`\chi^2` ball of radius :math:`\rho`.

    .. math::
        \sup_{D_{\chi^2}(q\|p)\le\rho} \mathbb E_q[\ell]
        = \min_\eta \Big\{ \sqrt{1 + 2\rho}\;
          \big\lVert (\ell - \eta)_+ \big\rVert_2 + \eta \Big\}.

    The induced example weights are :math:`w_i \propto (\ell_i - \eta)_+`,
    i.e. linear rather than 0/1 in the excess loss.  That single change is why
    :math:`\chi^2`-DRO tolerates label noise where CVaR does not: a study with
    a wrong label gets a *large but finite* weight instead of the entire
    budget.
    """

    def __init__(self, rho: float = 1.0, *, init_eta: float = 0.0) -> None:
        super().__init__()
        self.rho = float(rho)
        self.eta = nn.Parameter(torch.tensor(float(init_eta)))

    def forward(self, per_example_loss: torch.Tensor) -> torch.Tensor:
        loss = per_example_loss.flatten()
        n = loss.numel()
        excess = F.relu(loss - self.eta)
        rms = torch.sqrt((excess**2).sum() / max(n, 1) + _EPS)
        return (1.0 + 2.0 * self.rho) ** 0.5 * rms + self.eta


class IRMPenalty(nn.Module):
    r"""IRMv1 gradient-norm penalty (Arjovsky et al., 2019).

    .. math::
        \sum_{e} \big\lVert \nabla_{w|w=1}\,
            \mathcal R_e(w \cdot f_\theta) \big\rVert^2

    with a dummy scalar classifier :math:`w = 1`.

    The environments here are *acquisition* environments -- the (site, scanner
    vendor, field strength) buckets stamped into ``StudyRecord.env_index`` --
    and the invariance we are asking for is that the optimal rescaling of the
    logits be the same at every centre.  A feature that needs a different gain
    at Site 3 than at Site 7 is a feature about Site 3, not about the knee, and
    on a 16-site test set that is exactly what does not transfer.

    Scope, stated plainly because the alternative is a claim we have not
    measured: this is applied to the image classifier, it is scheduled in S4
    only, and we have no ablation of our own that isolates its contribution --
    the justification is Arjovsky et al.'s, plus the fact that the competition
    metric is macro-AUC over a site distribution that differs from training.

    Ramp the weight in slowly (0 for the first 2 epochs, then to 1e2–1e4): a
    large IRM penalty from step 0 prevents the model from learning anything at
    all, a failure mode that looks like a bad learning rate.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        env_id: torch.Tensor,
        *,
        base_loss: str = "bce",
    ) -> torch.Tensor:
        scale = torch.ones(1, device=logits.device, dtype=logits.dtype, requires_grad=True)
        y = torch.nan_to_num(targets, nan=0.0)
        valid = torch.isfinite(targets).to(logits.dtype)

        penalties = []
        for e in torch.unique(env_id):
            sel = env_id == e
            if sel.sum() < 2:
                continue
            z = logits[sel] * scale
            if base_loss == "bce":
                le = F.binary_cross_entropy_with_logits(
                    z, y[sel], weight=valid[sel], reduction="sum"
                ) / valid[sel].sum().clamp_min(1.0)
            else:
                le = ((z - y[sel]) ** 2 * valid[sel]).sum() / valid[sel].sum().clamp_min(1.0)
            (g,) = torch.autograd.grad(le, [scale], create_graph=True)
            penalties.append((g**2).sum())

        if not penalties:
            return logits.sum() * 0.0
        return self.weight * torch.stack(penalties).mean()
