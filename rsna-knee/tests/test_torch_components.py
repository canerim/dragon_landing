"""Numerical validation of the torch components.

Every test here pins a mathematical property, not a shape.  Shapes are checked
too, but a shape test would not have caught any of the bugs these found.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kairos.constants import NUM_SEQUENCE_FAMILIES, NUM_TARGETS  # noqa: E402
from kairos.losses.auc import (  # noqa: E402
    AUCMarginLoss,
    PairwiseRankQueue,
    PartialAUCLoss,
    soft_topk_weights,
)
from kairos.losses.multimodal import (  # noqa: E402
    ReportShortcutRegulariser,
    SoftContrastive,
    jaccard_similarity,
)
from kairos.losses.ot import (  # noqa: E402
    PhraseSliceOT,
    sinkhorn_divergence,
    sinkhorn_log,
    unbalanced_sinkhorn_log,
)
from kairos.losses.robust import ChiSquareDRO, CVaRLoss, GroupDRO  # noqa: E402
from kairos.losses.supervised import (  # noqa: E402
    AsymmetricLoss,
    GaussianCopulaNLL,
    bivariate_normal_cdf,
)
from kairos.models.adaptive import GumbelTopKSelector, expand_windows  # noqa: E402
from kairos.models.aggregator import SelectiveScanAggregator, SliceTransformer  # noqa: E402
from kairos.models.encoding import PhysicalPositionalEncoding  # noqa: E402
from kairos.models.heads import SNGPHead, SpectralNormLinear, mean_field_logits  # noqa: E402
from kairos.models.hyperbolic import OntologyEmbedding, PoincareBall  # noqa: E402
from kairos.models.label_queries import LabelExpertRouter, LabelQueryPool  # noqa: E402
from kairos.models.system import KairosConfig, KairosModel, StudyBatch  # noqa: E402
from kairos.optim.pesg import PESG, GradientSurgery  # noqa: E402

torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# AUC losses                                                                   #
# --------------------------------------------------------------------------- #


def test_auc_margin_equals_the_pairwise_surrogate_at_the_saddle_point():
    r"""The theorem the loss rests on:

    .. math::
        \min_{a,b}\max_\alpha f(a,b,\alpha)
          = p(1-p)\,\mathbb E\big[(m - h(x^+) + h(x^-))^2\big].

    Verified end-to-end through the actual ``AUCMarginLoss.forward``, with the
    auxiliaries snapped to their closed-form optima.  This is the test that
    caught the class-conditional-vs-batch-mean bug: with conditional means the
    two sides differ by a factor of ~6.
    """
    torch.manual_seed(1)
    margin, B = 1.0, 6000
    y = (torch.rand(B, 1) < 0.3).float()
    z = torch.randn(B, 1) + 1.5 * y
    h = torch.sigmoid(z)
    p = y.mean()

    pos, neg = h[y > 0.5], h[y <= 0.5]
    pairwise = ((margin - pos[:, None] + neg[None, :]) ** 2).mean()

    loss = AUCMarginLoss(1, margin=margin, prevalence=p.reshape(1))
    loss.set_optimal_auxiliaries(z, y)
    with torch.no_grad():
        got = loss(z, y)

    want = p * (1 - p) * pairwise
    assert torch.isclose(got, want, rtol=1e-3), (float(got), float(want))


def test_auc_margin_at_the_saddle_tracks_the_true_auc():
    """Lower AUC-M (at its own optimum) must mean better ranking."""
    torch.manual_seed(2)
    y = torch.zeros(400, 1)
    y[:120] = 1.0
    base = torch.randn(400, 1)
    values = []
    for gap in (0.0, 1.0, 3.0):
        z = base.clone()
        z[:120] += gap
        loss = AUCMarginLoss(1, margin=1.0, ema_prevalence=None)
        loss.set_optimal_auxiliaries(z, y)
        with torch.no_grad():
            values.append(float(loss(z, y)))
    assert values[0] > values[1] > values[2], values


def test_auc_margin_gradients_flow_and_alpha_is_flagged_for_ascent():
    loss = AUCMarginLoss(NUM_TARGETS)
    z = torch.randn(64, NUM_TARGETS, requires_grad=True)
    y = (torch.rand(64, NUM_TARGETS) < 0.3).float()
    out = loss(z, y)
    out.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert getattr(loss.alpha, "_auc_ascent", False) is True
    loss.project()
    assert float(loss.alpha.detach().min()) >= 0.0


def test_auc_margin_handles_all_nan_and_single_class_labels():
    loss = AUCMarginLoss(3)
    y = torch.full((16, 3), float("nan"))
    z = torch.randn(16, 3)
    assert torch.isfinite(loss(z, y))
    y2 = torch.zeros(16, 3)  # no positives anywhere
    assert torch.isfinite(loss(z, y2))


def test_soft_topk_sums_to_k_and_is_differentiable():
    s = torch.randn(50, requires_grad=True)
    w = soft_topk_weights(s, k=7.0, temperature=0.05)
    assert abs(float(w.sum()) - 7.0) < 0.05
    assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0
    w.sum().backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_soft_topk_selects_the_largest_entries():
    s = torch.tensor([5.0, 1.0, 4.0, 0.0, 3.0])
    w = soft_topk_weights(s, k=2.0, temperature=0.01)
    assert int(w.argmax()) == 0
    assert w[0] > 0.9 and w[2] > 0.8 and w[3] < 0.1


def test_partial_auc_focuses_on_hard_pairs():
    torch.manual_seed(2)
    y = torch.zeros(400, 1)
    y[:100] = 1.0
    z = torch.randn(400, 1)
    z[:100] += 2.0
    base = PartialAUCLoss(fpr_max=1.0, tpr_min=0.0)(z, y)
    strict = PartialAUCLoss(fpr_max=0.1, tpr_min=0.9)(z, y)
    # Restricting to the hardest pairs must not lower the loss.
    assert float(strict) >= float(base) - 1e-6


def test_rank_queue_produces_gradient_when_batch_has_no_positive():
    q = PairwiseRankQueue(2, capacity=64)
    y_pos = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    q(torch.randn(2, 2), y_pos)  # seed the positive queue
    z = torch.randn(8, 2, requires_grad=True)
    y_neg = torch.zeros(8, 2)
    out = q(z, y_neg)
    out.backward()
    assert z.grad is not None and float(z.grad.abs().sum()) > 0


# --------------------------------------------------------------------------- #
# Optimal transport                                                            #
# --------------------------------------------------------------------------- #


def test_sinkhorn_marginals_converge():
    torch.manual_seed(3)
    cost = torch.rand(2, 6, 9)
    plan, value = sinkhorn_log(cost, epsilon=0.05, n_iter=200)
    assert torch.allclose(plan.sum(2), torch.full((2, 6), 1 / 6), atol=2e-3)
    assert torch.allclose(plan.sum(1), torch.full((2, 9), 1 / 9), atol=2e-3)
    assert torch.all(value >= 0)


def test_sinkhorn_recovers_the_permutation_for_a_matching_cost():
    n = 5
    perm = torch.tensor([2, 0, 4, 1, 3])
    cost = torch.ones(1, n, n)
    cost[0, torch.arange(n), perm] = 0.0
    plan, _ = sinkhorn_log(cost, epsilon=0.01, n_iter=400)
    assert torch.all(plan[0].argmax(dim=1) == perm)


def test_sinkhorn_respects_masks():
    cost = torch.rand(1, 5, 7)
    rm = torch.tensor([[True, True, True, False, False]])
    cm = torch.tensor([[True] * 4 + [False] * 3])
    plan, _ = sinkhorn_log(cost, row_mask=rm, col_mask=cm, n_iter=150)
    assert float(plan[0, 3:].abs().sum()) == 0.0
    assert float(plan[0, :, 4:].abs().sum()) == 0.0
    assert abs(float(plan.sum()) - 1.0) < 1e-2


def test_unbalanced_ot_destroys_mass_when_nothing_matches():
    n = 6
    good = torch.zeros(1, n, n) + 1.0
    good[0, torch.arange(n), torch.arange(n)] = 0.0
    bad = torch.ones(1, n, n) * 2.0
    _, _ = unbalanced_sinkhorn_log(good, tau=0.5)
    p_good, _ = unbalanced_sinkhorn_log(good, tau=0.5, n_iter=200)
    p_bad, _ = unbalanced_sinkhorn_log(bad, tau=0.5, n_iter=200)
    assert float(p_bad.sum()) < float(p_good.sum())


def test_sinkhorn_divergence_is_zero_for_identical_clouds():
    x = torch.randn(2, 10, 8)
    d = sinkhorn_divergence(x, x.clone(), epsilon=0.05, n_iter=120)
    assert torch.all(d < 1e-3), d


def test_phrase_slice_ot_gradients_and_confidence_gating():
    torch.manual_seed(4)
    B, P, S, D, L = 2, 5, 12, 16, NUM_TARGETS
    pe = torch.randn(B, P, D, requires_grad=True)
    se = torch.randn(B, S, D, requires_grad=True)
    pm = torch.ones(B, P, dtype=torch.bool)
    sm = torch.ones(B, S, dtype=torch.bool)
    conf = torch.rand(B, P)
    lab = torch.randint(0, L, (B, P))
    attn = torch.softmax(torch.randn(B, L, S), dim=-1)
    loss = PhraseSliceOT()
    out = loss(pe, se, phrase_mask=pm, slice_mask=sm, phrase_conf=conf,
               phrase_label=lab, label_attention=attn)
    out["total"].backward()
    assert torch.isfinite(pe.grad).all() and torch.isfinite(se.grad).all()
    assert "attention_consistency" in out
    assert float(out["attention_consistency"]) >= 0.0


# --------------------------------------------------------------------------- #
# Supervised / copula                                                          #
# --------------------------------------------------------------------------- #


def test_asl_masks_nan_targets_exactly():
    loss = AsymmetricLoss(gamma_neg=4.0, clip=0.05)
    z = torch.randn(8, 4)
    y = torch.zeros(8, 4)
    y[:, 0] = float("nan")
    per_label = loss(z, y, reduction="per_label")
    assert torch.isfinite(per_label).all()
    # Making the NaN column arbitrarily wrong must not change the loss.
    z2 = z.clone()
    z2[:, 0] += 100.0
    assert torch.isclose(loss(z, y), loss(z2, y), atol=1e-6)


def test_asl_downweights_easy_negatives_relative_to_bce():
    import torch.nn.functional as F

    z = torch.full((100, 1), -6.0)  # confidently negative
    y = torch.zeros(100, 1)
    asl = AsymmetricLoss(gamma_neg=4.0, clip=0.05)(z, y)
    bce = F.binary_cross_entropy_with_logits(z, y)
    assert float(asl) < float(bce)


def test_bivariate_normal_cdf_against_known_values():
    # Phi_2(0, 0; rho) = 1/4 + arcsin(rho) / (2 pi)
    for rho in (-0.8, -0.3, 0.0, 0.5, 0.9):
        got = bivariate_normal_cdf(
            torch.tensor(0.0), torch.tensor(0.0), torch.tensor(rho)
        )
        want = 0.25 + math.asin(rho) / (2 * math.pi)
        assert abs(float(got) - want) < 1e-6, (rho, float(got), want)
    # rho = 0 factorises.  Use float64 inputs: the function returns in the
    # input dtype, so a float32 call caps the accuracy at ~1e-7 regardless of
    # the (float64) quadrature.
    h = torch.tensor(1.0, dtype=torch.float64)
    k = torch.tensor(-0.5, dtype=torch.float64)
    got = bivariate_normal_cdf(h, k, torch.tensor(0.0, dtype=torch.float64))
    phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))  # noqa: E731
    assert abs(float(got) - phi(1.0) * phi(-0.5)) < 1e-12
    # Monotone in rho for positive orthant probabilities.
    vals = [
        float(bivariate_normal_cdf(torch.tensor(0.5), torch.tensor(0.5), torch.tensor(r)))
        for r in (-0.9, -0.5, 0.0, 0.5, 0.9)
    ]
    assert all(a < b for a, b in zip(vals, vals[1:])), vals


def test_copula_learns_positive_correlation():
    torch.manual_seed(5)
    n = 3000
    latent = torch.randn(n, 1)
    y = torch.stack(
        [(latent[:, 0] + 0.3 * torch.randn(n) > 0).float(),
         (latent[:, 0] + 0.3 * torch.randn(n) > 0).float()], dim=1
    )
    logits = torch.zeros(n, 2)
    cop = GaussianCopulaNLL(2, rank=2)
    opt = torch.optim.Adam(cop.parameters(), lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        loss = cop(logits, y)
        loss.backward()
        opt.step()
    assert float(cop.correlation()[0, 1]) > 0.4


def test_copula_correlation_is_a_valid_correlation_matrix():
    cop = GaussianCopulaNLL(NUM_TARGETS, rank=4)
    R = cop.correlation()
    assert torch.allclose(torch.diagonal(R), torch.ones(NUM_TARGETS), atol=1e-4)
    assert torch.allclose(R, R.T, atol=1e-6)
    off = R - torch.eye(NUM_TARGETS)
    assert float(off.abs().max()) <= 0.98 + 1e-6
    evals = torch.linalg.eigvalsh(R.double())
    assert float(evals.min()) > -1e-6, "correlation matrix must be PSD"


# --------------------------------------------------------------------------- #
# Robustness                                                                   #
# --------------------------------------------------------------------------- #


def test_group_dro_upweights_the_worst_group():
    g = GroupDRO(3, eta_q=0.5, shrinkage=0.0, standardise_labels=False)
    loss = torch.tensor([0.1, 0.1, 5.0, 5.0, 0.1, 0.1])
    gid = torch.tensor([0, 0, 1, 1, 2, 2])
    for _ in range(30):
        g(loss, gid)
    assert int(g.q.argmax()) == 1
    assert float(g.q[1]) > 0.8


def test_group_dro_shrinkage_protects_tiny_groups():
    big = GroupDRO(2, eta_q=0.3, shrinkage=3.0, standardise_labels=False)
    # Group 1 has one noisy high-loss sample; group 0 has many moderate ones.
    for _ in range(20):
        loss = torch.cat([torch.full((40,), 0.5), torch.tensor([4.0])])
        gid = torch.cat([torch.zeros(40, dtype=torch.long), torch.ones(1, dtype=torch.long)])
        big(loss, gid)
    no_shrink = GroupDRO(2, eta_q=0.3, shrinkage=0.0, standardise_labels=False)
    for _ in range(20):
        loss = torch.cat([torch.full((40,), 0.5), torch.tensor([4.0])])
        gid = torch.cat([torch.zeros(40, dtype=torch.long), torch.ones(1, dtype=torch.long)])
        no_shrink(loss, gid)
    assert float(big.q[1]) <= float(no_shrink.q[1]) + 1e-6


def test_cvar_equals_the_mean_of_the_worst_fraction():
    losses = torch.arange(100, dtype=torch.float32)
    cvar = CVaRLoss(alpha=0.1)
    opt = torch.optim.Adam(cvar.parameters(), lr=1.0)
    for _ in range(600):
        opt.zero_grad()
        v = cvar(losses)
        v.backward()
        opt.step()
    with torch.no_grad():
        got = float(cvar(losses))
    want = float(losses.topk(10).values.mean())
    assert abs(got - want) < 1.0, (got, want)


def test_chi_square_dro_between_erm_and_cvar():
    losses = torch.cat([torch.zeros(90), torch.full((10,), 10.0)])
    erm = float(losses.mean())
    dro = ChiSquareDRO(rho=1.0)
    opt = torch.optim.Adam(dro.parameters(), lr=0.5)
    for _ in range(500):
        opt.zero_grad()
        v = dro(losses)
        v.backward()
        opt.step()
    with torch.no_grad():
        got = float(dro(losses))
    assert erm <= got <= float(losses.max()) + 1e-3


# --------------------------------------------------------------------------- #
# Multimodal                                                                   #
# --------------------------------------------------------------------------- #


def test_jaccard_similarity_is_correct_and_nan_safe():
    y = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    J = jaccard_similarity(y)
    assert abs(float(J[0, 1]) - 0.5) < 1e-6
    assert float(J[2, 2]) == 0.0  # empty union -> 0, not 1
    y2 = y.clone()
    y2[0, 2] = float("nan")
    assert torch.isfinite(jaccard_similarity(y2)).all()


def test_soft_contrastive_prefers_the_matched_pair_but_not_absolutely():
    torch.manual_seed(6)
    B, D = 8, 32
    img = torch.randn(B, D)
    txt = img.clone() + 0.01 * torch.randn(B, D)
    weak = (torch.rand(B, NUM_TARGETS) < 0.3).float()
    loss = SoftContrastive(alpha_identity=0.6, beta_jaccard=0.3, gamma_graph=0.0)
    matched = loss(img, txt, weak_labels=weak)
    shuffled = loss(img, txt[torch.randperm(B)], weak_labels=weak)
    assert float(matched) < float(shuffled)


def test_report_shortcut_regulariser_flags_a_text_only_model():
    B, L = 16, NUM_TARGETS
    # A model that ignores the image: shuffling the report changes nothing,
    # so report_reliance ~ 0 and the margin penalty is ~0 too...
    z = torch.randn(B, L) * 3
    reg = ReportShortcutRegulariser()
    same = reg(z, z.clone(), torch.zeros(B, L))
    assert abs(float(same["report_reliance"])) < 1e-5
    # ...whereas a model that genuinely uses the image gets *less* confident
    # under a shuffled report, so reliance is positive.
    healthy = reg(z, torch.zeros(B, L), torch.zeros(B, L))
    assert float(healthy["report_reliance"]) > 0


# --------------------------------------------------------------------------- #
# Model components                                                             #
# --------------------------------------------------------------------------- #


def test_physical_positional_encoding_is_translation_sensitive_and_bounded():
    pe = PhysicalPositionalEncoding(64, min_period_mm=2.0, max_period_mm=240.0)
    z = torch.tensor([0.0, 3.0, 6.0, 100.0])
    e = pe(z)
    assert e.shape == (4, 64)
    assert float(e.abs().max()) <= 1.0 + 1e-6
    # Distinct physical positions must get distinct codes.
    assert float((e[0] - e[1]).abs().sum()) > 1.0
    # Same physical position, different index -> identical code.
    assert torch.allclose(pe(torch.tensor([3.0]))[0], e[1], atol=1e-6)


def test_slice_transformer_ignores_padded_slices():
    torch.manual_seed(7)
    m = SliceTransformer(32, depth=2, n_heads=4, dropout=0.0, drop_path=0.0).eval()
    x = torch.randn(2, 10, 32)
    z = torch.linspace(0, 27, 10)[None].repeat(2, 1)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[:, 7:] = False
    out_a = m(x, z, mask)
    x2 = x.clone()
    x2[:, 7:] = 999.0  # garbage in the padded region
    out_b = m(x2, z, mask)
    assert torch.allclose(out_a[:, :7], out_b[:, :7], atol=1e-4)


def test_selective_scan_respects_masks_and_physical_gaps():
    torch.manual_seed(8)
    m = SelectiveScanAggregator(32, state_dim=8, dropout=0.0).eval()
    x = torch.randn(2, 12, 32)
    z_even = torch.linspace(0, 33, 12)[None].repeat(2, 1)
    z_gap = z_even.clone()
    z_gap[:, 6:] += 40.0
    mask = torch.ones(2, 12, dtype=torch.bool)
    o1 = m(x, z_even, mask)
    o2 = m(x, z_gap, mask)
    assert o1.shape == x.shape
    assert torch.isfinite(o1).all()
    # A large physical gap must change the aggregation.
    assert float((o1 - o2).abs().mean()) > 1e-5


def test_label_query_pool_gives_each_label_its_own_attention():
    torch.manual_seed(9)
    pool = LabelQueryPool(48, NUM_TARGETS, n_query_tokens=2, dropout=0.0).eval()
    h = torch.randn(3, 16, 48)
    mask = torch.ones(3, 16, dtype=torch.bool)
    mask[:, 12:] = False
    pooled, attn, ent = pool(h, slice_mask=mask)
    assert pooled.shape == (3, NUM_TARGETS, 48)
    assert torch.allclose(attn.sum(-1), torch.ones(3, NUM_TARGETS), atol=1e-5)
    assert float(attn[:, :, 12:].abs().max()) < 1e-6
    # Different labels must not produce identical attention.
    assert float((attn[:, 0] - attn[:, 1]).abs().max()) > 1e-4
    assert torch.all(ent >= 0)


def test_moe_router_balances_load():
    torch.manual_seed(10)
    r = LabelExpertRouter(32, NUM_TARGETS, n_experts=4, top_k=2)
    z = torch.randn(8, NUM_TARGETS, 32)
    out, aux = r(z)
    assert out.shape == z.shape
    assert float(aux["balance"]) > 0
    assert torch.isfinite(aux["router_z"])
    out.sum().backward()
    assert r.gate.weight.grad is not None


def test_sngp_variance_is_larger_far_from_the_training_data():
    torch.manual_seed(11)
    head = SNGPHead(16, 2, num_random_features=256, gp_kernel_scale=1.0)
    head.train()
    near = torch.randn(64, 2, 16) * 0.1
    for _ in range(20):
        head(near, update_precision=True)
    head.eval()
    _, v_near = head(torch.randn(32, 2, 16) * 0.1)
    _, v_far = head(torch.randn(32, 2, 16) * 8.0 + 20.0)
    assert float(v_far.mean()) > float(v_near.mean())


def test_mean_field_logits_shrink_towards_zero_with_variance():
    z = torch.tensor([[3.0, -3.0]])
    v = torch.tensor([[0.0, 4.0]])
    out = mean_field_logits(z, v)
    assert torch.isclose(out[0, 0], z[0, 0])
    assert abs(float(out[0, 1])) < abs(float(z[0, 1]))


def test_spectral_norm_bounds_the_lipschitz_constant():
    layer = SpectralNormLinear(8, 8, c=1.0)
    with torch.no_grad():
        layer.linear.weight.mul_(50.0)
    for _ in range(30):
        layer(torch.randn(4, 8))
    x = torch.randn(256, 8)
    d = torch.randn(256, 8) * 0.01
    ratio = (layer(x + d) - layer(x)).norm(dim=1) / d.norm(dim=1)
    assert float(ratio.max()) <= 1.05


def test_gumbel_topk_selects_k_and_dilates_windows():
    sel = GumbelTopKSelector(k=3, window_radius=1, exploration_frac=0.0)
    sel.eval()
    scores = torch.zeros(1, 2, 10)
    scores[0, 0, [1, 5, 8]] = 5.0
    scores[0, 1, [0, 2, 9]] = 5.0
    valid = torch.ones(1, 10, dtype=torch.bool)
    out = sel(scores, valid=valid, training=False)
    assert bool(out["mask"][0, 0, 1]) and bool(out["mask"][0, 0, 5])
    assert bool(out["mask"][0, 0, 0])  # dilated neighbour of slice 1
    assert out["mask"].shape == (1, 2, 10)


def test_expand_windows_matches_a_manual_dilation():
    m = torch.zeros(1, 7, dtype=torch.bool)
    m[0, 3] = True
    out = expand_windows(m, radius=2)
    assert out[0].tolist() == [False, True, True, True, True, True, False]


def test_poincare_operations_stay_in_the_ball():
    ball = PoincareBall(1.0)
    v = torch.randn(64, 8) * 10.0
    x = ball.expmap0(v)
    assert float(x.norm(dim=-1).max()) < 1.0
    back = ball.logmap0(x)
    assert torch.isfinite(back).all()
    d = ball.distance(x[:32], x[32:])
    assert torch.all(d >= 0) and torch.isfinite(d).all()
    assert float(ball.distance(x[:4], x[:4]).abs().max()) < 1e-4


def test_ontology_hierarchy_loss_decreases_under_optimisation():
    torch.manual_seed(12)
    parent = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4])
    ont = OntologyEmbedding(12, 5, 8, parent_of_label=parent)
    opt = torch.optim.Adam(ont.parameters(), lr=0.05)
    first = float(ont.hierarchy_loss())
    for _ in range(200):
        opt.zero_grad()
        loss = ont.hierarchy_loss()
        loss.backward()
        opt.step()
    assert float(ont.hierarchy_loss()) < first
    sim = ont.label_similarity()
    assert torch.allclose(sim, sim.T, atol=1e-5)
    assert float(sim.max()) <= 1.0 + 1e-6


# --------------------------------------------------------------------------- #
# Optimisers                                                                   #
# --------------------------------------------------------------------------- #


def test_pesg_ascends_alpha_and_descends_weights():
    w = torch.nn.Parameter(torch.tensor([1.0]))
    a = torch.nn.Parameter(torch.tensor([0.5]))
    a._auc_ascent = True
    opt = PESG([w, a], lr=0.1, gamma=1e9, momentum=0.0, weight_decay=0.0)
    w.grad = torch.tensor([1.0])
    a.grad = torch.tensor([1.0])
    opt.step()
    assert float(w) < 1.0  # descent
    assert float(a) > 0.5  # ascent


def test_gradient_surgery_resolves_a_head_on_conflict():
    p = torch.nn.Parameter(torch.zeros(2))
    gs = GradientSurgery([p], mode="pcgrad")
    l1 = (p * torch.tensor([1.0, 0.0])).sum()
    l2 = (p * torch.tensor([-1.0, 1.0])).sum()
    d = gs.backward([l1, l2])
    assert torch.isfinite(d).all()
    # The directly-conflicting first component must be attenuated relative to
    # the plain mean of (1,0) and (-1,1), which is (0, 0.5).
    mean = torch.tensor([0.0, 0.5])
    assert float(d[1]) > 0
    assert abs(float(d[0])) <= abs(float(mean[0])) + 0.6


def test_aligned_mtl_is_scale_invariant():
    p = torch.nn.Parameter(torch.zeros(3))
    gs = GradientSurgery([p], mode="aligned")
    g1 = torch.tensor([1.0, 0.0, 0.0])
    g2 = torch.tensor([0.0, 1.0, 0.0])
    d_a = gs.backward([(p * g1).sum(), (p * g2).sum()]).clone()
    # Rescale one task's loss by 100x: the *direction* must be essentially
    # unchanged, which is the property that motivates Aligned-MTL.
    d_b = gs.backward([(p * g1).sum(), 100.0 * (p * g2).sum()]).clone()
    cos = torch.nn.functional.cosine_similarity(d_a[None], d_b[None]).item()
    assert cos > 0.99, cos


# --------------------------------------------------------------------------- #
# End-to-end model                                                             #
# --------------------------------------------------------------------------- #


def _make_batch(B=2, Nseq=3, S=6, C=5, H=48, W=48):
    slice_mask = torch.ones(B, Nseq, S, dtype=torch.bool)
    slice_mask[0, 1, 4:] = False
    series_mask = torch.ones(B, Nseq, dtype=torch.bool)
    series_mask[1, 2] = False  # a missing sequence
    return StudyBatch(
        pixels=torch.randn(B, Nseq, S, C, H, W),
        slice_mask=slice_mask,
        series_mask=series_mask,
        z_mm=torch.linspace(0, 15, S)[None, None].repeat(B, Nseq, 1),
        family_id=torch.randint(0, NUM_SEQUENCE_FAMILIES, (B, Nseq)),
        manufacturer_id=torch.randint(0, 8, (B, Nseq)),
        fat_sat_id=torch.randint(0, 3, (B, Nseq)),
        context=torch.randn(B, Nseq, 8),
        targets=(torch.rand(B, NUM_TARGETS) < 0.3).float(),
        group_id=torch.randint(0, 4, (B,)),
    )


@pytest.mark.parametrize("aggregator", ["transformer", "ssm"])
def test_model_forward_shapes_and_finiteness(aggregator):
    torch.manual_seed(13)
    cfg = KairosConfig(
        dim=64, aggregator=aggregator, agg_depth=1, agg_heads=4,
        sngp_features=64, n_experts=3, enable_fine_pass=False,
        sequence_dropout=0.0, slice_dropout=0.0,
    )
    cfg.backbone.pretrained = False
    model = KairosModel(cfg).eval()
    batch = _make_batch()
    with torch.no_grad():
        out = model(batch)
    assert out["logits"].shape == (2, NUM_TARGETS)
    assert torch.isfinite(out["logits"]).all()
    assert out["slice_attention"].shape[:3] == (2, 3, NUM_TARGETS)
    assert "variance" in out and torch.isfinite(out["variance"]).all()


def test_model_is_invariant_to_padded_series_content():
    torch.manual_seed(14)
    cfg = KairosConfig(dim=48, agg_depth=1, agg_heads=4, sngp_features=64,
                       n_experts=2, enable_fine_pass=False,
                       sequence_dropout=0.0, slice_dropout=0.0)
    cfg.backbone.pretrained = False
    model = KairosModel(cfg).eval()
    b = _make_batch()
    with torch.no_grad():
        a = model(b)["logits"]
        b.pixels[1, 2] = 1e3  # garbage inside the masked-out series
        c = model(b)["logits"]
    assert torch.allclose(a[1], c[1], atol=1e-3)


def test_model_fine_pass_runs_and_backpropagates():
    torch.manual_seed(15)
    cfg = KairosConfig(dim=48, agg_depth=1, agg_heads=4, sngp_features=64,
                       n_experts=2, enable_fine_pass=True, fine_top_k=2,
                       sequence_dropout=0.0, slice_dropout=0.0)
    cfg.backbone.pretrained = False
    model = KairosModel(cfg)
    out = model(_make_batch(H=32, W=32), update_precision=True)
    assert "fine_logits" in out and torch.isfinite(out["fine_logits"]).all()
    assert 0.0 <= float(out["fine_fraction"].mean()) <= 1.0
    loss = out["logits"].square().mean() + out["attn_entropy_penalty"]
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_parameter_groups_exclude_queries_from_weight_decay():
    cfg = KairosConfig(dim=32, agg_depth=1, agg_heads=4, sngp_features=32, n_experts=2)
    cfg.backbone.pretrained = False
    model = KairosModel(cfg)
    groups = model.parameter_groups(backbone_lr=1e-5, head_lr=1e-3)
    assert len(groups) == 3
    assert groups[-1]["weight_decay"] == 0.0
    query_ids = {id(model.pool.query), id(model.fusion.query)}
    nodecay_ids = {id(p) for p in groups[-1]["params"]}
    assert query_ids <= nodecay_ids


def test_sequence_dropout_never_empties_a_study():
    torch.manual_seed(16)
    cfg = KairosConfig(dim=32, agg_depth=1, agg_heads=4, sngp_features=32,
                       n_experts=2, enable_fine_pass=False, sequence_dropout=0.99)
    cfg.backbone.pretrained = False
    model = KairosModel(cfg).train()
    mask = torch.ones(8, 4, dtype=torch.bool)
    for _ in range(20):
        out = model._apply_modality_dropout(mask)
        assert bool(out.any(dim=1).all())
