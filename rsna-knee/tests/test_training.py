"""Training wiring: the objective registry, schedule validation, and learning.

The centrepiece is :func:`test_model_can_overfit_a_tiny_dataset`. Everything
else in the suite checks that a component is individually correct; that test
checks that the *assembled* system has an intact gradient path from the loss
back through the SNGP head, the MoE router, the ontology prior, the
cross-sequence fusion, the label queries, the slice aggregator, the FiLM
conditioning and the backbone. A model that cannot overfit twelve studies has a
break somewhere in that chain, and no hyperparameter search will find it.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kairos.constants import NUM_TARGETS, Plane  # noqa: E402
from kairos.data.dataset import SeriesRecord, StudyRecord, collate_studies  # noqa: E402
from kairos.data.loader import (  # noqa: E402
    ClassAwareBatchSampler,
    SamplerConfig,
    StudyDataset,
    infer_prevalence,
)
from kairos.data.transforms import (  # noqa: E402
    MEDIAL_LATERAL_SWAP,
    AugmentConfig,
    SeriesAugmenter,
    bias_field,
)
from kairos.eval.metrics import macro_auc  # noqa: E402
from kairos.losses.supervised import AsymmetricLoss  # noqa: E402
from kairos.models.backbones import BackboneSpec  # noqa: E402
from kairos.models.system import KairosConfig, KairosModel  # noqa: E402
from kairos.train.curriculum import LossSchedule, default_plan  # noqa: E402
from kairos.train.loop import TrainConfig, Trainer  # noqa: E402
from kairos.train.objectives import (  # noqa: E402
    MissingObjectiveError,
    ObjectiveConfig,
    build_objectives,
    validate_schedule,
)
from kairos.train.ssl import CrossPlaneConsistency, MaskedTokenModelling  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _record(i, labels, *, size=32, n_slices=6, n_series=1, planted=True, seed=0):
    rng = np.random.default_rng(seed + i)
    series = []
    for s in range(n_series):
        v = rng.normal(0, 0.3, (n_slices, size, size)).astype(np.float32)
        if planted:
            for l in range(NUM_TARGETS):
                if labels[i, l] > 0.5:
                    r0 = (l * 2) % (size - 4)
                    v[:, r0 : r0 + 3, 3:8] += 4.0
        series.append(SeriesRecord(
            f"{i}.{s}", 1 + s, Plane.SAGITTAL, v,
            np.arange(n_slices, dtype=np.float32) * 3.0, 3.0, 0.5,
        ))
    r = StudyRecord(study_uid=f"s{i}", series=series)
    r.labels = labels[i]
    # Stamp the cohort indices the way the dataloader does, so Group-DRO and
    # IRM are actually feedable in tests that build batches by hand.
    r.group_index = i % 3
    r.env_index = i % 3
    return r


def _tiny_model(**kw):
    cfg = KairosConfig(
        backbone=BackboneSpec(pretrained=False, in_chans=5),
        dim=kw.pop("dim", 64), agg_depth=1, agg_heads=4, sngp_features=64,
        n_experts=2, enable_fine_pass=kw.pop("fine", False),
        sequence_dropout=0.0, slice_dropout=0.0, dropout=0.0, **kw,
    )
    return KairosModel(cfg)


# --------------------------------------------------------------------------- #
# The decisive test                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_model_can_overfit_a_tiny_dataset():
    """The whole assembled graph must be able to memorise 12 studies.

    Not a performance claim -- a wiring claim. If this regresses, a gradient
    has been detached somewhere between the loss and the backbone.
    """
    torch.manual_seed(0)
    torch.set_num_threads(4)
    rng = np.random.default_rng(0)
    N = 12
    labels = (rng.random((N, NUM_TARGETS)) < 0.4).astype(np.float32)
    batch = collate_studies([_record(i, labels) for i in range(N)], device="cpu")

    model = _tiny_model(dim=96)
    opt = torch.optim.AdamW(
        model.parameter_groups(backbone_lr=1e-3, head_lr=5e-3), betas=(0.9, 0.95)
    )
    asl = AsymmetricLoss(gamma_neg=2.0, clip=0.0)

    first_loss = None
    for step in range(101):
        out = model(batch, update_precision=(step % 20 == 0))
        loss = asl(out["logits"], batch.targets) + 0.02 * out["attn_entropy_penalty"]
        if first_loss is None:
            first_loss = float(loss.detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        auc = macro_auc(labels, torch.sigmoid(model(batch)["logits"]).numpy())
    assert float(loss.detach()) < 0.5 * first_loss, "loss must fall substantially"
    assert auc > 0.90, f"assembled model cannot overfit (macro-AUC {auc:.4f})"


def test_every_registered_module_receives_gradient():
    """No registered parameter may be orphaned from the loss."""
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    labels = (rng.random((4, NUM_TARGETS)) < 0.4).astype(np.float32)
    batch = collate_studies([_record(i, labels) for i in range(4)], device="cpu")

    model = _tiny_model(fine=True)
    out = model(batch, update_precision=True)
    loss = (
        out["logits"].square().mean()
        + out["attn_entropy_penalty"]
        + out["ontology_hierarchy"]
        + out["moe_balance"]
        + out["selector_budget"]
    )
    loss.backward()

    orphans = [
        n for n, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
    ]
    # The GP random-feature projection is a fixed buffer by design; anything
    # else without a gradient is a bug.
    orphans = [n for n in orphans if "rff_" not in n]
    assert not orphans, f"parameters with no finite gradient: {orphans[:10]}"


# --------------------------------------------------------------------------- #
# Objective registry and schedule validation                                   #
# --------------------------------------------------------------------------- #


def test_registry_covers_every_curriculum_term():
    reg = build_objectives(ObjectiveConfig(prevalence=[0.2] * NUM_TARGETS))
    missing = set(LossSchedule.TERMS) - set(reg)
    assert not missing, f"curriculum terms with no objective: {sorted(missing)}"


def test_validation_raises_when_a_scheduled_term_is_unregistered():
    plan = default_plan(steps_per_epoch=4, budget="small")
    sched = LossSchedule(plan)
    reg = build_objectives(ObjectiveConfig(prevalence=[0.2] * NUM_TARGETS))
    crippled = {k: v for k, v in reg.items() if k != "asl"}
    with pytest.raises(MissingObjectiveError, match="asl"):
        validate_schedule(crippled, sched, strict=True)


def test_validation_catches_a_scheduled_term_the_batch_cannot_feed():
    """The bug this exists for: kd scheduled, no teacher logits, silent skip."""
    rng = np.random.default_rng(2)
    labels = (rng.random((2, NUM_TARGETS)) < 0.4).astype(np.float32)
    batch = collate_studies([_record(i, labels) for i in range(2)], device="cpu")
    assert batch.teacher_logits is None

    sched = LossSchedule(default_plan(steps_per_epoch=4, budget="small"))
    reg = build_objectives(ObjectiveConfig(prevalence=[0.2] * NUM_TARGETS))
    problems = validate_schedule(reg, sched, sample_batch=batch, strict=False)
    names = {p.split("'")[1] for p in problems}
    assert "kd" in names
    assert any("teacher_logits" in p for p in problems)


def test_trainer_refuses_to_start_on_an_unfeedable_schedule():
    rng = np.random.default_rng(3)
    labels = (rng.random((2, NUM_TARGETS)) < 0.4).astype(np.float32)
    batch = collate_studies([_record(i, labels) for i in range(2)], device="cpu")
    plan = default_plan(steps_per_epoch=2, budget="small")
    reg = build_objectives(ObjectiveConfig(prevalence=[0.2] * NUM_TARGETS))
    model = _tiny_model()

    with pytest.raises(MissingObjectiveError):
        Trainer(model, plan, TrainConfig(), loss_terms=reg,
                fold_hash="abc", sample_batch=batch)

    # ... and accepts the same schedule once the gaps are declared.
    t = Trainer(model, plan, TrainConfig(), loss_terms=reg, fold_hash="abc",
                sample_batch=batch,
                disabled_terms=["kd", "ot_ground", "weak_label"])
    assert t.disabled_terms == {"kd", "ot_ground", "weak_label"}


def test_trainer_requires_a_fold_hash():
    plan = default_plan(steps_per_epoch=2, budget="small")
    reg = build_objectives(ObjectiveConfig(prevalence=[0.2] * NUM_TARGETS))
    with pytest.raises(ValueError, match="fold_hash"):
        Trainer(_tiny_model(), plan, TrainConfig(), loss_terms=reg,
                fold_hash="", validate=False)


def test_disabled_terms_contribute_nothing_to_the_loss():
    rng = np.random.default_rng(4)
    labels = (rng.random((4, NUM_TARGETS)) < 0.4).astype(np.float32)
    batch = collate_studies([_record(i, labels) for i in range(4)], device="cpu")
    plan = default_plan(steps_per_epoch=2, budget="small")
    reg = build_objectives(ObjectiveConfig(prevalence=[0.2] * NUM_TARGETS))
    model = _tiny_model()
    t = Trainer(model, plan, TrainConfig(), loss_terms=reg, fold_hash="h",
                sample_batch=batch,
                disabled_terms=["kd", "ot_ground", "weak_label", "asl"])
    out = model(batch)
    weights = dict.fromkeys(LossSchedule.TERMS, 0.0)
    weights["asl"] = 1.0
    total, logs, parts = t.compute_losses(out, batch, weights, per_label=True)
    assert float(total) == 0.0
    assert "loss/asl" not in logs
    # A disabled term also contributes no per-label decomposition, or gradient
    # surgery would cancel a gradient that was never accumulated.
    assert parts is None


def test_objective_terms_are_finite_on_a_real_batch():
    rng = np.random.default_rng(5)
    labels = (rng.random((6, NUM_TARGETS)) < 0.35).astype(np.float32)
    recs = [_record(i, labels, n_series=2) for i in range(6)]
    for k, r in enumerate(recs):
        r.group_index = k % 3
        r.env_index = k % 3
    batch = collate_studies(recs, device="cpu")
    model = _tiny_model(fine=True)
    out = model(batch, update_precision=True)

    reg = build_objectives(
        ObjectiveConfig(prevalence=[0.35] * NUM_TARGETS, num_groups=3, ssl_dim=64)
    )
    computed = {}
    for name, term in reg.items():
        v = term(out, batch, {"step": 0})
        if v is None:
            continue
        assert torch.isfinite(v), f"{name} produced a non-finite value"
        computed[name] = float(v)

    # Everything that does not need report/teacher data must have fired.
    for expected in ("asl", "auc_margin", "pauc", "copula", "group_dro", "chi2_dro",
                     "attn_entropy", "moe_balance", "ontology", "mim", "cross_plane",
                     "consistency", "selector_budget", "irm", "rank_queue"):
        assert expected in computed, f"{expected} returned None on a full batch"


# --------------------------------------------------------------------------- #
# SSL heads                                                                    #
# --------------------------------------------------------------------------- #


def test_masked_token_modelling_learns_and_does_not_collapse():
    torch.manual_seed(6)
    B, Nseq, S, D = 4, 2, 10, 32
    head = MaskedTokenModelling(D, mask_ratio=0.4, depth=1, n_heads=4)
    tokens = torch.randn(B, Nseq, S, D)
    mask = torch.ones(B, Nseq, S, dtype=torch.bool)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)

    first = None
    for _ in range(60):
        o = head(tokens, mask)
        if first is None:
            first = float(o["loss"].detach())
        opt.zero_grad()
        o["loss"].backward()
        opt.step()
    assert float(o["loss"].detach()) < first
    # Collapse guard: the predictions must retain spread.
    assert float(o["variance"]) > 0.05, "masked-token predictions collapsed"
    assert 0.0 < float(o["masked_fraction"]) < 1.0


def test_masked_token_modelling_never_masks_an_entire_series():
    head = MaskedTokenModelling(16, mask_ratio=1.0, depth=1, n_heads=4)
    tokens = torch.randn(2, 1, 5, 16)
    mask = torch.ones(2, 1, 5, dtype=torch.bool)
    o = head(tokens, mask)
    assert torch.isfinite(o["loss"])
    assert float(o["masked_fraction"]) < 1.0


def test_cross_plane_consistency_penalises_disagreement():
    torch.manual_seed(7)
    B, S, D = 6, 8, 32
    head = CrossPlaneConsistency(D, proj_dim=32).eval()
    # family 1 = sagittal, 5 = coronal, 8 = axial
    family = torch.tensor([[1, 5, 8]]).repeat(B, 1)
    smask = torch.ones(B, 3, S, dtype=torch.bool)
    sermask = torch.ones(B, 3, dtype=torch.bool)

    shared = torch.randn(B, 1, S, D).repeat(1, 3, 1, 1)
    agree = head(shared + 0.01 * torch.randn(B, 3, S, D), smask, sermask, family)
    disagree = head(torch.randn(B, 3, S, D) * 3.0, smask, sermask, family)
    assert float(agree["invariance"]) < float(disagree["invariance"])


def test_cross_plane_skips_single_plane_studies():
    head = CrossPlaneConsistency(16, proj_dim=16)
    family = torch.tensor([[1, 1]])  # both sagittal -> only one plane present
    o = head(torch.randn(1, 2, 4, 16), torch.ones(1, 2, 4, dtype=torch.bool),
             torch.ones(1, 2, dtype=torch.bool), family)
    assert float(o["loss"]) == 0.0


# --------------------------------------------------------------------------- #
# Sampler and augmentation                                                     #
# --------------------------------------------------------------------------- #


def test_class_aware_sampler_puts_rare_positives_in_every_batch():
    rng = np.random.default_rng(8)
    N = 400
    prev = np.array([0.30, 0.25, 0.20, 0.02])  # last one is rare
    y = (rng.random((N, 4)) < prev).astype(float)
    y[:, 3] = 0.0
    y[rng.choice(N, 8, replace=False), 3] = 1.0  # exactly 8 positives

    s = ClassAwareBatchSampler(y, SamplerConfig(batch_size=8, rare_threshold=0.10))
    batches = list(s)
    assert len(batches) == N // 8
    with_rare = sum(1 for b in batches if y[b, 3].sum() > 0)
    # Without the quota this would be ~2% of batches.
    assert with_rare / len(batches) > 0.9


def test_sampler_respects_the_repeat_cap():
    rng = np.random.default_rng(9)
    y = np.zeros((100, 2))
    y[:, 0] = (rng.random(100) < 0.4).astype(float)
    y[[3, 7], 1] = 1.0  # two rare positives only
    s = ClassAwareBatchSampler(y, SamplerConfig(batch_size=5, max_repeat=3.0,
                                                rare_threshold=0.10))
    counts = np.zeros(100)
    for b in s:
        counts[b] += 1
    # A cap of 3 with 20 batches must keep the two rare studies well below the
    # "appears in every batch" degenerate case.
    assert counts[[3, 7]].max() <= len(list(s)) * 0.75


def test_sampler_batches_are_the_requested_size_and_unique():
    rng = np.random.default_rng(10)
    y = (rng.random((120, 5)) < 0.2).astype(float)
    for b in ClassAwareBatchSampler(y, SamplerConfig(batch_size=6)):
        assert len(b) == 6
        assert len(set(b)) == 6


def test_infer_prevalence_is_nan_aware_and_bounded():
    y = np.array([[1.0, np.nan], [0.0, 1.0], [1.0, np.nan]])
    p = infer_prevalence(y)
    assert abs(p[0] - 2 / 3) < 1e-9
    assert abs(p[1] - 1.0) < 0.4  # floored away from exactly 1
    assert np.all((p > 0) & (p < 1))


def test_augmenter_applies_one_transform_to_the_whole_series():
    """Adjacent 2.5D channels must remain adjacent anatomy."""
    cfg = AugmentConfig(seed=0, p_geometric=1.0, p_intensity=0.0,
                        rotation_deg=10.0, slice_dropout=0.0, noise_sigma=0.0)
    aug = SeriesAugmenter(cfg)
    # Identical slices in -> identical slices out, iff one shared transform.
    base = np.random.default_rng(0).normal(size=(1, 40, 40)).astype(np.float32)
    vol = np.repeat(base, 5, axis=0)
    out, _ = aug(vol)
    for s in range(1, 5):
        assert np.allclose(out[0], out[s], atol=1e-5), "per-slice transform detected"


def test_horizontal_flip_permutes_medial_and_lateral_labels():
    from kairos.constants import TARGETS

    idx = {t: i for i, t in enumerate(TARGETS)}
    assert MEDIAL_LATERAL_SWAP[idx["Medial OA"]] == idx["Lateral OA"]
    assert MEDIAL_LATERAL_SWAP[idx["Lateral Meniscus"]] == idx["Medial Meniscus"]
    assert MEDIAL_LATERAL_SWAP[idx["ACL"]] == idx["ACL"]

    cfg = AugmentConfig(seed=0, p_geometric=0.0, p_intensity=0.0,
                        horizontal_flip_prob=1.0, slice_dropout=0.0)
    aug = SeriesAugmenter(cfg)
    vol = np.random.default_rng(0).normal(size=(3, 16, 16)).astype(np.float32)
    labels = np.zeros(NUM_TARGETS, dtype=np.float32)
    labels[idx["Medial OA"]] = 1.0
    out, new_labels = aug(vol, labels=labels)
    assert np.allclose(out, vol[:, :, ::-1])
    assert new_labels[idx["Lateral OA"]] == 1.0
    assert new_labels[idx["Medial OA"]] == 0.0


def test_flip_is_off_by_default():
    assert AugmentConfig().horizontal_flip_prob == 0.0


def test_bias_field_is_positive_and_smooth():
    rng = np.random.default_rng(0)
    b = bias_field((32, 32), 0.3, rng)
    assert np.all(b > 0)
    assert 0.5 < b.mean() < 2.0
    # Smoothness: neighbouring pixels must be close.
    assert np.abs(np.diff(b, axis=0)).max() < 0.2


def test_augmentation_preserves_shape_and_finiteness():
    aug = SeriesAugmenter(AugmentConfig(seed=3))
    vol = np.random.default_rng(1).normal(size=(8, 32, 32)).astype(np.float32)
    for _ in range(10):
        out, _ = aug(vol)
        assert out.shape == vol.shape
        assert np.isfinite(out).all()


def test_study_dataset_stamps_group_and_env_indices():
    """``load_study`` never sets these; the dataset must, or Group-DRO is inert."""
    rng = np.random.default_rng(11)
    labels = (rng.random((4, NUM_TARGETS)) < 0.4).astype(np.float32)

    def load(uid):
        r = _record(int(uid[1:]), labels)
        r.group_index = r.env_index = None  # as load_study leaves them
        return r

    ds = StudyDataset(
        [f"s{i}" for i in range(4)],
        load_fn=load, labels=labels, env_id=[0, 1, 2, 1],
    )
    recs = [ds[i] for i in range(4)]
    assert [r.group_index for r in recs] == [0, 1, 2, 1]
    batch = collate_studies(recs, device="cpu")
    assert batch.group_id is not None
    assert batch.group_id.tolist() == [0, 1, 2, 1]


def test_study_dataset_does_not_override_a_group_the_loader_already_knows():
    rng = np.random.default_rng(12)
    labels = (rng.random((2, NUM_TARGETS)) < 0.4).astype(np.float32)

    def load(uid):
        r = _record(int(uid[1:]), labels)
        r.group_index = 7  # a cache that already carries the bucketing
        return r

    ds = StudyDataset([f"s{i}" for i in range(2)], load_fn=load,
                      labels=labels, env_id=[0, 1])
    assert [ds[i].group_index for i in range(2)] == [7, 7]


# --------------------------------------------------------------------------- #
# Gradient surgery                                                             #
# --------------------------------------------------------------------------- #


def test_label_contrib_is_the_exact_decomposition_of_the_mean():
    """``sum_l contrib_l == mean``.  Surgery cancels the plain sum of these
    against the gradient ``backward()`` accumulated; a decomposition that only
    *approximately* sums to the term biases every update by the residual."""
    from kairos.losses.auc import AUCMarginLoss

    torch.manual_seed(0)
    logits = torch.randn(16, NUM_TARGETS)
    targets = (torch.rand(16, NUM_TARGETS) < 0.4).float()
    targets[3, 2] = float("nan")          # unobserved labels must not break it
    targets[7, :] = float("nan")

    asl = AsymmetricLoss()
    c = asl(logits, targets, reduction="label_contrib")
    assert c.shape == (NUM_TARGETS,)
    assert torch.allclose(c.sum(), asl(logits, targets), atol=1e-6)

    aucm = AUCMarginLoss(NUM_TARGETS, prevalence=torch.full((NUM_TARGETS,), 0.4))
    ca = aucm(logits, targets, reduction="label_contrib")
    assert torch.allclose(ca.sum(), aucm(logits, targets), atol=1e-6)


def test_aligned_surgery_is_homogeneous_and_survives_rank_deficiency():
    from kairos.optim.pesg import GradientSurgery

    torch.manual_seed(0)
    G = torch.randn(5, 40)
    d = GradientSurgery._aligned(G)
    assert torch.allclose(GradientSurgery._aligned(3.0 * G), 3.0 * d, rtol=1e-4)

    # Orthogonal rows: rescaling a task that is not the smallest leaves the
    # combined direction exactly where it was.  This is the scale-invariance
    # that makes 'aligned' the default over pcgrad/cagrad.
    Q = torch.linalg.qr(torch.randn(40, 5))[0].T
    Go = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])[:, None] * Q
    da = GradientSurgery._aligned(Go)
    Gb = Go.clone()
    Gb[3] *= 7.0
    assert torch.allclose(GradientSurgery._aligned(Gb), da, atol=1e-5)

    # Two labels with (near-)collinear gradients make G rank-deficient.  The
    # naive sigma_min prefactor would scale the whole update to zero and
    # training would stop silently.
    Gr = torch.randn(4, 30)
    Gr[2] = Gr[0] * (1.0 + 1e-7)
    dr = GradientSurgery._aligned(Gr)
    assert torch.isfinite(dr).all()
    assert dr.norm() > 0.05 * Gr.norm(dim=1).min()

    # No usable gradient at all -> the plain sum, not zero.
    assert torch.allclose(
        GradientSurgery._aligned(torch.zeros(3, 10)), torch.zeros(10)
    )


def test_surgery_correct_composes_with_the_rest_of_the_objective():
    """``correct_`` must leave ``g_other`` intact and replace only the plain
    sum of the surgery losses."""
    from kairos.optim.pesg import GradientSurgery

    torch.manual_seed(0)
    head, other = torch.nn.Linear(8, 4), torch.nn.Linear(8, 1)
    x = torch.randn(6, 8)
    params = list(head.parameters())

    def per_task():
        z = head(x)
        return [(z[:, l] ** 2).mean() for l in range(4)]

    surgery = GradientSurgery(params, mode="aligned")
    per = per_task()
    (sum(per) + 3.0 * (other(x) ** 2).mean()).backward(retain_graph=True)
    logs = surgery.correct_(per)
    got = torch.cat([p.grad.reshape(-1) for p in params])

    head.zero_grad(set_to_none=True)
    G = torch.stack([
        torch.cat([g.reshape(-1) for g in torch.autograd.grad(
            l, params, retain_graph=True)])
        for l in per_task()
    ])
    assert torch.allclose(got, GradientSurgery._aligned(G), atol=1e-6)
    assert logs["surgery/skipped"] == 0.0
    assert logs["surgery/active_tasks"] == 4.0
    # Parameters outside the surgery set keep the ordinary summed gradient.
    assert other.weight.grad is not None and torch.isfinite(other.weight.grad).all()


def test_surgery_skips_when_fewer_than_two_tasks_are_active():
    from kairos.optim.pesg import GradientSurgery

    torch.manual_seed(0)
    head = torch.nn.Linear(8, 4)
    x = torch.randn(6, 8)
    z = head(x)
    # One live task; the other three are constants (a rare label with no
    # positive in the minibatch looks exactly like this).
    per = [(z[:, 0] ** 2).mean()] + [z.sum() * 0.0 for _ in range(3)]
    sum(per).backward(retain_graph=True)
    before = head.weight.grad.clone()
    logs = GradientSurgery(list(head.parameters())).correct_(per)
    assert logs["surgery/skipped"] == 1.0
    assert torch.equal(head.weight.grad, before)  # untouched


def test_trainer_actually_applies_gradient_surgery():
    """The whole point of the wiring: with surgery on, the update direction on
    the fusion/head block differs from the plain summed gradient."""
    rng = np.random.default_rng(21)
    labels = (rng.random((4, NUM_TARGETS)) < 0.4).astype(np.float32)
    batch = collate_studies([_record(i, labels) for i in range(4)], device="cpu")
    plan = default_plan(steps_per_epoch=1, budget="small")
    common = dict(loss_terms=None, fold_hash="h", sample_batch=batch,
                  disabled_terms=["kd", "ot_ground", "weak_label"])

    grads = {}
    for mode in ("none", "aligned"):
        torch.manual_seed(3)
        model = _tiny_model()
        reg = build_objectives(
            ObjectiveConfig(prevalence=[0.4] * NUM_TARGETS), model=model
        )
        t = Trainer(model, plan, TrainConfig(gradient_surgery=mode, use_ema=False),
                    **{**common, "loss_terms": reg})
        weights = dict.fromkeys(LossSchedule.TERMS, 0.0)
        weights["asl"] = 1.0
        out = model(batch, update_precision=True)
        loss, _, parts = t.compute_losses(out, batch, weights,
                                          per_label=t.surgery is not None)
        loss.backward(retain_graph=t.surgery is not None)
        if t.surgery is not None:
            logs = t.surgery.correct_(list(parts))
            assert logs["surgery/skipped"] == 0.0
            assert logs["surgery/active_tasks"] >= 2
        grads[mode] = torch.cat([
            p.grad.reshape(-1) for p in t._surgery_params() if p.grad is not None
        ])

    assert grads["none"].numel() == grads["aligned"].numel() > 0
    rel = (grads["aligned"] - grads["none"]).norm() / grads["none"].norm()
    assert rel > 1e-3, f"surgery changed nothing (rel={float(rel):.2e})"


def test_train_epoch_runs_with_surgery_enabled():
    rng = np.random.default_rng(22)
    labels = (rng.random((4, NUM_TARGETS)) < 0.4).astype(np.float32)
    recs = [_record(i, labels) for i in range(4)]
    batch = collate_studies(recs, device="cpu")
    model = _tiny_model()
    reg = build_objectives(
        ObjectiveConfig(prevalence=[0.4] * NUM_TARGETS), model=model
    )
    plan = default_plan(steps_per_epoch=2, budget="small")
    t = Trainer(model, plan,
                TrainConfig(gradient_surgery="aligned", accum_steps=2, use_ema=False),
                loss_terms=reg, fold_hash="h", sample_batch=batch,
                disabled_terms=["kd", "ot_ground", "weak_label"])
    stats = t.train_epoch([batch, batch])
    assert "surgery/mean_pairwise_cos" in stats
    assert np.isfinite(stats["loss/total"])
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_prepare_apply_matches_correct_and_needs_no_retained_graph():
    """The loop measures before backward and applies after, so the big
    backward can free as it goes.  Both orders must give the same gradient."""
    from kairos.optim.pesg import GradientSurgery

    def run(split: bool):
        torch.manual_seed(5)
        head, other = torch.nn.Linear(8, 4), torch.nn.Linear(8, 1)
        x = torch.randn(6, 8)
        surgery = GradientSurgery(list(head.parameters()), mode="aligned")
        z = head(x)
        per = [(z[:, l] ** 2).mean() for l in range(4)]
        loss = sum(per) + 3.0 * (other(x) ** 2).mean()
        if split:
            delta, logs = surgery.prepare(per)
            loss.backward()          # no retain_graph needed
            surgery.apply_(delta)
        else:
            loss.backward(retain_graph=True)
            logs = surgery.correct_(per)
        return torch.cat([p.grad.reshape(-1) for p in head.parameters()]), logs

    g_split, l_split = run(True)
    g_once, l_once = run(False)
    assert torch.allclose(g_split, g_once, atol=1e-6)
    assert l_split == l_once


# --------------------------------------------------------------------------- #
# Min-max block ownership                                                      #
# --------------------------------------------------------------------------- #


def test_auc_auxiliaries_are_owned_by_pesg_and_nothing_else():
    """a, b and alpha must be in exactly one optimiser -- PESG.

    The failure this guards: alpha was excluded from AdamW (right) but PESG was
    only built inside the ranking *stage* (wrong), so for most of the run alpha
    accumulated a gradient that nothing ever applied.
    """
    rng = np.random.default_rng(31)
    labels = (rng.random((4, NUM_TARGETS)) < 0.4).astype(np.float32)
    batch = collate_studies([_record(i, labels) for i in range(4)], device="cpu")
    model = _tiny_model()
    reg = build_objectives(ObjectiveConfig(prevalence=[0.4] * NUM_TARGETS), model=model)
    t = Trainer(model, default_plan(steps_per_epoch=2, budget="small"), TrainConfig(),
                loss_terms=reg, fold_hash="h", sample_batch=batch,
                disabled_terms=["kd", "ot_ground", "weak_label"])

    aucm = reg["auc_margin"].module
    minimax = {id(p) for p in (aucm.a, aucm.b, aucm.alpha)}
    assert t.pesg is not None
    assert {id(p) for g in t.pesg.param_groups for p in g["params"]} == minimax
    in_adamw = {id(p) for g in t.opt.param_groups for p in g["params"]}
    assert not (minimax & in_adamw), "min-max auxiliaries must not be in AdamW"


def test_alpha_is_ascended_in_every_stage_where_auc_margin_is_live():
    rng = np.random.default_rng(32)
    labels = (rng.random((6, NUM_TARGETS)) < 0.4).astype(np.float32)
    batch = collate_studies([_record(i, labels) for i in range(6)], device="cpu")
    model = _tiny_model()
    reg = build_objectives(ObjectiveConfig(prevalence=[0.4] * NUM_TARGETS), model=model)
    plan = default_plan(steps_per_epoch=2, budget="small")
    t = Trainer(model, plan, TrainConfig(use_ema=False), loss_terms=reg,
                fold_hash="h", sample_batch=batch,
                disabled_terms=["kd", "ot_ground", "weak_label"])
    aucm = reg["auc_margin"].module

    # Both S3 ('ranking' in the name) and S4 ('robust + distil') schedule
    # auc_margin; the old stage-name test stopped stepping alpha at S4.
    for stage in plan.stages:
        w = (stage.weights_end or stage.weights).get("auc_margin", 0.0)
        if w == 0.0:
            continue
        weights = dict.fromkeys(LossSchedule.TERMS, 0.0)
        weights["auc_margin"] = w
        assert t._pesg_wanted(weights), f"PESG idle during {stage.name}"

        before = aucm.alpha.detach().clone()
        out = model(batch, update_precision=True)
        loss, _, _ = t.compute_losses(out, batch, weights)
        loss.backward()
        t._optimizer_step(batch, weights, stage)
        assert not torch.allclose(aucm.alpha, before), f"alpha frozen in {stage.name}"
        assert (aucm.alpha >= 0).all(), "alpha must stay in its feasible set"


def test_lr_schedule_reaches_pesg_too():
    rng = np.random.default_rng(33)
    labels = (rng.random((2, NUM_TARGETS)) < 0.4).astype(np.float32)
    batch = collate_studies([_record(i, labels) for i in range(2)], device="cpu")
    model = _tiny_model()
    reg = build_objectives(ObjectiveConfig(prevalence=[0.4] * NUM_TARGETS), model=model)
    plan = default_plan(steps_per_epoch=8, budget="small")
    t = Trainer(model, plan, TrainConfig(), loss_terms=reg, fold_hash="h",
                sample_batch=batch, disabled_terms=["kd", "ot_ground", "weak_label"])
    seen = []
    for step in (0, plan.total_steps // 3, plan.total_steps // 2,
                 plan.total_steps - 1):
        t.step = step
        mult = t._set_lr()
        pesg_lr = t.pesg.param_groups[0]["lr"]
        # PESG must be on exactly the same multiplier as AdamW -- warmup,
        # cosine and the stage's lr_scale, not a constant of its own.
        assert pesg_lr == pytest.approx(t.cfg.pesg_lr * mult, rel=1e-9)
        assert t.opt.param_groups[0]["lr"] == pytest.approx(
            t.base_lrs[0] * mult, rel=1e-9)
        seen.append(pesg_lr)
    assert len(set(seen)) > 1, "the schedule never moved"
    assert seen[-1] < max(seen), "PESG must decay with everything else"


def test_validator_catches_a_term_the_model_cannot_feed():
    """The blind spot: a fully-registered term with a module and no batch
    requirement, dead because the model emits none of its inputs."""
    from kairos.train.curriculum import CurriculumPlan, Stage

    reg = build_objectives(ObjectiveConfig(prevalence=[0.2] * NUM_TARGETS))
    plan = CurriculumPlan(
        stages=(Stage(name="s", epochs=1, weights={"asl": 1.0, "shortcut": 0.5}),),
        steps_per_epoch=2,
    )
    sched = LossSchedule(plan)
    # Without sample_outputs the validator cannot know -- that is the old
    # behaviour and it is why the term went unnoticed.
    assert not any("shortcut" in p for p in
                   validate_schedule(reg, sched, strict=False))
    problems = validate_schedule(
        reg, sched, sample_outputs={"logits": torch.zeros(2, NUM_TARGETS)},
        strict=False,
    )
    assert any("shortcut" in p and "logits_report" in p for p in problems)


def test_shipped_curriculum_has_no_dead_terms():
    """Every term the default plan switches on must be computable by the
    shipped model on a real batch."""
    rng = np.random.default_rng(34)
    labels = (rng.random((4, NUM_TARGETS)) < 0.4).astype(np.float32)
    recs = [_record(i, labels, n_series=2) for i in range(4)]
    batch = collate_studies(recs, device="cpu")
    model = _tiny_model(fine=True)
    with torch.no_grad():
        out = model(batch, update_precision=False)
    reg = build_objectives(
        ObjectiveConfig(prevalence=[0.4] * NUM_TARGETS, num_groups=3, ssl_dim=64),
        model=model,
    )
    sched = LossSchedule(default_plan(steps_per_epoch=4, budget="medium"))
    problems = validate_schedule(reg, sched, sample_outputs=out, strict=False)
    # Only the genuinely batch-conditional terms may be listed, and only
    # because this hand-built batch carries no reports or teacher.
    assert not problems, problems


def _ema_step(aucm, p_ema, targets):
    """One EMA prevalence decay of ``aucm`` applied to ``p_ema`` by hand."""
    y = torch.nan_to_num(targets, nan=0.0)
    valid = torch.isfinite(targets).float()
    p_batch = (y * valid).sum(0) / valid.sum(0).clamp_min(1.0)
    d = aucm.ema_prevalence
    return (p_ema * d + p_batch * (1 - d)).clamp(1e-4, 1 - 1e-4)


def test_per_label_mode_does_not_evaluate_a_term_twice():
    """AUCMarginLoss decays a prevalence EMA inside forward(); calling it once
    for the scalar and again for the decomposition would decay it twice."""
    rng = np.random.default_rng(35)
    labels = (rng.random((6, NUM_TARGETS)) < 0.4).astype(np.float32)
    batch = collate_studies([_record(i, labels) for i in range(6)], device="cpu")
    model = _tiny_model()
    # prevalence=None -> the EMA path is live, which is what we are guarding.
    reg = build_objectives(ObjectiveConfig(), model=model)
    t = Trainer(model, default_plan(steps_per_epoch=2, budget="small"),
                TrainConfig(gradient_surgery="aligned"), loss_terms=reg,
                fold_hash="h", validate=False)
    aucm = reg["auc_margin"].module
    weights = dict.fromkeys(LossSchedule.TERMS, 0.0)
    weights["auc_margin"] = 1.0

    out = model(batch, update_precision=True)
    t.compute_losses(out, batch, weights, per_label=True)
    once = aucm._p_ema.clone()
    t.compute_losses(out, batch, weights, per_label=True)
    assert torch.allclose(aucm._p_ema, _ema_step(aucm, once, batch.targets), atol=1e-6)


def test_scalar_equals_the_sum_of_the_decomposition_in_the_trainer():
    rng = np.random.default_rng(36)
    labels = (rng.random((6, NUM_TARGETS)) < 0.4).astype(np.float32)
    batch = collate_studies([_record(i, labels) for i in range(6)], device="cpu")
    model = _tiny_model()
    reg = build_objectives(ObjectiveConfig(prevalence=[0.4] * NUM_TARGETS), model=model)
    t = Trainer(model, default_plan(steps_per_epoch=2, budget="small"),
                TrainConfig(gradient_surgery="aligned"), loss_terms=reg,
                fold_hash="h", validate=False)
    weights = dict.fromkeys(LossSchedule.TERMS, 0.0)
    weights["asl"], weights["auc_margin"] = 1.0, 0.2

    out = model(batch, update_precision=True)
    plain, logs_a, none_parts = t.compute_losses(out, batch, weights)
    dec, logs_b, parts = t.compute_losses(out, batch, weights, per_label=True)
    assert none_parts is None
    assert parts is not None and parts.shape == (NUM_TARGETS,)
    assert torch.allclose(plain, dec, atol=1e-5)
    assert torch.allclose(parts.sum(), dec, atol=1e-5)
    for k in ("loss/asl", "loss/auc_margin"):
        assert logs_a[k] == pytest.approx(logs_b[k], abs=1e-5)


def test_minimax_routing_survives_a_device_move():
    """A Python attribute on a Parameter does not survive ``.to(device)``.

    ``nn.Module._apply`` keeps the Parameter object only when
    ``torch._has_compatible_shallow_copy_type`` holds -- true for a dtype cast,
    false for any device change, where it constructs a fresh
    ``Parameter(...)`` and drops every user attribute.  Routing the min-max
    block on such a marker worked on CPU, passed its CPU tests, and on GPU put
    alpha in AdamW: descent on an objective that is concave in alpha, pinned to
    0 by ``project()``, so the whole A3 margin block was identically zero while
    the reported loss went *down*.

    ``meta`` takes the same ``_apply`` branch as ``cuda`` and needs no GPU.
    """
    from kairos.losses.auc import AUCMarginLoss

    plain = AUCMarginLoss(NUM_TARGETS)
    moved = build_objectives(ObjectiveConfig(), device="meta")["auc_margin"].module

    for m in (plain, moved):
        desc, asc = m.minimax_parameters()
        assert [id(p) for p in desc] == [id(m.a), id(m.b)]
        assert [id(p) for p in asc] == [id(m.alpha)]
        # The tags are re-applied by the _apply override, for anything that
        # still reads them.
        assert getattr(m.alpha, "_auc_ascent", False)
        assert all(getattr(p, "_minimax", False) for p in desc + asc)

    # A dtype cast (which _does_ preserve the object) must not break either.
    cast = AUCMarginLoss(NUM_TARGETS).to(torch.float64)
    assert getattr(cast.alpha, "_auc_ascent", False)


def test_pesg_ascends_alpha_without_any_marker_attribute():
    """The explicit ``ascent_params`` channel must work on its own, because it
    is the only one a device move cannot destroy."""
    from kairos.optim.pesg import PESG

    alpha = torch.nn.Parameter(torch.tensor([0.5]))
    theta = torch.nn.Parameter(torch.tensor([0.5]))
    assert not hasattr(alpha, "_auc_ascent")  # exactly the post-move state

    opt = PESG([alpha, theta], lr=0.1, weight_decay=0.0, ascent_params=[alpha])
    alpha.grad = torch.tensor([1.0])
    theta.grad = torch.tensor([1.0])
    opt.step()
    assert float(alpha) > 0.5, "alpha must ascend"
    assert float(theta) < 0.5, "theta must descend"


def test_trainer_routes_the_minimax_block_after_a_device_move():
    reg = build_objectives(ObjectiveConfig(prevalence=[0.4] * NUM_TARGETS),
                           device="meta")
    aucm = reg["auc_margin"].module
    model = _tiny_model()
    t = Trainer(model, default_plan(steps_per_epoch=2, budget="small"),
                TrainConfig(), loss_terms=reg, fold_hash="h", validate=False)
    minimax = {id(aucm.a), id(aucm.b), id(aucm.alpha)}
    assert {id(p) for g in t.pesg.param_groups for p in g["params"]} == minimax
    assert not (minimax & {id(p) for g in t.opt.param_groups for p in g["params"]})
    assert t.pesg._is_ascent(aucm.alpha)
    assert not t.pesg._is_ascent(aucm.a)
