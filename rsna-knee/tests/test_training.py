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
    total, logs = t.compute_losses(out, batch, weights)
    assert float(total) == 0.0
    assert "loss/asl" not in logs


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
