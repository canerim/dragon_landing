#!/usr/bin/env python3
"""End-to-end smoke test on synthetic data. No DICOM, no GPU, no weights.

    python scripts/99_smoke_test.py

Exercises the real code path: manifest → folds → dataset collation → model
forward/backward under the curriculum → OOF evaluation → audit suite →
ensembling → calibration → conformal → submission. Every stage runs the
production function, not a stub.

Its purpose is not to validate accuracy (there is no signal in random noise) but
to prove that the pieces fit together and that nothing raises on shapes, dtypes,
masks or file formats -- which is exactly what breaks after a refactor and what
you do not want to discover on Kaggle.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

t0 = time.monotonic()


def step(msg: str) -> None:
    print(f"[{time.monotonic() - t0:6.2f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
step("1/10  synthetic cohort + patient-safe folds")
# --------------------------------------------------------------------------- #
from kairos.constants import NUM_TARGETS, TARGETS  # noqa: E402
from kairos.data.folds import FoldSpec, fold_report, make_folds  # noqa: E402

rng = np.random.default_rng(7)
N_PATIENTS = 240
prev = np.array([0.22, 0.14, 0.30, 0.18, 0.25, 0.16, 0.20, 0.35, 0.06, 0.11, 0.09, 0.03])

gid, labels, site, lang, uids = [], [], [], [], []
for p in range(N_PATIENTS):
    base = (rng.random(NUM_TARGETS) < prev).astype(float)
    s, l = f"site{rng.integers(0, 16)}", f"lang{rng.integers(0, 12)}"
    for k in range(1 + int(rng.random() < 0.15)):
        y = base.copy()
        flip = rng.random(NUM_TARGETS) < 0.05
        y[flip] = 1 - y[flip]
        gid.append(f"p{p}")
        labels.append(y)
        site.append(s)
        lang.append(l)
        uids.append(f"1.2.826.0.1.{p}.{k}")
labels = np.array(labels)
N = len(uids)

folds = make_folds(
    group_id=gid,
    labels=labels,
    covariates={"site": site, "language": lang},
    spec=FoldSpec(n_folds=5, n_anneal_steps=6000, seed=1),
)
print(fold_report(folds, TARGETS))
assert len({f for g in set(gid) for f in {folds.row_fold[i] for i, x in enumerate(gid) if x == g}}) <= 5
by_group: dict[str, set] = {}
for g, f in zip(gid, folds.row_fold):
    by_group.setdefault(g, set()).add(int(f))
assert all(len(v) == 1 for v in by_group.values()), "patient split across folds"
step(f"       {N} studies, {N_PATIENTS} patients, hash {folds.fold_hash[:12]}")


# --------------------------------------------------------------------------- #
step("2/10  synthetic studies through the real collation path")
# --------------------------------------------------------------------------- #
import torch  # noqa: E402

from kairos.constants import COARSE_SIZE, Plane  # noqa: E402
from kairos.data.dataset import SeriesRecord, StudyRecord, collate_studies  # noqa: E402

SIZE = 40  # small enough to run on CPU in seconds


def fake_study(i: int) -> StudyRecord:
    n_series = int(rng.integers(2, 5))
    series = []
    for s in range(n_series):
        n_slices = int(rng.integers(8, 16))
        vol = rng.normal(0, 1, size=(n_slices, SIZE, SIZE)).astype(np.float32)
        vol += labels[i, s % NUM_TARGETS] * 0.8  # a little planted signal
        spacing = float(rng.uniform(2.0, 4.0))
        series.append(
            SeriesRecord(
                series_uid=f"s{i}.{s}",
                family_index=int(rng.integers(1, 12)),
                plane=Plane.SAGITTAL,
                pixels=vol,
                z_mm=np.cumsum(rng.uniform(spacing * 0.9, spacing * 1.1, n_slices)).astype(np.float32),
                spacing_mm=spacing,
                in_plane_mm=float(rng.uniform(0.3, 0.7)),
                field_strength=float(rng.choice([1.5, 3.0])),
                te_ms=float(rng.uniform(10, 90)),
                tr_ms=float(rng.uniform(500, 4000)),
            )
        )
    r = StudyRecord(study_uid=uids[i], series=series)
    r.labels = labels[i]
    return r


batch = collate_studies([fake_study(i) for i in range(4)], device="cpu")
step(f"       batch pixels {tuple(batch.pixels.shape)}, "
     f"series_mask {tuple(batch.series_mask.shape)}")
assert batch.pixels.shape[3] == 5, "2.5D channel stacking"
assert batch.targets is not None and batch.targets.shape == (4, NUM_TARGETS)


# --------------------------------------------------------------------------- #
step("3/10  model construction + forward (coarse and coarse→fine)")
# --------------------------------------------------------------------------- #
from kairos.models.system import KairosConfig, KairosModel  # noqa: E402

cfg = KairosConfig(
    dim=64, agg_depth=1, agg_heads=4, aggregator="transformer",
    sngp_features=128, n_experts=3, moe_top_k=2,
    enable_fine_pass=False, sequence_dropout=0.0, slice_dropout=0.0,
    fine_top_k=3,
)
cfg.backbone.pretrained = False
cfg.backbone.in_chans = 5
model = KairosModel(cfg)
n_params = sum(p.numel() for p in model.parameters())
step(f"       {n_params / 1e6:.2f}M parameters")

model.eval()
with torch.no_grad():
    out_coarse = model(batch)
assert out_coarse["logits"].shape == (4, NUM_TARGETS)
assert torch.isfinite(out_coarse["logits"]).all()

model.cfg.enable_fine_pass = True
with torch.no_grad():
    out_fine = model(batch)
assert "fine_logits" in out_fine
step(f"       coarse ok; fine fraction {float(out_fine['fine_fraction'].mean()):.3f}")


# --------------------------------------------------------------------------- #
step("4/10  curriculum + composed losses + backward")
# --------------------------------------------------------------------------- #
from kairos.losses.auc import AUCMarginLoss, PartialAUCLoss  # noqa: E402
from kairos.losses.robust import GroupDRO  # noqa: E402
from kairos.losses.supervised import AsymmetricLoss, GaussianCopulaNLL  # noqa: E402
from kairos.train.curriculum import LossSchedule, default_plan  # noqa: E402

plan = default_plan(steps_per_epoch=4, budget="small")
sched = LossSchedule(plan)
print(sched.describe())

asl = AsymmetricLoss(gamma_neg=4.0, clip=0.05)
aucm = AUCMarginLoss(NUM_TARGETS, margin=1.0)
pauc = PartialAUCLoss()
copula = GaussianCopulaNLL(NUM_TARGETS, rank=3)
dro = GroupDRO(4, num_labels=NUM_TARGETS)

model.train()
model.cfg.enable_fine_pass = True
groups = model.parameter_groups(backbone_lr=1e-4, head_lr=1e-3)
opt = torch.optim.AdamW(groups)

for stage_probe in (0, plan.total_steps // 2, plan.total_steps - 1):
    w = sched(stage_probe)
    out = model(batch, update_precision=True)
    z, y = out["logits"], batch.targets
    total = (
        w["asl"] * asl(z, y)
        + w["auc_margin"] * aucm(z, y)
        + w["pauc"] * pauc(z, y)
        + w["copula"] * copula(z, y)
        + w["attn_entropy"] * out["attn_entropy_penalty"]
        + w["moe_balance"] * out.get("moe_balance", torch.zeros(()))
        + w["ontology"] * out.get("ontology_hierarchy", torch.zeros(()))
        + w["selector_budget"] * out.get("selector_budget", torch.zeros(()))
        + w["group_dro"] * dro(
            asl(z, y, reduction="per_example"), torch.randint(0, 4, (4,))
        )
    )
    assert torch.isfinite(total), f"non-finite loss at step {stage_probe}"
    opt.zero_grad(set_to_none=True)
    total.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(gn), "non-finite gradient norm"
    opt.step()
    active = ", ".join(f"{k}={v:g}" for k, v in sorted(w.items()) if v > 0)
    step(f"       step {stage_probe:>3}: loss {float(total):.4f}  grad {float(gn):.3f}")
    step(f"                  active: {active}")


# --------------------------------------------------------------------------- #
step("5/10  synthetic OOF + evaluation")
# --------------------------------------------------------------------------- #
from kairos.eval.metrics import evaluate  # noqa: E402

oof_logits = labels * 1.4 + rng.normal(0, 1.0, size=labels.shape)
report = evaluate(
    labels, 1 / (1 + np.exp(-oof_logits)),
    group_id=np.array(gid), fold=folds.row_fold, site=np.array(site), n_boot=200,
)
print(report.to_text(TARGETS))


# --------------------------------------------------------------------------- #
step("6/10  audit suite")
# --------------------------------------------------------------------------- #
from kairos.eval.leakage import content_hash, run_audit_suite  # noqa: E402

scores = 1 / (1 + np.exp(-oof_logits))
suite = run_audit_suite(
    y=labels,
    scores=scores,
    fold=folds.row_fold,
    group_id=gid,
    metadata=np.stack([
        np.array([int(s[4:]) for s in site], dtype=float),
        np.array([int(l[4:]) for l in lang], dtype=float),
        rng.normal(size=N),
    ], axis=1),
    site=site,
    hashes=[content_hash(rng.random((4, 32, 32))) for _ in range(N)],
    embeddings=rng.normal(size=(N, 24)),
)
print(suite.to_text())
assert not suite.blocking, "smoke data must not trigger a blocking audit"


# --------------------------------------------------------------------------- #
step("7/10  ensembling with the nested honesty check")
# --------------------------------------------------------------------------- #
from kairos.constants import GROUP_OF_LABEL, LABEL_GROUPS  # noqa: E402
from kairos.ensemble.weights import nested_evaluate  # noqa: E402

members = np.stack([
    labels * 1.4 + rng.normal(0, 1.0, size=labels.shape),
    labels * 1.1 + rng.normal(0, 1.2, size=labels.shape),
    labels * 1.6 + rng.normal(0, 1.4, size=labels.shape),
])
gnames = list(LABEL_GROUPS)
gidx = np.array([gnames.index(GROUP_OF_LABEL[t]) for t in TARGETS])
ew = nested_evaluate(
    labels, members, folds.row_fold,
    model_names=["a", "b", "c"], group_of_label=gidx, n_steps=60,
)
print(ew.summary(TARGETS))


# --------------------------------------------------------------------------- #
step("8/10  calibration + conformal risk control")
# --------------------------------------------------------------------------- #
from kairos.calibrate.conformal import (  # noqa: E402
    BetaCalibrator,
    ConformalRiskController,
    TemperatureScaler,
)
from kairos.eval.metrics import macro_auc  # noqa: E402

cal, test = folds.row_fold != 0, folds.row_fold == 0
ts = TemperatureScaler(NUM_TARGETS).fit(oof_logits[cal], labels[cal])
calibrated = ts.predict_proba(oof_logits)
assert abs(macro_auc(labels, calibrated) - macro_auc(labels, scores)) < 1e-9, \
    "temperature scaling must not move AUC"

bc = BetaCalibrator(NUM_TARGETS).fit(scores[cal], labels[cal])
beta_p = bc.predict_proba(scores)
assert abs(macro_auc(labels, beta_p) - macro_auc(labels, scores)) < 1e-6, \
    "beta calibration must not move AUC"

crc = ConformalRiskController(alpha=0.15).fit(scores[cal], labels[cal])
risk = crc.empirical_risk(scores[test], labels[test])
step(f"       temperatures {np.round(ts.temperature, 3)}")
step(f"       conformal thresholds {np.round(crc.lambdas, 3)}")
step(f"       realised risk (target ≤0.15) mean {np.nanmean(risk):.3f} "
     f"max {np.nanmax(risk):.3f}")


# --------------------------------------------------------------------------- #
step("9/10  runtime governor + Pareto frontier")
# --------------------------------------------------------------------------- #
from kairos.infer.budget import CostModel, RuntimeGovernor, pareto_frontier  # noqa: E402

frontier = pareto_frontier(
    y_true=labels,
    coarse_prob=scores,
    fine_prob=1 / (1 + np.exp(-(labels * 1.8 + rng.normal(0, 0.9, labels.shape)))),
    uncertainty=rng.random(labels.shape),
    n_slices=rng.integers(20, 45, N),
    cost=CostModel(),
    time_budget_s=9 * 3600,
)
step(f"       {len(frontier)} non-dominated policies")
for p in frontier[: min(4, len(frontier))]:
    step(f"         AUC {p.macro_auc:.4f}  {p.seconds / 60:6.1f} min  "
         f"models={p.n_models} tta={p.n_tta} k={p.top_k} fine={p.fine_fraction:.2f}")

gov = RuntimeGovernor(budget_s=600.0, n_studies=100, start_escalation=0.5)
for _ in range(20):
    gov.end_study(8.0)
step(f"       {gov.report()}")
assert gov.escalation < 0.5, "governor must back off when behind schedule"


# --------------------------------------------------------------------------- #
step("10/10 submission write + validate")
# --------------------------------------------------------------------------- #
import pandas as pd  # noqa: E402

from kairos.infer.submission import build_submission, validate_submission  # noqa: E402

from kairos.ensemble.weights import combine_members, rank_transform  # noqa: E402

# Write what the notebook would write, not a convenient stand-in.  The members
# are combined through the *deployment* combiner, so this stage would catch a
# drift between how the weights are fitted (rank space) and how they are
# applied -- which is exactly the bug that used to live in the notebook.
final = combine_members(members, ew.weights)
ref = np.einsum("lm,mnl->nl", ew.weights,
                np.stack([rank_transform(members[m]) for m in range(members.shape[0])]))
assert abs(macro_auc(labels, final) - macro_auc(labels, ref)) < 1e-6, \
    "the probability rescale must be monotone in the combined rank"
step(f"       combined macro-AUC {macro_auc(labels, final):.5f} "
     f"(uniform {macro_auc(labels, np.stack([rank_transform(members[m]) for m in range(members.shape[0])]).mean(0)):.5f})")

with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "submission.csv"
    sample = pd.DataFrame({"StudyInstanceUID": uids, **{t: 0.5 for t in TARGETS}})
    df = build_submission(uids, final, output_path=out, sample_submission=sample)
    info = validate_submission(out, expected_uids=uids)
    step(f"       {info['n_rows']} rows, sha256 {info['sha256'][:16]}")
    assert list(df.columns) == ["StudyInstanceUID", *TARGETS]

print()
print("=" * 72)
print(f"ALL 10 STAGES PASSED in {time.monotonic() - t0:.1f}s")
print("=" * 72)
sys.exit(0)
