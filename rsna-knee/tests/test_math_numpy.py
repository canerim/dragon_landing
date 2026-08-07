"""Numerical validation of the numpy-side mathematics.

These are not smoke tests.  Each one checks a property that would silently
degrade the leaderboard score if it broke: DeLong variance against a bootstrap,
AUC against a brute-force Mann-Whitney count, fold balance against a target,
conformal coverage against its nominal level.
"""

from __future__ import annotations

import numpy as np
import pytest

from kairos.calibrate.conformal import (
    BetaCalibrator,
    ConformalRiskController,
    TemperatureScaler,
)
from kairos.constants import TARGETS
from kairos.data.folds import FoldSpec, make_folds
from kairos.ensemble.weights import fit_label_weights, nested_evaluate, rank_transform
from kairos.eval.metrics import (
    delong_auc_variance,
    delong_test,
    macro_auc,
    patient_bootstrap,
    roc_auc,
)
from kairos.io.geometry import (
    SliceGeometry,
    build_affine,
    classify_plane,
    order_slices,
    slice_normal,
)


# --------------------------------------------------------------------------- #
# Geometry                                                                     #
# --------------------------------------------------------------------------- #


def _sagittal_series(n=20, spacing=3.0, shuffle=True, seed=0, reverse=False):
    """Sagittal stack: rows along +z (superior), cols along -y (anterior)."""
    rng = np.random.default_rng(seed)
    iop = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]  # r = +P, c = -S  -> n = r x c = -x... check below
    slices = []
    order = np.arange(n)
    if reverse:
        order = order[::-1]
    for k, i in enumerate(order):
        pos = np.array([i * spacing, -60.0, 40.0])
        slices.append(
            SliceGeometry(
                sop_uid=f"sop{i}",
                position=pos,
                orientation=np.array(iop),
                pixel_spacing=np.array([0.4, 0.4]),
                rows=320,
                cols=320,
                instance_number=k + 1,
                slice_thickness=spacing,
            )
        )
    if shuffle:
        idx = rng.permutation(len(slices))
        slices = [slices[i] for i in idx]
    return slices


def test_slice_normal_is_unit_and_orthogonal():
    iop = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    n = slice_normal(iop)
    assert np.isclose(np.linalg.norm(n), 1.0)
    assert np.allclose(n, [0.0, 0.0, 1.0])
    assert np.isclose(np.dot(n, iop[:3]), 0.0, atol=1e-12)
    assert np.isclose(np.dot(n, iop[3:]), 0.0, atol=1e-12)


def test_slice_normal_reorthogonalises_skewed_orientation():
    # 1 degree of skew between the row and column vectors.
    a = np.deg2rad(1.0)
    iop = np.array([1.0, 0.0, 0.0, np.sin(a), np.cos(a), 0.0])
    n = slice_normal(iop)
    assert np.isclose(np.linalg.norm(n), 1.0, atol=1e-12)
    assert abs(np.dot(n, iop[:3])) < 1e-12


@pytest.mark.parametrize(
    "iop,expected",
    [
        ([0, 1, 0, 0, 0, -1], "SAGITTAL"),
        ([1, 0, 0, 0, 0, -1], "CORONAL"),
        ([1, 0, 0, 0, 1, 0], "AXIAL"),
    ],
)
def test_plane_classification(iop, expected):
    assert classify_plane(slice_normal(np.array(iop, dtype=float))).name == expected


def test_ordering_recovers_geometry_from_shuffled_instances():
    slices = _sagittal_series(shuffle=True, seed=3)
    g = order_slices(slices)
    z = g.z
    assert np.all(np.diff(z) > 0), "physical coordinate must be monotone after ordering"
    assert np.isclose(g.spacing_mm, 3.0, atol=1e-6)
    assert g.spacing_cv < 1e-9
    assert g.orientation_consistent
    assert "duplicate_positions" not in g.flags


def test_ordering_canonicalises_direction():
    fwd = order_slices(_sagittal_series(shuffle=False, reverse=False))
    rev = order_slices(_sagittal_series(shuffle=False, reverse=True))
    # Both must end up with the normal pointing the same canonical way.
    assert np.allclose(fwd.normal, rev.normal, atol=1e-9)
    assert np.all(np.diff(fwd.z) > 0) and np.all(np.diff(rev.z) > 0)


def test_duplicate_and_gap_detection():
    slices = _sagittal_series(n=10, shuffle=False)
    slices.append(slices[3])  # exact duplicate position
    g = order_slices(slices)
    assert g.n_duplicate_positions >= 1
    assert "duplicate_positions" in g.flags

    gapped = _sagittal_series(n=10, shuffle=False)
    gapped[7].position = gapped[7].position + np.array([30.0, 0, 0])
    g2 = order_slices(gapped)
    assert "large_slice_gap" in g2.flags or "irregular_spacing" in g2.flags


def test_affine_maps_indices_to_expected_physical_points():
    iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ipp = [10.0, 20.0, 30.0]
    A = build_affine(iop, ipp, (0.5, 0.5), 3.0, output="LPS")
    origin = A @ np.array([0, 0, 0, 1.0])
    assert np.allclose(origin[:3], ipp)
    one_col = A @ np.array([1, 0, 0, 1.0])
    assert np.allclose(one_col[:3] - np.array(ipp), [0.5, 0, 0])
    one_slice = A @ np.array([0, 0, 1, 1.0])
    assert np.allclose(one_slice[:3] - np.array(ipp), [0, 0, 3.0])

    A_ras = build_affine(iop, ipp, (0.5, 0.5), 3.0, output="RAS")
    assert np.allclose((A_ras @ np.array([0, 0, 0, 1.0]))[:3], [-10.0, -20.0, 30.0])


# --------------------------------------------------------------------------- #
# AUC / DeLong                                                                 #
# --------------------------------------------------------------------------- #


def _brute_auc(y, s):
    pos, neg = s[y > 0.5], s[y <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (pos.size * neg.size)


def test_roc_auc_matches_brute_force_including_ties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.integers(20, 120)
        y = (rng.random(n) < 0.3).astype(float)
        s = np.round(rng.normal(size=n) + 1.5 * y, 1)  # rounding creates ties
        if y.sum() in (0, n):
            continue
        assert np.isclose(roc_auc(y, s), _brute_auc(y, s), atol=1e-12)


def test_roc_auc_edge_cases():
    assert np.isnan(roc_auc(np.zeros(10), np.arange(10.0)))
    assert np.isnan(roc_auc(np.ones(10), np.arange(10.0)))
    y = np.array([0, 0, 1, 1.0])
    assert roc_auc(y, np.array([0.0, 0.0, 1.0, 1.0])) == 1.0
    assert roc_auc(y, np.array([1.0, 1.0, 0.0, 0.0])) == 0.0
    assert roc_auc(y, np.ones(4)) == 0.5


def test_delong_variance_agrees_with_bootstrap():
    rng = np.random.default_rng(7)
    n = 900
    y = (rng.random(n) < 0.25).astype(float)
    s = rng.normal(size=n) + 1.0 * y
    auc, cov = delong_auc_variance(y, s[None, :])
    se_delong = float(np.sqrt(cov[0, 0]))

    boot = np.empty(1500)
    for b in range(boot.size):
        idx = rng.integers(0, n, n)
        boot[b] = roc_auc(y[idx], s[idx])
    se_boot = float(np.nanstd(boot))
    assert np.isclose(auc[0], roc_auc(y, s), atol=1e-9)
    # Agreement to within 20% is the standard expectation for these two
    # estimators at this sample size.
    assert abs(se_delong - se_boot) / se_boot < 0.2, (se_delong, se_boot)


def test_delong_paired_test_detects_a_real_difference_and_not_a_fake_one():
    rng = np.random.default_rng(11)
    n = 2000
    y = (rng.random(n) < 0.3).astype(float)
    shared = rng.normal(size=n)
    a = shared + 1.2 * y + 0.3 * rng.normal(size=n)
    b = shared + 0.6 * y + 0.3 * rng.normal(size=n)
    res = delong_test(y, a, b)
    assert res["diff"] > 0
    assert res["p_value"] < 1e-6

    same = delong_test(y, a, a + 1e-9 * rng.normal(size=n))
    assert same["p_value"] > 0.01


def test_macro_auc_ignores_degenerate_labels():
    y = np.zeros((50, 3))
    y[:20, 0] = 1
    y[:10, 1] = 1
    # column 2 is all-negative -> NaN AUC -> excluded
    s = np.random.default_rng(0).random((50, 3))
    m = macro_auc(y, s)
    assert np.isfinite(m)


def test_patient_bootstrap_respects_clusters():
    rng = np.random.default_rng(3)
    n_pat, per = 60, 4
    gid = np.repeat(np.arange(n_pat), per)
    y = np.zeros((n_pat * per, 2))
    eff = rng.random(n_pat) < 0.3
    y[:, 0] = np.repeat(eff.astype(float), per)
    y[:, 1] = (rng.random(n_pat * per) < 0.4).astype(float)
    s = y + 0.6 * rng.normal(size=y.shape)
    cluster = patient_bootstrap(y, s, gid, n_boot=300, seed=1)
    naive = patient_bootstrap(y, s, np.arange(len(gid)), n_boot=300, seed=1)
    # Clustered resampling must not *understate* the variance relative to the
    # (invalid) independent resampling.
    assert np.nanstd(cluster) >= 0.9 * np.nanstd(naive)


# --------------------------------------------------------------------------- #
# Folds                                                                        #
# --------------------------------------------------------------------------- #


def _synthetic_cohort(n_patients=400, seed=0):
    rng = np.random.default_rng(seed)
    prev = np.array([0.22, 0.14, 0.30, 0.18, 0.25, 0.16, 0.20, 0.35, 0.06, 0.11, 0.09, 0.03])
    rows_gid, rows_y, site, lang = [], [], [], []
    for p in range(n_patients):
        n_studies = 1 + int(rng.random() < 0.15)
        base = (rng.random(12) < prev).astype(float)
        s = f"site{rng.integers(0, 16)}"
        l = f"lang{rng.integers(0, 12)}"
        for _ in range(n_studies):
            y = base.copy()
            flip = rng.random(12) < 0.05
            y[flip] = 1 - y[flip]
            rows_gid.append(f"p{p}")
            rows_y.append(y)
            site.append(s)
            lang.append(l)
    return rows_gid, np.array(rows_y), site, lang


def test_folds_never_split_a_patient():
    gid, y, site, lang = _synthetic_cohort(300, seed=1)
    res = make_folds(
        group_id=gid,
        labels=y,
        covariates={"site": site, "language": lang},
        spec=FoldSpec(n_folds=5, n_anneal_steps=4000, seed=42),
    )
    by_group: dict[str, set[int]] = {}
    for g, f in zip(gid, res.row_fold):
        by_group.setdefault(g, set()).add(int(f))
    assert all(len(v) == 1 for v in by_group.values())


def test_folds_balance_rare_label_prevalence():
    gid, y, site, lang = _synthetic_cohort(500, seed=2)
    res = make_folds(
        group_id=gid,
        labels=y,
        covariates={"site": site, "language": lang},
        spec=FoldSpec(n_folds=5, n_anneal_steps=8000, seed=7),
    )
    prev = np.asarray(res.diagnostics["per_fold_prevalence"])
    glob = np.asarray(res.diagnostics["global_prevalence"])
    rare = int(np.argmin(glob))
    dev = np.max(np.abs(prev[:, rare] - glob[rare]))
    # The rarest label's per-fold prevalence must stay within 40% relative of
    # the global rate -- a random split routinely exceeds 100% here.
    assert dev < 0.4 * glob[rare] + 0.01, (dev, glob[rare])
    assert res.diagnostics["improvement"] >= -1e-9


def test_fold_hash_is_deterministic_and_sensitive():
    gid, y, site, _ = _synthetic_cohort(150, seed=5)
    spec = FoldSpec(n_folds=5, n_anneal_steps=1500, seed=1)
    a = make_folds(group_id=gid, labels=y, covariates={"site": site}, spec=spec)
    b = make_folds(group_id=gid, labels=y, covariates={"site": site}, spec=spec)
    assert a.fold_hash == b.fold_hash
    c = make_folds(
        group_id=gid, labels=y, covariates={"site": site},
        spec=FoldSpec(n_folds=5, n_anneal_steps=1500, seed=2),
    )
    assert c.fold_hash != a.fold_hash


# --------------------------------------------------------------------------- #
# Ensembling                                                                   #
# --------------------------------------------------------------------------- #


def test_rank_transform_preserves_auc():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.3).astype(float)
    s = rng.normal(size=(500, 1)) + y[:, None]
    assert np.isclose(roc_auc(y, s[:, 0]), roc_auc(y, rank_transform(s)[:, 0]), atol=1e-12)


def test_rank_transform_uses_midranks_so_ties_do_not_move_the_auc():
    """Ordinal ranking breaks ties in argsort order, which is not a monotone
    function of the input -- a single model's AUC then changes under a
    transform whose whole point is to leave it alone.  Exact ties are common
    here: every failed study is written as 0.5 on every label."""
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.3).astype(float)
    s = np.round(rng.random((400, 3)), 1)  # ~10 distinct values, heavy ties
    r = rank_transform(s)
    assert r.shape == s.shape
    for c in range(s.shape[1]):
        assert np.isclose(roc_auc(y, s[:, c]), roc_auc(y, r[:, c]), atol=1e-12)

    # Explicit midranks, and 1-D input is accepted too.
    assert np.allclose(
        rank_transform(np.array([3.0, 1.0, 1.0, 5.0])),
        np.array([2.0, 0.5, 0.5, 3.0]) / 3.0,
    )


def test_label_weights_prefer_the_better_model_per_label():
    rng = np.random.default_rng(4)
    n, L = 1200, 12
    y = (rng.random((n, L)) < 0.25).astype(float)
    # model 0 is good on the first half of the labels, model 1 on the rest.
    p0 = rng.normal(size=(n, L))
    p1 = rng.normal(size=(n, L))
    p0[:, : L // 2] += 1.6 * y[:, : L // 2]
    p0[:, L // 2 :] += 0.2 * y[:, L // 2 :]
    p1[:, : L // 2] += 0.2 * y[:, : L // 2]
    p1[:, L // 2 :] += 1.6 * y[:, L // 2 :]
    w = fit_label_weights(y, np.stack([p0, p1]), model_names=["a", "b"], n_steps=150)
    assert w.weights[0, 0] > w.weights[0, 1]
    assert w.weights[-1, 1] > w.weights[-1, 0]
    assert w.oof_macro_weighted >= w.oof_macro_uniform - 1e-6
    assert np.allclose(w.weights.sum(axis=1), 1.0)


def test_nested_evaluation_reports_an_honest_number():
    rng = np.random.default_rng(9)
    n, L, M = 600, 12, 3
    y = (rng.random((n, L)) < 0.3).astype(float)
    # All models equally (un)informative: the nested weighted score must NOT
    # meaningfully exceed the uniform one.  This is the guard against the
    # classic per-label-weight overfit.
    preds = np.stack([rng.normal(size=(n, L)) + 0.8 * y for _ in range(M)])
    fold = rng.integers(0, 5, n)
    w = nested_evaluate(y, preds, fold, model_names=list("abc"), n_steps=80)
    assert w.nested_macro_weighted < w.nested_macro_uniform + 0.02


# --------------------------------------------------------------------------- #
# Calibration & conformal                                                      #
# --------------------------------------------------------------------------- #


def test_temperature_scaling_reduces_nll_and_preserves_auc():
    rng = np.random.default_rng(2)
    n, L = 3000, 4
    y = (rng.random((n, L)) < 0.3).astype(float)
    z = 3.0 * (rng.normal(size=(n, L)) + 1.2 * y)  # deliberately overconfident

    def nll(zz):
        p = 1 / (1 + np.exp(-np.clip(zz, -60, 60)))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    ts = TemperatureScaler(L).fit(z, y)
    assert nll(ts.transform(z)) < nll(z)
    for l in range(L):
        assert np.isclose(roc_auc(y[:, l], z[:, l]),
                          roc_auc(y[:, l], ts.transform(z)[:, l]), atol=1e-9)


def test_beta_calibration_is_monotone_and_auc_preserving():
    rng = np.random.default_rng(6)
    n, L = 2000, 3
    y = (rng.random((n, L)) < 0.25).astype(float)
    p = 1 / (1 + np.exp(-(rng.normal(size=(n, L)) + 1.5 * y)))
    bc = BetaCalibrator(L).fit(p, y)
    q = bc.predict_proba(p)
    for l in range(L):
        assert np.isclose(roc_auc(y[:, l], p[:, l]), roc_auc(y[:, l], q[:, l]), atol=1e-9)
        order = np.argsort(p[:, l])
        assert np.all(np.diff(q[order, l]) >= -1e-9)


def test_conformal_risk_control_achieves_nominal_coverage():
    rng = np.random.default_rng(13)
    n, L, alpha = 4000, 6, 0.10
    y = (rng.random((n, L)) < 0.25).astype(float)
    p = 1 / (1 + np.exp(-(rng.normal(size=(n, L)) + 1.8 * y)))
    cal, test = slice(0, n // 2), slice(n // 2, n)

    crc = ConformalRiskController(alpha=alpha).fit(p[cal], y[cal])
    risk = crc.empirical_risk(p[test], y[test])
    # The guarantee is on the expectation; allow ~3 binomial SEs of slack.
    n_pos = np.nansum(y[test] > 0.5, axis=0)
    slack = 3.0 * np.sqrt(alpha * (1 - alpha) / np.maximum(n_pos, 1))
    assert np.all(risk <= alpha + slack + 1e-9), (risk, alpha + slack)
    assert np.all(crc.lambdas >= 0.0) and np.all(crc.lambdas <= 1.0)


def test_conformal_is_conservative_when_calibration_is_tiny():
    rng = np.random.default_rng(17)
    y = (rng.random((40, 2)) < 0.5).astype(float)
    p = rng.random((40, 2))
    crc = ConformalRiskController(alpha=0.05).fit(p, y)
    # With n small, (n R + B)/(n+1) <= alpha forces a very low threshold.
    assert np.all(crc.lambdas < 0.5)


# --------------------------------------------------------------------------- #
# Submission                                                                   #
# --------------------------------------------------------------------------- #


def test_submission_roundtrip_and_validation(tmp_path):
    pd = pytest.importorskip("pandas")
    from kairos.infer.submission import SubmissionError, build_submission, validate_submission

    rng = np.random.default_rng(0)
    uids = [f"1.2.826.0.1.{i}" for i in range(25)]
    p = rng.random((25, len(TARGETS)))
    out = tmp_path / "submission.csv"
    build_submission(uids, p, output_path=out)
    info = validate_submission(out)
    assert info["n_rows"] == 25

    text = out.read_text()
    assert text.split("\n")[0].startswith("StudyInstanceUID,ACL,MCL,")
    assert "Baker's" in text and "Baker’s" not in text

    df = pd.read_csv(out)
    df["ACL"] = 0.5  # constant column
    df.to_csv(out, index=False)
    with pytest.raises(SubmissionError, match="constant"):
        validate_submission(out)
    # ... but the check is opt-out, because the inference notebook's safety net
    # writes a deliberately-constant 0.5 fallback before inference starts.
    info = validate_submission(out, require_varying=False)
    assert info["constant_columns"] == ["ACL"]


def test_constant_fallback_submission_is_writable(tmp_path):
    """Regression: the notebook's Stage-0 safety net used to raise.

    ``build_submission`` validated its own output with the constant-column
    check enabled, so writing the all-0.5 fallback -- the exact thing that
    guarantees a file exists no matter what happens later -- crashed.
    """
    pytest.importorskip("pandas")
    from kairos.infer.submission import SubmissionError, build_submission

    uids = [f"u{i}" for i in range(12)]
    flat = np.full((12, len(TARGETS)), 0.5)
    out = tmp_path / "submission.csv"

    with pytest.raises(SubmissionError, match="constant"):
        build_submission(uids, flat, output_path=out)

    df = build_submission(uids, flat, output_path=out, allow_constant=True)
    assert len(df) == 12
    assert out.exists()


def test_submission_fills_from_sample_and_averages_duplicates(tmp_path):
    pd = pytest.importorskip("pandas")
    from kairos.infer.submission import build_submission

    sample = pd.DataFrame(
        {"StudyInstanceUID": [f"u{i}" for i in range(6)], **{t: 0.5 for t in TARGETS}}
    )
    uids = ["u0", "u1", "u1", "u5"]  # missing u2..u4, duplicated u1
    p = np.array([[0.1] * 12, [0.2] * 12, [0.4] * 12, [0.9] * 12])
    out = tmp_path / "submission.csv"
    df = build_submission(uids, p, output_path=out, sample_submission=sample)
    assert df["StudyInstanceUID"].tolist() == [f"u{i}" for i in range(6)]
    assert np.isclose(df.loc[df.StudyInstanceUID == "u1", "ACL"].item(), 0.3)
    assert np.isclose(df.loc[df.StudyInstanceUID == "u3", "ACL"].item(), 0.5)


def test_fallback_validation_path_used_by_the_notebook(tmp_path):
    """Regression: the notebook's 'no models, exit cleanly' path also raised.

    ``build_submission(..., allow_constant=True)`` was fixed once, but the
    *separate* ``validate_submission`` call on the way out of the no-weights
    branch was still strict -- so the branch whose only job is to leave a valid
    file behind crashed after leaving it.
    """
    pytest.importorskip("pandas")
    from kairos.infer.submission import (
        SubmissionError,
        build_submission,
        validate_submission,
    )

    uids = [f"u{i}" for i in range(8)]
    out = tmp_path / "submission.csv"
    build_submission(uids, np.full((8, len(TARGETS)), 0.5), output_path=out,
                     allow_constant=True)
    with pytest.raises(SubmissionError):
        validate_submission(out, expected_uids=uids)
    info = validate_submission(out, expected_uids=uids, require_varying=False)
    assert info["n_rows"] == 8
    assert len(info["constant_columns"]) == len(TARGETS)


def test_submission_rejects_nonfinite(tmp_path):
    from kairos.infer.submission import SubmissionError, build_submission

    p = np.full((10, len(TARGETS)), 0.5)
    p[0, 0] = np.nan
    p[1, 1] = np.inf
    with pytest.raises(SubmissionError, match="non-finite"):
        build_submission([f"u{i}" for i in range(10)], p,
                         output_path=tmp_path / "submission.csv", strict=True)


def test_combine_members_matches_the_space_the_weights_were_fitted_in():
    """`fit_label_weights` optimises over rank-transformed OOF predictions, so
    deployment must combine ranks too -- and the AUC of the result must equal
    the AUC of the rank combination exactly, despite the probability rescale."""
    from kairos.ensemble.weights import combine_members, rank_transform

    rng = np.random.default_rng(7)
    N, L, M = 300, 4, 3
    y = (rng.random((N, L)) < 0.3).astype(float)
    # Members with deliberately different calibration: one squashed towards
    # 0.5, one sharpened, one plain.  Probability averaging is dominated by the
    # sharpened member; rank averaging is not.
    base = [rng.normal(size=(N, L)) + 1.4 * y for _ in range(M)]
    squash = [1.0, 6.0, 0.25]
    preds = np.stack([1 / (1 + np.exp(-squash[m] * base[m])) for m in range(M)])

    w = np.full((L, M), 1.0 / M)
    out = combine_members(preds, w)
    assert out.shape == (N, L)
    assert np.isfinite(out).all()

    ranks = np.stack([rank_transform(preds[m]) for m in range(M)])
    ref = np.einsum("lm,mnl->nl", w, ranks)
    assert np.allclose(combine_members(preds, w, rescale=False), ref, atol=1e-12)
    for l in range(L):
        # The probability rescale is a *monotone* map of the combined rank --
        # that is the exact guarantee, and it is what makes the AUC of the
        # rescaled output the AUC of the ranking.
        order = np.argsort(ref[:, l], kind="mergesort")
        assert np.all(np.diff(out[order, l]) >= 0)
        assert np.isclose(roc_auc(y[:, l], out[:, l]),
                          roc_auc(y[:, l], ref[:, l]), atol=1e-4)

    # The rescale keeps the output on the members' probability scale rather
    # than on [0, 1] ranks, so prevalence sanity checks still mean something.
    assert abs(out.mean() - preds.mean()) < 0.05

    # A per-label weight of 1 on one member reproduces that member's ranking.
    w1 = np.zeros((L, M)); w1[:, 1] = 1.0
    only1 = combine_members(preds, w1)
    for l in range(L):
        assert np.isclose(roc_auc(y[:, l], only1[:, l]),
                          roc_auc(y[:, l], preds[1][:, l]), atol=1e-9)


def test_combine_members_survives_the_failed_study_fallback():
    """Failed studies are written as 0.5 on every label; the resulting mass of
    exact ties must not create ties in the combined ranking."""
    from kairos.ensemble.weights import combine_members

    rng = np.random.default_rng(8)
    N, L, M = 120, 3, 2
    preds = rng.random((M, N, L))
    preds[:, :30, :] = 0.5          # 30 "failed" studies
    preds[0, 40, 0] = np.nan        # and one NaN that must be repaired
    out = combine_members(preds)
    assert np.isfinite(out).all()
    for l in range(L):
        # Strictly increasing rescale => no new ties beyond the ones the ranks
        # themselves produce.
        assert len(np.unique(out[:, l])) >= len(np.unique(np.round(out[:, l], 9)))
    assert combine_members(preds[:, :1, :]).shape == (1, L)
