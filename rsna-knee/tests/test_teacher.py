"""The cross-fitted teacher artefact.

``07_make_teacher.py`` is the piece that makes stage S4's ``kd`` term
computable at all.  Its guards are the whole point: a teacher assembled across
two different splits, or from a run that predicted the same study twice, is a
teacher that saw the study it is teaching about -- and the resulting leak shows
up as an OOF score nobody can later reconstruct.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from kairos.constants import NUM_TARGETS, TARGETS

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "07_make_teacher.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("make_teacher", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_teacher"] = mod
    spec.loader.exec_module(mod)
    return mod


mt = _load_module()


def _run(uids, logits, *, fold=0, fold_hash="h"):
    return {"path": Path(f"run{fold}/oof.npz"), "logits": np.asarray(logits, float),
            "uids": list(uids), "fold": fold, "fold_hash": fold_hash}


def test_folds_concatenate_into_one_prediction_per_study():
    rng = np.random.default_rng(0)
    runs = [
        _run([f"s{5 * k + i}" for i in range(5)], rng.normal(size=(5, NUM_TARGETS)),
             fold=k)
        for k in range(4)
    ]
    uids, logits, stats = mt.build_teacher(runs)
    assert uids == sorted(f"s{i}" for i in range(20))
    assert logits.shape == (20, NUM_TARGETS)
    assert stats["members_per_study_max"] == 1
    assert np.isfinite(logits).all()


def test_averaging_is_in_probability_space_not_logit_space():
    """Ranks are not probabilities and neither are logits: a distillation
    target has to be a probability, so the mean is taken there."""
    a = np.full((1, NUM_TARGETS), -4.0)
    b = np.full((1, NUM_TARGETS), 4.0)
    _, logits, _ = mt.build_teacher([_run(["s0"], a, fold=0), _run(["s0"], b, fold=1)])
    # mean of sigmoid(-4) and sigmoid(+4) is 0.5 -> logit 0.  A logit-space mean
    # would also give 0 here, so use an asymmetric pair to tell them apart.
    assert np.allclose(logits, 0.0, atol=1e-6)

    a = np.full((1, NUM_TARGETS), 0.0)     # p = 0.5
    b = np.full((1, NUM_TARGETS), 4.0)     # p = 0.982
    _, logits, _ = mt.build_teacher([_run(["s0"], a, fold=0), _run(["s0"], b, fold=1)])
    expected_p = 0.5 * (0.5 + 1 / (1 + np.exp(-4.0)))
    assert np.allclose(mt._sigmoid(logits), expected_p, atol=1e-6)
    assert not np.allclose(logits, 2.0, atol=1e-3), "that would be a logit-space mean"


def test_mixing_fold_hashes_is_refused():
    rng = np.random.default_rng(1)
    runs = [
        _run(["s0"], rng.normal(size=(1, NUM_TARGETS)), fold=0, fold_hash="aaa"),
        _run(["s1"], rng.normal(size=(1, NUM_TARGETS)), fold=1, fold_hash="bbb"),
    ]
    with pytest.raises(SystemExit, match="more than one fold hash"):
        mt.build_teacher(runs)


def test_a_run_that_predicted_a_study_twice_is_refused(tmp_path):
    """Not something to silently average: it means that run's OOF writer is
    emitting a study it also trained on."""
    p = tmp_path / "oof.npz"
    np.savez(p, logits=np.zeros((2, NUM_TARGETS)),
             study_uid=np.array(["s0", "s0"], dtype=object), fold=0, fold_hash="h")
    with pytest.raises(ValueError, match="more than once"):
        mt.load_oof(p)


def test_shape_mismatch_is_refused(tmp_path):
    p = tmp_path / "oof.npz"
    np.savez(p, logits=np.zeros((3, NUM_TARGETS)),
             study_uid=np.array(["s0", "s1"], dtype=object), fold=0, fold_hash="h")
    with pytest.raises(ValueError, match="do not match"):
        mt.load_oof(p)


def test_sigmoid_is_stable_at_the_magnitudes_auc_margin_produces():
    z = np.array([[-800.0, 800.0] + [0.0] * (NUM_TARGETS - 2)])
    p = mt._sigmoid(z)
    assert np.isfinite(p).all()
    assert p[0, 0] == pytest.approx(0.0) and p[0, 1] == pytest.approx(1.0)
    # And the round trip stays finite, which is what the parquet has to carry.
    assert np.isfinite(mt._logit(p)).all()


def test_end_to_end_write_and_reload(tmp_path):
    """The written table must be exactly what 04_train.py's loader expects."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(2)
    runs = [_run([f"s{i}"], rng.normal(size=(1, NUM_TARGETS)), fold=i) for i in range(3)]
    uids, logits, _ = mt.build_teacher(runs)
    df = pd.DataFrame(logits, columns=list(TARGETS))
    df.insert(0, "StudyInstanceUID", uids)
    out = tmp_path / "teacher.parquet"
    df.to_parquet(out, index=False)

    back = pd.read_parquet(out)
    assert list(back.columns) == ["StudyInstanceUID", *TARGETS]
    assert len(back) == 3
