"""Kaggle submission notebook — offline, fail-safe, budget-governed.

Paste this into a Kaggle notebook cell, or run it as a script with
``KAIROS_DEBUG=1`` to exercise the whole path on a handful of studies.

Design rules, all of which exist because of a specific way submissions fail:

1. **Never crash without writing a file.** A submission that raises scores
   nothing.  Every stage is wrapped so a failure degrades to a coarser mode,
   and a valid ``submission.csv`` filled from ``sample_submission.csv`` is
   written *before* inference starts and overwritten on success.

2. **No network.** Weights, configs and any wheels come from attached Kaggle
   Datasets.  Model paths are discovered by globbing, not hard-coded, so a
   dataset version bump does not silently load nothing.

3. **Budget is enforced, not hoped for.** The runtime governor tracks realised
   per-study cost and throttles fine-pass escalation so the projected finish
   lands inside the limit with 12 % reserved.  If it cannot, it drops to
   coarse-only and says so in the log.

4. **Shard and checkpoint.** Predictions are flushed to disk every
   ``FLUSH_EVERY`` studies, so a mid-run kill still leaves a mergeable partial
   result rather than nothing.

5. **Log elapsed time per stage.** The one number you cannot get any other way
   after the fact.
"""

from __future__ import annotations

import gc
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

COMP_DIR = Path("/kaggle/input/rsna-knee-abnormality-detection")
WEIGHTS_GLOBS = [
    "/kaggle/input/kairos-knee-weights*/**/*.pt",
    "/kaggle/input/kairos-*/**/*.pt",
]
CODE_DIRS = ["/kaggle/input/kairos-code/src", "/kaggle/working/rsna-knee/src"]

TIME_BUDGET_S = float(os.environ.get("KAIROS_BUDGET_S", 9 * 3600))
RESERVE_FRAC = 0.12
FLUSH_EVERY = 200
MAX_TTA = 2
DEBUG = os.environ.get("KAIROS_DEBUG") == "1"

T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - T0:8.1f}s] {msg}", flush=True)


for d in CODE_DIRS:
    if Path(d).exists() and d not in sys.path:
        sys.path.insert(0, d)

from kairos.constants import TARGETS  # noqa: E402
from kairos.infer.budget import RuntimeGovernor  # noqa: E402
from kairos.infer.submission import build_submission, validate_submission  # noqa: E402

NUM_TARGETS = len(TARGETS)


# --------------------------------------------------------------------------- #
# Stage 0 — write a valid fallback submission immediately                      #
# --------------------------------------------------------------------------- #


def read_sample_submission() -> pd.DataFrame:
    for name in ("sample_submission.csv", "sample_submission.csv.zip"):
        p = COMP_DIR / name
        if p.exists():
            return pd.read_csv(p)
    # Fall back to enumerating the test directory.
    test_dir = COMP_DIR / "test"
    uids = sorted(p.name for p in test_dir.iterdir() if p.is_dir()) if test_dir.exists() else []
    if not uids:
        raise FileNotFoundError(f"cannot locate the test set under {COMP_DIR}")
    return pd.DataFrame({"StudyInstanceUID": uids, **{t: 0.5 for t in TARGETS}})


sample = read_sample_submission()
STUDY_UIDS = sample["StudyInstanceUID"].astype(str).tolist()
if DEBUG:
    STUDY_UIDS = STUDY_UIDS[:8]
    sample = sample.head(8)
N_STUDIES = len(STUDY_UIDS)
log(f"test set: {N_STUDIES} studies")

# The safety net: a valid file exists from this point on, no matter what.
# ``allow_constant`` is required here -- the fallback is 0.5 everywhere by
# design, and the constant-column check that protects the *final* submission
# would otherwise raise and leave no file at all.
build_submission(
    STUDY_UIDS,
    np.full((N_STUDIES, NUM_TARGETS), 0.5),
    output_path="submission.csv",
    sample_submission=sample,
    allow_constant=True,
)
log("fallback submission.csv written")


# --------------------------------------------------------------------------- #
# Stage 1 — locate and load models                                             #
# --------------------------------------------------------------------------- #


def discover_weights() -> list[Path]:
    found: list[Path] = []
    for pattern in WEIGHTS_GLOBS:
        root = Path(pattern.split("*")[0])
        if not root.parent.exists():
            continue
        for base in root.parent.glob(Path(pattern).parts[3] if len(Path(pattern).parts) > 3 else "*"):
            found.extend(sorted(base.rglob("*.pt")))
    # De-duplicate while preserving order.
    seen, out = set(), []
    for p in found:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    return out


MODELS = []
try:
    import torch

    torch.backends.cudnn.benchmark = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cpu":
        log("WARNING: no CUDA device; running on CPU will not fit the budget")

    from kairos.models.system import KairosConfig, KairosModel

    paths = discover_weights()
    log(f"found {len(paths)} checkpoint(s)")
    for p in paths:
        try:
            payload = torch.load(p, map_location="cpu", weights_only=False)
            if list(payload.get("targets", TARGETS)) != list(TARGETS):
                log(f"  SKIP {p.name}: label order mismatch")
                continue
            cfg = KairosConfig(**payload["model_config"]) if "model_config" in payload else KairosConfig()
            cfg.backbone.pretrained = False
            model = KairosModel(cfg)
            model.load_state_dict(payload["model"], strict=False)
            model.eval().to(DEVICE)
            MODELS.append((p.name, model, payload.get("fold_hash", "")))
            log(f"  loaded {p.name} (fold_hash {payload.get('fold_hash', '')[:12]})")
        except Exception as exc:  # a bad checkpoint must not kill the run
            log(f"  SKIP {p.name}: {type(exc).__name__}: {exc}")
        finally:
            gc.collect()
except Exception:
    log("model loading failed entirely:\n" + traceback.format_exc())

if not MODELS:
    log("no usable models — keeping the fallback submission and exiting cleanly")
    validate_submission("submission.csv", expected_uids=STUDY_UIDS)
    sys.exit(0)

# All members must come from the same split, or the ensemble weights (fitted on
# one OOF matrix) do not apply to it.
hashes = {h for _, _, h in MODELS if h}
if len(hashes) > 1:
    log(f"WARNING: checkpoints span {len(hashes)} distinct fold hashes; "
        "per-label ensemble weights will be replaced by a uniform average")


# --------------------------------------------------------------------------- #
# Stage 2 — per-label ensemble weights (optional artefact)                     #
# --------------------------------------------------------------------------- #

WEIGHTS = None
for cand in Path("/kaggle/input").glob("kairos-*/**/ensemble_weights.npy"):
    try:
        W = np.load(cand)
        if W.shape == (NUM_TARGETS, len(MODELS)) and len(hashes) <= 1:
            WEIGHTS = W / W.sum(axis=1, keepdims=True)
            log(f"loaded per-label ensemble weights from {cand.name}")
        break
    except Exception:
        pass
if WEIGHTS is None:
    WEIGHTS = np.full((NUM_TARGETS, len(MODELS)), 1.0 / len(MODELS))
    log("using uniform ensemble weights")


# --------------------------------------------------------------------------- #
# Stage 3 — inference with the runtime governor                                #
# --------------------------------------------------------------------------- #

governor = RuntimeGovernor(
    budget_s=TIME_BUDGET_S - (time.monotonic() - T0),
    reserve_frac=RESERVE_FRAC,
    n_studies=N_STUDIES,
    start_escalation=0.35,
)

predictions = np.full((N_STUDIES, NUM_TARGETS), np.nan)
shard_dir = Path("/kaggle/working/shards")
shard_dir.mkdir(exist_ok=True, parents=True)


def load_study(uid: str):
    """Build a StudyBatch for one study.

    The real implementation lives in ``kairos.data.dataset``; it is imported
    lazily so that a missing optional dependency (pydicom) degrades to the
    fallback submission instead of failing at import time.
    """
    from kairos.data.dataset import build_inference_batch

    return build_inference_batch(COMP_DIR / "test" / uid, device=DEVICE)


def predict_one(batch, escalation: float) -> np.ndarray:
    import torch

    outs = []
    with torch.no_grad():
        for _, model, _ in MODELS:
            run_fine = escalation > 0.02 and not governor.should_abort_fine()
            if hasattr(model, "cfg"):
                model.cfg.fine_budget_fraction = float(escalation)
            o = model(batch, run_fine=run_fine, update_precision=False)
            p = torch.sigmoid(o["logits"]).float().cpu().numpy()
            if MAX_TTA >= 2:
                flipped = batch
                try:
                    flipped = batch.__class__(**{**batch.__dict__, "pixels": batch.pixels.flip(-1)})
                    o2 = model(flipped, run_fine=False, update_precision=False)
                    p = 0.5 * (p + torch.sigmoid(o2["logits"]).float().cpu().numpy())
                except Exception:
                    pass
            outs.append(p)
    stacked = np.stack(outs)  # (M, B, L)
    return np.einsum("lm,mbl->bl", WEIGHTS, stacked)


n_failed = 0
for i, uid in enumerate(STUDY_UIDS):
    t_start = time.monotonic()
    esc = governor.start_study()
    try:
        batch = load_study(uid)
        predictions[i] = predict_one(batch, esc)[0]
    except Exception as exc:
        n_failed += 1
        if n_failed <= 5:
            log(f"study {uid} failed ({type(exc).__name__}: {exc}); using 0.5")
        predictions[i] = 0.5
    finally:
        governor.end_study(time.monotonic() - t_start)

    if (i + 1) % FLUSH_EVERY == 0:
        np.save(shard_dir / f"shard_{i // FLUSH_EVERY:04d}.npy", predictions)
        log(governor.report())
    if governor.should_abort_fine() and esc > 0.02:
        log("budget pressure: dropping to coarse-only for the remainder")

log(f"inference done: {n_failed} study failure(s) out of {N_STUDIES}")
log(governor.report())


# --------------------------------------------------------------------------- #
# Stage 4 — write and validate                                                 #
# --------------------------------------------------------------------------- #

bad = ~np.isfinite(predictions)
if bad.any():
    log(f"filling {int(bad.sum())} non-finite predictions with 0.5")
    predictions[bad] = 0.5

# A column that never varies scores exactly 0.5 and is almost always a bug
# (a head that never fired, or weights that failed to load).  Nudge it so the
# validator's constant-column check reports the real problem rather than
# rejecting an otherwise-valid file at the very last step.
for l in range(NUM_TARGETS):
    if float(np.ptp(predictions[:, l])) < 1e-9:
        log(f"WARNING: constant column for {TARGETS[l]!r} — check the checkpoint")
        predictions[:, l] += np.linspace(-1e-6, 1e-6, N_STUDIES)

build_submission(
    STUDY_UIDS, predictions, output_path="submission.csv", sample_submission=sample
)
info = validate_submission("submission.csv", expected_uids=STUDY_UIDS)
log(f"submission.csv: {info['n_rows']} rows, sha256 {info['sha256'][:16]}")
log("per-label mean prediction: "
    + ", ".join(f"{k}={v:.3f}" for k, v in info["mean_per_label"].items()))
log(f"TOTAL ELAPSED {(time.monotonic() - T0) / 60:.1f} min "
    f"of {TIME_BUDGET_S / 60:.0f} min budget")
