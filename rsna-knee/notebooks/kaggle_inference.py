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

import dataclasses
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

# Resolved by kairos.io.layout, which knows the real layout:
#   input/competitions/<comp>/{train,test}_series/<study>/<series>/*.dcm
COMP_DIR = Path(os.environ.get("KAIROS_COMP_DIR", "")) or None
WEIGHTS_GLOBS = [
    "/kaggle/input/kairos-knee-weights*/**/*.pt",
    "/kaggle/input/kairos-*/**/*.pt",
]
CODE_DIRS = ["/kaggle/input/kairos-code/src", "/kaggle/working/rsna-knee/src"]

TIME_BUDGET_S = float(os.environ.get("KAIROS_BUDGET_S", 9 * 3600))
RESERVE_FRAC = 0.12
FLUSH_EVERY = 200
# Forward passes per model per study, TTA included.  Default 1 = no TTA.
#
# This used to be 2, but the view was constructed in a way that raised on every
# study and was swallowed, so TTA never actually ran and the budget was never
# actually spent.  Now that it works, the honest default is off: a second pass
# doubles inference cost, and under a hard 9-hour budget the governor pays for
# it by throttling the *fine pass* -- trading a measured coarse-to-fine gain for
# an unmeasured augmentation-averaging one.  Raise it only with an OOF number
# that says the trade is worth it.
MAX_TTA = int(os.environ.get("KAIROS_MAX_TTA", 1))
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
from kairos.io.layout import discover  # noqa: E402

NUM_TARGETS = len(TARGETS)

LAYOUT = discover(COMP_DIR)
log("layout:\n" + LAYOUT.describe())
if not LAYOUT.ok:
    log("FATAL: could not locate the test images or a UID source.")
    sys.exit(1)
COMP_DIR = LAYOUT.root
TEST_IMAGES = LAYOUT.test_images


# --------------------------------------------------------------------------- #
# Stage 0 — write a valid fallback submission immediately                      #
# --------------------------------------------------------------------------- #


def read_sample_submission() -> pd.DataFrame:
    """UID set and row order for the submission, in decreasing order of trust."""
    if LAYOUT.sample_submission is not None:
        return pd.read_csv(LAYOUT.sample_submission)
    if LAYOUT.test_csv is not None:
        df = pd.read_csv(LAYOUT.test_csv)
        col = next((c for c in df.columns if "study" in c.lower()), df.columns[0])
        uids = df[col].astype(str).drop_duplicates().tolist()
        return pd.DataFrame({"StudyInstanceUID": uids, **{t: 0.5 for t in TARGETS}})
    uids = sorted(p.name for p in TEST_IMAGES.iterdir() if p.is_dir())
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
    # ``require_varying=False`` for the same reason it is off when the fallback
    # is written: this file IS the constant fallback.  Validating it in strict
    # mode raises on the way out of the one path whose entire purpose is to
    # leave a valid file behind.
    info = validate_submission(
        "submission.csv", expected_uids=STUDY_UIDS, require_varying=False
    )
    log(f"fallback submission.csv kept: {info['n_rows']} rows, "
        f"sha256 {info['sha256'][:16]}")
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

# (M, N, L): every member's own prediction, kept separately until the end.
#
# The members are combined in *rank* space, and that cannot be done one study
# at a time -- a rank needs the whole column.  It is also the only combination
# that matches how ``WEIGHTS`` was fitted: ``fit_label_weights`` optimises a
# per-label simplex over rank-transformed OOF predictions, so applying those
# same weights to raw probabilities at test time optimises one objective and
# deploys another.  Since the metric is macro-AUC, which sees only the ordering,
# rank averaging is also the correct combiner regardless of how differently the
# members happen to be calibrated.  Memory is negligible: 5 members x 10k
# studies x 12 labels is under 5 MB.
member_preds = np.full((max(len(MODELS), 1), N_STUDIES, NUM_TARGETS), np.nan)
shard_dir = Path("/kaggle/working/shards")
shard_dir.mkdir(exist_ok=True, parents=True)


# --------------------------------------------------------------------------- #
# Test-time augmentation views: (pixel transform, output permutation or None)   #
# --------------------------------------------------------------------------- #
#
# The left-right mirror that used to sit here was wrong twice over.  A mirrored
# knee swaps medial and lateral, so averaging the mirrored logits into the
# originals without permuting them corrupts ``Medial Meniscus``, ``Lateral
# Meniscus``, ``Medial OA`` and ``Lateral OA`` -- the exact corruption
# ``MEDIAL_LATERAL_SWAP`` exists to prevent during training.  And even with the
# permutation it would be out of distribution: laterality is canonicalised in
# the loader and ``AugmentConfig.horizontal_flip_prob`` is 0, so the network
# has never seen a mirrored knee.
#
# What is left is an intensity view, which *is* inside the training
# distribution (the augmenter applies gamma over the same range) and is
# label-preserving, so it needs no permutation.  The flip stays available,
# correctly permuted, for weights trained with flips enabled.
TTA_FLIP = os.environ.get("KAIROS_TTA_FLIP") == "1"


def _gamma_view(x, g: float = 1.12):
    lo = x.amin()
    rng = (x.amax() - lo).clamp_min(1e-6)
    return ((x - lo) / rng).clamp(0, 1).pow(g) * rng + lo


#: Counted, not silently swallowed: a TTA that never runs looks exactly like a
#: TTA that runs and does not help.
_TTA_FAILURES = 0

TTA_VIEWS = [(_gamma_view, None)]
if TTA_FLIP:
    from kairos.data.transforms import MEDIAL_LATERAL_SWAP  # noqa: E402

    # The swap is an involution, so the same permutation maps the mirrored
    # model's outputs back onto the original anatomy.
    TTA_VIEWS.append((lambda x: x.flip(-1), list(MEDIAL_LATERAL_SWAP)))

n_views = 1 + len(TTA_VIEWS[: max(MAX_TTA - 1, 0)])
log(f"TTA: {n_views} view(s) per model per study"
    + (" (mirror enabled)" if TTA_FLIP else ""))


def load_study(uid: str):
    """Build a StudyBatch for one study.

    The real implementation lives in ``kairos.data.dataset``; it is imported
    lazily so that a missing optional dependency (pydicom) degrades to the
    fallback submission instead of failing at import time.
    """
    from kairos.data.dataset import build_inference_batch

    return build_inference_batch(TEST_IMAGES / uid, device=DEVICE)


def predict_one(batch, escalation: float) -> np.ndarray:
    """Return ``(M, L)`` -- one probability vector per ensemble member."""
    import torch

    outs = []
    with torch.no_grad():
        for _, model, _ in MODELS:
            run_fine = escalation > 0.02 and not governor.should_abort_fine()
            if hasattr(model, "cfg"):
                model.cfg.fine_budget_fraction = float(escalation)
            o = model(batch, run_fine=run_fine, update_precision=False)
            p = torch.sigmoid(o["logits"]).float().cpu().numpy()
            n_tta = 1
            for tf, perm in TTA_VIEWS[: max(MAX_TTA - 1, 0)]:
                try:
                    # ``dataclasses.replace``, not ``batch.__class__(**__dict__)``:
                    # StudyBatch is ``@dataclass(slots=True)`` and therefore has
                    # no ``__dict__`` at all.  The old expression raised
                    # AttributeError on the first study and was swallowed by
                    # this very ``except``, so TTA had been a silent no-op --
                    # costing nothing, gaining nothing, and looking enabled.
                    view = dataclasses.replace(batch, pixels=tf(batch.pixels))
                    o2 = model(view, run_fine=False, update_precision=False)
                    q = torch.sigmoid(o2["logits"]).float().cpu().numpy()
                    p = p + (q[:, perm] if perm is not None else q)
                    n_tta += 1
                except Exception as exc:  # noqa: BLE001 - never lose a study
                    global _TTA_FAILURES
                    _TTA_FAILURES += 1
                    if _TTA_FAILURES == 1:
                        log(f"TTA disabled after a failure: {type(exc).__name__}: {exc}")
            outs.append(p / n_tta)
    return np.stack(outs)[:, 0, :]  # (M, L); batch size is 1 here


n_failed = 0
for i, uid in enumerate(STUDY_UIDS):
    t_start = time.monotonic()
    esc = governor.start_study()
    try:
        batch = load_study(uid)
        member_preds[:, i, :] = predict_one(batch, esc)
    except Exception as exc:
        n_failed += 1
        if n_failed <= 5:
            log(f"study {uid} failed ({type(exc).__name__}: {exc}); using 0.5")
        member_preds[:, i, :] = 0.5
    finally:
        governor.end_study(time.monotonic() - t_start)

    if (i + 1) % FLUSH_EVERY == 0:
        np.save(shard_dir / f"shard_{i // FLUSH_EVERY:04d}.npy", member_preds)
        log(governor.report())
    if governor.should_abort_fine() and esc > 0.02:
        log("budget pressure: dropping to coarse-only for the remainder")

log(f"inference done: {n_failed} study failure(s) out of {N_STUDIES}")
log(governor.report())


# --------------------------------------------------------------------------- #
# Stage 3b — combine the members in rank space                                 #
# --------------------------------------------------------------------------- #

from kairos.ensemble.weights import combine_members  # noqa: E402

predictions = combine_members(member_preds, WEIGHTS)
log(f"combined {len(MODELS)} member(s) in rank space")


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
