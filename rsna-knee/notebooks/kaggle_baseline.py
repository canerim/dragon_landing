"""Kaggle BASELINE notebook — no model, no weights, no training required.

Submit this **first**, before any model exists. It does three jobs, and all
three are on the critical path:

1. **Proves the submission plumbing works.** Column names (including the ASCII
   apostrophe in ``Baker's``), UID set and order, dtypes, no index column, the
   filename. A submission that scores 0.5 is worth far more on day 3 than a
   submission that errors on day 77.

2. **Discovers the actual data layout.** Nothing here assumes a directory
   structure — it *reports* what it finds: how studies are nested, how many
   series and slices they have, which DICOM tags are present, whether reports
   ship with the test set, what the pixel encodings are. Every one of those is
   an assumption the real pipeline makes, and every one is cheaper to check now.

3. **Measures the runtime budget.** It profiles header reads, pixel decode and
   preprocessing on a sample of studies and extrapolates to the full test set.
   That number decides how many ensemble members and how much fine-resolution
   escalation fit in nine hours. This is the week-2 gate in ``docs/DESIGN.md``
   §9 and it is the single most commonly deferred — and most commonly
   regretted — measurement in a code competition.

The submission it produces is the ``sample_submission`` benchmark plus
imperceptible jitter (1e-6), so it scores ~0.5 and is a valid, non-degenerate
file. Read the log, not the score.

Usage: paste into a Kaggle notebook, or run locally with
``KAIROS_COMP_DIR=/path/to/data python notebooks/kaggle_baseline.py``.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.monotonic()
COMP_DIR = os.environ.get("KAIROS_COMP_DIR") or None
N_PROFILE = int(os.environ.get("KAIROS_N_PROFILE", 24))
TIME_BUDGET_S = 9 * 3600


def log(msg: str = "") -> None:
    print(f"[{time.monotonic() - T0:7.1f}s] {msg}", flush=True)


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


for d in ("/kaggle/input/kairos-code/src", "/kaggle/working/rsna-knee/src",
          str(Path(__file__).resolve().parent.parent / "src")):
    if Path(d).exists() and d not in sys.path:
        sys.path.insert(0, d)

try:
    from kairos.constants import TARGETS
    from kairos.infer.submission import build_submission, validate_submission
    from kairos.io.layout import discover
    HAVE_KAIROS = True
except Exception:
    HAVE_KAIROS = False
    TARGETS = (
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
        "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
        "Contusion", "Fracture",
    )
    log("NOTE: kairos package not importable; using the built-in fallback writer")


# --------------------------------------------------------------------------- #
rule("1. WHAT IS ACTUALLY IN THE INPUT DIRECTORY")
# --------------------------------------------------------------------------- #

if HAVE_KAIROS:
    LAYOUT = discover(COMP_DIR)
    COMP_DIR = LAYOUT.root
    log(LAYOUT.describe())
else:
    LAYOUT = None
    COMP_DIR = Path(COMP_DIR or "/kaggle/input/competitions/rsna-knee-abnormality-detection")

if not COMP_DIR.exists():
    log(f"FATAL: {COMP_DIR} does not exist.")
    log("Attach the competition dataset, or set KAIROS_COMP_DIR.")
    sys.exit(1)

top = sorted(COMP_DIR.iterdir())
log(f"{COMP_DIR}  ({len(top)} entries)")
for p in top[:30]:
    kind = "dir " if p.is_dir() else "file"
    size = "" if p.is_dir() else f"  {p.stat().st_size / 1e6:8.2f} MB"
    log(f"  {kind} {p.name}{size}")
if len(top) > 30:
    log(f"  ... and {len(top) - 30} more")

for csv in sorted(COMP_DIR.glob("*.csv")):
    try:
        df = pd.read_csv(csv, nrows=5)
        log(f"\n  {csv.name}: {list(df.columns)[:16]}")
        log(f"    first row: {df.iloc[0].to_dict() if len(df) else '(empty)'}")
    except Exception as exc:
        log(f"  {csv.name}: unreadable ({exc})")


# --------------------------------------------------------------------------- #
rule("2. TEST SET AND SUBMISSION TEMPLATE")
# --------------------------------------------------------------------------- #

ss_path = (LAYOUT.sample_submission if LAYOUT else None) or (
    COMP_DIR / "sample_submission.csv" if (COMP_DIR / "sample_submission.csv").exists() else None
)
sample = pd.read_csv(ss_path) if ss_path else None
if sample is not None:
    log(f"sample_submission: {len(sample)} rows, {len(sample.columns)} columns")

if sample is None:
    cand = [d for d in COMP_DIR.iterdir() if d.is_dir() and d.name.lower().startswith("test")]
    root = cand[0] if cand else COMP_DIR
    uids = sorted(p.name for p in root.iterdir() if p.is_dir())
    log(f"no sample_submission; enumerated {len(uids)} study dirs under {root.name}")
    sample = pd.DataFrame({"StudyInstanceUID": uids, **{t: 0.5 for t in TARGETS}})
else:
    got, want = list(sample.columns), ["StudyInstanceUID", *TARGETS]
    if got == want:
        log("column names match kairos.constants.TARGETS exactly")
    else:
        log("!! COLUMN MISMATCH — update kairos/constants.py TARGETS before anything else")
        log(f"   sample : {got}")
        log(f"   kairos : {want}")
        for a, b in zip(got, want):
            if a != b:
                log(f"   first difference: {a!r} vs {b!r}  "
                    f"(codepoints {[hex(ord(c)) for c in a]} vs {[hex(ord(c)) for c in b]})")
                break

STUDY_UIDS = sample["StudyInstanceUID"].astype(str).tolist()
N = len(STUDY_UIDS)
log(f"test studies: {N}")
log("NOTE: the public test set is usually much smaller than the private one.")
log("      Scale every timing below by the ratio you expect, not by 1.")


# --------------------------------------------------------------------------- #
rule("3. STUDY LAYOUT — how are series and slices nested?")
# --------------------------------------------------------------------------- #

test_root = LAYOUT.test_images if LAYOUT else None
if test_root is None:
    for cand in ("test_series", "test_images", "test"):
        p = COMP_DIR / cand
        if p.is_dir():
            test_root = p
            break
if test_root is None:
    dirs = [d for d in COMP_DIR.iterdir() if d.is_dir()]
    test_root = dirs[0] if dirs else COMP_DIR
log(f"test image root: {test_root}")

probe = []
for uid in STUDY_UIDS[: min(N_PROFILE, N)]:
    d = test_root / uid
    if not d.exists():
        continue
    subdirs = [x for x in d.iterdir() if x.is_dir()]
    files = [x for x in d.rglob("*") if x.is_file()]
    probe.append((uid, len(subdirs), len(files), Counter(x.suffix.lower() for x in files)))

if not probe:
    log(f"!! no study directories found under {test_root} for the first "
        f"{min(N_PROFILE, N)} UIDs — the layout is not study-per-directory.")
    log("   Inspect the tree above and adjust kairos/data/dataset.py:load_study.")
else:
    n_series = [p[1] for p in probe]
    n_files = [p[2] for p in probe]
    exts = Counter()
    for p in probe:
        exts.update(p[3])
    log(f"probed {len(probe)} studies")
    log(f"  series per study : min {min(n_series)}  median {int(np.median(n_series))}  "
        f"max {max(n_series)}")
    log(f"  files  per study : min {min(n_files)}  median {int(np.median(n_files))}  "
        f"max {max(n_files)}")
    log(f"  file extensions  : {dict(exts)}")
    if min(n_series) == 0:
        log("  layout is FLAT (files directly under the study dir); load_study "
            "handles this, but grouping by SeriesInstanceUID is then required")


# --------------------------------------------------------------------------- #
rule("4. DICOM TAGS — which of the pipeline's assumptions hold?")
# --------------------------------------------------------------------------- #

REQUIRED = [
    "ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing",
    "Rows", "Columns", "SeriesInstanceUID", "SOPInstanceUID",
]
USEFUL = [
    "InstanceNumber", "SliceThickness", "SpacingBetweenSlices", "RepetitionTime",
    "EchoTime", "InversionTime", "ScanningSequence", "ScanOptions",
    "SeriesDescription", "ProtocolName", "Manufacturer", "ManufacturerModelName",
    "MagneticFieldStrength", "ImageLaterality", "Laterality", "BodyPartExamined",
    "PatientID", "TransferSyntaxUID",
]

try:
    import pydicom

    found, missing_required = Counter(), Counter()
    transfer, sample_desc = Counter(), []
    n_read = 0
    for uid, *_ in probe[:12]:
        for f in sorted((test_root / uid).rglob("*")):
            if not f.is_file():
                continue
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            except Exception:
                continue
            n_read += 1
            for tag in REQUIRED:
                (found if hasattr(ds, tag) else missing_required)[tag] += 1
            for tag in USEFUL:
                if hasattr(ds, tag):
                    found[tag] += 1
            try:
                transfer[str(ds.file_meta.TransferSyntaxUID)] += 1
            except Exception:
                pass
            if hasattr(ds, "SeriesDescription"):
                sample_desc.append(str(ds.SeriesDescription))
            break  # one slice per series is enough for a tag census

    log(f"read {n_read} headers")
    for tag in REQUIRED:
        ok = found[tag]
        flag = "OK  " if missing_required[tag] == 0 else "MISS"
        log(f"  [{flag}] {tag:<28} present in {ok}/{n_read}")
    log("")
    for tag in USEFUL:
        log(f"  {'yes' if found[tag] else ' no'}  {tag:<28} {found[tag]}/{n_read}")
    if transfer:
        log(f"\n  transfer syntaxes: {dict(transfer)}")
        if any("91" in k or "90" in k for k in transfer):
            log("  -> JPEG2000/HTJ2K present: nvJPEG2000 GPU decode is worth ~5x here")
    if sample_desc:
        log(f"\n  example SeriesDescription values: {sample_desc[:10]}")
    if missing_required:
        log("\n!! Missing REQUIRED tags break geometric slice ordering. "
            "Check kairos/io/geometry.py fallbacks before training.")
except ImportError:
    log("pydicom not available; skipping the tag census")
except Exception:
    log("tag census failed:\n" + traceback.format_exc())


# --------------------------------------------------------------------------- #
rule("5. RUNTIME BUDGET — the number that decides the whole design")
# --------------------------------------------------------------------------- #

timings = {"headers": [], "decode": [], "preprocess": [], "slices": []}
try:
    import pydicom

    for uid, *_ in probe[: min(12, len(probe))]:
        files = [f for f in sorted((test_root / uid).rglob("*")) if f.is_file()]
        if not files:
            continue

        t = time.monotonic()
        for f in files:
            try:
                pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            except Exception:
                pass
        timings["headers"].append(time.monotonic() - t)

        t, arrs = time.monotonic(), []
        for f in files[:40]:
            try:
                arrs.append(pydicom.dcmread(str(f), force=True).pixel_array)
            except Exception:
                pass
        timings["decode"].append(time.monotonic() - t)
        timings["slices"].append(max(len(arrs), 1))

        if arrs and HAVE_KAIROS:
            from kairos.data.dataset import resample_to_spacing, robust_normalise

            vol = np.stack([a.astype(np.float32) for a in arrs
                            if a.shape == arrs[0].shape])
            t = time.monotonic()
            resample_to_spacing(robust_normalise(vol), 0.5, 0.7, 256)
            timings["preprocess"].append(time.monotonic() - t)

    if timings["headers"]:
        n_sl = float(np.mean(timings["slices"]))
        hdr = float(np.mean(timings["headers"]))
        dec = float(np.mean(timings["decode"]))
        pre = float(np.mean(timings["preprocess"])) if timings["preprocess"] else 0.0
        per_study = hdr + dec + pre
        log(f"per study (mean over {len(timings['headers'])} studies, "
            f"~{n_sl:.0f} decoded slices):")
        log(f"  header read   {hdr:7.3f} s")
        log(f"  pixel decode  {dec:7.3f} s   ({dec / max(n_sl, 1) * 1000:.2f} ms/slice)")
        log(f"  preprocess    {pre:7.3f} s")
        log(f"  TOTAL I/O     {per_study:7.3f} s")
        log("")
        total = per_study * N
        log(f"extrapolated I/O only, {N} studies: {total / 60:.1f} min "
            f"({total / TIME_BUDGET_S:.1%} of the 9 h budget)")
        for mult, label in ((1, "public"), (3, "3x private"), (5, "5x private")):
            t_io = total * mult
            left = TIME_BUDGET_S * 0.88 - t_io
            log(f"  {label:<12} I/O {t_io / 60:7.1f} min -> "
                f"{left / 60:7.1f} min left for models"
                + ("" if left > 0 else "   <-- I/O ALONE BLOWS THE BUDGET"))
        log("")
        log("Rule of thumb: if I/O exceeds ~25% of the budget, GPU-side model")
        log("optimisation is not where the runtime is. Fix decoding first.")
    else:
        log("no timings collected")
except Exception:
    log("profiling failed:\n" + traceback.format_exc())


# --------------------------------------------------------------------------- #
rule("6. DO REPORTS SHIP WITH THE TEST SET?")
# --------------------------------------------------------------------------- #
# This single fact decides whether the system runs image-only at inference or
# gates on a text branch. docs/DESIGN.md §4.3 prepares both; this tells you
# which one to build out.

report_hits = []
for csv in sorted(COMP_DIR.rglob("*.csv")):
    try:
        cols = pd.read_csv(csv, nrows=1).columns.str.lower()
    except Exception:
        continue
    if any(k in c for c in cols for k in ("report", "impression", "finding", "text",
                                          "narrative", "conclusion")):
        report_hits.append((csv.name, list(pd.read_csv(csv, nrows=1).columns)))

if report_hits:
    for name, cols in report_hits:
        log(f"  {name}: {cols}")
    log("\n  Reports appear to be present. Check the RULES before using them at")
    log("  test time; either way they remain valid as training supervision.")
else:
    log("  No report-like columns found in any CSV under the input directory.")
    log("  => Plan for IMAGE-ONLY inference. Reports stay a training-time signal")
    log("     (contrastive pretraining, weak labels, distillation).")


# --------------------------------------------------------------------------- #
rule("7. WRITE AND VALIDATE THE SUBMISSION")
# --------------------------------------------------------------------------- #

rng = np.random.default_rng(0)
# Constant columns score 0.5 but are rejected by the validator (they normally
# mean a head failed to load). Imperceptible jitter keeps the file valid and
# the score identical to the benchmark.
preds = np.full((N, len(TARGETS)), 0.5) + rng.normal(0, 1e-6, size=(N, len(TARGETS)))
preds = np.clip(preds, 0.0, 1.0)

if HAVE_KAIROS:
    build_submission(STUDY_UIDS, preds, output_path="submission.csv",
                     sample_submission=sample)
    info = validate_submission("submission.csv", expected_uids=STUDY_UIDS)
    log(f"submission.csv: {info['n_rows']} rows, sha256 {info['sha256'][:16]}")
else:
    out = pd.DataFrame(preds, columns=list(TARGETS))
    out.insert(0, "StudyInstanceUID", STUDY_UIDS)
    out.to_csv("submission.csv", index=False)
    log(f"submission.csv: {len(out)} rows (written without kairos validation)")

head = Path("submission.csv").read_text(encoding="utf-8").split("\n")[0]
log(f"header: {head}")
assert "Baker's" in head, "apostrophe must be U+0027"

Path("baseline_probe.json").write_text(json.dumps({
    "n_studies": N,
    "test_root": str(test_root),
    "series_per_study_median": int(np.median([p[1] for p in probe])) if probe else None,
    "files_per_study_median": int(np.median([p[2] for p in probe])) if probe else None,
    "io_seconds_per_study": (
        float(np.mean(timings["headers"]) + np.mean(timings["decode"]))
        if timings["headers"] else None
    ),
    "reports_present": bool(report_hits),
    "csvs": {p.name: list(pd.read_csv(p, nrows=1).columns)
             for p in sorted(COMP_DIR.glob("*.csv"))},
}, indent=2))

rule(f"BASELINE COMPLETE in {(time.monotonic() - T0) / 60:.1f} min")
print("Expected score: ~0.5 (the sample_submission benchmark).")
print("The score is not the point. The log above is: it tells you the data")
print("layout, which DICOM tags exist, whether reports ship at test time, and")
print("how much of the 9 h budget I/O alone consumes.")
print("Probe written to baseline_probe.json.")
