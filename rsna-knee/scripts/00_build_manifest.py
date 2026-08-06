#!/usr/bin/env python3
"""Scan the DICOM tree once and write the manifest every later stage reads.

    python scripts/00_build_manifest.py --root data/train \
        --labels data/train_labels.csv --out artifacts/manifest.parquet --workers 8

This is stage zero for a reason: nothing downstream is trustworthy until the
manifest exists. It reads **headers only** (never pixels), so a full scan of a
5 000-study dataset is minutes rather than hours, and it records per-series QC
so that a bad series is excluded by a documented rule instead of by whatever
the training loop happens to do when it hits a decode error.

Columns written, one row per series:

  StudyInstanceUID, SeriesInstanceUID, PatientID
  plane, sequence_family, weighting, fat_sat, n_slices
  spacing_mm, spacing_cv, max_gap_ratio, in_plane_mm, extent_mm
  rows, cols, manufacturer, model, field_strength, te_ms, tr_ms, ti_ms
  laterality, body_part, transfer_syntax
  qc_flags, usable
  + the twelve target columns, joined from --labels

and a study-level rollup (``--out`` with ``_studies`` appended) that
``01_make_folds.py`` consumes directly.

The QC rule for ``usable``: a series is excluded when it has fewer than
``--min-slices`` slices, an inconsistent orientation, or a degenerate z-extent.
Irregular spacing and large gaps are *flagged but kept* -- they are common at
several sites and dropping them would bias the cohort towards the sites with
tidy protocols, which is precisely the domain shift we are trying to survive.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from kairos.constants import SEQUENCE_FAMILIES, TARGETS


def scan_study(args) -> list[dict]:
    """Header-only scan of one study directory.  Runs in a worker process."""
    study_dir, min_slices = args
    study_dir = Path(study_dir)
    import pydicom

    from kairos.data.sequence_taxonomy import classify_series, infer_fat_sat, infer_weighting
    from kairos.io.geometry import SliceGeometry, order_slices, spacing_diagnostics

    series_dirs = [d for d in sorted(study_dir.iterdir()) if d.is_dir()] or [study_dir]
    rows: list[dict] = []

    for sd in series_dirs:
        files = [p for p in sorted(sd.rglob("*")) if p.is_file()]
        if not files:
            continue
        geoms, first = [], None
        transfer = ""
        for p in files:
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            except Exception:
                continue
            if first is None:
                first = ds
                try:
                    transfer = str(ds.file_meta.TransferSyntaxUID)
                except Exception:
                    transfer = ""
            try:
                geoms.append(SliceGeometry(
                    sop_uid=str(getattr(ds, "SOPInstanceUID", p.name)),
                    position=np.asarray(ds.ImagePositionPatient, dtype=np.float64),
                    orientation=np.asarray(ds.ImageOrientationPatient, dtype=np.float64),
                    pixel_spacing=np.asarray(
                        getattr(ds, "PixelSpacing", [1.0, 1.0]), dtype=np.float64),
                    rows=int(getattr(ds, "Rows", 0)),
                    cols=int(getattr(ds, "Columns", 0)),
                    instance_number=int(getattr(ds, "InstanceNumber", 0) or 0),
                    slice_thickness=float(getattr(ds, "SliceThickness", 0) or 0) or None,
                ))
            except Exception:
                continue

        if first is None:
            rows.append({"StudyInstanceUID": study_dir.name,
                         "SeriesInstanceUID": sd.name, "usable": False,
                         "qc_flags": "no_readable_headers", "n_slices": 0})
            continue

        row = {
            "StudyInstanceUID": str(getattr(first, "StudyInstanceUID", study_dir.name)),
            "SeriesInstanceUID": str(getattr(first, "SeriesInstanceUID", sd.name)),
            "PatientID": str(getattr(first, "PatientID", "") or study_dir.name),
            "series_description": str(getattr(first, "SeriesDescription", "") or ""),
            "manufacturer": str(getattr(first, "Manufacturer", "") or ""),
            "model": str(getattr(first, "ManufacturerModelName", "") or ""),
            "field_strength": float(getattr(first, "MagneticFieldStrength", 0) or 0),
            "te_ms": float(getattr(first, "EchoTime", 0) or 0),
            "tr_ms": float(getattr(first, "RepetitionTime", 0) or 0),
            "ti_ms": float(getattr(first, "InversionTime", 0) or 0),
            "laterality": str(getattr(first, "ImageLaterality", "")
                              or getattr(first, "Laterality", "") or ""),
            "body_part": str(getattr(first, "BodyPartExamined", "") or ""),
            "transfer_syntax": transfer,
            "rows": int(getattr(first, "Rows", 0)),
            "cols": int(getattr(first, "Columns", 0)),
            "n_files": len(files),
        }

        if len(geoms) < 3:
            row.update(n_slices=len(geoms), usable=False, qc_flags="too_few_slices")
            rows.append(row)
            continue

        try:
            geo = order_slices(geoms)
        except Exception as exc:
            row.update(n_slices=len(geoms), usable=False,
                       qc_flags=f"ordering_failed:{type(exc).__name__}")
            rows.append(row)
            continue

        diag = spacing_diagnostics(geo.z)
        fam = classify_series(first, plane=geo.plane)
        flags = list(geo.flags)
        usable = (
            len(geoms) >= min_slices
            and geo.orientation_consistent
            and "degenerate_z_extent" not in flags
        )
        row.update(
            plane=geo.plane.name,
            sequence_family=fam.name,
            sequence_family_index=fam.index,
            weighting=infer_weighting(first).name,
            fat_sat=bool(infer_fat_sat(first)),
            n_slices=len(geoms),
            spacing_mm=float(geo.spacing_mm),
            spacing_cv=float(geo.spacing_cv),
            max_gap_ratio=float(geo.max_gap_ratio),
            in_plane_mm=float(np.mean(geoms[0].pixel_spacing)),
            extent_mm=float(diag["extent_mm"]),
            n_duplicate_positions=int(geo.n_duplicate_positions),
            reversed_to_canonical=bool(geo.reversed_to_canonical),
            qc_flags="|".join(flags),
            usable=bool(usable),
        )
        rows.append(row)
    return rows


def _merge_coalescing(left, right, *, on: str):
    """Left-join, filling rather than duplicating columns that appear in both.

    ``train.csv`` carries ``PatientID`` and so does the DICOM rollup.  A plain
    ``merge`` renames both to ``PatientID_x`` / ``PatientID_y``, and the next
    stage -- which needs ``PatientID`` to group folds by patient -- dies with a
    bare ``KeyError``.  Worse, if it did not die, a surrogate group column would
    silently split a patient across folds.

    So: keep the left value where it is present, fall back to the right, and
    leave a single column with the original name.
    """
    import pandas as pd

    shared = [c for c in right.columns if c in left.columns and c != on]
    merged = left.merge(right, on=on, how="left", suffixes=("", "__rhs"))
    for c in shared:
        rhs = f"{c}__rhs"
        if rhs not in merged.columns:
            continue
        lhs_empty = merged[c].isna() | (merged[c].astype(str).str.strip() == "")
        merged[c] = merged[c].where(~lhs_empty, merged[rhs])
        merged = merged.drop(columns=[rhs])
    if shared:
        print(f"note: coalesced overlapping column(s) {shared} from {on}-join")
    return merged


def write_table(df, path: Path) -> Path:
    """Write parquet, falling back to CSV when no parquet engine is installed.

    Losing a completed corpus scan to a missing optional dependency is a bad
    trade; the caller gets a file either way, and a clear note about which.
    """
    try:
        df.to_parquet(path, index=False)
        return path
    except ImportError:
        alt = path.with_suffix(".csv")
        df.to_csv(alt, index=False)
        print("note: no parquet engine installed (pip install pyarrow); wrote CSV")
        return alt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="directory of study dirs")
    ap.add_argument("--labels", type=Path, help="CSV with StudyInstanceUID + 12 targets")
    ap.add_argument("--reports", type=Path,
                    help="CSV with StudyInstanceUID + report text (adds language/length)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-slices", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="scan only N studies (debug)")
    args = ap.parse_args()

    try:
        import pandas as pd
        import pydicom  # noqa: F401
    except ImportError as exc:
        print(f"missing dependency: {exc}. pip install -e '.[train]'", file=sys.stderr)
        return 2

    studies = [d for d in sorted(args.root.iterdir()) if d.is_dir()]
    if args.limit:
        studies = studies[: args.limit]
    if not studies:
        print(f"no study directories under {args.root}", file=sys.stderr)
        return 2
    print(f"scanning {len(studies)} studies with {args.workers} workers")

    t0 = time.monotonic()
    rows: list[dict] = []
    tasks = [(str(d), args.min_slices) for d in studies]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(scan_study, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs)):
                try:
                    rows.extend(fut.result())
                except Exception as exc:
                    print(f"  study failed: {exc}", file=sys.stderr)
                if (i + 1) % 200 == 0:
                    rate = (i + 1) / (time.monotonic() - t0)
                    print(f"  {i + 1}/{len(studies)}  {rate:.1f} studies/s  "
                          f"eta {(len(studies) - i - 1) / max(rate, 1e-9) / 60:.1f} min",
                          flush=True)
    else:
        for t in tasks:
            rows.extend(scan_study(t))

    df = pd.DataFrame(rows)
    print(f"\nscanned in {(time.monotonic() - t0) / 60:.1f} min: "
          f"{len(df)} series over {df['StudyInstanceUID'].nunique()} studies")

    # -- QC summary: the numbers that decide whether to trust the pipeline -- #
    usable = df.get("usable", pd.Series(dtype=bool)).fillna(False)
    print(f"usable series: {int(usable.sum())}/{len(df)} ({usable.mean():.1%})")
    flags = Counter()
    for f in df.get("qc_flags", pd.Series(dtype=str)).fillna(""):
        flags.update(x for x in str(f).split("|") if x)
    for k, v in flags.most_common():
        print(f"  {k:<32} {v:>6}  ({v / max(len(df), 1):.2%})")
    if "sequence_family" in df:
        print("\nsequence families:")
        for k, v in df["sequence_family"].value_counts().items():
            print(f"  {k:<20} {v:>6}")
        unknown = int((df["sequence_family"] == "unknown").sum())
        if unknown > 0.15 * len(df):
            print(f"\n!! {unknown / len(df):.1%} of series are 'unknown'. The taxonomy "
                  "needs the actual SeriesDescription vocabulary; run with --limit 200 "
                  "and inspect series_description before training.")

    # -- study-level rollup ---------------------------------------------- #
    g = df[usable] if usable.any() else df
    studies_df = (
        g.groupby("StudyInstanceUID")
        .agg(
            PatientID=("PatientID", "first"),
            n_series=("SeriesInstanceUID", "nunique"),
            n_slices=("n_slices", "sum"),
            manufacturer=("manufacturer", "first"),
            field_strength=("field_strength", "first"),
            laterality=("laterality", "first"),
            median_in_plane_mm=("in_plane_mm", "median"),
            median_slice_mm=("spacing_mm", "median"),
            families=("sequence_family", lambda s: "|".join(sorted(set(s)))),
        )
        .reset_index()
    )
    # ``site`` is not a DICOM tag; the closest available proxy is the
    # (manufacturer, model, field strength) triple.  Named honestly so nobody
    # mistakes it for the organisers' site identifier.
    studies_df["scanner_proxy"] = (
        studies_df["manufacturer"].astype(str) + "/"
        + studies_df["field_strength"].astype(str)
    )
    studies_df["field_strength_bucket"] = pd.cut(
        studies_df["field_strength"], [-1, 0.9, 1.9, 2.5, 99],
        labels=["unknown", "1.5T", "3T", "high"],
    ).astype(str)

    if args.labels and args.labels.exists():
        lab = pd.read_csv(args.labels)
        missing = [t for t in TARGETS if t not in lab.columns]
        if missing:
            print(f"!! labels CSV missing target columns: {missing}", file=sys.stderr)
        studies_df = _merge_coalescing(studies_df, lab, on="StudyInstanceUID")
        have = studies_df[list(set(TARGETS) & set(studies_df.columns))].notna().all(axis=1)
        print(f"\nlabelled studies: {int(have.sum())}/{len(studies_df)}")

    if args.reports and args.reports.exists():
        rep = pd.read_csv(args.reports)
        text_col = next(
            (c for c in rep.columns
             if any(k in c.lower() for k in ("report", "text", "impression", "finding"))),
            None,
        )
        if text_col:
            rep["report_length"] = rep[text_col].astype(str).str.len()
            keep = ["StudyInstanceUID", "report_length"]
            if "language" in rep.columns:
                keep.append("language")
            studies_df = studies_df.merge(rep[keep], on="StudyInstanceUID", how="left")
            print(f"reports joined on column {text_col!r}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    series_path = write_table(df, args.out.with_name(args.out.stem + "_series.parquet"))
    studies_path = write_table(studies_df, args.out)
    args.out.with_suffix(".summary.json").write_text(json.dumps({
        "n_series": int(len(df)),
        "n_studies": int(df["StudyInstanceUID"].nunique()),
        "usable_fraction": float(usable.mean()) if len(df) else 0.0,
        "qc_flags": dict(flags),
        "families": (df["sequence_family"].value_counts().to_dict()
                     if "sequence_family" in df else {}),
        "scan_minutes": (time.monotonic() - t0) / 60,
    }, indent=2, default=str))

    print(f"\nwrote {studies_path} ({len(studies_df)} studies)")
    print(f"      {series_path} ({len(df)} series)")
    print(f"      {args.out.with_suffix('.summary.json')}")
    print(f"\nNext: python scripts/01_make_folds.py --manifest {args.out} "
          "--out artifacts/folds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
