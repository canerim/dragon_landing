#!/usr/bin/env python3
"""Build the immutable cross-validation split.

Run this ONCE. The output (``folds.parquet`` + ``folds.json``) is a competition
artefact: every checkpoint records its ``fold_hash``, and the trainer refuses to
load a checkpoint whose hash disagrees with the current split. Regenerating the
split invalidates every OOF matrix you have.

    python scripts/01_make_folds.py \
        --manifest artifacts/manifest.parquet \
        --out artifacts/folds \
        --n-folds 5 --anneal-steps 60000

The manifest must contain, per study row: ``StudyInstanceUID``, a patient/group
id, the twelve target columns, and whatever covariates you want balanced
(``site``, ``report_language``, ``manufacturer``, ``field_strength``,
``n_sequences``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kairos.constants import TARGETS
from kairos.data.folds import FoldSpec, fold_report, make_folds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--anneal-steps", type=int, default=60_000)
    ap.add_argument("--group-col", default="PatientID")
    ap.add_argument(
        "--covariates",
        nargs="*",
        default=["site", "report_language", "manufacturer", "field_strength_bucket"],
    )
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing split (invalidates every OOF matrix)")
    args = ap.parse_args()

    out_dir = args.out
    if (out_dir / "folds.json").exists() and not args.force:
        print(f"REFUSING: {out_dir / 'folds.json'} already exists.\n"
              "A split is an immutable artefact; pass --force only if you intend to\n"
              "invalidate every OOF matrix and checkpoint that references it.")
        return 2

    df = pd.read_parquet(args.manifest) if args.manifest.suffix == ".parquet" \
        else pd.read_csv(args.manifest)
    print(f"manifest: {len(df)} rows, {df[args.group_col].nunique()} groups")

    missing = [t for t in TARGETS if t not in df.columns]
    if missing:
        raise SystemExit(f"manifest is missing target columns: {missing}")

    labels = df[list(TARGETS)].to_numpy(dtype=np.float64)
    covariates = {c: df[c].tolist() for c in args.covariates if c in df.columns}
    skipped = [c for c in args.covariates if c not in df.columns]
    if skipped:
        print(f"note: covariates absent from the manifest, skipped: {skipped}")

    spec = FoldSpec(
        n_folds=args.n_folds, seed=args.seed, n_anneal_steps=args.anneal_steps
    )
    result = make_folds(
        group_id=df[args.group_col].astype(str).tolist(),
        labels=labels,
        covariates=covariates,
        spec=spec,
    )

    print()
    print(fold_report(result, TARGETS))
    print()
    print(f"seed objective  {result.diagnostics['seed_objective']:.6f}")
    print(f"final objective {result.diagnostics['final_objective']:.6f}  "
          f"(improvement {result.diagnostics['improvement']:.6f})")

    out_dir.mkdir(parents=True, exist_ok=True)
    df_out = df[[c for c in (["StudyInstanceUID", args.group_col] + list(covariates))
                 if c in df.columns]].copy()
    df_out["fold"] = result.row_fold
    df_out.to_parquet(out_dir / "folds.parquet", index=False)

    (out_dir / "folds.json").write_text(
        json.dumps(
            {
                "fold_hash": result.fold_hash,
                "objective": result.objective,
                "spec": vars(spec),
                "diagnostics": result.diagnostics,
                "group_col": args.group_col,
                "covariates": list(covariates),
            },
            indent=2,
            default=float,
        )
    )
    print(f"\nwrote {out_dir}/folds.parquet and folds.json")
    print(f"FOLD HASH: {result.fold_hash}")
    print("Record this hash. Every checkpoint and OOF matrix must carry it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
