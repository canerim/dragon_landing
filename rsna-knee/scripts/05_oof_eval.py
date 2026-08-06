#!/usr/bin/env python3
"""Evaluate an OOF matrix, run the audit suite, and gate the pipeline.

    python scripts/05_oof_eval.py --oof artifacts/oof/convnext_s.npz \
        --folds artifacts/folds/folds.parquet --report artifacts/reports/convnext_s.txt

Exit codes:
    0  everything passed
    1  a **blocking** audit failed -- do not build a submission from this model
    2  bad inputs

The blocking behaviour is the point. An audit suite that only warns is an audit
suite that gets ignored at 2 a.m. on the last day.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from kairos.constants import TARGETS
from kairos.eval.leakage import run_audit_suite
from kairos.eval.metrics import delong_test, evaluate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", required=True, type=Path,
                    help="npz with keys: logits (N,12), targets (N,12), study_uid, "
                         "and optionally embeddings, hashes")
    ap.add_argument("--folds", required=True, type=Path)
    ap.add_argument("--baseline-oof", type=Path,
                    help="a second OOF matrix; runs paired DeLong tests against it")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--allow-blocking-failures", action="store_true",
                    help="exit 0 even on a blocking failure (development only)")
    args = ap.parse_args()

    import pandas as pd

    z = np.load(args.oof, allow_pickle=True)
    logits = z["logits"].astype(np.float64)
    targets = z["targets"].astype(np.float64)
    uids = [str(u) for u in z["study_uid"]]
    fold_hash = str(z["fold_hash"]) if "fold_hash" in z else ""

    folds = pd.read_parquet(args.folds).set_index("StudyInstanceUID")
    try:
        rows = folds.loc[uids]
    except KeyError as exc:
        print(f"OOF contains study UIDs absent from the fold table: {exc}", file=sys.stderr)
        return 2

    fold = rows["fold"].to_numpy()
    group_col = next((c for c in ("PatientID", "patient_id", "group_id") if c in rows), None)
    group = rows[group_col].astype(str).to_numpy() if group_col else np.array(uids)
    site = rows["site"].to_numpy() if "site" in rows else None

    scores = 1.0 / (1.0 + np.exp(-logits))

    report = evaluate(
        targets, scores, group_id=group, fold=fold, site=site, n_boot=args.n_boot
    )
    text = [report.to_text(TARGETS), ""]

    metadata = None
    meta_cols = [c for c in rows.columns if c not in {"fold", group_col}]
    if meta_cols:
        metadata = np.stack(
            [pd.factorize(rows[c])[0].astype(float) for c in meta_cols], axis=1
        )

    suite = run_audit_suite(
        y=targets,
        scores=scores,
        fold=fold,
        group_id=group,
        metadata=metadata,
        site=site,
        hashes=[str(h) for h in z["hashes"]] if "hashes" in z else None,
        embeddings=z["embeddings"] if "embeddings" in z else None,
    )
    text += ["=" * 72, "AUDIT SUITE", "=" * 72, suite.to_text(), ""]

    if args.baseline_oof is not None:
        b = np.load(args.baseline_oof, allow_pickle=True)
        b_scores = 1.0 / (1.0 + np.exp(-b["logits"].astype(np.float64)))
        text += ["=" * 72, "PAIRED DeLONG vs BASELINE", "=" * 72]
        w = max(len(t) for t in TARGETS) + 2
        text.append("label".ljust(w) + "   this    base     diff      se       p")
        n_better = 0
        for l, name in enumerate(TARGETS):
            r = delong_test(targets[:, l], scores[:, l], b_scores[:, l])
            flag = "*" if r["p_value"] < 0.05 else " "
            n_better += int(r["diff"] > 0 and r["p_value"] < 0.05)
            text.append(
                name.ljust(w)
                + f" {r['auc_a']:.4f}  {r['auc_b']:.4f}  {r['diff']:+.4f}"
                + f"  {r['se']:.4f}  {r['p_value']:.4f}{flag}"
            )
        text.append(f"\nsignificantly better on {n_better}/{len(TARGETS)} labels "
                    "(uncorrected; treat < 3 as noise)")

    text.append("")
    text.append(f"fold_hash in OOF: {fold_hash or '(absent -- this is a problem)'}")
    out = "\n".join(text)
    print(out)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(out)
        print(f"\nwrote {args.report}")

    if not fold_hash:
        print("\nERROR: this OOF matrix carries no fold hash and cannot be safely "
              "ensembled.", file=sys.stderr)
        return 1
    if suite.blocking and not args.allow_blocking_failures:
        print("\nERROR: blocking audit failure -- do not build a submission from this "
              "model until it is resolved.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
