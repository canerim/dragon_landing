#!/usr/bin/env python3
"""Fit per-label ensemble weights, honestly.

    python scripts/06_ensemble.py --oof artifacts/oof/*.npz \
        --folds artifacts/folds/folds.parquet --out artifacts/ensemble

The script's job is as much to *refuse* weighting as to fit it. It reports the
nested leave-one-fold-out comparison and, unless ``--force-weighted`` is given,
writes uniform weights whenever the nested weighted macro-AUC does not beat the
nested uniform macro-AUC by more than one bootstrap standard error.

That rule is the difference between a per-label ensemble that gains 0.003 on the
private split and one that loses 0.003 on it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kairos.constants import GROUP_OF_LABEL, LABEL_GROUPS, TARGETS
from kairos.ensemble.weights import (
    greedy_selection,
    nested_evaluate,
    rank_transform,
)
from kairos.eval.metrics import macro_auc, patient_bootstrap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", nargs="+", required=True, type=Path)
    ap.add_argument("--folds", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--anchor-lambda", type=float, default=0.15)
    ap.add_argument("--use-ranks", action="store_true", default=True)
    ap.add_argument("--force-weighted", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    import pandas as pd

    mats, names, hashes, uid_ref, targets = [], [], set(), None, None
    for p in args.oof:
        z = np.load(p, allow_pickle=True)
        uids = [str(u) for u in z["study_uid"]]
        if uid_ref is None:
            uid_ref, targets = uids, z["targets"].astype(np.float64)
        elif uids != uid_ref:
            # Re-order rather than fail: OOF matrices are written per model and
            # the row order legitimately differs.
            order = {u: i for i, u in enumerate(uids)}
            idx = [order[u] for u in uid_ref]
            z = {k: (z[k][idx] if getattr(z[k], "shape", (0,))[:1] == (len(uids),) else z[k])
                 for k in z.files}
        mats.append(np.asarray(z["logits"], dtype=np.float64))
        names.append(p.stem)
        hashes.add(str(z["fold_hash"]) if "fold_hash" in z else "")

    if len(hashes) > 1:
        raise SystemExit(
            f"OOF matrices span {len(hashes)} distinct fold hashes: {hashes}\n"
            "They were produced under different splits and cannot be ensembled."
        )

    preds = np.stack(mats)  # (M, N, L)
    M, N, L = preds.shape
    print(f"{M} models × {N} studies × {L} labels")

    folds = pd.read_parquet(args.folds).set_index("StudyInstanceUID")
    rows = folds.loc[uid_ref]
    fold = rows["fold"].to_numpy()
    group_col = next((c for c in ("PatientID", "patient_id", "group_id") if c in rows), None)
    group = rows[group_col].astype(str).to_numpy() if group_col else np.array(uid_ref)

    group_names = list(LABEL_GROUPS)
    gidx = np.array([group_names.index(GROUP_OF_LABEL[t]) for t in TARGETS])

    ew = nested_evaluate(
        targets, preds, fold,
        model_names=names,
        anchor_lambda=args.anchor_lambda,
        group_of_label=gidx,
        use_ranks=args.use_ranks,
    )
    print()
    print(ew.summary(TARGETS))

    ranked = np.stack([rank_transform(preds[m]) for m in range(M)])
    uniform = ranked.mean(axis=0)
    boot = patient_bootstrap(targets, uniform, group, n_boot=args.n_boot)
    se = float(np.nanstd(boot))
    delta = ew.nested_macro_weighted - ew.nested_macro_uniform
    print(f"\nbootstrap SE of the macro-AUC: {se:.5f}")
    print(f"nested gain from weighting:    {delta:+.5f}  "
          f"({delta / max(se, 1e-9):+.2f} SE)")

    chosen, greedy_score = greedy_selection(targets, preds, max_size=min(3 * M, 24))
    print(f"greedy selection picks {[names[i] for i in chosen]} -> {greedy_score:.5f}")

    use_weighted = args.force_weighted or delta > se
    if use_weighted:
        W = ew.weights
        print("\nDECISION: shipping PER-LABEL weights (nested gain exceeds 1 SE)")
    else:
        W = np.full((L, M), 1.0 / M)
        print("\nDECISION: shipping the UNIFORM average.")
        print("  The nested gain does not exceed one bootstrap SE, so the per-label")
        print("  weights are fitting OOF noise. Pass --force-weighted to override.")

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "ensemble_weights.npy", W)
    (args.out / "ensemble.json").write_text(
        json.dumps(
            {
                "models": names,
                "fold_hash": hashes.pop() if hashes else "",
                "weighted": bool(use_weighted),
                "oof_macro_uniform": ew.oof_macro_uniform,
                "oof_macro_weighted": ew.oof_macro_weighted,
                "nested_macro_uniform": ew.nested_macro_uniform,
                "nested_macro_weighted": ew.nested_macro_weighted,
                "bootstrap_se": se,
                "greedy_selection": [names[i] for i in chosen],
                "greedy_macro": greedy_score,
                "final_macro": macro_auc(targets, np.einsum("lm,mnl->nl", W, ranked)),
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}/ensemble_weights.npy and ensemble.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
