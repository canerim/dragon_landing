#!/usr/bin/env python3
"""Build the cross-fitted teacher that stage S4 distils from.

    python scripts/07_make_teacher.py \
        --oof runs/convnext_s_f*/oof.npz runs/swin_f*/oof.npz \
        --out artifacts/teacher.parquet

    python scripts/04_train.py ... --teacher artifacts/teacher.parquet

Why this script exists
----------------------

The curriculum schedules ``kd`` at weight 0.6 through the whole of S4 and names
its teacher "the cross-fitted OOF ensemble".  Nothing produced that artefact, so
every run had to pass ``--disable kd`` -- a scheduled objective that could never
be computed, which is precisely the failure the objective validator exists to
make loud.

What "cross-fitted" buys, and what it does not
----------------------------------------------

Each ``oof.npz`` holds the predictions a fold-*k* run made on **its own
validation fold**, i.e. on studies that run never trained on.  Concatenating the
folds therefore gives, for every study, a prediction from a model that has not
seen it.  That is the property distillation needs: without it the teacher's
logit for a training study is partly a memorised label, and the student learns
to memorise it too.

The residual, stated plainly rather than glossed over: the student for fold *k*
trains on folds ``!= k``, and the teacher logit for a study in fold *m* comes
from the run that validated on *m* -- a run that trained on folds ``!= m``,
which **includes** fold *k*.  So the teacher's parameters encode fold-*k*
information, and a trace of it can reach the student through the teacher's
outputs on training studies.  Removing that would need leave-two-out teachers
(one run per ordered fold pair, i.e. 20 runs instead of 5).  We do not pay that,
and the consequence is that a KD-trained student's fold-*k* OOF is mildly
optimistic relative to a no-KD student's.  Compare the two when deciding whether
KD helped; do not compare a KD student's OOF against a published number.

Averaging is in **probability** space, not rank space.  Ranks are not
probabilities and a distillation target has to be one -- this is the one place
in the pipeline where rank averaging is the wrong answer (see
``kairos.ensemble.weights`` for the other direction).

Guards
------

* Every input must carry the **same fold hash**.  Mixing splits means some
  study's "out-of-fold" prediction came from a model that trained on it, which
  is the leak this whole construction exists to prevent.
* A study predicted twice by the *same run* is a bug in that run's OOF writer,
  not something to silently average.
* Studies whose teacher is missing are simply absent from the output; the
  dataloader leaves them NaN and the ``kd`` term drops them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from kairos.constants import NUM_TARGETS, TARGETS

_EPS = 1e-6


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Stable logistic; an AUC-margin objective produces large |z|."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def load_oof(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    for key in ("logits", "study_uid"):
        if key not in d:
            raise KeyError(f"{path}: missing '{key}'")
    logits = np.asarray(d["logits"], dtype=np.float64)
    uids = [str(u) for u in np.asarray(d["study_uid"], dtype=object).ravel()]
    if logits.shape != (len(uids), NUM_TARGETS):
        raise ValueError(
            f"{path}: logits {logits.shape} do not match {len(uids)} uids "
            f"x {NUM_TARGETS} targets"
        )
    if len(set(uids)) != len(uids):
        raise ValueError(
            f"{path}: the same study appears more than once in one run's OOF; "
            "that is an OOF-writer bug, not something to average away"
        )
    return {
        "path": path,
        "logits": logits,
        "uids": uids,
        "fold": int(d["fold"]) if "fold" in d else -1,
        "fold_hash": str(d["fold_hash"]) if "fold_hash" in d else "",
    }


def build_teacher(runs: list[dict]) -> tuple[list[str], np.ndarray, dict]:
    """Average member probabilities per study; return (uids, logits, stats)."""
    hashes = {r["fold_hash"] for r in runs if r["fold_hash"]}
    if len(hashes) > 1:
        raise SystemExit(
            "the OOF files span more than one fold hash:\n  "
            + "\n  ".join(sorted(hashes))
            + "\n\nA teacher built across splits is not out-of-fold: some study's "
            "'teacher' came from a model that trained on it."
        )

    acc: dict[str, list[np.ndarray]] = {}
    for r in runs:
        probs = _sigmoid(r["logits"])
        for uid, p in zip(r["uids"], probs):
            acc.setdefault(uid, []).append(p)

    uids = sorted(acc)
    n_members = np.array([len(acc[u]) for u in uids])
    mean_p = np.stack([np.mean(acc[u], axis=0) for u in uids])
    stats = {
        "n_studies": len(uids),
        "n_runs": len(runs),
        "members_per_study_min": int(n_members.min()) if len(uids) else 0,
        "members_per_study_max": int(n_members.max()) if len(uids) else 0,
        "members_per_study_mean": float(n_members.mean()) if len(uids) else 0.0,
        "fold_hash": (hashes.pop() if hashes else ""),
        "mean_prob_per_label": {
            t: float(mean_p[:, i].mean()) for i, t in enumerate(TARGETS)
        },
    }
    return uids, _logit(mean_p), stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--oof", type=Path, nargs="+", required=True,
                    help="oof.npz files, one per (backbone, fold) run")
    ap.add_argument("--out", type=Path, default=Path("artifacts/teacher.parquet"))
    ap.add_argument("--min-members", type=int, default=1,
                    help="drop studies covered by fewer runs than this")
    args = ap.parse_args()

    runs = []
    for p in args.oof:
        try:
            runs.append(load_oof(p))
        except Exception as exc:
            print(f"!! SKIP {p}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if not runs:
        print("no usable OOF files", file=sys.stderr)
        return 1

    print(f"{len(runs)} run(s):")
    for r in runs:
        print(f"  {r['path'].parent.name}/{r['path'].name}  fold {r['fold']}  "
              f"{len(r['uids'])} studies  hash {r['fold_hash'][:12]}")

    uids, logits, stats = build_teacher(runs)

    if args.min_members > 1:
        counts = {}
        for r in runs:
            for u in r["uids"]:
                counts[u] = counts.get(u, 0) + 1
        keep = [i for i, u in enumerate(uids) if counts[u] >= args.min_members]
        dropped = len(uids) - len(keep)
        if dropped:
            print(f"dropping {dropped} study(ies) covered by "
                  f"< {args.min_members} run(s)")
        uids = [uids[i] for i in keep]
        logits = logits[keep]
        stats["n_studies"] = len(uids)

    # A teacher that is 0.5 everywhere is not a teacher; catch it here rather
    # than after a wasted S4.
    spread = float(np.ptp(logits, axis=0).min()) if len(uids) else 0.0
    if spread < 1e-6:
        print("!! at least one teacher column is constant -- check the OOF files",
              file=sys.stderr)

    import pandas as pd

    df = pd.DataFrame(logits, columns=list(TARGETS))
    df.insert(0, "StudyInstanceUID", uids)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(args.out, index=False)
        written = args.out
    except Exception:
        written = args.out.with_suffix(".csv")
        df.to_csv(written, index=False)
    (args.out.with_suffix(".json")).write_text(json.dumps(stats, indent=2))

    print(f"\nwrote {written}  ({stats['n_studies']} studies, "
          f"{stats['members_per_study_mean']:.2f} members/study)")
    print("teacher mean probability per label:")
    print("  " + ", ".join(f"{t.split()[0][:6]}={v:.3f}"
                           for t, v in stats["mean_prob_per_label"].items()))
    print(f"\nNext: python scripts/04_train.py ... --teacher {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
