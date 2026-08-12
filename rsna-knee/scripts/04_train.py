#!/usr/bin/env python3
"""Train one fold of KAIROS.

    python scripts/04_train.py \
        --folds artifacts/folds/folds.parquet \
        --manifest artifacts/manifest.parquet \
        --fold 0 --budget medium --out runs/convnext_s_f0

    # dry run on synthetic studies -- no data, no GPU, ~1 minute
    python scripts/04_train.py --synthetic --fold 0 --budget small --epochs-cap 2

What this does, in order:

1. Loads the fold artefact and **records its hash**; every checkpoint carries
   it and ``Trainer.load`` refuses a mismatch.
2. Builds the model, the curriculum plan and the objective registry.
3. **Validates the schedule against a real batch.** If the curriculum activates
   a term the dataloader cannot feed -- ``kd`` without teacher logits,
   ``ot_ground`` without phrase embeddings -- the run stops here with a message
   naming the missing fields, instead of silently training a smaller objective.
   Pass ``--disable term,term`` to proceed deliberately; the names are written
   into the run manifest.
4. Trains, evaluating out-of-fold every ``--eval-every`` epochs and keeping the
   best checkpoint by a **multi-objective** criterion (macro-AUC with a
   worst-label floor), not by macro-AUC alone.
5. Writes ``oof.npz`` (logits, targets, uids, fold hash) ready for
   ``05_oof_eval.py`` and ``06_ensemble.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from kairos.constants import NUM_TARGETS, TARGETS
from kairos.data.folds import resolve_environments
from kairos.data.loader import SamplerConfig, StudyDataset, build_dataloader, infer_prevalence
from kairos.data.transforms import AugmentConfig
from kairos.eval.metrics import per_label_auc
from kairos.models.backbones import BackboneSpec
from kairos.models.system import KairosConfig, KairosModel
from kairos.train.curriculum import default_plan, student_plan
from kairos.train.loop import RunManifest, TrainConfig, Trainer
from kairos.train.objectives import ObjectiveConfig, build_objectives


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def package_versions() -> dict:
    import platform

    out = {"python": platform.python_version(), "torch": torch.__version__}
    for mod in ("numpy", "pandas", "timm", "pydicom"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "absent"
    return out


# --------------------------------------------------------------------------- #
# Synthetic data path (for --synthetic)                                        #
# --------------------------------------------------------------------------- #


#: The two producers do not agree on a column convention and should not have
#: to: ``02_parse_reports.py`` writes ``weak_<target>`` / ``conf_<target>``
#: (the same table also carries ``state_<target>``), while
#: ``07_make_teacher.py`` writes bare target names.  Resolving prefixes here,
#: once, is much safer than making either producer guess what the consumer
#: wants -- a silent mismatch would leave the term looking present and reading
#: nothing.
_VALUE_PREFIXES = ("", "weak_", "teacher_")
_CONF_PREFIXES = ("conf_", "confidence_")


def _pick_columns(df, prefixes, suffix: str = ""):
    for pre in prefixes:
        cols = [f"{pre}{t}{suffix}" for t in TARGETS]
        if all(c in df.columns for c in cols):
            return cols
    return None


def _load_per_study(path, uids, what: str, *, want_confidence: bool = False):
    """Align a per-study, per-label table to ``uids``.

    Returns ``(values, confidence)``; missing studies are NaN, which is the
    "unobserved" convention every masked loss in this codebase already honours.
    Alignment is by ``StudyInstanceUID`` and never by row order -- a positional
    join between two artefacts built at different times is the single most
    destructive silent bug in a pipeline like this, and it is invisible in every
    metric until the private leaderboard.
    """
    if path is None:
        return None, None
    import pandas as pd

    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    uid_col = next((c for c in df.columns if "study" in c.lower()), df.columns[0])
    value_cols = _pick_columns(df, _VALUE_PREFIXES)
    if value_cols is None:
        raise SystemExit(
            f"{path}: {what} table has no complete set of target columns. "
            f"Looked for prefixes {_VALUE_PREFIXES} over {len(TARGETS)} targets; "
            f"found {list(df.columns)[:8]}..."
        )

    df[uid_col] = df[uid_col].astype(str)
    df = df.drop_duplicates(subset=[uid_col], keep="first").set_index(uid_col)
    values = df.reindex(list(uids))[value_cols].to_numpy(dtype=np.float32)
    covered = int(np.isfinite(values).any(axis=1).sum())
    print(f"{what}: {covered}/{len(uids)} studies covered ({path})")
    if covered == 0:
        raise SystemExit(
            f"{path}: no study UID matched the cohort. Check that this artefact "
            "was built from the same manifest."
        )

    conf = None
    if want_confidence:
        conf_cols = _pick_columns(df, _CONF_PREFIXES) or _pick_columns(
            df, ("",), suffix="_confidence"
        )
        if conf_cols is not None:
            conf = df.reindex(list(uids))[conf_cols].to_numpy(dtype=np.float32)
        else:
            # No per-label confidence shipped: treat an extracted label as
            # fully confident where it exists and absent where it does not.
            # The weak-label term gates on confidence, so 0 is the correct
            # value for a missing entry -- not 1.
            conf = np.where(np.isfinite(values), 1.0, 0.0).astype(np.float32)
        conf = np.nan_to_num(conf, nan=0.0)
    return values, conf


def synthetic_cohort(n_studies: int, size: int, seed: int):
    from kairos.constants import Plane
    from kairos.data.dataset import SeriesRecord, StudyRecord

    rng = np.random.default_rng(seed)
    prev = np.array([0.22, 0.14, 0.30, 0.18, 0.25, 0.16,
                     0.20, 0.35, 0.06, 0.11, 0.09, 0.03])
    labels = (rng.random((n_studies, NUM_TARGETS)) < prev).astype(np.float32)
    uids = [f"synth.{i}" for i in range(n_studies)]
    groups = [f"p{i // 2}" for i in range(n_studies)]

    def load(uid: str) -> StudyRecord:
        i = int(uid.split(".")[1])
        r = np.random.default_rng(seed + i)
        series = []
        for s in range(int(r.integers(2, 4))):
            n = int(r.integers(8, 13))
            vol = r.normal(0, 1, (n, size, size)).astype(np.float32)
            # Plant a weak, label-dependent signal so the loss can actually go
            # down -- a smoke run on pure noise cannot distinguish "training
            # works" from "training does nothing".
            vol += 0.9 * labels[i, s % NUM_TARGETS]
            series.append(SeriesRecord(
                series_uid=f"{uid}.{s}", family_index=int(r.integers(1, 12)),
                plane=Plane.SAGITTAL, pixels=vol,
                z_mm=np.cumsum(r.uniform(2.7, 3.3, n)).astype(np.float32),
                spacing_mm=3.0, in_plane_mm=0.5,
                field_strength=3.0, te_ms=35.0, tr_ms=3000.0,
            ))
        rec = StudyRecord(study_uid=uid, series=series)
        rec.labels = labels[i]
        return rec

    return uids, labels, groups, load


# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=Path, help="folds.parquet from 01_make_folds.py")
    ap.add_argument("--fold-json", type=Path, help="folds.json (defaults next to --folds)")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--images", type=Path,
                    help="training image root; defaults to the discovered "
                         "<root>/train_series")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/kairos"))
    ap.add_argument("--budget", choices=("small", "medium", "large"), default="medium")
    ap.add_argument("--student", action="store_true", help="efficiency-track plan")

    ap.add_argument("--backbone", default="convnext_small.fb_in22k_ft_in1k")
    ap.add_argument("--allow-fallback-backbone", action="store_true",
                    help="permit the random-init FallbackEncoder if the named "
                         "backbone cannot be built (default: refuse and exit)")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--aggregator", choices=("transformer", "ssm", "both"),
                    default="transformer")
    ap.add_argument("--image-size", type=int, default=256)

    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--backbone-lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=4e-4)
    ap.add_argument("--amp", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--asam", action="store_true")
    ap.add_argument("--gradient-surgery", default="aligned",
                    choices=("none", "aligned", "pcgrad", "cagrad"))

    ap.add_argument("--disable", default="",
                    help="comma-separated objective terms to switch off deliberately")
    ap.add_argument("--teacher", type=Path,
                    help="teacher.parquet from 07_make_teacher.py; makes the "
                         "S4 `kd` term computable")
    ap.add_argument("--weak-labels", type=Path,
                    help="weak_labels.parquet from 02_parse_reports.py; makes "
                         "the S2 `weak_label` term computable")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--epochs-cap", type=int, default=0,
                    help="stop after N epochs regardless of the plan (debugging)")
    ap.add_argument("--synthetic", action="store_true",
                    help="run on generated studies; no data or GPU required")
    ap.add_argument("--synthetic-n", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # -- fold artefact ---------------------------------------------------- #
    if args.synthetic:
        size = min(args.image_size, 64)
        uids, labels, groups, load_fn = synthetic_cohort(args.synthetic_n, size, args.seed)
        fold_of = np.array([i % 5 for i in range(len(uids))])
        fold_hash = "synthetic-" + "0" * 8
        # Two environments, so group_dro / irm are exercised rather than
        # silently degenerate in the smoke path.
        sites = np.arange(len(uids)) % 2
        site_col = "synthetic"
    else:
        if args.folds is None:
            ap.error("--folds is required unless --synthetic is given")
        import pandas as pd

        fold_json = args.fold_json or args.folds.with_name("folds.json")
        if not fold_json.exists():
            ap.error(f"{fold_json} not found; it carries the fold hash and is required")
        fold_hash = json.loads(fold_json.read_text())["fold_hash"]

        folds = pd.read_parquet(args.folds)
        man = pd.read_parquet(args.manifest) if args.manifest else folds
        df = folds.merge(man, on="StudyInstanceUID", how="left", suffixes=("", "_m"))
        uids = df["StudyInstanceUID"].astype(str).tolist()
        missing = [t for t in TARGETS if t not in df.columns]
        if missing:
            ap.error(f"manifest is missing target columns: {missing}")
        labels = df[list(TARGETS)].to_numpy(dtype=np.float32)
        gcol = next((c for c in ("PatientID", "patient_id", "group_id") if c in df), None)
        groups = df[gcol].astype(str).tolist() if gcol else uids
        fold_of = df["fold"].to_numpy()
        sites, site_col = resolve_environments(df)

        from kairos.data.dataset import load_study
        from kairos.io.layout import discover

        # The images are under <root>/train_series/<study>/<series>/*.dcm, and
        # <root>/train.csv sits next to it -- so a bare "<root>/train" guess
        # finds nothing.  Discovery handles that in one place.
        images = args.images
        if images is None:
            layout = discover()
            images = layout.train_images
        if images is None or not Path(images).is_dir():
            ap.error(
                "could not locate the training images; pass --images explicitly "
                "(expected <root>/train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm)"
            )
        images = Path(images)
        print(f"training images: {images}")

        def load_fn(uid: str):
            return load_study(images / uid, out_size=args.image_size)

    available = sorted(int(f) for f in np.unique(fold_of))
    if args.fold not in available:
        ap.error(
            f"--fold {args.fold} is not present in the split (folds: {available}). "
            "Training would otherwise use 100% of the data and report an empty "
            "validation set as success."
        )
    train_idx = np.flatnonzero(fold_of != args.fold)
    val_idx = np.flatnonzero(fold_of == args.fold)
    if val_idx.size == 0 or train_idx.size == 0:
        ap.error(f"fold {args.fold} leaves {train_idx.size} train / {val_idx.size} val")
    print(f"fold {args.fold}: {len(train_idx)} train / {len(val_idx)} val studies")
    print(f"fold hash: {fold_hash}")

    prevalence = infer_prevalence(labels[train_idx])
    print("train prevalence: " + ", ".join(
        f"{t}={p:.3f}" for t, p in zip(TARGETS, prevalence)))

    # -- data ------------------------------------------------------------- #
    weak, weak_conf = _load_per_study(args.weak_labels, uids, "weak labels",
                                      want_confidence=True)
    teacher, _ = _load_per_study(args.teacher, uids, "teacher logits")

    def subset(idx, augment):
        take = lambda a: None if a is None else a[idx]  # noqa: E731
        return StudyDataset(
            [uids[i] for i in idx],
            load_fn=load_fn,
            labels=labels[idx],
            augment=augment,
            group_id=[groups[i] for i in idx],
            env_id=sites[idx],
            weak_labels=take(weak),
            weak_confidence=take(weak_conf),
            teacher_logits=take(teacher),
        )

    train_ds = subset(train_idx, AugmentConfig(seed=args.seed))
    val_ds = subset(val_idx, None)

    scfg = SamplerConfig(batch_size=args.batch_size, seed=args.seed)
    train_loader = build_dataloader(train_ds, sampler_cfg=scfg,
                                    num_workers=args.workers, device=args.device)
    val_loader = build_dataloader(
        val_ds, sampler_cfg=SamplerConfig(batch_size=args.batch_size, shuffle=False,
                                          drop_last=False),
        num_workers=args.workers, device=args.device, balanced=False,
    )

    # -- model ------------------------------------------------------------ #
    mcfg = KairosConfig(
        backbone=BackboneSpec(name=args.backbone, pretrained=not args.no_pretrained,
                              in_chans=5,
                              allow_fallback=args.allow_fallback_backbone),
        dim=args.dim, aggregator=args.aggregator,
    )
    try:
        model = KairosModel(mcfg).to(args.device)
    except RuntimeError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Report what was *built*, not what was requested.  Printing args.backbone
    # is how a silently-substituted encoder stayed invisible: every log line
    # named convnext_small while a 4.2M random-init CNN was training.
    built = args.backbone if model.backbone.is_timm else "FallbackEncoder (RANDOM INIT)"
    print(f"model: {built}, {n_par / 1e6:.1f}M trainable parameters")

    # Optimiser steps, not batches.  Trainer.step increments once per
    # *optimiser* step, so counting batches makes the curriculum advance
    # accum_steps times too slowly: with accum=4 a plan that nominally covers
    # 30 epochs only ever reaches a quarter of the way through its stages, and
    # the ranking and robustness stages never run at all.
    batches_per_epoch = max(1, len(train_idx) // max(args.batch_size, 1))
    steps_per_epoch = max(1, batches_per_epoch // max(args.accum, 1))
    plan = (student_plan(steps_per_epoch=steps_per_epoch) if args.student
            else default_plan(steps_per_epoch=steps_per_epoch, budget=args.budget))
    print(f"curriculum: {plan.total_epochs} epochs, {plan.total_steps} optimiser "
          f"steps ({batches_per_epoch} batches/epoch, accum {args.accum})")

    objectives = build_objectives(
        ObjectiveConfig(prevalence=prevalence.tolist(),
                        num_groups=max(int(sites.max()) + 1, 2),
                        ssl_dim=args.dim),
        model=model, device=args.device,
    )

    # -- schedule validation against a REAL batch ------------------------- #
    sample = next(iter(train_loader))
    disabled = {t.strip() for t in args.disable.split(",") if t.strip()}
    # Terms whose inputs this invocation cannot supply are disabled *here*,
    # loudly, and recorded in the run manifest -- rather than left to trip the
    # schedule validator with an error the user then silences by hand.  The
    # distinction that matters is between "omitted on purpose, written down"
    # and "omitted silently"; this keeps the first and never allows the second.
    if site_col is None or int(sites.max()) == 0:
        disabled |= {"group_dro", "irm"}
        print("!! single acquisition environment: disabling group_dro and irm")
    if teacher is None:
        disabled.add("kd")
        print("!! no --teacher: disabling kd "
              "(build one with scripts/07_make_teacher.py)")
    if weak is None:
        disabled.add("weak_label")
        print("!! no --weak-labels: disabling weak_label "
              "(build one with scripts/02_parse_reports.py)")
    if sample.text_embedding is None:
        disabled.add("contrastive")
    if sample.phrase_embedding is None:
        disabled.add("ot_ground")
    tcfg = TrainConfig(
        fold=args.fold, seed=args.seed, backbone_lr=args.backbone_lr,
        head_lr=args.head_lr, accum_steps=args.accum, amp_dtype=args.amp,
        use_asam=args.asam, gradient_surgery=args.gradient_surgery,
        out_dir=str(args.out),
    )
    try:
        trainer = Trainer(
            model, plan, tcfg, loss_terms=objectives, fold_hash=fold_hash,
            device=args.device, sample_batch=sample, disabled_terms=disabled,
        )
    except Exception as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        print("Hint: --disable kd,ot_ground,contrastive,weak_label,irm "
              "runs image-only.", file=sys.stderr)
        return 2

    if disabled:
        print(f"deliberately disabled terms: {sorted(disabled)}")

    from dataclasses import asdict as _asdict

    manifest = RunManifest(
        run_id=f"{args.backbone}_f{args.fold}_{int(time.time())}",
        git_commit=git_commit(),
        config={
            "args": {k: str(v) for k, v in vars(args).items()},
            # asdict, not vars: KairosConfig is a slots dataclass.
            "model": _asdict(mcfg),
            "train": _asdict(tcfg),
            "curriculum": [s_.name for s_ in plan.stages],
        },
        fold_hash=fold_hash, fold=args.fold, seed=args.seed,
        package_versions=package_versions(),
    )

    # -- train ------------------------------------------------------------ #
    best = {"score": -np.inf, "macro": -np.inf, "worst": -np.inf, "epoch": -1}
    max_epochs = args.epochs_cap or plan.total_epochs
    history = []

    for epoch in range(max_epochs):
        stats = trainer.train_epoch(train_loader)
        line = (f"epoch {epoch:>3}/{max_epochs}  [{stats['stage']}]  "
                f"loss {stats.get('loss/total', float('nan')):.4f}  "
                f"grad {stats.get('grad_norm', float('nan')):.2f}  "
                f"{stats['epoch_seconds']:.0f}s")
        print(line, flush=True)
        history.append(stats)

        if (epoch + 1) % args.eval_every and epoch != max_epochs - 1:
            continue

        logits, targets, val_uids = trainer.predict(val_loader)
        if getattr(trainer, "n_unusable_predicted", 0):
            # Loud, because a systematic decode failure must not read as a bad
            # epoch: these studies had no usable series at all and were dropped
            # from the OOF rather than scored from zero padding.
            print(f"           !! {trainer.n_unusable_predicted} validation "
                  f"study(ies) had no usable series and were excluded")
        if len(logits) == 0:
            continue
        # Stable sigmoid: exp(-logits) overflows for the large-magnitude
        # negative logits an AUC-margin objective happily produces.
        probs = np.where(logits >= 0, 1.0 / (1.0 + np.exp(-np.abs(logits))),
                         np.exp(-np.abs(logits)) / (1.0 + np.exp(-np.abs(logits))))
        aucs = per_label_auc(targets, probs)
        macro = float(np.nanmean(aucs))
        worst = float(np.nanmin(aucs)) if np.isfinite(aucs).any() else float("nan")
        # Multi-objective selection: macro-AUC with a worst-label floor.  On a
        # macro metric, a model that gains 0.01 overall while losing 0.05 on one
        # label has not improved -- it has moved the variance around.
        score = macro + 0.5 * worst
        print(f"           OOF macro {macro:.5f}   worst {worst:.5f}   "
              f"score {score:.5f}" + ("   <- best" if score > best["score"] else ""))
        print("           " + "  ".join(
            f"{t.split()[0][:6]}={a:.3f}" for t, a in zip(TARGETS, aucs)))

        if score > best["score"]:
            best = {"score": score, "macro": macro, "worst": worst, "epoch": epoch}
            manifest.best_macro_auc = macro
            manifest.best_worst_label_auc = worst
            trainer.save(args.out / "checkpoint.pt", manifest)
            np.savez_compressed(
                args.out / "oof.npz",
                logits=logits, targets=targets,
                study_uid=np.array(val_uids, dtype=object),
                fold_hash=fold_hash, fold=args.fold,
            )

    (args.out / "history.json").write_text(json.dumps(history, indent=2, default=str))
    print(f"\nbest epoch {best['epoch']}: macro {best['macro']:.5f}, "
          f"worst {best['worst']:.5f}")
    print(f"wrote {args.out}/checkpoint.pt, oof.npz, history.json, checkpoint.manifest.json")
    print("\nNext: python scripts/05_oof_eval.py --oof "
          f"{args.out}/oof.npz --folds {args.folds or '<folds.parquet>'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
