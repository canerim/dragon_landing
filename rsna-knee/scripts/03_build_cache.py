#!/usr/bin/env python3
"""Decode every study once and cache the preprocessed tensors.

    python scripts/03_build_cache.py \
        --root /kaggle/input/competitions/rsna-knee-abnormality-detection/train_series \
        --out /tmp/cache --image-size 192 --max-series 4 --workers 4

    python scripts/04_train.py --cache /tmp/cache ...

Why this exists
---------------

``StudyDataset`` takes ``load_fn`` as an injection precisely so that the same
class can serve real DICOM directories *or* a pre-built tensor cache.  Only the
DICOM path was ever written, and on a machine with few cores that path makes
training impossible rather than merely slow:

A single epoch over 3517 training studies re-reads and re-resamples roughly
580 000 DICOM slices.  Measured on a Kaggle T4x2 session (4 vCPU), that is
about 40 minutes of *pure I/O* per epoch, with both GPUs sitting at 0 %
utilisation the entire time -- the 36-epoch ``medium`` curriculum would need
23 hours against a 12-hour session limit.  Decoding is also perfectly
redundant: the DICOMs do not change between epochs, only the augmentation
does, and augmentation is applied after ``load_fn`` returns.

So: decode once, write ``float16`` tensors, and let every subsequent epoch --
and every subsequent *fold*, which is where this really pays -- read them back.

Sizing
------

Cache size is ``n_studies x series x slices x size^2 x 2`` bytes.  At the
defaults here (192 px, 4 series) that is roughly 6 MB per study and ~27 GB for
4407 studies, which fits Kaggle's 57.6 GB scratch disk.  At the full 256 px and
8 series it would be ~95 GB and would not.  ``--max-series`` is the cheaper
knob: series are kept in the order ``load_study`` returns them, which is its
own informativeness ordering.

Write the cache to ``/tmp``, **not** to ``/kaggle/working``: the latter is the
notebook's output, which has a far smaller size limit than the scratch disk and
would make the commit fail after the work is done.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kairos.constants import Plane  # noqa: E402


def cache_path(cache_dir: Path, uid: str) -> Path:
    """Shard so that no single directory holds thousands of entries.

    On the *last* two characters, not the first: every StudyInstanceUID in this
    competition begins ``1.2.826.0.1.3680043.8.498.``, so a prefix shard puts
    all 4407 files in one directory and buys nothing.
    """
    return Path(cache_dir) / str(uid)[-2:] / f"{uid}.npz"


def _encode(rec) -> dict:
    """Flatten a :class:`StudyRecord` into arrays ``np.savez`` can hold.

    ``pixels`` goes to float16: it is already robust-normalised to roughly
    unit scale, so the precision loss is far below the noise floor of the
    acquisition, and it halves both the disk footprint and the read time --
    which is the entire point of the exercise.
    """
    out: dict[str, np.ndarray] = {
        "study_uid": np.array(rec.study_uid),
        "n_series": np.array(len(rec.series)),
    }
    for i, s in enumerate(rec.series):
        out[f"px_{i}"] = np.asarray(s.pixels, dtype=np.float16)
        out[f"z_{i}"] = np.asarray(s.z_mm, dtype=np.float32)
        out[f"meta_{i}"] = np.array(
            [s.family_index, int(s.plane), s.spacing_mm, s.in_plane_mm,
             s.manufacturer_id, s.fat_sat_id, s.field_strength, s.te_ms,
             s.tr_ms, float(s.mirrored)], dtype=np.float32,
        )
        out[f"uid_{i}"] = np.array(s.series_uid)
    return out


def decode_one(args_tuple) -> tuple[str, bool, str]:
    study_dir, cache_dir, size, target_mm, max_series, max_slices = args_tuple
    from kairos.data.dataset import load_study

    uid = Path(study_dir).name
    dest = cache_path(Path(cache_dir), uid)
    if dest.exists():
        return uid, True, "cached"
    try:
        rec = load_study(study_dir, out_size=size, target_mm=target_mm,
                         max_series=max_series, max_slices=max_slices)
    except Exception as exc:
        return uid, False, f"{type(exc).__name__}: {exc}"
    if not rec.series:
        # Written anyway, as an empty record.  A study with no usable series is
        # a fact about the data, and the collator already marks it invalid and
        # drops it from the OOF -- re-decoding it every epoch to rediscover
        # that would be the same waste this script exists to remove.
        return _write(dest, rec, uid, note="no usable series")
    return _write(dest, rec, uid)


def _write(dest: Path, rec, uid: str, note: str = "") -> tuple[str, bool, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # The suffix must stay ``.npz``: ``np.savez`` silently *appends* ``.npz``
    # to any path that lacks it, so a ``.npz.tmp`` target is written to
    # ``.npz.tmp.npz`` and the rename below fails on a file that is not there.
    tmp = dest.with_name(dest.name + ".tmp.npz")
    # Write-then-rename, so a half-written file from an interrupted session is
    # not indistinguishable from a good one -- the resume path skips anything
    # whose final name exists, and would skip a truncated file forever.
    np.savez(tmp, **_encode(rec))
    tmp.rename(dest)
    return uid, True, note


def load_cached_study(cache_dir: str | Path, uid: str):
    """Rebuild a :class:`StudyRecord` from the cache. The training-time reader."""
    from kairos.data.dataset import SeriesRecord, StudyRecord

    p = cache_path(Path(cache_dir), uid)
    rec = StudyRecord(study_uid=str(uid))
    if not p.exists():
        rec.errors.append(f"not in cache: {p}")
        return rec
    with np.load(p, allow_pickle=False) as z:
        for i in range(int(z["n_series"])):
            m = z[f"meta_{i}"]
            rec.series.append(SeriesRecord(
                series_uid=str(z[f"uid_{i}"]),
                family_index=int(m[0]),
                plane=Plane(int(m[1])),
                # float32 on the way out: the model and every loss run in
                # float32 (autocast handles the rest), and handing them a
                # float16 array silently changes accumulation dtype.
                pixels=np.asarray(z[f"px_{i}"], dtype=np.float32),
                z_mm=np.asarray(z[f"z_{i}"], dtype=np.float32),
                spacing_mm=float(m[2]), in_plane_mm=float(m[3]),
                manufacturer_id=int(m[4]), fat_sat_id=int(m[5]),
                field_strength=float(m[6]), te_ms=float(m[7]), tr_ms=float(m[8]),
                mirrored=bool(m[9]),
            ))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="directory of study directories, e.g. <comp>/train_series")
    ap.add_argument("--out", required=True, type=Path, help="cache directory (use /tmp)")
    ap.add_argument("--image-size", type=int, default=192)
    ap.add_argument("--target-mm", type=float, default=0.70)
    ap.add_argument("--max-series", type=int, default=4)
    ap.add_argument("--max-slices", type=int, default=48)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="cache only N studies (debug)")
    args = ap.parse_args()

    studies = [d for d in sorted(args.root.iterdir()) if d.is_dir()]
    if args.limit:
        studies = studies[: args.limit]
    if not studies:
        print(f"no study directories under {args.root}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"caching {len(studies)} studies -> {args.out}")
    print(f"  {args.image_size}px, {args.target_mm}mm, "
          f"max {args.max_series} series x {args.max_slices} slices, "
          f"{args.workers} workers")

    tasks = [(str(d), str(args.out), args.image_size, args.target_mm,
              args.max_series, args.max_slices) for d in studies]
    t0 = time.monotonic()
    done = failed = skipped = 0
    notes: list[str] = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(decode_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs)):
            uid, ok, note = fut.result()
            done += ok
            failed += not ok
            skipped += note == "cached"
            if note and note != "cached":
                notes.append(f"{uid}: {note}")
            if (i + 1) % 200 == 0:
                rate = (i + 1) / (time.monotonic() - t0)
                print(f"  {i + 1}/{len(tasks)}  {rate:.1f} studies/s  "
                      f"eta {(len(tasks) - i - 1) / max(rate, 1e-9) / 60:.1f} min",
                      flush=True)

    total_bytes = sum(p.stat().st_size for p in args.out.rglob("*.npz"))
    print(f"\ncached {done}/{len(tasks)} in {(time.monotonic() - t0) / 60:.1f} min "
          f"({skipped} already present, {failed} failed)")
    print(f"cache size: {total_bytes / 2**30:.1f} GiB "
          f"({total_bytes / max(done, 1) / 2**20:.1f} MiB per study)")
    if notes:
        print(f"\n{len(notes)} study(ies) with notes; first few:")
        for n in notes[:10]:
            print(f"  {n}")
    if failed:
        print(f"\n!! {failed} studies failed to decode and are absent from the "
              "cache. Training will mark them invalid and drop them from the "
              "OOF rather than score them from padding.", file=sys.stderr)
    print(f"\nNext: python scripts/04_train.py --cache {args.out} "
          f"--image-size {args.image_size} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
