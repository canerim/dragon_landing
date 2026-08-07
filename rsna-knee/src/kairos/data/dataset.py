r"""Study loading: DICOM → geometry-corrected, metric-resampled 2.5D tensors.

The pipeline per series:

1. read headers only, build :class:`~kairos.io.geometry.SliceGeometry` records
2. order geometrically, drop duplicate physical positions, record QC flags
3. decode pixels for the retained slices
4. rescale (``RescaleSlope``/``Intercept``), then robust intensity
   normalisation *inside a foreground mask*
5. resample in-plane to a **physical** target spacing and centre-crop
6. canonicalise laterality to a right-knee frame
7. stack ``NEIGHBOUR_SLICES`` adjacent planes into the channel dimension

Two choices deserve their own note.

**Intensity.** MRI has no absolute intensity scale: the same tissue is 300 at
one centre and 1800 at another. We clip to the 0.5/99.5 percentiles *of the
foreground* and z-score within the series. Computing percentiles over the whole
image instead lets a large black background dominate, so a study with a big
field of view is normalised differently from a tightly-cropped one showing the
same knee. The foreground mask is a simple Otsu-style threshold followed by a
largest-component filter, which is enough — the failure mode we are avoiding is
gross, not subtle.

**Resampling.** To millimetres, never to a fixed pixel grid. See
``docs/DESIGN.md`` §2.2.

Everything here is numpy; ``pydicom`` is imported lazily so the module can be
imported (and its pure functions tested) without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..constants import (
    COARSE_SIZE,
    COARSE_SPACING_MM,
    FINE_SIZE,
    FINE_SPACING_MM,
    INTENSITY_PERCENTILES,
    MAX_SERIES_PER_STUDY,
    MAX_SLICES_PER_SERIES,
    NEIGHBOUR_SLICES,
    NUM_TARGETS,
    Plane,
)
from ..io.geometry import (
    SliceGeometry,
    order_slices,
    physical_z_coordinates,
)
from .sequence_taxonomy import classify_series

__all__ = [
    "SeriesRecord",
    "StudyRecord",
    "foreground_mask",
    "robust_normalise",
    "resample_to_spacing",
    "stack_neighbours",
    "select_slices",
    "load_study",
    "build_inference_batch",
]


# --------------------------------------------------------------------------- #
# Records                                                                      #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SeriesRecord:
    series_uid: str
    family_index: int
    plane: Plane
    pixels: np.ndarray  # (S, H, W) float32, normalised
    z_mm: np.ndarray  # (S,) physical coordinate
    spacing_mm: float
    in_plane_mm: float
    flags: tuple[str, ...] = ()
    manufacturer_id: int = 0
    fat_sat_id: int = 2
    field_strength: float = 0.0
    te_ms: float = 0.0
    tr_ms: float = 0.0
    mirrored: bool = False


@dataclass(slots=True)
class StudyRecord:
    study_uid: str
    series: list[SeriesRecord] = field(default_factory=list)
    labels: np.ndarray | None = None
    report: str | None = None
    site: str | None = None
    language: str | None = None
    laterality: str | None = None
    errors: list[str] = field(default_factory=list)
    #: Integer bucket used by Group-DRO (site x language x scanner) and by the
    #: IRM environment split.  Populated by the dataloader, not by load_study,
    #: because the bucketing is a property of the *cohort*, not of the study.
    group_index: int | None = None
    env_index: int | None = None
    #: Auxiliary supervision, all optional and all *cohort*-level like the
    #: indices above: the dataloader stamps them from artefacts built by
    #: ``02_parse_reports.py`` and ``07_make_teacher.py``.  Without these
    #: fields the curriculum's ``weak_label`` and ``kd`` terms are registered,
    #: scheduled, and permanently uncomputable -- which is the failure mode the
    #: objective validator exists to make loud, and the reason every run so far
    #: had to pass ``--disable weak_label,kd``.
    weak_labels: np.ndarray | None = None      # (L,) in [0, 1], NaN = no mention
    weak_confidence: np.ndarray | None = None  # (L,) in [0, 1]
    teacher_logits: np.ndarray | None = None   # (L,) cross-fitted OOF teacher


# --------------------------------------------------------------------------- #
# Intensity                                                                    #
# --------------------------------------------------------------------------- #


def foreground_mask(volume: np.ndarray, *, min_fraction: float = 0.05) -> np.ndarray:
    r"""Otsu threshold over the whole series, with a sanity floor.

    A per-slice threshold is tempting and wrong: the end slices of a knee stack
    contain much less tissue, so a per-slice Otsu puts the threshold in the
    noise there and marks background as foreground.  One threshold for the
    series is both simpler and more stable.

    ``min_fraction`` guards the degenerate case (a nearly-uniform series) where
    Otsu picks a threshold that selects almost nothing; there we fall back to a
    percentile.
    """
    v = np.asarray(volume, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.ones(volume.shape, dtype=bool)

    # Build the histogram over a *robust* range.  A susceptibility blowout or a
    # metal artefact can be four orders of magnitude above tissue; with a
    # full-range histogram every real voxel lands in bin 0 and the threshold
    # ends up describing the artefact rather than the anatomy.
    lo_h, hi_h = np.percentile(v, INTENSITY_PERCENTILES)
    if not np.isfinite(lo_h) or not np.isfinite(hi_h) or hi_h <= lo_h:
        lo_h, hi_h = float(v.min()), float(v.max())
    if hi_h <= lo_h:
        return np.ones(volume.shape, dtype=bool)
    v = np.clip(v, lo_h, hi_h)

    hist, edges = np.histogram(v, bins=256, range=(lo_h, hi_h))
    centres = 0.5 * (edges[:-1] + edges[1:])
    w = np.cumsum(hist).astype(np.float64)
    total = w[-1]
    if total <= 0:
        return np.ones(volume.shape, dtype=bool)
    mu = np.cumsum(hist * centres)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_t * w / total - mu) ** 2 / (w * (total - w) / total)
    # ``posinf``/``neginf`` are NOT optional here.  numpy's default maps +inf to
    # 1.8e308, so the final bin -- where the denominator is zero -- always wins
    # the argmax and every threshold comes back at the top of the range.  The
    # symptom is a mask that selects nothing and silently falls through to the
    # percentile branch below, which looks like it works.
    between = np.nan_to_num(between, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    thr = float(centres[int(np.argmax(between))])

    mask = np.asarray(volume) > thr
    if mask.mean() < min_fraction:
        thr = float(np.percentile(v, 100 * (1 - min_fraction)))
        mask = np.asarray(volume) > thr
    return mask


def robust_normalise(
    volume: np.ndarray,
    *,
    percentiles: tuple[float, float] = INTENSITY_PERCENTILES,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Percentile-clip inside the foreground, then robust z-score.

    Two passes.  The first clips at the *global* percentiles, which is what
    removes a susceptibility blowout or metal artefact -- those occupy well
    under 0.5 % of voxels, so they sit outside the global range but *inside*
    the foreground range (where they can be several percent of the selected
    voxels and therefore survive a foreground-only percentile).  The second
    pass then refines within the foreground.

    Limitation, stated rather than hidden: an artefact occupying more than
    ~0.5 % of the volume survives pass one.  Those studies are caught by the
    QC flags instead, not here.
    """
    v = np.asarray(volume, dtype=np.float32)
    g_lo, g_hi = np.percentile(v[np.isfinite(v)], percentiles) if np.isfinite(v).any() else (0.0, 1.0)
    if np.isfinite(g_lo) and np.isfinite(g_hi) and g_hi > g_lo:
        v = np.clip(v, g_lo, g_hi)

    m = foreground_mask(v) if mask is None else mask
    fg = v[m]
    if fg.size < 16:
        fg = v.ravel()
    lo, hi = np.percentile(fg, percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(v)), float(np.max(v))
        if hi <= lo:
            return np.zeros_like(v)
    v = np.clip(v, lo, hi)
    med = float(np.median(v[m])) if m.any() else float(np.median(v))
    # Median absolute deviation, scaled to be a consistent estimator of sigma
    # for Gaussian data.  Far more stable than std() when a metal artefact or a
    # bright fluid collection occupies part of the field.
    mad = float(np.median(np.abs(v[m] - med))) if m.any() else float(np.median(np.abs(v - med)))
    scale = 1.4826 * mad
    if scale < 1e-6:
        scale = float(np.std(v)) or 1.0
    return ((v - med) / scale).astype(np.float32)


# --------------------------------------------------------------------------- #
# Spatial                                                                      #
# --------------------------------------------------------------------------- #


def _bilinear_resize(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Dependency-free bilinear resize of a 2-D array."""
    h, w = img.shape
    if (h, w) == (out_h, out_w):
        return img.astype(np.float32)
    ys = (np.arange(out_h) + 0.5) * h / out_h - 0.5
    xs = (np.arange(out_w) + 0.5) * w / out_w - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = np.clip(ys - y0, 0, 1)[:, None]
    wx = np.clip(xs - x0, 0, 1)[None, :]
    a = img[np.ix_(y0, x0)]
    b = img[np.ix_(y0, x1)]
    c = img[np.ix_(y1, x0)]
    d = img[np.ix_(y1, x1)]
    top = a * (1 - wx) + b * wx
    bot = c * (1 - wx) + d * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def resample_to_spacing(
    volume: np.ndarray,
    in_plane_mm: float,
    target_mm: float,
    out_size: int,
    *,
    centre: tuple[float, float] | None = None,
) -> np.ndarray:
    r"""Resample each slice to ``target_mm`` and centre-crop/pad to ``out_size``.

    The field of view of the output is exactly ``out_size * target_mm``
    millimetres regardless of the input matrix or spacing, which is the whole
    point: identical anatomy occupies identical pixels at every site.
    """
    v = np.asarray(volume, dtype=np.float32)
    if v.ndim == 2:
        v = v[None]
    S, H, W = v.shape
    scale = float(in_plane_mm) / float(target_mm)
    new_h, new_w = max(1, int(round(H * scale))), max(1, int(round(W * scale)))

    out = np.zeros((S, out_size, out_size), dtype=np.float32)
    cy = new_h / 2.0 if centre is None else centre[0] * scale
    cx = new_w / 2.0 if centre is None else centre[1] * scale
    top = int(round(cy - out_size / 2.0))
    left = int(round(cx - out_size / 2.0))

    for s in range(S):
        r = _bilinear_resize(v[s], new_h, new_w)
        y0, x0 = max(top, 0), max(left, 0)
        y1, x1 = min(top + out_size, new_h), min(left + out_size, new_w)
        if y1 <= y0 or x1 <= x0:
            continue
        oy0, ox0 = y0 - top, x0 - left
        out[s, oy0 : oy0 + (y1 - y0), ox0 : ox0 + (x1 - x0)] = r[y0:y1, x0:x1]
    return out


def stack_neighbours(volume: np.ndarray, n: int = NEIGHBOUR_SLICES) -> np.ndarray:
    r"""``(S, H, W)`` → ``(S, n, H, W)`` with edge-clamped neighbours.

    Edge clamping rather than zero padding: a zero-padded first slice looks
    like a slice through air, and the 2.5D stem learns to treat the stack
    boundary as an anatomical feature.
    """
    v = np.asarray(volume, dtype=np.float32)
    S = v.shape[0]
    half = n // 2
    idx = np.clip(np.arange(S)[:, None] + np.arange(-half, half + 1)[None, :], 0, S - 1)
    return v[idx]


def select_slices(
    z_mm: np.ndarray, max_slices: int = MAX_SLICES_PER_SERIES
) -> np.ndarray:
    r"""Choose at most ``max_slices`` indices, uniform in **physical** space.

    Uniform-in-index subsampling of a series with a spacing discontinuity puts
    most of the retained slices on one side of the gap.  Uniform-in-millimetres
    keeps the anatomical coverage even, which is what the aggregator's metric
    positional encoding assumes.
    """
    z = np.asarray(z_mm, dtype=np.float64)
    S = z.size
    if S <= max_slices:
        return np.arange(S)
    targets = np.linspace(z.min(), z.max(), max_slices)
    idx = np.unique(np.abs(z[None, :] - targets[:, None]).argmin(axis=1))
    if idx.size < max_slices:  # ties collapsed: top up with the densest region
        extra = np.setdiff1d(np.arange(S), idx)
        idx = np.sort(np.concatenate([idx, extra[: max_slices - idx.size]]))
    return idx


# --------------------------------------------------------------------------- #
# DICOM loading                                                                #
# --------------------------------------------------------------------------- #


def _read_headers(paths: Sequence[Path]):
    import pydicom

    out = []
    for p in paths:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            out.append((p, ds))
        except Exception:
            continue
    return out


def _to_slice_geometry(ds, path: Path) -> SliceGeometry | None:
    try:
        iop = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
        ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
        ps = np.asarray(getattr(ds, "PixelSpacing", [1.0, 1.0]), dtype=np.float64)
    except Exception:
        return None
    return SliceGeometry(
        sop_uid=str(getattr(ds, "SOPInstanceUID", path.name)),
        position=ipp,
        orientation=iop,
        pixel_spacing=ps,
        rows=int(getattr(ds, "Rows", 0)),
        cols=int(getattr(ds, "Columns", 0)),
        instance_number=int(getattr(ds, "InstanceNumber", 0) or 0),
        slice_thickness=float(getattr(ds, "SliceThickness", 0) or 0) or None,
    )


def _pixel_array(path: Path) -> np.ndarray | None:
    import pydicom

    try:
        ds = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        return arr * slope + inter
    except Exception:
        return None


def load_study(
    study_dir: str | Path,
    *,
    target_mm: float = COARSE_SPACING_MM,
    out_size: int = COARSE_SIZE,
    max_series: int = MAX_SERIES_PER_STUDY,
    max_slices: int = MAX_SLICES_PER_SERIES,
    laterality: str | None = None,
) -> StudyRecord:
    """Load and preprocess every series of one study directory."""
    study_dir = Path(study_dir)
    rec = StudyRecord(study_uid=study_dir.name)

    series_dirs = [d for d in sorted(study_dir.iterdir()) if d.is_dir()]
    if not series_dirs:  # flat layout: group by SeriesInstanceUID
        series_dirs = [study_dir]

    candidates: list[SeriesRecord] = []
    for sd in series_dirs:
        files = sorted(p for p in sd.rglob("*") if p.is_file())
        if not files:
            continue
        headers = _read_headers(files)
        if not headers:
            rec.errors.append(f"{sd.name}: no readable headers")
            continue

        geoms, kept = [], []
        for p, ds in headers:
            g = _to_slice_geometry(ds, p)
            if g is not None:
                geoms.append(g)
                kept.append(p)
        if len(geoms) < 3:
            rec.errors.append(f"{sd.name}: fewer than 3 usable slices")
            continue

        try:
            geo = order_slices(geoms)
        except Exception as exc:
            rec.errors.append(f"{sd.name}: ordering failed ({exc})")
            continue

        ordered_paths = [kept[i] for i in geo.order]
        z = geo.z
        # Drop exact duplicate physical positions, keeping the first.
        keep = np.concatenate([[True], np.abs(np.diff(z)) > 1e-3])
        ordered_paths = [p for p, k in zip(ordered_paths, keep) if k]
        z = z[keep]

        sel = select_slices(z, max_slices)
        ordered_paths = [ordered_paths[i] for i in sel]
        z = z[sel]

        arrays = [_pixel_array(p) for p in ordered_paths]
        good = [(a, zz) for a, zz in zip(arrays, z) if a is not None and a.ndim == 2]
        if len(good) < 3:
            rec.errors.append(f"{sd.name}: pixel decode failed")
            continue
        shapes = {a.shape for a, _ in good}
        if len(shapes) > 1:  # keep the modal shape; mixed-matrix series exist
            from collections import Counter

            modal = Counter(a.shape for a, _ in good).most_common(1)[0][0]
            good = [(a, zz) for a, zz in good if a.shape == modal]
        vol = np.stack([a for a, _ in good])
        z = np.array([zz for _, zz in good], dtype=np.float64)

        vol = robust_normalise(vol)
        ds0 = headers[0][1]
        in_plane = float(np.mean(geoms[0].pixel_spacing))
        vol = resample_to_spacing(vol, in_plane, target_mm, out_size)

        fam = classify_series(ds0, plane=geo.plane)
        mirrored = (laterality or str(getattr(ds0, "ImageLaterality", "") or
                                      getattr(ds0, "Laterality", ""))).upper().startswith("L")
        if mirrored:
            # Canonical = right knee.  Sagittal stacks mirror along the slice
            # axis (through-plane is medial-lateral); coronal/axial along cols.
            if geo.plane == Plane.SAGITTAL:
                vol = vol[::-1].copy()
                z = -z[::-1].copy()
            else:
                vol = vol[:, :, ::-1].copy()

        candidates.append(
            SeriesRecord(
                series_uid=str(getattr(ds0, "SeriesInstanceUID", sd.name)),
                family_index=fam.index,
                plane=geo.plane,
                pixels=vol,
                z_mm=z.astype(np.float32),
                spacing_mm=float(geo.spacing_mm),
                in_plane_mm=in_plane,
                flags=geo.flags,
                fat_sat_id=1 if fam.fat_sat else 0,
                field_strength=float(getattr(ds0, "MagneticFieldStrength", 0) or 0),
                te_ms=float(getattr(ds0, "EchoTime", 0) or 0),
                tr_ms=float(getattr(ds0, "RepetitionTime", 0) or 0),
                mirrored=mirrored,
            )
        )

    # Rank by QC quality then by family informativeness, keep the best N.
    candidates.sort(key=lambda s: (len(s.flags), -s.pixels.shape[0]))
    rec.series = candidates[:max_series]
    rec.laterality = laterality
    if not rec.series:
        rec.errors.append("no usable series")
    return rec


# --------------------------------------------------------------------------- #
# Batch assembly                                                               #
# --------------------------------------------------------------------------- #


def collate_studies(records: Iterable[StudyRecord], *, device="cpu", fine: bool = False):
    """Pad a list of :class:`StudyRecord` into a ``StudyBatch``."""
    import torch

    from ..models.encoding import build_context_vector
    from ..models.system import StudyBatch

    records = list(records)
    B = len(records)
    Nseq = max(1, max(len(r.series) for r in records))
    S = max(
        1,
        max((s.pixels.shape[0] for r in records for s in r.series), default=1),
    )
    # The spatial size is a *run-level* parameter, so read it from the first
    # record that actually has a series -- not from ``records[0]``, which may
    # be a study whose DICOMs all failed to decode.  Sampling record 0 made the
    # allocation batch-order dependent: at ``--image-size 320`` the batch
    # ``[failed, good]`` allocated 256 and died on the assignment below, while
    # ``[good, failed]`` was fine, so the crash surfaced at a random step deep
    # into training rather than at startup.
    sizes = {s.pixels.shape[-1] for r in records for s in r.series}
    if len(sizes) > 1:
        raise ValueError(
            f"series in one batch have different in-plane sizes {sorted(sizes)}; "
            "they must all be resampled to the same grid before collation"
        )
    size = sizes.pop() if sizes else COARSE_SIZE
    C = NEIGHBOUR_SLICES

    pixels = torch.zeros(B, Nseq, S, C, size, size)
    slice_mask = torch.zeros(B, Nseq, S, dtype=torch.bool)
    series_mask = torch.zeros(B, Nseq, dtype=torch.bool)
    z_mm = torch.zeros(B, Nseq, S)
    family = torch.zeros(B, Nseq, dtype=torch.long)
    manu = torch.zeros(B, Nseq, dtype=torch.long)
    fat = torch.full((B, Nseq), 2, dtype=torch.long)
    cont = torch.zeros(B, Nseq, 8)
    targets = torch.full((B, NUM_TARGETS), float("nan"))

    for b, r in enumerate(records):
        for k, s in enumerate(r.series[:Nseq]):
            n = min(s.pixels.shape[0], S)
            stacked = stack_neighbours(s.pixels[:n], C)  # (n, C, H, W)
            pixels[b, k, :n] = torch.from_numpy(np.ascontiguousarray(stacked))
            slice_mask[b, k, :n] = True
            series_mask[b, k] = True
            z_mm[b, k, :n] = torch.from_numpy(np.ascontiguousarray(s.z_mm[:n]))
            family[b, k] = int(s.family_index)
            manu[b, k] = int(s.manufacturer_id)
            fat[b, k] = int(s.fat_sat_id)
            extent = float(s.z_mm[:n].max() - s.z_mm[:n].min()) if n > 1 else 0.0
            cont[b, k] = build_context_vector(
                in_plane_mm=torch.tensor(s.in_plane_mm),
                slice_mm=torch.tensor(s.spacing_mm),
                extent_mm=torch.tensor(extent),
                field_strength_t=torch.tensor(s.field_strength),
                te_ms=torch.tensor(s.te_ms),
                tr_ms=torch.tensor(s.tr_ms),
                n_slices=torch.tensor(float(n)),
                spacing_cv=torch.tensor(0.0),
            )
        if r.labels is not None:
            targets[b] = torch.from_numpy(np.asarray(r.labels, dtype=np.float32))

    # A study with no usable series is all-padding.  Left alone it is *worse*
    # than useless: the fusion attention over an all-False mask is uniform over
    # padded slots, the head emits finite logits from zeros, and those logits
    # are scored against the study's real targets -- so it contributes pure
    # padding gradient in training and a fabricated row in the OOF matrix.
    # (This is reachable in practice: 00_build_manifest decides usability from
    # headers with stop_before_pixels=True, while load_study additionally
    # rejects on a pixel-decode failure, so a missing codec plugin or a
    # truncated PixelData passes QC and only dies here.)
    #
    # Marking it invalid and NaN-ing its targets makes every masked loss skip
    # it by the same mechanism that already handles unobserved labels.
    study_valid = series_mask.any(dim=1)
    if not bool(study_valid.all()):
        targets[~study_valid] = float("nan")

    def _stack(attr: str, fill: float):
        """(B, L) tensor of a per-study auxiliary field, or None if unused."""
        if not any(getattr(r, attr, None) is not None for r in records):
            return None
        out = torch.full((B, NUM_TARGETS), fill)
        for b, r in enumerate(records):
            v = getattr(r, attr, None)
            if v is not None:
                out[b] = torch.from_numpy(np.asarray(v, dtype=np.float32))
        return out

    # Weak labels default to NaN (= "the report did not mention it"), teacher
    # logits to 0 with zero confidence -- both are the neutral element for the
    # term that reads them, so a partially-covered cohort degrades smoothly
    # instead of supervising the uncovered studies with a fabricated value.
    weak = _stack("weak_labels", float("nan"))
    weak_conf = _stack("weak_confidence", 0.0)
    # NaN, not 0: a teacher logit of 0 is "p = 0.5 for every label", which is a
    # confident and wrong target, not a missing one.  The kd term selects the
    # finite rows; a batch with no teacher at all carries None so the term is
    # skipped rather than fed a matrix of NaNs.
    teacher = _stack("teacher_logits", float("nan"))
    if teacher is not None and not bool(torch.isfinite(teacher).any()):
        teacher = None

    have_group = any(r.group_index is not None for r in records)
    have_env = any(r.env_index is not None for r in records)
    group_id = (
        torch.tensor([int(r.group_index or 0) for r in records], dtype=torch.long)
        if have_group else None
    )
    env_id = (
        torch.tensor([int(r.env_index or 0) for r in records], dtype=torch.long)
        if have_env else None
    )

    return StudyBatch(
        pixels=pixels.to(device),
        slice_mask=slice_mask.to(device),
        series_mask=series_mask.to(device),
        z_mm=z_mm.to(device),
        family_id=family.to(device),
        manufacturer_id=manu.to(device),
        fat_sat_id=fat.to(device),
        context=cont.to(device),
        targets=targets.to(device) if any(r.labels is not None for r in records) else None,
        study_uid=[r.study_uid for r in records],
        study_valid=study_valid.to(device),
        weak_labels=None if weak is None else weak.to(device),
        weak_confidence=None if weak_conf is None else weak_conf.to(device),
        teacher_logits=None if teacher is None else teacher.to(device),
        fine_pixels=None if not fine else pixels.to(device),
        group_id=None if group_id is None else group_id.to(device),
        env_id=None if env_id is None else env_id.to(device),
    )


def build_inference_batch(study_dir: str | Path, *, device="cpu"):
    """Single-study batch for the Kaggle notebook.

    On total failure this returns a batch of zeros with one masked-in series
    rather than raising, so the notebook records a 0.5 prediction for the study
    and keeps going -- one unreadable study must not cost the whole submission.
    """
    rec = load_study(study_dir)
    if not rec.series:
        rec.series = [
            SeriesRecord(
                series_uid="empty",
                family_index=0,
                plane=Plane.UNKNOWN,
                pixels=np.zeros((3, COARSE_SIZE, COARSE_SIZE), dtype=np.float32),
                z_mm=np.zeros(3, dtype=np.float32),
                spacing_mm=1.0,
                in_plane_mm=COARSE_SPACING_MM,
                flags=("empty_fallback",),
            )
        ]
    return collate_studies([rec], device=device)


def fine_crop_size() -> tuple[float, int]:
    """The fine pass's (spacing, size), kept next to the coarse defaults."""
    return FINE_SPACING_MM, FINE_SIZE
