"""Pure-numpy DICOM geometry.

No pydicom import here on purpose: every function takes plain arrays so the
whole module is unit-testable without a DICOM fixture, and so the Kaggle
inference notebook can call it without paying an import cost per study.

The three things that go wrong in practice, and what this module does about
them:

1. **Slice order.** ``InstanceNumber`` is a *transmission* index, not a spatial
   one.  Interleaved acquisitions, re-sent series and multi-echo scans all
   break it.  The only reliable ordering is the projection of
   ``ImagePositionPatient`` onto the slice normal
   :math:`\\mathbf n = \\hat{\\mathbf r} \\times \\hat{\\mathbf c}`, where
   :math:`\\hat{\\mathbf r}, \\hat{\\mathbf c}` are the first and second triples
   of ``ImageOrientationPatient``.

2. **Anatomical direction.** Two sites can acquire the same sagittal series in
   opposite directions (medial→lateral vs lateral→medial).  A model that has
   learnt "ACL lives around slice 12" from one site is then wrong at the other.
   We fix a canonical direction by the *sign of the normal in patient space*,
   not by acquisition order.

3. **Laterality.** Left and right knees are mirror images.  Training on both
   without canonicalisation forces the network to spend capacity learning a
   reflection symmetry it can be handed for free.  We map every study to a
   canonical (right-knee) frame and record the flip so that "medial" and
   "lateral" labels stay on the correct side.

DICOM patient coordinates are LPS: ``+x`` = patient Left, ``+y`` = Posterior,
``+z`` = Superior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from ..constants import Plane

__all__ = [
    "SliceGeometry",
    "SeriesGeometry",
    "unit",
    "slice_normal",
    "classify_plane",
    "order_slices",
    "spacing_diagnostics",
    "build_affine",
    "canonical_flip",
    "physical_z_coordinates",
]

_EPS = 1e-8


# --------------------------------------------------------------------------- #
# Basic vector helpers                                                         #
# --------------------------------------------------------------------------- #


def unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normalise, leaving (near-)zero vectors untouched instead of NaN."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, _EPS)


def slice_normal(orientation: Sequence[float] | np.ndarray) -> np.ndarray:
    """Slice normal from a 6-element ``ImageOrientationPatient``.

    ``orientation = [r_x, r_y, r_z, c_x, c_y, c_z]`` where ``r`` is the
    direction of increasing *column* index (image x) and ``c`` the direction of
    increasing *row* index (image y), both expressed in LPS patient
    coordinates.  The normal is :math:`\\mathbf n = \\hat r \\times \\hat c`,
    which points along increasing slice index for a right-handed volume.
    """
    o = np.asarray(orientation, dtype=np.float64).reshape(6)
    r, c = unit(o[:3]), unit(o[3:])
    # Re-orthogonalise c against r (Gram-Schmidt): some vendors emit
    # orientations with a fraction of a degree of skew, which is harmless for
    # display but accumulates when we invert the affine.
    c = unit(c - np.dot(c, r) * r)
    return unit(np.cross(r, c))


def classify_plane(normal: np.ndarray, oblique_tol_deg: float = 30.0) -> Plane:
    """Map a slice normal to an acquisition plane.

    A series is called sagittal/coronal/axial when its normal is within
    ``oblique_tol_deg`` of the LPS x/y/z axis respectively, and OBLIQUE
    otherwise.  Knee protocols routinely tilt the sagittal stack along the ACL
    (so-called oblique-sagittal), so the default tolerance is deliberately
    generous -- an oblique-sagittal series is far more useful labelled
    ``SAGITTAL`` than dumped into ``OBLIQUE``.
    """
    n = np.abs(unit(np.asarray(normal, dtype=np.float64)))
    axis = int(np.argmax(n))
    cos_tol = float(np.cos(np.deg2rad(oblique_tol_deg)))
    if n[axis] < cos_tol:
        return Plane.OBLIQUE
    return (Plane.SAGITTAL, Plane.CORONAL, Plane.AXIAL)[axis]


# --------------------------------------------------------------------------- #
# Per-slice / per-series geometry records                                      #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SliceGeometry:
    sop_uid: str
    position: np.ndarray  # (3,) ImagePositionPatient, LPS mm
    orientation: np.ndarray  # (6,) ImageOrientationPatient
    pixel_spacing: np.ndarray  # (2,) (row_mm, col_mm)
    rows: int
    cols: int
    instance_number: int | None = None
    slice_thickness: float | None = None


@dataclass(slots=True)
class SeriesGeometry:
    """Result of ordering + validating a series."""

    order: np.ndarray  # (N,) int, permutation into the original list
    z: np.ndarray  # (N,) float, physical coordinate along the normal (mm)
    normal: np.ndarray  # (3,)
    plane: Plane
    spacing_mm: float  # robust (median) inter-slice spacing
    spacing_cv: float  # coefficient of variation of the gaps
    max_gap_ratio: float  # largest gap / median gap
    n_duplicate_positions: int
    orientation_consistent: bool
    reversed_to_canonical: bool
    flags: tuple[str, ...]


def order_slices(
    slices: Sequence[SliceGeometry],
    *,
    duplicate_tol_mm: float = 1e-3,
    orientation_tol_deg: float = 5.0,
) -> SeriesGeometry:
    """Order a series geometrically and report everything that looks wrong.

    Ordering is by :math:`z_i = \\langle \\mathbf p_i, \\mathbf n \\rangle`
    with :math:`\\mathbf n` the *median* normal over the series (robust to a
    handful of corrupt headers).  ``InstanceNumber`` is used only to break exact
    ties, and only as a last resort.
    """
    if len(slices) == 0:
        raise ValueError("cannot order an empty series")

    flags: list[str] = []

    normals = np.stack([slice_normal(s.orientation) for s in slices])
    # Vendors sometimes flip the sign of the in-plane vectors halfway through a
    # series; align every normal to the first before taking a median so the
    # median is not the average of +n and -n.
    ref = normals[0]
    signs = np.sign(normals @ ref)
    signs[signs == 0] = 1.0
    aligned = normals * signs[:, None]
    normal = unit(np.median(aligned, axis=0))

    cos_dev = np.abs(aligned @ normal)
    orientation_consistent = bool(
        np.all(cos_dev >= np.cos(np.deg2rad(orientation_tol_deg)))
    )
    if not orientation_consistent:
        flags.append("inconsistent_orientation")

    positions = np.stack([np.asarray(s.position, dtype=np.float64) for s in slices])
    z = positions @ normal

    # Duplicate detection *before* sorting: two slices at the same physical
    # location are either a re-sent instance or a second echo.  Both must be
    # collapsed, otherwise the aggregator sees the same anatomy twice and the
    # attention mass over that location is doubled.
    order = np.lexsort(
        (
            np.array(
                [s.instance_number if s.instance_number is not None else 0 for s in slices]
            ),
            z,
        )
    )
    z_sorted = z[order]
    gaps = np.diff(z_sorted)
    n_dupes = int(np.sum(np.abs(gaps) <= duplicate_tol_mm))
    if n_dupes:
        flags.append("duplicate_positions")

    positive = gaps[gaps > duplicate_tol_mm]
    if positive.size == 0:
        spacing = float(
            slices[0].slice_thickness if slices[0].slice_thickness else 1.0
        )
        spacing_cv = 0.0
        max_gap_ratio = 1.0
        flags.append("degenerate_z_extent")
    else:
        spacing = float(np.median(positive))
        spacing_cv = float(np.std(positive) / max(spacing, _EPS))
        max_gap_ratio = float(np.max(positive) / max(spacing, _EPS))
        if spacing_cv > 0.15:
            flags.append("irregular_spacing")
        if max_gap_ratio > 2.5:
            flags.append("large_slice_gap")

    # Canonical direction: we require the physical coordinate to increase along
    # +normal, and we require the normal to point towards patient Left (x),
    # Anterior (-y) or Superior (z) depending on the plane.  This makes slice 0
    # mean the same anatomy at every site.
    plane = classify_plane(normal)
    canonical_axis = {
        Plane.SAGITTAL: np.array([1.0, 0.0, 0.0]),
        Plane.CORONAL: np.array([0.0, -1.0, 0.0]),
        Plane.AXIAL: np.array([0.0, 0.0, 1.0]),
    }.get(plane, normal)
    reversed_to_canonical = bool(np.dot(normal, canonical_axis) < 0)
    if reversed_to_canonical:
        # Flip the normal *and* re-project: z is defined as <p, n>, so negating
        # n negates z, and reversing the order then leaves z increasing again.
        normal = -normal
        z = positions @ normal
        order = order[::-1].copy()
        z_sorted = z[order]

    in_plane = np.stack(
        [np.asarray(s.pixel_spacing, dtype=np.float64) for s in slices]
    )
    if np.ptp(in_plane, axis=0).max() > 1e-3:
        flags.append("variable_in_plane_spacing")

    shapes = {(s.rows, s.cols) for s in slices}
    if len(shapes) > 1:
        flags.append("variable_matrix_size")

    return SeriesGeometry(
        order=order.astype(np.int64),
        z=z_sorted.astype(np.float64),
        normal=normal,
        plane=plane,
        spacing_mm=spacing,
        spacing_cv=spacing_cv,
        max_gap_ratio=max_gap_ratio,
        n_duplicate_positions=n_dupes,
        orientation_consistent=orientation_consistent,
        reversed_to_canonical=reversed_to_canonical,
        flags=tuple(flags),
    )


def spacing_diagnostics(z: np.ndarray) -> dict[str, float]:
    """Summary statistics of the physical slice grid, for the manifest."""
    z = np.asarray(z, dtype=np.float64)
    if z.size < 2:
        return {"extent_mm": 0.0, "spacing_mm": 0.0, "spacing_cv": 0.0, "n": float(z.size)}
    gaps = np.diff(np.sort(z))
    gaps = gaps[gaps > 1e-6]
    if gaps.size == 0:
        return {"extent_mm": 0.0, "spacing_mm": 0.0, "spacing_cv": 0.0, "n": float(z.size)}
    med = float(np.median(gaps))
    return {
        "extent_mm": float(z.max() - z.min()),
        "spacing_mm": med,
        "spacing_cv": float(np.std(gaps) / max(med, _EPS)),
        "n": float(z.size),
    }


# --------------------------------------------------------------------------- #
# Affine construction                                                          #
# --------------------------------------------------------------------------- #


def build_affine(
    orientation: Sequence[float] | np.ndarray,
    position: Sequence[float] | np.ndarray,
    pixel_spacing: Sequence[float] | np.ndarray,
    slice_spacing: float,
    *,
    output: Literal["LPS", "RAS"] = "RAS",
) -> np.ndarray:
    """Voxel-index → patient-space affine (4×4).

    Index convention is ``(col, row, slice)`` i.e. ``(i, j, k)`` with ``i``
    along ``ImageOrientationPatient[:3]``.  ``pixel_spacing`` follows the DICOM
    convention ``(row_spacing, col_spacing)``.
    """
    o = np.asarray(orientation, dtype=np.float64).reshape(6)
    r, c = unit(o[:3]), unit(o[3:])
    c = unit(c - np.dot(c, r) * r)
    n = unit(np.cross(r, c))
    ps = np.asarray(pixel_spacing, dtype=np.float64).reshape(2)
    row_mm, col_mm = float(ps[0]), float(ps[1])

    a = np.eye(4, dtype=np.float64)
    a[:3, 0] = r * col_mm
    a[:3, 1] = c * row_mm
    a[:3, 2] = n * float(slice_spacing)
    a[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)

    if output == "RAS":
        # LPS -> RAS is a sign flip on the first two axes.
        flip = np.diag([-1.0, -1.0, 1.0, 1.0])
        a = flip @ a
    return a


def physical_z_coordinates(z: np.ndarray) -> np.ndarray:
    """Map physical slice coordinates to ``[-1, 1]`` with the *centre of mass*
    of the acquired stack at 0.

    Used as the positional signal fed to the slice encoder.  We deliberately do
    **not** use ``index / n_slices``: two studies with the same anatomy but
    different field-of-view would then get different positional codes for the
    same structure.  Normalising by physical extent makes the code
    field-of-view invariant up to a scale, and the residual scale is handed to
    the network separately as the log-extent so it can undo it if useful.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.size == 0:
        return z
    lo, hi = float(z.min()), float(z.max())
    if hi - lo < 1e-6:
        return np.zeros_like(z)
    return 2.0 * (z - lo) / (hi - lo) - 1.0


def canonical_flip(
    laterality: str | None,
    normal: np.ndarray,
    plane: Plane,
) -> bool:
    """Should this series be mirrored to reach the canonical (right-knee) frame?

    Returns ``True`` when a left–right mirror is required.  For sagittal series
    the mirror is applied along the *slice* axis (the through-plane direction is
    the medial–lateral axis); for coronal and axial series it is applied along
    the in-plane column axis.  Callers must apply the flip consistently and
    record it, because ``Medial OA`` and ``Lateral OA`` swap meaning under an
    unrecorded mirror -- a silent version of this bug will cost more AUC on
    those two labels than any architecture change will win back.
    """
    if laterality is None:
        return False
    side = laterality.strip().upper()[:1]
    if side not in {"L", "R"}:
        return False
    # Canonical = right knee.  A left knee is mirrored about the patient's
    # sagittal midplane, i.e. the LPS x axis.
    return side == "L"
