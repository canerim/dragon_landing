"""Competition-level constants, the knee pathology ontology, and sequence taxonomy.

Everything downstream imports its label order from here.  The order of
:data:`TARGETS` is the *exact* column order required by the submission file and
must never be permuted -- several modules index label-specific parameters
positionally (label queries, per-label ensemble weights, per-label temperature).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Final

# --------------------------------------------------------------------------- #
# Competition targets                                                          #
# --------------------------------------------------------------------------- #

TARGETS: Final[tuple[str, ...]] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)

NUM_TARGETS: Final[int] = len(TARGETS)
TARGET_INDEX: Final[dict[str, int]] = {name: i for i, name in enumerate(TARGETS)}

SUBMISSION_COLUMNS: Final[tuple[str, ...]] = ("StudyInstanceUID",) + TARGETS


class Target(IntEnum):
    """Positional enum so model code can say ``Target.ACL`` instead of ``0``."""

    ACL = 0
    MCL = 1
    MEDIAL_MENISCUS = 2
    LATERAL_MENISCUS = 3
    MEDIAL_OA = 4
    LATERAL_OA = 5
    PF_OA = 6
    EFFUSION = 7
    SYNOVITIS = 8
    BAKERS = 9
    CONTUSION = 10
    FRACTURE = 11


# --------------------------------------------------------------------------- #
# Anatomical / semantic structure over the label set                           #
# --------------------------------------------------------------------------- #
# The 12 targets are not exchangeable.  They live on a shallow tree
#
#     knee
#     ├── ligament ────────── ACL, MCL
#     ├── meniscus ────────── Medial Meniscus, Lateral Meniscus
#     ├── degenerative ────── Medial OA, Lateral OA, PF OA
#     ├── inflammatory ────── Effusion, Synovitis, Baker's
#     └── osseous ─────────── Contusion, Fracture
#
# and additionally on a *compartment* axis (medial / lateral / patellofemoral /
# global).  Both structures are exploited:
#   * the tree drives the hyperbolic ontology prior (``models.hyperbolic``),
#   * the compartment axis drives label-query spatial priors and the
#     mixture-of-experts routing bias (``models.label_queries``).

LABEL_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "ligament": ("ACL", "MCL"),
    "meniscus": ("Medial Meniscus", "Lateral Meniscus"),
    "degenerative": ("Medial OA", "Lateral OA", "PF OA"),
    "inflammatory": ("Effusion", "Synovitis", "Baker's"),
    "osseous": ("Contusion", "Fracture"),
}

GROUP_OF_LABEL: Final[dict[str, str]] = {
    label: group for group, labels in LABEL_GROUPS.items() for label in labels
}


class Compartment(IntEnum):
    GLOBAL = 0
    MEDIAL = 1
    LATERAL = 2
    PATELLOFEMORAL = 3
    CENTRAL = 4  # intercondylar notch (ACL/PCL)
    POSTERIOR = 5  # popliteal fossa (Baker's cyst)


LABEL_COMPARTMENT: Final[dict[str, Compartment]] = {
    "ACL": Compartment.CENTRAL,
    "MCL": Compartment.MEDIAL,
    "Medial Meniscus": Compartment.MEDIAL,
    "Lateral Meniscus": Compartment.LATERAL,
    "Medial OA": Compartment.MEDIAL,
    "Lateral OA": Compartment.LATERAL,
    "PF OA": Compartment.PATELLOFEMORAL,
    "Effusion": Compartment.GLOBAL,
    "Synovitis": Compartment.GLOBAL,
    "Baker's": Compartment.POSTERIOR,
    "Contusion": Compartment.GLOBAL,
    "Fracture": Compartment.GLOBAL,
}

NUM_COMPARTMENTS: Final[int] = len(Compartment)


# --------------------------------------------------------------------------- #
# MRI sequence taxonomy                                                        #
# --------------------------------------------------------------------------- #
# ``SeriesDescription`` is free text authored independently at 16 sites in ~12
# languages; it is a *hint*, never a key.  We map every series into a small
# closed vocabulary using DICOM acquisition parameters plus the description, and
# we always keep an explicit UNKNOWN bucket rather than force-fitting.


class Plane(IntEnum):
    UNKNOWN = 0
    SAGITTAL = 1
    CORONAL = 2
    AXIAL = 3
    OBLIQUE = 4


class Weighting(IntEnum):
    UNKNOWN = 0
    PD = 1  # proton density
    T2 = 2
    T1 = 3
    STIR = 4  # short-tau inversion recovery (fat suppressed by inversion)
    GRE = 5  # gradient echo / T2*
    OTHER = 6


@dataclass(frozen=True, slots=True)
class SequenceFamily:
    """A (plane, weighting, fat-saturation) triple with a stable integer id."""

    index: int
    plane: Plane
    weighting: Weighting
    fat_sat: bool
    name: str
    # Which targets this family is *a priori* informative for.  Used only as a
    # soft prior on cross-sequence attention logits -- never as a hard gate,
    # because atypical protocols exist and hard gating would silently destroy
    # recall on the sites that use them.
    informative_for: tuple[str, ...] = field(default=())


SEQUENCE_FAMILIES: Final[tuple[SequenceFamily, ...]] = (
    SequenceFamily(
        0, Plane.UNKNOWN, Weighting.UNKNOWN, False, "unknown",
        informative_for=TARGETS,
    ),
    SequenceFamily(
        1, Plane.SAGITTAL, Weighting.PD, True, "sag_pd_fs",
        informative_for=(
            "ACL", "Medial Meniscus", "Lateral Meniscus", "PF OA",
            "Effusion", "Contusion", "Fracture", "Synovitis",
        ),
    ),
    SequenceFamily(
        2, Plane.SAGITTAL, Weighting.T2, True, "sag_t2_fs",
        informative_for=(
            "ACL", "Medial Meniscus", "Lateral Meniscus", "Effusion",
            "Contusion", "Fracture", "Synovitis", "Baker's",
        ),
    ),
    SequenceFamily(
        3, Plane.SAGITTAL, Weighting.PD, False, "sag_pd",
        informative_for=(
            "ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "PF OA",
        ),
    ),
    SequenceFamily(
        4, Plane.SAGITTAL, Weighting.T1, False, "sag_t1",
        informative_for=("Fracture", "Contusion", "Medial OA", "Lateral OA"),
    ),
    SequenceFamily(
        5, Plane.CORONAL, Weighting.PD, True, "cor_pd_fs",
        informative_for=(
            "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "Contusion", "Fracture", "Effusion",
        ),
    ),
    SequenceFamily(
        6, Plane.CORONAL, Weighting.T2, True, "cor_t2_fs",
        informative_for=(
            "MCL", "Medial Meniscus", "Lateral Meniscus", "Contusion",
            "Fracture", "Effusion", "Synovitis",
        ),
    ),
    SequenceFamily(
        7, Plane.CORONAL, Weighting.T1, False, "cor_t1",
        informative_for=("Fracture", "Medial OA", "Lateral OA", "Contusion"),
    ),
    SequenceFamily(
        8, Plane.AXIAL, Weighting.PD, True, "ax_pd_fs",
        informative_for=(
            "PF OA", "Effusion", "Synovitis", "Baker's", "MCL", "Contusion",
        ),
    ),
    SequenceFamily(
        9, Plane.AXIAL, Weighting.T2, True, "ax_t2_fs",
        informative_for=("PF OA", "Effusion", "Synovitis", "Baker's"),
    ),
    SequenceFamily(
        10, Plane.AXIAL, Weighting.T1, False, "ax_t1",
        informative_for=("PF OA", "Fracture"),
    ),
    SequenceFamily(
        11, Plane.SAGITTAL, Weighting.GRE, False, "sag_gre",
        informative_for=("Medial OA", "Lateral OA", "PF OA", "Synovitis"),
    ),
    SequenceFamily(
        12, Plane.CORONAL, Weighting.STIR, True, "cor_stir",
        informative_for=("Contusion", "Fracture", "MCL", "Effusion"),
    ),
    SequenceFamily(
        13, Plane.SAGITTAL, Weighting.STIR, True, "sag_stir",
        informative_for=("Contusion", "Fracture", "Effusion", "Baker's"),
    ),
    SequenceFamily(
        14, Plane.OBLIQUE, Weighting.UNKNOWN, False, "oblique_other",
        informative_for=TARGETS,
    ),
)

NUM_SEQUENCE_FAMILIES: Final[int] = len(SEQUENCE_FAMILIES)
FAMILY_BY_NAME: Final[dict[str, SequenceFamily]] = {
    f.name: f for f in SEQUENCE_FAMILIES
}

# Boolean prior matrix  P[label, family] used to bias cross-sequence attention.
SEQUENCE_LABEL_PRIOR: Final[tuple[tuple[bool, ...], ...]] = tuple(
    tuple(target in fam.informative_for for fam in SEQUENCE_FAMILIES)
    for target in TARGETS
)


# --------------------------------------------------------------------------- #
# Preprocessing defaults                                                       #
# --------------------------------------------------------------------------- #

#: Robust intensity clipping percentiles applied inside the foreground mask.
INTENSITY_PERCENTILES: Final[tuple[float, float]] = (0.5, 99.5)

#: Target in-plane spacing (mm) for the coarse and fine passes.  We resample to
#: *physical* spacing rather than to a fixed pixel grid so that a 3 mm meniscal
#: root tear occupies the same number of pixels at every one of the 16 sites.
COARSE_SPACING_MM: Final[float] = 0.70
FINE_SPACING_MM: Final[float] = 0.35

COARSE_SIZE: Final[int] = 256
FINE_SIZE: Final[int] = 384

#: Number of adjacent slices stacked into the 2.5D channel dimension.
NEIGHBOUR_SLICES: Final[int] = 5

#: Cap on slices retained per series after physical resampling.
MAX_SLICES_PER_SERIES: Final[int] = 48

#: Cap on series retained per study (after QC ranking).
MAX_SERIES_PER_STUDY: Final[int] = 8


# --------------------------------------------------------------------------- #
# Weak-label states extracted from reports                                     #
# --------------------------------------------------------------------------- #


class Assertion(IntEnum):
    """Four-state assertion for a concept mention in a radiology report."""

    NOT_MENTIONED = 0
    NEGATIVE = 1
    UNCERTAIN = 2
    POSITIVE = 3


ASSERTION_SOFT_TARGET: Final[dict[Assertion, float]] = {
    Assertion.NOT_MENTIONED: float("nan"),  # masked out of the weak-label loss
    Assertion.NEGATIVE: 0.02,
    Assertion.UNCERTAIN: 0.50,
    Assertion.POSITIVE: 0.98,
}


__all__ = [
    "TARGETS",
    "NUM_TARGETS",
    "TARGET_INDEX",
    "SUBMISSION_COLUMNS",
    "Target",
    "LABEL_GROUPS",
    "GROUP_OF_LABEL",
    "Compartment",
    "LABEL_COMPARTMENT",
    "NUM_COMPARTMENTS",
    "Plane",
    "Weighting",
    "SequenceFamily",
    "SEQUENCE_FAMILIES",
    "NUM_SEQUENCE_FAMILIES",
    "FAMILY_BY_NAME",
    "SEQUENCE_LABEL_PRIOR",
    "INTENSITY_PERCENTILES",
    "COARSE_SPACING_MM",
    "FINE_SPACING_MM",
    "COARSE_SIZE",
    "FINE_SIZE",
    "NEIGHBOUR_SLICES",
    "MAX_SLICES_PER_SERIES",
    "MAX_SERIES_PER_STUDY",
    "Assertion",
    "ASSERTION_SOFT_TARGET",
]
