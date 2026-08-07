r"""Mapping a DICOM series to a (plane, weighting, fat-sat) family.

``SeriesDescription`` is free text authored independently at sixteen sites in
twelve languages.  It is informative but it is not a key: the same sequence is
``"Sag PD FS"``, ``"SAG DP SAT GRASA"``, ``"sag_pdw_fs_tse"``,
``"矢状位 PD 压脂"`` and ``"Sagittal PD Fatsat"`` depending on where you are.

So we classify on the *physics* first -- repetition time, echo time, inversion
time, scanning sequence, scan options, sequence variant -- and use the
description only to break ties and to catch fat saturation when the header does
not declare it.

The decision rules below are the standard MR contrast heuristics:

===================  ============================  =========================
weighting            TR (ms)                       TE (ms)
===================  ============================  =========================
T1                   < 800                         < 30
PD                   > 1500                        < 40
T2                   > 1500                        > 60
STIR                 any, with TI ≈ 120–180 ms     any
GRE                  ``ScanningSequence`` = ``GR``  any
===================  ============================  =========================

The band between PD and T2 (TE 40–60 ms) is genuinely ambiguous -- many knee
protocols run an "intermediate-weighted" sequence there -- and we assign it to
PD, which is what it is used for clinically.  Anything the rules cannot place
goes to ``unknown`` rather than being forced, because a mislabelled family
poisons the (label × family) attention prior at exactly the sites with unusual
protocols.
"""

from __future__ import annotations

import re
from typing import Any

from ..constants import FAMILY_BY_NAME, SEQUENCE_FAMILIES, Plane, SequenceFamily, Weighting

__all__ = ["classify_series", "infer_weighting", "infer_fat_sat", "FAT_SAT_PATTERNS"]


FAT_SAT_PATTERNS = re.compile(
    r"\b(fs|fat[\s_-]?sat|fatsat|spair|spir|stir|sat|frfse[\s_-]?fs|"
    r"tirm|dixon|chess|yağ|grasa|graisse|fettsat|脂肪抑制|压脂|脂肪抑制)\b",
    re.IGNORECASE,
)

_STIR_PATTERN = re.compile(r"\b(stir|tirm)\b", re.IGNORECASE)
_GRE_PATTERN = re.compile(r"\b(gre|grass|fisp|medic|merge|t2\*|gradient)\b", re.IGNORECASE)
_T1_PATTERN = re.compile(r"\bt1\b", re.IGNORECASE)
_T2_PATTERN = re.compile(r"\bt2\b", re.IGNORECASE)
_PD_PATTERN = re.compile(r"\b(pd|dp|proton|intermediate|iw)\b", re.IGNORECASE)


def _get(ds: Any, name: str, default=None):
    v = getattr(ds, name, default)
    if v is None or v == "":
        return default
    return v


def _float(ds: Any, name: str, default: float = 0.0) -> float:
    try:
        return float(_get(ds, name, default))
    except (TypeError, ValueError):
        return default


_SEPARATORS = re.compile(r"[_\-./\\+]+")


def _description(ds: Any) -> str:
    """Concatenate the free-text fields, with separators turned into spaces.

    ``sag_pdw_fs_tse`` is a real and common series description.  Python's
    ``\\b`` treats ``_`` as a word character, so ``\\bfs\\b`` does *not* match
    inside it and the series is silently classified as non-fat-suppressed --
    which then routes it to the wrong sequence family and the wrong attention
    prior.  Normalising separators first is the whole fix.
    """
    parts = [
        str(_get(ds, "SeriesDescription", "") or ""),
        str(_get(ds, "ProtocolName", "") or ""),
        str(_get(ds, "SequenceName", "") or ""),
    ]
    return _SEPARATORS.sub(" ", " ".join(p for p in parts if p))


def infer_fat_sat(ds: Any) -> bool:
    """Fat suppression from ``ScanOptions``/``ScanningSequence`` then description."""
    opts = _get(ds, "ScanOptions", "") or ""
    if isinstance(opts, (list, tuple)):
        opts = " ".join(str(o) for o in opts)
    opts = str(opts)
    if re.search(r"\b(FS|SP|FATSAT|SAT)\b", opts, re.IGNORECASE):
        return True
    seq = _get(ds, "ScanningSequence", "") or ""
    if isinstance(seq, (list, tuple)):
        seq = " ".join(str(s) for s in seq)
    if "IR" in str(seq) and 100.0 <= _float(ds, "InversionTime", 0.0) <= 200.0:
        return True  # STIR: fat-suppressed by inversion
    return bool(FAT_SAT_PATTERNS.search(_description(ds)))


def infer_weighting(ds: Any) -> Weighting:
    """Contrast weighting from acquisition parameters, then description."""
    tr = _float(ds, "RepetitionTime", 0.0)
    te = _float(ds, "EchoTime", 0.0)
    ti = _float(ds, "InversionTime", 0.0)
    seq = _get(ds, "ScanningSequence", "") or ""
    if isinstance(seq, (list, tuple)):
        seq = " ".join(str(s) for s in seq)
    seq = str(seq)
    desc = _description(ds)

    if 100.0 <= ti <= 200.0 or _STIR_PATTERN.search(desc):
        return Weighting.STIR
    if "GR" in seq or _GRE_PATTERN.search(desc):
        return Weighting.GRE

    if tr > 0 and te > 0:
        if tr < 800 and te < 30:
            return Weighting.T1
        if tr > 1500:
            # 40-60 ms is the intermediate-weighted band; clinically it is read
            # as PD, so that is where it goes.
            return Weighting.T2 if te > 60 else Weighting.PD

    if _T1_PATTERN.search(desc):
        return Weighting.T1
    if _T2_PATTERN.search(desc):
        return Weighting.T2
    if _PD_PATTERN.search(desc):
        return Weighting.PD
    return Weighting.UNKNOWN


def classify_series(ds: Any, *, plane: Plane | None = None) -> SequenceFamily:
    """Map a DICOM dataset to one of :data:`kairos.constants.SEQUENCE_FAMILIES`.

    ``plane`` should come from the *geometry* (``classify_plane`` on the slice
    normal), not from the description -- the description lies about orientation
    surprisingly often, and the geometry never does.
    """
    if plane is None:
        plane = Plane.UNKNOWN
    w = infer_weighting(ds)
    fs = infer_fat_sat(ds)

    if plane == Plane.OBLIQUE:
        return FAMILY_BY_NAME["oblique_other"]

    for fam in SEQUENCE_FAMILIES:
        if fam.index == 0:
            continue
        if fam.plane == plane and fam.weighting == w and fam.fat_sat == fs:
            return fam

    # Graceful degradation: same plane and weighting, ignoring fat-sat.
    for fam in SEQUENCE_FAMILIES:
        if fam.index == 0:
            continue
        if fam.plane == plane and fam.weighting == w:
            return fam
    return SEQUENCE_FAMILIES[0]
