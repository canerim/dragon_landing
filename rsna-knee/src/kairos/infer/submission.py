r"""Submission assembly with fail-fast validation.

Every competition loses a handful of teams to a malformed CSV on the last day.
The failure modes are boring and entirely preventable: a stray index column, an
apostrophe normalised by a spreadsheet, a UID set that differs from the sample
by one row, a NaN from a study whose DICOM failed to decode.

This module writes the file and then *re-reads it and checks it*, because the
only thing that matters is what is on disk.  Validation is strict by default:
it raises rather than warns, so the notebook fails during the commit run rather
than producing a scored-but-wrong submission.

The column names are taken verbatim from :data:`kairos.constants.TARGETS`
including ``Baker's`` with a straight ASCII apostrophe (U+0027).  A typographic
apostrophe (U+2019) is a different column name and scores zero; the validator
checks the exact code points.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..constants import SUBMISSION_COLUMNS, TARGETS

__all__ = ["build_submission", "validate_submission", "SubmissionError"]


class SubmissionError(RuntimeError):
    """Raised when a submission fails validation.  Never downgraded to a warning."""


def build_submission(
    study_uids: list[str] | np.ndarray,
    probabilities: np.ndarray,
    *,
    output_path: str | Path = "submission.csv",
    sample_submission: "object | None" = None,
    fill_missing: float = 0.5,
    strict: bool = True,
    allow_constant: bool = False,
):
    """Write ``submission.csv`` and validate it by reading it back.

    Parameters
    ----------
    sample_submission
        Optional ``pandas.DataFrame`` (or path) of the competition's
        ``sample_submission.csv``.  When given, the output is *reindexed onto
        its UID set and row order*, and any UID we failed to predict is filled
        with ``fill_missing``.  This is the single most valuable safety net in
        the whole pipeline: a study that crashes DICOM decoding then costs a
        0.5 prediction instead of an invalid file.
    allow_constant
        Permit columns with zero variance.  Off by default, because a constant
        column in a *real* submission means a head never fired or a checkpoint
        failed to load, and it scores exactly 0.5 on that label.

        It must be ``True`` for the deliberately-constant fallback the
        inference notebook writes before inference starts -- otherwise the
        safety net raises and there is no file at all, which is the exact
        failure the safety net exists to prevent.  (This is not hypothetical:
        the notebook shipped with that bug until a test caught it.)
    """
    import pandas as pd

    uids = [str(u) for u in np.asarray(study_uids).ravel()]
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != len(TARGETS):
        raise SubmissionError(
            f"probabilities must be (N, {len(TARGETS)}), got {p.shape}"
        )
    if p.shape[0] != len(uids):
        raise SubmissionError(f"{len(uids)} uids but {p.shape[0]} prediction rows")

    n_bad = int((~np.isfinite(p)).sum())
    if n_bad:
        if strict and n_bad > 0.01 * p.size:
            raise SubmissionError(f"{n_bad} non-finite predictions ({n_bad / p.size:.2%})")
        p = np.nan_to_num(p, nan=fill_missing, posinf=1.0, neginf=0.0)
    p = np.clip(p, 0.0, 1.0)

    df = pd.DataFrame(p, columns=list(TARGETS))
    df.insert(0, "StudyInstanceUID", uids)

    if len(set(uids)) != len(uids):
        # Duplicate study rows: average them rather than dropping, because a
        # duplicate usually means the study was processed by two shards.
        df = df.groupby("StudyInstanceUID", as_index=False, sort=False).mean()

    if sample_submission is not None:
        sample = (
            pd.read_csv(sample_submission)
            if isinstance(sample_submission, (str, Path))
            else sample_submission
        )
        want = sample["StudyInstanceUID"].astype(str).tolist()
        df = (
            pd.DataFrame({"StudyInstanceUID": want})
            .merge(df, on="StudyInstanceUID", how="left")
            .fillna(fill_missing)
        )
        missing = int(df[list(TARGETS)].isna().sum().sum())
        if missing:
            raise SubmissionError("NaNs survived the sample-submission merge")

    df = df[list(SUBMISSION_COLUMNS)]
    path = Path(output_path)
    df.to_csv(path, index=False)

    validate_submission(
        path,
        expected_uids=None if sample_submission is None else want,
        require_varying=not allow_constant,
    )
    return df


def validate_submission(
    path: str | Path,
    *,
    expected_uids: list[str] | None = None,
    require_varying: bool = True,
) -> dict:
    """Re-read a submission file and assert every format invariant."""
    import pandas as pd

    path = Path(path)
    if path.name != "submission.csv":
        raise SubmissionError(f"file must be named submission.csv, got {path.name!r}")

    raw = path.read_text(encoding="utf-8")
    header = raw.split("\n", 1)[0].rstrip("\r")
    expected_header = ",".join(SUBMISSION_COLUMNS)
    if header != expected_header:
        raise SubmissionError(
            "header mismatch\n"
            f"  expected: {expected_header!r}\n"
            f"  found   : {header!r}\n"
            "  (check the apostrophe in Baker's is U+0027, not U+2019)"
        )

    df = pd.read_csv(path)
    if list(df.columns) != list(SUBMISSION_COLUMNS):
        raise SubmissionError(f"column mismatch after read-back: {list(df.columns)}")
    if df["StudyInstanceUID"].duplicated().any():
        dup = df.loc[df["StudyInstanceUID"].duplicated(), "StudyInstanceUID"].tolist()[:5]
        raise SubmissionError(f"duplicate StudyInstanceUID: {dup}")

    vals = df[list(TARGETS)].to_numpy(dtype=np.float64)
    if not np.isfinite(vals).all():
        raise SubmissionError(f"{int((~np.isfinite(vals)).sum())} non-finite values on disk")
    if vals.min() < 0.0 or vals.max() > 1.0:
        raise SubmissionError(f"values out of [0,1]: [{vals.min()}, {vals.max()}]")

    if expected_uids is not None:
        got = df["StudyInstanceUID"].astype(str).tolist()
        if got != [str(u) for u in expected_uids]:
            missing = set(map(str, expected_uids)) - set(got)
            extra = set(got) - set(map(str, expected_uids))
            raise SubmissionError(
                f"UID set/order mismatch: {len(missing)} missing, {len(extra)} extra"
            )

    # A constant column scores exactly 0.5 AUC and is almost always a bug --
    # a head that never fired, or a fold whose weights failed to load.
    constant = [t for t in TARGETS if float(np.ptp(df[t].to_numpy())) < 1e-9]
    if constant and require_varying:
        raise SubmissionError(f"constant prediction column(s): {constant}")

    return {
        "n_rows": int(len(df)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "constant_columns": constant,
        "mean_per_label": {t: float(df[t].mean()) for t in TARGETS},
    }
