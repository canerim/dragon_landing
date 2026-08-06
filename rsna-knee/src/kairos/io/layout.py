r"""Competition file layout discovery.

The observed layout on Kaggle::

    /kaggle/input/competitions/rsna-knee-abnormality-detection/
        train.csv                 study-level (labels)
        test.csv                  study-level
        train_series.csv          series-level metadata
        test_series.csv           series-level metadata
        sample_submission.csv
        train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
        test_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm

Two things about it are easy to get wrong and both cost a whole submission:

* the root is under ``input/competitions/``, not ``input/`` -- the shorter path
  is what a Kaggle dataset attachment looks like, and it is what most public
  notebooks show, so it is the natural wrong guess;
* the images live in ``{split}_series/``, and ``{split}.csv`` exists alongside,
  so a naive ``COMP_DIR / split`` check finds nothing and a naive glob finds the
  CSV.

Rather than repeat those two facts in the baseline notebook, the inference
notebook and the training script -- three places to drift out of sync -- they
live here once, behind :func:`discover`, with the candidate lists ordered so
that the *observed* layout wins and the others remain as fallbacks. If the
organisers reshuffle the directory mid-competition, this is the single file to
change.

Everything is discovery, not assertion: :class:`CompetitionLayout` reports what
it found and what it did not, and the caller decides whether that is fatal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["CompetitionLayout", "discover", "iter_study_dirs", "ROOT_CANDIDATES"]


#: Ordered most- to least-likely.  The first entry is the layout observed in
#: the actual competition container.
ROOT_CANDIDATES: tuple[str, ...] = (
    "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    "/kaggle/input/rsna-knee-abnormality-detection",
    "/kaggle/input/rsna-knee-abnormalities-detection",
    "data",
)

#: Image directory names to try per split, in order.
_IMAGE_DIR_CANDIDATES = ("{split}_series", "{split}_images", "{split}")


@dataclass(slots=True)
class CompetitionLayout:
    root: Path
    train_images: Path | None = None
    test_images: Path | None = None
    train_csv: Path | None = None
    test_csv: Path | None = None
    train_series_csv: Path | None = None
    test_series_csv: Path | None = None
    sample_submission: Path | None = None
    extra_csv: dict[str, Path] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #

    def images_for(self, split: str) -> Path | None:
        return self.train_images if split.startswith("train") else self.test_images

    def csv_for(self, split: str) -> Path | None:
        return self.train_csv if split.startswith("train") else self.test_csv

    @property
    def ok(self) -> bool:
        """Enough was found to run inference: test images and a UID source."""
        return self.test_images is not None and (
            self.sample_submission is not None or self.test_csv is not None
        )

    def describe(self) -> str:
        lines = [f"root: {self.root}"]
        for name in ("train_images", "test_images", "train_csv", "test_csv",
                     "train_series_csv", "test_series_csv", "sample_submission"):
            v = getattr(self, name)
            lines.append(f"  {name:<20} {v if v else '(not found)'}")
        for k, v in sorted(self.extra_csv.items()):
            lines.append(f"  {'extra:' + k:<20} {v}")
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


def _find_images(root: Path, split: str) -> tuple[Path | None, str | None]:
    for pattern in _IMAGE_DIR_CANDIDATES:
        p = root / pattern.format(split=split)
        # is_dir() matters: ``test.csv`` exists next to ``test_series/``, so a
        # bare exists() check on the "test" candidate would match the CSV's
        # stem in some layouts and a directory in others.
        if p.is_dir():
            return p, None
    # Last resort: any directory whose name mentions the split and which
    # contains at least one nested directory (study -> series).
    for p in sorted(root.iterdir()) if root.is_dir() else []:
        if p.is_dir() and split in p.name.lower():
            if any(c.is_dir() for c in list(p.iterdir())[:5]):
                return p, f"{split} images found by fallback scan at {p.name}"
    return None, f"no {split} image directory under {root}"


def discover(root: str | Path | None = None) -> CompetitionLayout:
    """Locate the competition files.

    ``root`` overrides the search; otherwise :data:`ROOT_CANDIDATES` is tried in
    order and the first existing directory wins.
    """
    if root is not None:
        base = Path(root)
    else:
        base = next((Path(c) for c in ROOT_CANDIDATES if Path(c).is_dir()),
                    Path(ROOT_CANDIDATES[0]))

    layout = CompetitionLayout(root=base)
    if not base.is_dir():
        layout.notes.append(f"root {base} does not exist")
        return layout

    for split, attr in (("train", "train_images"), ("test", "test_images")):
        found, note = _find_images(base, split)
        setattr(layout, attr, found)
        if note:
            layout.notes.append(note)

    known = {
        "train.csv": "train_csv",
        "test.csv": "test_csv",
        "train_series.csv": "train_series_csv",
        "test_series.csv": "test_series_csv",
        "sample_submission.csv": "sample_submission",
    }
    for csv in sorted(base.glob("*.csv")):
        attr = known.get(csv.name)
        if attr:
            setattr(layout, attr, csv)
        else:
            layout.extra_csv[csv.stem] = csv

    if layout.sample_submission is None:
        layout.notes.append(
            "no sample_submission.csv; the submission UID set must come from "
            "test.csv or from enumerating the test image directory"
        )
    return layout


def iter_study_dirs(images_root: Path, uids: list[str] | None = None):
    """Yield ``(uid, study_dir)`` for the studies present under ``images_root``.

    When ``uids`` is given the order follows it and missing studies are yielded
    with a non-existent path, so the caller can count them rather than silently
    processing a shorter list than the submission requires.
    """
    images_root = Path(images_root)
    if uids is not None:
        for u in uids:
            yield str(u), images_root / str(u)
        return
    for d in sorted(images_root.iterdir()):
        if d.is_dir():
            yield d.name, d
