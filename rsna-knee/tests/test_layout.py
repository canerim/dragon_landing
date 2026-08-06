"""Competition file-layout discovery, against the observed Kaggle structure.

The real layout is

    input/competitions/rsna-knee-abnormality-detection/
        {train,test}.csv  {train,test}_series.csv  sample_submission.csv
        {train,test}_series/<StudyUID>/<SeriesUID>/<SOPUID>.dcm

Two things about it are easy to get wrong and both cost a submission: the root
sits under ``competitions/``, and the images live in ``{split}_series/`` while
``{split}.csv`` sits right next to it. These tests pin both.
"""

from __future__ import annotations

import pytest

from kairos.constants import TARGETS
from kairos.io.layout import ROOT_CANDIDATES, discover, iter_study_dirs


def _build(tmp_path, *, splits=("train", "test"), csvs=True, suffix="_series"):
    root = tmp_path / "competitions" / "rsna-knee-abnormality-detection"
    root.mkdir(parents=True)
    uids = {}
    for split in splits:
        uids[split] = []
        for st in range(3):
            uid = f"1.2.826.{split}.{st}"
            uids[split].append(uid)
            for se in range(2):
                d = root / f"{split}{suffix}" / uid / f"1.2.826.{split}.{st}.{se}"
                d.mkdir(parents=True)
                (d / "0001.dcm").write_bytes(b"\0" * 8)
    if csvs:
        header = "StudyInstanceUID," + ",".join(TARGETS)
        for split in splits:
            (root / f"{split}.csv").write_text(
                "StudyInstanceUID\n" + "\n".join(uids[split])
            )
            (root / f"{split}_series.csv").write_text(
                "StudyInstanceUID,SeriesInstanceUID,SeriesDescription\n"
            )
        (root / "sample_submission.csv").write_text(
            header + "\n" + "\n".join(f"{u}," + ",".join(["0.5"] * 12)
                                      for u in uids.get("test", []))
        )
    return root, uids


def test_discovers_the_observed_layout(tmp_path):
    root, uids = _build(tmp_path)
    lay = discover(root)
    assert lay.ok
    assert lay.test_images == root / "test_series"
    assert lay.train_images == root / "train_series"
    assert lay.train_csv == root / "train.csv"
    assert lay.test_csv == root / "test.csv"
    assert lay.train_series_csv == root / "train_series.csv"
    assert lay.test_series_csv == root / "test_series.csv"
    assert lay.sample_submission == root / "sample_submission.csv"
    assert lay.extra_csv == {}


def test_split_csv_is_not_mistaken_for_the_image_directory(tmp_path):
    """``test.csv`` sits next to ``test_series/``; a bare "test" probe must not win."""
    root, _ = _build(tmp_path)
    lay = discover(root)
    assert lay.test_images is not None
    assert lay.test_images.is_dir()
    assert lay.test_images.name == "test_series"


def test_falls_back_to_split_images_and_bare_split_names(tmp_path):
    root, _ = _build(tmp_path, suffix="_images", csvs=False)
    assert discover(root).test_images == root / "test_images"

    root2, _ = _build(tmp_path / "b", suffix="", csvs=False)
    assert discover(root2).test_images == root2 / "test"


def test_reports_rather_than_raises_on_a_missing_root(tmp_path):
    lay = discover(tmp_path / "nope")
    assert not lay.ok
    assert lay.test_images is None
    assert any("does not exist" in n for n in lay.notes)
    assert "nope" in lay.describe()


def test_missing_sample_submission_is_noted_not_fatal(tmp_path):
    root, _ = _build(tmp_path, csvs=False)
    lay = discover(root)
    assert lay.sample_submission is None
    assert any("sample_submission" in n for n in lay.notes)
    # test.csv is absent too here, so ok is False -- but test_images was found.
    assert lay.test_images is not None


def test_test_csv_alone_is_enough_to_proceed(tmp_path):
    root, uids = _build(tmp_path, csvs=False)
    (root / "test.csv").write_text("StudyInstanceUID\n" + "\n".join(uids["test"]))
    lay = discover(root)
    assert lay.ok
    assert lay.sample_submission is None
    assert lay.test_csv is not None


def test_unknown_csvs_are_surfaced_not_dropped(tmp_path):
    root, _ = _build(tmp_path)
    (root / "train_reports.csv").write_text("StudyInstanceUID,report\n")
    lay = discover(root)
    assert "train_reports" in lay.extra_csv


def test_default_search_order_prefers_the_competitions_path():
    assert ROOT_CANDIDATES[0] == (
        "/kaggle/input/competitions/rsna-knee-abnormality-detection"
    )
    assert "/kaggle/input/rsna-knee-abnormality-detection" in ROOT_CANDIDATES


def test_images_for_and_csv_for_route_by_split(tmp_path):
    root, _ = _build(tmp_path)
    lay = discover(root)
    assert lay.images_for("train") == lay.train_images
    assert lay.images_for("test") == lay.test_images
    assert lay.csv_for("train") == lay.train_csv


def test_iter_study_dirs_enumerates_and_preserves_a_requested_order(tmp_path):
    root, uids = _build(tmp_path)
    lay = discover(root)

    found = [u for u, _ in iter_study_dirs(lay.test_images)]
    assert sorted(found) == sorted(uids["test"])

    wanted = list(reversed(uids["test"])) + ["missing-uid"]
    got = list(iter_study_dirs(lay.test_images, wanted))
    assert [u for u, _ in got] == wanted
    # A requested-but-absent study is yielded with a non-existent path so the
    # caller can count it rather than silently producing a shorter submission.
    assert not got[-1][1].exists()


def test_study_dirs_contain_series_subdirectories(tmp_path):
    root, _ = _build(tmp_path)
    lay = discover(root)
    _, d = next(iter(iter_study_dirs(lay.test_images)))
    series = [c for c in d.iterdir() if c.is_dir()]
    assert series, "study -> series -> *.dcm nesting is what load_study assumes"
    assert list(series[0].glob("*.dcm"))


@pytest.mark.parametrize("split", ["train", "test"])
def test_load_study_accepts_the_discovered_paths(tmp_path, split):
    """The paths discovery returns must be the ones load_study can consume."""
    pytest.importorskip("pydicom")
    root, uids = _build(tmp_path, splits=(split,), csvs=False)
    lay = discover(root)
    images = lay.images_for(split)
    assert images is not None
    _, study_dir = next(iter(iter_study_dirs(images)))
    # The fixture writes stub bytes rather than valid DICOM, so load_study must
    # degrade to a recorded error instead of raising.
    from kairos.data.dataset import load_study

    rec = load_study(study_dir)
    assert rec.errors, "unreadable studies must be reported, not silently empty"
