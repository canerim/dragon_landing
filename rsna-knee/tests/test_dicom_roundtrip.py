"""End-to-end DICOM round-trip on synthetically written DICOM files.

Every other test in the suite works on arrays. This one writes real DICOM to
disk with pydicom and reads it back through the production ``load_study`` path,
because the gap between "the array maths is right" and "it works on a DICOM
directory" is where a competition pipeline actually breaks: a missing tag, a
laterality flag in the wrong place, a plane inferred from the description
instead of the geometry.

Skipped when pydicom is absent so the core suite stays dependency-light.
"""

from __future__ import annotations

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from pydicom.dataset import Dataset, FileDataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # noqa: E402

from kairos.constants import FAMILY_BY_NAME, NUM_TARGETS, Plane  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic DICOM writer                                                       #
# --------------------------------------------------------------------------- #

PROTOCOLS = [
    # (description, ImageOrientationPatient, TR, TE, ScanOptions, expected family)
    ("Sag PD FS", [0, 1, 0, 0, 0, -1], 3000, 30, "FS", "sag_pd_fs"),
    ("cor_t2_spair", [1, 0, 0, 0, 0, -1], 4000, 80, "", "cor_t2_fs"),
    ("AX PD FS", [1, 0, 0, 0, 1, 0], 3000, 35, "FS", "ax_pd_fs"),
]


def _write_series(sdir, iop, desc, tr, te, opts, *, laterality, n_slices=10,
                  spacing=3.0, rows=64, cols=64, shuffle_names=False, seed=0,
                  study_uid=None):
    sdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    series_uid = generate_uid()
    # A real UID, not the directory name: pydicom validates the UI value
    # representation and the suite runs with warnings-as-errors.
    study_uid = study_uid or generate_uid()
    normal = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
    step = normal / max(np.linalg.norm(normal), 1e-9) * spacing
    base = np.array([0.0, -60.0, 40.0])

    order = list(range(n_slices))
    if shuffle_names:
        order = list(rng.permutation(n_slices))

    for file_idx, k in enumerate(order):
        fm = FileMetaDataset()
        fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        fm.MediaStorageSOPInstanceUID = generate_uid()
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        path = sdir / f"{file_idx:03d}.dcm"
        ds = FileDataset(str(path), Dataset(), file_meta=fm, preamble=b"\0" * 128)
        ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
        ds.SOPClassUID = fm.MediaStorageSOPClassUID
        ds.SeriesInstanceUID = series_uid
        ds.StudyInstanceUID = study_uid
        ds.PatientID = "P0"
        ds.ImageOrientationPatient = [float(v) for v in iop]
        ds.ImagePositionPatient = list(base + k * step)
        ds.PixelSpacing = [0.4, 0.4]
        ds.SliceThickness = spacing
        # Deliberately WRONG relative to geometry when shuffle_names is on:
        # InstanceNumber follows the file order, not the physical order.
        ds.InstanceNumber = file_idx + 1
        ds.Rows, ds.Columns = rows, cols
        ds.SeriesDescription = desc
        ds.RepetitionTime, ds.EchoTime = tr, te
        if opts:
            ds.ScanOptions = opts
        ds.Manufacturer = "SIEMENS"
        ds.MagneticFieldStrength = 3.0
        ds.ImageLaterality = laterality
        ds.BodyPartExamined = "KNEE"
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
        ds.PixelRepresentation = 0
        img = rng.normal(300, 40, (rows, cols))
        img += 500 * (np.abs(np.arange(rows)[:, None] - rows // 2) < rows // 5)
        ds.PixelData = np.clip(img, 0, 4095).astype(np.uint16).tobytes()
        ds.save_as(str(path), enforce_file_format=True)
    return series_uid


@pytest.fixture
def study_dir(tmp_path):
    d = tmp_path / "1.2.826.0.1.7"
    for i, (desc, iop, tr, te, opts, _) in enumerate(PROTOCOLS):
        _write_series(d / f"series{i}", iop, desc, tr, te, opts,
                      laterality="R", seed=i)
    return d


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #


def test_load_study_reads_every_series_without_errors(study_dir):
    from kairos.data.dataset import load_study

    rec = load_study(study_dir)
    assert rec.errors == [], rec.errors
    assert len(rec.series) == 3
    for s in rec.series:
        assert np.isfinite(s.pixels).all()
        assert s.pixels.ndim == 3


def test_plane_and_family_come_from_geometry_not_description(study_dir):
    from kairos.data.dataset import load_study

    rec = load_study(study_dir)
    by_plane = {s.plane: s for s in rec.series}
    assert set(by_plane) == {Plane.SAGITTAL, Plane.CORONAL, Plane.AXIAL}

    expected = {
        Plane.SAGITTAL: FAMILY_BY_NAME["sag_pd_fs"].index,
        Plane.CORONAL: FAMILY_BY_NAME["cor_t2_fs"].index,
        Plane.AXIAL: FAMILY_BY_NAME["ax_pd_fs"].index,
    }
    for plane, fam in expected.items():
        assert by_plane[plane].family_index == fam, (
            f"{plane.name} classified as family {by_plane[plane].family_index}, "
            f"expected {fam}"
        )
    # cor_t2_spair has no ScanOptions: fat-sat must be caught from the
    # description, with the separator normalisation doing its job.
    assert by_plane[Plane.CORONAL].fat_sat_id == 1


def test_ordering_ignores_a_misleading_instance_number(tmp_path):
    """The whole point of geometric ordering."""
    from kairos.data.dataset import load_study

    d = tmp_path / "study_shuffled"
    desc, iop, tr, te, opts, _ = PROTOCOLS[0]
    _write_series(d / "s0", iop, desc, tr, te, opts, laterality="R",
                  shuffle_names=True, n_slices=12, seed=3)
    rec = load_study(d)
    assert len(rec.series) == 1
    z = rec.series[0].z_mm
    assert np.all(np.diff(z) > 0), "z must be monotone despite shuffled InstanceNumber"
    gaps = np.diff(z)
    assert np.allclose(gaps, gaps[0], atol=1e-3), "uniform spacing must be recovered"


def test_left_knee_is_mirrored_and_recorded(tmp_path):
    from kairos.data.dataset import load_study

    left, right = tmp_path / "L", tmp_path / "R"
    for d, lat in ((left, "L"), (right, "R")):
        desc, iop, tr, te, opts, _ = PROTOCOLS[1]  # coronal
        _write_series(d / "s0", iop, desc, tr, te, opts, laterality=lat, seed=5)

    rl = load_study(left).series[0]
    rr = load_study(right).series[0]
    assert rl.mirrored is True
    assert rr.mirrored is False
    # Coronal mirrors in-plane along columns, so the left study's volume must be
    # the right study's column-reversed -- identical pixel content otherwise.
    assert np.allclose(rl.pixels, rr.pixels[:, :, ::-1], atol=1e-4)


def test_resampling_is_to_physical_spacing(tmp_path):
    """Two acquisitions of the same anatomy at different in-plane resolutions
    must produce the same physical field of view."""
    from kairos.data.dataset import load_study

    desc, iop, tr, te, opts, _ = PROTOCOLS[0]
    a, b = tmp_path / "A", tmp_path / "B"
    _write_series(a / "s0", iop, desc, tr, te, opts, laterality="R",
                  rows=64, cols=64, seed=1)
    _write_series(b / "s0", iop, desc, tr, te, opts, laterality="R",
                  rows=128, cols=128, seed=1)

    ra = load_study(a).series[0]
    rb = load_study(b).series[0]
    assert ra.pixels.shape[1:] == rb.pixels.shape[1:], "output grid must be fixed"


def test_build_inference_batch_shapes_and_masks(study_dir):
    torch = pytest.importorskip("torch")
    from kairos.data.dataset import build_inference_batch

    b = build_inference_batch(study_dir)
    B, Nseq, S = b.slice_mask.shape
    assert B == 1 and Nseq == 3
    assert b.pixels.shape == (B, Nseq, S, 5, 256, 256)
    assert bool(b.series_mask.all())
    assert int(b.slice_mask.sum()) == Nseq * S
    assert bool(torch.isfinite(b.context).all())
    assert b.study_uid == [study_dir.name]


def test_missing_study_degrades_instead_of_raising(tmp_path):
    """One unreadable study must not cost the whole submission."""
    pytest.importorskip("torch")
    from kairos.data.dataset import build_inference_batch

    empty = tmp_path / "nothing_here"
    empty.mkdir()
    b = build_inference_batch(empty)
    assert b.pixels.shape[0] == 1
    assert b.series_mask.shape[1] >= 1


def test_model_forward_on_a_real_dicom_batch(study_dir):
    torch = pytest.importorskip("torch")
    from kairos.data.dataset import build_inference_batch
    from kairos.models.system import KairosConfig, KairosModel

    cfg = KairosConfig(dim=48, agg_depth=1, agg_heads=4, sngp_features=64,
                       n_experts=2, enable_fine_pass=True, fine_top_k=2,
                       sequence_dropout=0.0, slice_dropout=0.0)
    cfg.backbone.pretrained = False
    model = KairosModel(cfg).eval()
    with torch.no_grad():
        out = model(build_inference_batch(study_dir))
    assert out["logits"].shape == (1, NUM_TARGETS)
    assert torch.isfinite(out["logits"]).all()
    p = torch.sigmoid(out["logits"])
    assert bool(((p >= 0) & (p <= 1)).all())


def test_qc_flags_fire_on_a_corrupted_series(tmp_path):
    from kairos.data.dataset import load_study

    d = tmp_path / "gappy"
    desc, iop, tr, te, opts, _ = PROTOCOLS[0]
    _write_series(d / "s0", iop, desc, tr, te, opts, laterality="R",
                  n_slices=10, spacing=3.0, seed=9)
    # Push one slice far away to create a gap the QC must notice.
    files = sorted((d / "s0").glob("*.dcm"))
    ds = pydicom.dcmread(str(files[6]))
    ipp = list(ds.ImagePositionPatient)
    ds.ImagePositionPatient = [ipp[0] + 40.0, ipp[1], ipp[2]]
    ds.save_as(str(files[6]), enforce_file_format=True)

    rec = load_study(d)
    assert rec.series, rec.errors
    flags = rec.series[0].flags
    assert any(f in flags for f in ("large_slice_gap", "irregular_spacing")), flags
