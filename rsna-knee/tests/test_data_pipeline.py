"""Preprocessing and sequence-taxonomy behaviour on synthetic inputs."""

from __future__ import annotations

import numpy as np
import pytest

from kairos.constants import Plane, Weighting
from kairos.data.dataset import (
    _bilinear_resize,
    foreground_mask,
    resample_to_spacing,
    robust_normalise,
    select_slices,
    stack_neighbours,
)
from kairos.data.sequence_taxonomy import classify_series, infer_fat_sat, infer_weighting


class FakeDS:
    """A minimal stand-in for a pydicom Dataset."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# --------------------------------------------------------------------------- #
# Intensity                                                                    #
# --------------------------------------------------------------------------- #


def test_foreground_mask_separates_tissue_from_background():
    rng = np.random.default_rng(0)
    v = rng.normal(20, 5, size=(6, 64, 64))
    v[:, 16:48, 16:48] += 400.0  # "tissue"
    m = foreground_mask(v)
    assert m[:, 24:40, 24:40].mean() > 0.95
    assert m[:, :8, :8].mean() < 0.05


def test_foreground_mask_falls_back_on_a_uniform_volume():
    v = np.full((4, 32, 32), 7.0)
    m = foreground_mask(v)
    assert m.shape == v.shape
    assert np.isfinite(m).all()


def test_robust_normalise_is_invariant_to_affine_intensity_rescaling():
    rng = np.random.default_rng(1)
    v = rng.normal(0, 1, size=(8, 48, 48))
    v[:, 12:36, 12:36] += 6.0
    a = robust_normalise(v)
    b = robust_normalise(v * 137.0 + 5000.0)
    # This is the property that makes the model transferable across vendors.
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.999
    assert abs(float(np.median(a[foreground_mask(v)]))) < 0.1


def test_robust_normalise_survives_a_metal_artefact():
    rng = np.random.default_rng(2)
    v = rng.normal(100, 10, size=(6, 40, 40))
    clean = robust_normalise(v.copy())
    v[3, 5:9, 5:9] = 1e6  # susceptibility blowout
    dirty = robust_normalise(v)
    # MAD-based scaling must not let four pixels rescale the whole series.
    assert abs(float(np.std(clean)) - float(np.std(dirty))) < 0.5


def test_robust_normalise_handles_a_constant_volume():
    out = robust_normalise(np.full((3, 16, 16), 4.0))
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Spatial                                                                      #
# --------------------------------------------------------------------------- #


def test_bilinear_resize_preserves_a_constant_and_is_shape_correct():
    img = np.full((17, 23), 3.5)
    out = _bilinear_resize(img, 40, 11)
    assert out.shape == (40, 11)
    assert np.allclose(out, 3.5, atol=1e-5)


def test_bilinear_resize_is_close_to_identity_for_a_ramp():
    img = np.tile(np.linspace(0, 1, 32), (32, 1))
    out = _bilinear_resize(_bilinear_resize(img, 64, 64), 32, 32)
    assert np.abs(out - img).max() < 0.05


def test_resample_gives_the_same_physical_field_of_view():
    """The core invariant: identical anatomy → identical pixels at any site."""
    # A 20 mm square object imaged at two different in-plane resolutions.
    fine = np.zeros((1, 200, 200), dtype=np.float32)  # 0.25 mm/px -> 50 mm FOV
    fine[0, 60:140, 60:140] = 1.0  # 20 mm object centred
    coarse = np.zeros((1, 50, 50), dtype=np.float32)  # 1.0 mm/px -> 50 mm FOV
    coarse[0, 15:35, 15:35] = 1.0  # the same 20 mm object

    a = resample_to_spacing(fine, 0.25, 0.5, 64)
    b = resample_to_spacing(coarse, 1.0, 0.5, 64)
    assert a.shape == b.shape == (1, 64, 64)
    # Object area in pixels must match to within a pixel row.
    assert abs(float(a.sum()) - float(b.sum())) / max(float(a.sum()), 1) < 0.1


def test_resample_pads_when_the_input_is_smaller_than_the_crop():
    v = np.ones((2, 10, 10), dtype=np.float32)
    out = resample_to_spacing(v, 1.0, 1.0, 32)
    assert out.shape == (2, 32, 32)
    assert out[:, :5, :5].sum() == 0.0  # padded region stays zero
    assert out.sum() == pytest.approx(2 * 100, rel=0.2)


def test_stack_neighbours_clamps_at_the_edges():
    v = np.arange(5 * 4 * 4, dtype=np.float32).reshape(5, 4, 4)
    s = stack_neighbours(v, 5)
    assert s.shape == (5, 5, 4, 4)
    # First slice: neighbours -2, -1 clamp to slice 0.
    assert np.allclose(s[0, 0], v[0]) and np.allclose(s[0, 1], v[0])
    assert np.allclose(s[0, 2], v[0]) and np.allclose(s[0, 3], v[1])
    # Last slice clamps upward.
    assert np.allclose(s[4, 4], v[4]) and np.allclose(s[4, 3], v[4])
    # Middle slice is a genuine window.
    assert np.allclose(s[2, 2], v[2]) and np.allclose(s[2, 0], v[0])


def test_select_slices_is_uniform_in_millimetres_not_in_index():
    # A stack that is dense at one end and sparse at the other.
    z = np.concatenate([np.linspace(0, 10, 40), np.linspace(40, 80, 10)])
    idx = select_slices(z, max_slices=10)
    chosen = z[idx]
    gaps = np.diff(np.sort(chosen))
    # Uniform-in-index would put 8 of 10 picks inside the first 10 mm.
    assert (chosen > 20).sum() >= 3
    assert gaps.std() / max(gaps.mean(), 1e-6) < 1.5


def test_select_slices_is_identity_when_under_the_cap():
    z = np.linspace(0, 30, 12)
    assert np.array_equal(select_slices(z, 48), np.arange(12))


# --------------------------------------------------------------------------- #
# Sequence taxonomy                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "tr,te,expected",
    [
        (500, 12, Weighting.T1),
        (3000, 30, Weighting.PD),
        (3500, 45, Weighting.PD),  # intermediate-weighted band -> PD
        (4000, 90, Weighting.T2),
    ],
)
def test_weighting_from_acquisition_parameters(tr, te, expected):
    assert infer_weighting(FakeDS(RepetitionTime=tr, EchoTime=te)) == expected


def test_stir_detected_from_inversion_time_regardless_of_description():
    ds = FakeDS(RepetitionTime=4000, EchoTime=40, InversionTime=150,
                ScanningSequence="IR", SeriesDescription="cor sequence 4")
    assert infer_weighting(ds) == Weighting.STIR
    assert infer_fat_sat(ds) is True


def test_gre_detected_from_scanning_sequence():
    assert infer_weighting(FakeDS(ScanningSequence="GR", RepetitionTime=30,
                                  EchoTime=10)) == Weighting.GRE


@pytest.mark.parametrize(
    "desc",
    ["Sag PD FS", "SAG DP SAT GRASA", "sag_pdw_fs_tse", "Sagittal PD Fatsat",
     "cor t2 spair", "AX PD SPIR", "Sag PD yağ baskılı", "矢状位 PD 压脂"],
)
def test_fat_sat_detected_across_languages(desc):
    assert infer_fat_sat(FakeDS(SeriesDescription=desc)) is True


def test_fat_sat_not_falsely_detected():
    assert infer_fat_sat(FakeDS(SeriesDescription="Sag PD TSE")) is False
    assert infer_fat_sat(FakeDS(SeriesDescription="Coronal T1")) is False


def test_classify_series_uses_the_geometric_plane_not_the_description():
    ds = FakeDS(RepetitionTime=3000, EchoTime=30, SeriesDescription="axial pd fs")
    fam = classify_series(ds, plane=Plane.SAGITTAL)
    assert fam.plane == Plane.SAGITTAL
    assert fam.name == "sag_pd_fs"


def test_classify_series_degrades_gracefully():
    # Unknown weighting -> unknown family, not a wrong one.
    assert classify_series(FakeDS(), plane=Plane.CORONAL).index in (0,)
    # Oblique geometry gets its own bucket.
    assert classify_series(FakeDS(RepetitionTime=3000, EchoTime=30),
                           plane=Plane.OBLIQUE).name == "oblique_other"


def test_classify_series_falls_back_when_fat_sat_variant_is_missing():
    # Axial T1 without fat-sat exists; axial T1 *with* fat-sat does not in the
    # taxonomy, so it must fall back to the same plane+weighting family.
    ds = FakeDS(RepetitionTime=500, EchoTime=12, SeriesDescription="ax t1 fs")
    fam = classify_series(ds, plane=Plane.AXIAL)
    assert fam.plane == Plane.AXIAL and fam.weighting == Weighting.T1
