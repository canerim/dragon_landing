"""Preprocessing and sequence-taxonomy behaviour on synthetic inputs."""

from __future__ import annotations

import numpy as np
import pytest

from kairos.constants import NUM_TARGETS, Plane, Weighting
from kairos.data.dataset import (
    SeriesRecord,
    StudyRecord,
    _bilinear_resize,
    foreground_mask,
    resample_to_spacing,
    robust_normalise,
    select_slices,
    stack_neighbours,
)


def _record_with_series(i, *, size=32, n_slices=6, planted=True):
    """One sagittal study, shaped the way the loader produces them."""
    rng = np.random.default_rng(i)
    v = rng.normal(0, 0.3, (n_slices, size, size)).astype(np.float32)
    if planted:
        v[:, 4:8, 3:8] += 4.0
    rec = StudyRecord(study_uid=f"s{i}", series=[SeriesRecord(
        f"{i}.0", 1, Plane.SAGITTAL, v,
        np.arange(n_slices, dtype=np.float32) * 3.0, 3.0, 0.5,
    )])
    rec.labels = np.zeros(NUM_TARGETS, dtype=np.float32)
    return rec
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


# --------------------------------------------------------------------------- #
# Studies that could not be decoded                                            #
# --------------------------------------------------------------------------- #


def test_collate_marks_an_unusable_study_invalid_and_nans_its_targets():
    """A study with no usable series is all padding.

    Left alone it is worse than useless: fusion attention over an all-False
    mask is uniform over padded slots, the head emits finite logits from zeros,
    and those logits get scored against the study's *real* targets -- pure
    padding gradient in training, a fabricated row in the OOF matrix.
    """
    torch = pytest.importorskip("torch")
    from kairos.data.dataset import collate_studies

    good = _record_with_series(0)
    bad = StudyRecord(study_uid="broken", series=[], errors=["no usable series"])
    bad.labels = np.zeros(NUM_TARGETS, dtype=np.float32)

    batch = collate_studies([good, bad], device="cpu")
    assert batch.study_valid.tolist() == [True, False]
    assert torch.isfinite(batch.targets[0]).all()
    assert torch.isnan(batch.targets[1]).all(), "padding must not be supervised"

    # And the reverse order, which used to allocate the wrong spatial size.
    batch2 = collate_studies([bad, good], device="cpu")
    assert batch2.study_valid.tolist() == [False, True]
    assert batch2.pixels.shape[-1] == batch.pixels.shape[-1]


def test_collate_reads_the_spatial_size_from_any_record_not_just_the_first():
    """`records[0]` may be the failed study; sampling it made the allocation
    batch-order dependent and blew up at a random step deep into training."""
    pytest.importorskip("torch")
    from kairos.data.dataset import collate_studies

    good = _record_with_series(1, size=48)
    bad = StudyRecord(study_uid="broken", series=[])
    assert collate_studies([bad, good], device="cpu").pixels.shape[-1] == 48
    assert collate_studies([good, bad], device="cpu").pixels.shape[-1] == 48


def test_collate_rejects_a_batch_with_mixed_spatial_sizes():
    pytest.importorskip("torch")
    from kairos.data.dataset import collate_studies

    with pytest.raises(ValueError, match="different in-plane sizes"):
        collate_studies([_record_with_series(0, size=32),
                         _record_with_series(1, size=48)], device="cpu")


def test_modality_dropout_does_not_invent_a_series_for_an_empty_study():
    """``argmax`` on an all-False row returns 0, so an unconditional restore
    force-marks padded slot 0 as a genuine series."""
    torch = pytest.importorskip("torch")
    from kairos.models.backbones import BackboneSpec
    from kairos.models.system import KairosConfig, KairosModel

    model = KairosModel(KairosConfig(
        backbone=BackboneSpec(pretrained=False, in_chans=5), dim=32, agg_depth=1,
        agg_heads=4, sngp_features=32, n_experts=2, sequence_dropout=1.0,
    ))
    model.train()
    mask = torch.tensor([[True, True], [False, False]])
    out = model._apply_modality_dropout(mask)
    assert bool(out[0].any()), "a real study must keep at least one series"
    assert not bool(out[1].any()), "an empty study must stay empty"


def test_rare_label_quota_is_honoured_when_a_pick_covers_two_labels():
    """The `covered` set marked a label fully satisfied after one co-occurring
    pick -- correct only at quota 1, and it also made the have/need accounting
    dead code that always saw have == 0."""
    from kairos.data.loader import ClassAwareBatchSampler, SamplerConfig

    rng = np.random.default_rng(0)
    N = 300
    y = np.zeros((N, NUM_TARGETS), dtype=np.float32)
    y[:, 0] = (rng.random(N) < 0.30)            # common, not rare
    a, b = 1, 2                                  # two rare labels
    a_idx = rng.choice(N, 12, replace=False)
    y[a_idx, a] = 1.0
    y[a_idx[0], b] = 1.0                         # b's only positive also has a
    y[rng.choice(N, 2, replace=False), b] = 1.0

    cfg = SamplerConfig(batch_size=12, seed=1, min_positives_per_rare_label=2)
    sampler = ClassAwareBatchSampler(y, cfg)
    short = sum(1 for batch in sampler if y[batch, a].sum() < 2)
    assert short == 0, f"{short} batches missed the quota for the co-occurring label"


def test_dataloader_workers_get_independent_augmentation_streams():
    """One numpy Generator bound in the parent is copied into every worker, and
    torch's per-worker seeding cannot reach a Generator held on the dataset --
    so all workers replayed the identical augmentation stream."""
    pytest.importorskip("torch")
    from kairos.data.loader import SamplerConfig, StudyDataset, build_dataloader
    from kairos.data.transforms import AugmentConfig

    labels = (np.random.default_rng(0).random((8, NUM_TARGETS)) < 0.4).astype(np.float32)

    def load(uid):
        # Every study has *identical* pixels, so any difference in the output
        # can only come from the augmentation draw.  With different base
        # content the test passes even when the streams are in lockstep, which
        # is exactly how this bug stayed hidden.
        r = _record_with_series(0, planted=False)
        r.study_uid = uid
        r.labels = labels[int(uid[1:])]
        return r

    ds = StudyDataset([f"s{i}" for i in range(8)], load_fn=load, labels=labels,
                      augment=AugmentConfig(seed=7, noise_sigma=1.0, blur_prob=0.0))
    loader = build_dataloader(
        ds, sampler_cfg=SamplerConfig(batch_size=1, shuffle=False, drop_last=False),
        num_workers=2, balanced=False,
    )
    firsts = [float(b.pixels[0, 0, 0, 0, 0, 0]) for b in loader]
    # Worker 0 handles samples 0,2,4,... and worker 1 handles 1,3,5,...  With a
    # shared Generator the two streams are byte-identical in pairs.
    pairs = [(firsts[i], firsts[i + 1]) for i in range(0, len(firsts) - 1, 2)]
    assert not all(a == b for a, b in pairs), f"workers are in lockstep: {firsts}"


# --------------------------------------------------------------------------- #
# Study cache (scripts/03_build_cache.py)                                      #
# --------------------------------------------------------------------------- #


def _cache_module():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "scripts"))
    from importlib import import_module

    return import_module("03_build_cache")


def _synthetic_study(uid: str, n_series: int = 3):
    import numpy as np

    from kairos.constants import Plane
    from kairos.data.dataset import SeriesRecord, StudyRecord

    rng = np.random.default_rng(0)
    rec = StudyRecord(study_uid=uid)
    for s in range(n_series):
        n = 7 + s
        rec.series.append(SeriesRecord(
            series_uid=f"{uid}.{s}", family_index=s + 1, plane=Plane.SAGITTAL,
            pixels=rng.normal(0, 1, (n, 32, 32)).astype(np.float32),
            z_mm=np.cumsum(rng.uniform(2.7, 3.3, n)).astype(np.float32),
            spacing_mm=3.0, in_plane_mm=0.5, manufacturer_id=2, fat_sat_id=1,
            field_strength=1.5, te_ms=35.0, tr_ms=3000.0, mirrored=True))
    return rec


def test_cache_round_trip_preserves_every_field(tmp_path):
    """The cache replaces the DICOM reader, so anything it drops is data the
    model silently stops seeing -- and nothing downstream would report it."""
    import numpy as np

    mod = _cache_module()
    uid = "1.2.826.0.1.3680043.8.498.1000487322909905386909332429219581726"
    rec = _synthetic_study(uid)

    mod._write(mod.cache_path(tmp_path, uid), rec, uid)
    back = mod.load_cached_study(tmp_path, uid)

    assert back.study_uid == uid
    assert len(back.series) == len(rec.series)
    for a, b in zip(rec.series, back.series):
        # float32 out, whatever the storage dtype: the model and the losses run
        # in float32 and a float16 array silently changes accumulation dtype.
        assert b.pixels.dtype == np.float32
        assert np.allclose(a.pixels, b.pixels, atol=1e-2)
        assert np.allclose(a.z_mm, b.z_mm)
        assert a.series_uid == b.series_uid
        assert (a.family_index, a.plane, a.mirrored) == (b.family_index, b.plane, b.mirrored)
        assert (a.spacing_mm, a.in_plane_mm) == (b.spacing_mm, b.in_plane_mm)
        assert (a.manufacturer_id, a.fat_sat_id) == (b.manufacturer_id, b.fat_sat_id)
        assert (a.field_strength, a.te_ms, a.tr_ms) == (b.field_strength, b.te_ms, b.tr_ms)


def test_cache_write_leaves_no_partial_file(tmp_path):
    """np.savez appends '.npz' to any path lacking it, so a '.npz.tmp' target
    lands at '.npz.tmp.npz' and the write-then-rename renames a missing file."""
    mod = _cache_module()
    uid = "1.2.826.0.1.3680043.8.498.42"
    mod._write(mod.cache_path(tmp_path, uid), _synthetic_study(uid), uid)

    assert mod.cache_path(tmp_path, uid).exists()
    assert not list(tmp_path.rglob("*.tmp*")), "temporary file left behind"


def test_cache_shards_on_the_variable_end_of_the_uid(tmp_path):
    """Every UID here begins '1.2.826.0.1.3680043.8.498.', so a prefix shard
    puts all 4407 studies in one directory."""
    mod = _cache_module()
    uids = [f"1.2.826.0.1.3680043.8.498.{i:040d}" for i in range(40)]
    for u in uids:
        mod._write(mod.cache_path(tmp_path, u), _synthetic_study(u, 1), u)
    assert len({p.name for p in tmp_path.iterdir() if p.is_dir()}) > 1


def test_missing_study_reports_an_error_rather_than_raising(tmp_path):
    """A study absent from the cache must arrive as an invalid record, so the
    collator drops it from the OOF instead of scoring it from padding."""
    mod = _cache_module()
    rec = mod.load_cached_study(tmp_path, "not.a.real.uid")
    assert rec.series == []
    assert rec.errors


def test_study_with_no_usable_series_round_trips(tmp_path):
    """No usable series is a fact about the data. Re-decoding it every epoch to
    rediscover that is exactly the waste the cache removes."""
    from kairos.data.dataset import StudyRecord

    mod = _cache_module()
    uid = "1.2.826.0.1.3680043.8.498.empty"
    mod._write(mod.cache_path(tmp_path, uid), StudyRecord(study_uid=uid), uid)
    back = mod.load_cached_study(tmp_path, uid)
    assert back.series == []
    assert not back.errors, "an empty-but-present study is not a cache miss"
