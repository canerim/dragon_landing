# KAIROS

**K**nee **A**bnormality **I**nference with **R**eport-**O**ptimised **S**upervision
— a system for the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) challenge (Kaggle, 2026).

Twelve clinically defined abnormalities on knee MRI, from sixteen centres across
five continents, each study paired with its original radiology report in one of
~twelve languages. Metric: macro-average ROC-AUC over the twelve targets.

- **Method:** [`docs/DESIGN.md`](docs/DESIGN.md) — the full specification, with derivations.
- **Turkish summary:** [`docs/OZET_TR.md`](docs/OZET_TR.md)

---

## What this is

Six claims, each implemented and each falsifiable on out-of-fold data:

1. **This is a domain-generalisation problem.** Sixteen centres and an unknown
   private site mix mean ERM optimises the wrong functional. We optimise a
   distributionally-robust objective and treat the *site AUC gap* as a headline
   metric, not a diagnostic.
2. **Physical geometry, not array indices.** Slice order from
   `ImagePositionPatient` projected on the slice normal; positional encoding as
   a Fourier feature of *millimetres*; metric distance bias in attention; the
   SSM's discretisation step scaled by the real inter-slice gap; resampling to
   physical spacing rather than to a pixel grid.
3. **Twelve pathologies, twelve queries.** Each target pools its own slice and
   sequence evidence, refined by a top-2 MoE routed from the anatomical
   mechanism taxonomy and regularised on the Poincaré ball.
4. **The report is privileged information, never a test-time shortcut.**
   Soft-target contrastive pretraining, unbalanced optimal-transport
   phrase→slice grounding without boxes, four-state multilingual weak labels,
   cross-fitted distillation — and a shortcut regulariser plus a *blocking*
   audit that refuses a multimodal teacher which is secretly a text classifier.
5. **Optimise the metric, after the classifier exists.** AUC min-max margin
   (verified against the pairwise surrogate at the saddle point), two-way
   partial AUC via implicit-differentiated soft top-k, and a momentum memory
   queue for the labels where a minibatch contains no positive.
6. **Compute follows pathology.** Coarse pass at 0.70 mm → per-(study, label)
   confidence-and-uncertainty gate → Gumbel top-k slice selection → fine pass at
   0.35 mm. Thresholds read off a measured runtime–AUC Pareto frontier, with a
   closed-loop governor that guarantees the notebook finishes.

---

## Install

```bash
pip install -e ".[train,dev]"     # torch, timm, pydicom, pyarrow, pytest
pip install -e .                  # numpy/pandas only: folds, metrics, ensembling, submission
```

The numpy-only surface imports without torch, so fold construction, evaluation,
ensembling, calibration and submission validation run in a lightweight job. The
torch surface is lazily imported.

## Verify

```bash
pytest -q                          # 212 tests
python scripts/99_smoke_test.py    # 10-stage end-to-end run, ~20 s, CPU only
```

The smoke test drives the real code path — folds → collation → model
forward/backward under the curriculum → OOF evaluation → audit suite →
ensembling → calibration → conformal → submission — on synthetic data. No
stubs.

## Use

The competition layout, confirmed on Kaggle and encoded in `kairos/io/layout.py`:

```
/kaggle/input/competitions/rsna-knee-abnormality-detection/
    train.csv  test.csv                 study-level
    train_series.csv  test_series.csv   series-level
    sample_submission.csv
    {train,test}_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
```

Note the two traps: the root is under `input/competitions/`, and the images are
in `{split}_series/` with `{split}.csv` sitting right beside it. Discovery lives
in one module so those two facts cannot drift between the notebooks and the
scripts.

```bash
# 0. scan the DICOM tree (headers only) -> manifest + QC + family census
python scripts/00_build_manifest.py \
    --root /kaggle/input/competitions/rsna-knee-abnormality-detection/train_series \
    --labels /kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv \
    --out artifacts/manifest.parquet --workers 8

# 1. the immutable fold artefact.  Records a hash every checkpoint must carry.
python scripts/01_make_folds.py --manifest artifacts/manifest.parquet \
    --out artifacts/folds --n-folds 5 --anneal-steps 60000

# 2. reports -> four-state weak labels.  Run --audit FIRST and grow the lexicon.
python scripts/02_parse_reports.py --reports data/train_reports.csv --audit --top 40
python scripts/02_parse_reports.py --reports data/train_reports.csv \
    --labels data/train_labels.csv --evaluate --out artifacts/weak_labels.parquet

# 4. train one fold
python scripts/04_train.py --folds artifacts/folds/folds.parquet \
    --manifest artifacts/manifest.parquet --fold 0 --budget medium \
    --out runs/convnext_s_f0

# 5. evaluate + audit.  Exits non-zero on a blocking leakage failure.
python scripts/05_oof_eval.py --oof runs/convnext_s_f0/oof.npz \
    --folds artifacts/folds/folds.parquet --report artifacts/reports/convnext_s.txt

# 6. ensemble.  Ships the uniform average unless the nested gain beats 1 SE.
python scripts/06_ensemble.py --oof runs/*/oof.npz \
    --folds artifacts/folds/folds.parquet --out artifacts/ensemble
```

Before any of that, submit `notebooks/kaggle_baseline.py`: it needs no weights,
scores ~0.5, and returns the three numbers that shape everything else — the real
data layout, the DICOM tag census, and how much of the nine-hour budget I/O
alone consumes.

A one-command dry run of the whole training path, no data and no GPU:

```bash
python scripts/04_train.py --synthetic --budget small --epochs-cap 2 \
    --no-pretrained --device cpu --disable kd,ot_ground,weak_label
```

`notebooks/kaggle_inference.py` is the offline submission notebook: it writes a
valid fallback `submission.csv` *before* inference starts, discovers weights by
globbing attached datasets, governs its own runtime, shards predictions to disk,
and validates the file it wrote.

---

## Layout

```
src/kairos/
  constants.py            12 targets, mechanism tree, compartments, 15 sequence families
  io/
    geometry.py           slice ordering, canonical direction/laterality, QC, affines
    layout.py             competition file discovery (the two path traps, once)
  data/
    folds.py              patient-safe multi-objective CV (iterative strat + annealing)
    dataset.py            DICOM → metric-resampled, robust-normalised 2.5D tensors
    sequence_taxonomy.py  (plane, weighting, fat-sat) from physics, not description
    loader.py             Dataset + class-aware batch sampler with a rare quota
    transforms.py         anatomy-safe augmentation; one affine per series
  text/ontology.py        multilingual knee lexicon, post-posed negation, compartments
  models/
    encoding.py           physical Fourier / rotary-in-mm / acquisition FiLM
    aggregator.py         metric-distance slice transformer; irregular-Δ selective SSM
    label_queries.py      12 label queries, cross-sequence fusion, top-2 MoE
    heads.py              SNGP (RFF + Laplace) with a soft spectral-norm trunk
    adaptive.py           Gumbel top-k, confidence gate, PonderNet halting
    hyperbolic.py         Poincaré ontology, entailment cones
    backbones.py          timm/DINOv3/MedSigLIP adapters, stem inflation, LoRA
    system.py             the assembled coarse-to-fine model
  losses/
    auc.py                AUC min-max margin, two-way pAUC, memory-queue ranking
    ot.py                 log-domain balanced/unbalanced Sinkhorn, phrase→slice OT
    supervised.py         ASL, pattern-rarity BCE, Gaussian copula (composite NLL)
    multimodal.py         soft-target contrastive, decoupled KD, shortcut regulariser
    robust.py             Group-DRO (+SE shrinkage), CVaR, χ²-DRO, IRMv1
  optim/pesg.py           PESG min-max, ASAM, PCGrad/CAGrad/Aligned-MTL
  train/
    curriculum.py         five-stage schedule as a pure step -> weight function
    objectives.py         curriculum term -> loss; REFUSES a schedule it cannot feed
    ssl.py                masked feature modelling, cross-plane VICReg (stage S0)
    loop.py               PESG/ASAM/EMA trainer, fold-hash-guarded checkpoints
  eval/
    metrics.py            DeLong (co)variance + paired test, patient-cluster bootstrap
    leakage.py            ten-audit shortcut suite, five of them blocking
  ensemble/weights.py     anchored mirror descent + James–Stein shrinkage + nested LOFO
  calibrate/conformal.py  Newton temperature, beta, conformal risk control, Mondrian
  infer/                  runtime–AUC Pareto, closed-loop governor, submission writer
```

---

## Design decisions worth knowing about

**The fold artefact is immutable.** `fold_hash` is stored in every checkpoint
manifest and `Trainer.load` refuses a mismatch. Silently ensembling across split
versions produces an OOF score that is optimistic by an amount nobody can later
reconstruct.

**The audit suite blocks.** Five audits exit the pipeline non-zero:
`shuffled_label`, `shuffled_report`, `duplicate_hash`, `embedding_neighbour`,
and the fold-hash check. An audit that only warns is an audit that gets ignored.

**Per-label ensemble weights must earn their place.** `06_ensemble.py` reports
the nested leave-one-fold-out comparison and ships the *uniform* average unless
the nested gain exceeds one bootstrap standard error. Overriding requires
`--force-weighted`.

**Calibration cannot move the leaderboard.** Only monotone calibrators
(temperature, beta) are on by default; isotonic is provided but off, because it
ties scores and does change AUC.

**A submission that raises scores nothing.** The notebook writes a valid
fallback first and degrades to coarse-only under budget pressure rather than
overrunning.

**A scheduled objective that cannot be computed is a hard error.** If the
curriculum activates `kd` and the dataloader emits no teacher logits, training
stops with a message naming the missing field. Skipping it silently would train
a strictly smaller objective while the loss curve looked healthy — and the only
symptom would arrive three GPU-days later as an OOF score that contradicts the
ablation. Terms you omit on purpose must be named with `--disable`, and they are
recorded in the run manifest.

---

## Tests

212 tests, pinning numerics rather than shapes. The decisive one is
`test_model_can_overfit_a_tiny_dataset`: the assembled graph drives 12 studies to
macro-AUC 1.000, which is what proves the gradient path is intact from the loss
back through the SNGP head, the MoE router, the ontology prior, the
cross-sequence fusion, the label queries, the aggregator, the FiLM conditioning
and the backbone. A selection of the rest:

- AUC-M equals `p(1-p)·E[(m − h(x⁺) + h(x⁻))²]` at the saddle point
- DeLong's SE agrees with a 1500-replicate bootstrap to within 20 %
- Sinkhorn recovers a known permutation; unbalanced OT destroys mass when
  nothing matches; masks are respected exactly
- `Φ₂(0,0;ρ) = ¼ + arcsin(ρ)/2π` to 1e-6, and `Φ₂(h,k;0) = Φ(h)Φ(k)` to 1e-12
- Beta calibration and temperature scaling leave AUC unchanged to 1e-9
- Conformal risk control achieves nominal coverage on held-out data
- Soft top-k sums to *k* and is differentiable through the implicit `ν`
- Spectral normalisation bounds the empirical Lipschitz ratio at ≤ 1.05
- SNGP variance is larger far from the training distribution
- Aligned-MTL is invariant to a 100× rescale of one task's loss
- Folds never split a patient; the rarest label's per-fold prevalence stays
  within 40 % relative
- Resampling gives the same physical field of view from 0.25 mm and 1.0 mm input
- Robust normalisation is invariant to affine intensity rescaling and survives
  a metal artefact
- Every audit fires on a deliberately planted defect and stays quiet otherwise
- Real DICOM written with pydicom and read back: geometric ordering beats a
  misleading `InstanceNumber`, plane comes from geometry not description,
  laterality mirrors, QC flags fire on a planted gap
- The schedule validator refuses a curriculum it cannot feed
- Turkish post-posed negation, Japanese post-posed negation, and compartment
  resolution all produce the right assertion

Ten real bugs were found by these tests during development, each documented in
the code at the site of its fix:

| bug | why it mattered |
|---|---|
| canonical-flip negated the normal but reused the old projection | physical coordinate came back decreasing |
| AUC-M used class-conditional means | broke the min-max identity by ~6× |
| `np.nan_to_num` maps `+inf` to 1.8e308 | Otsu's argmax always chose the last bin; mask silently fell through |
| `\b` does not split on `_` | `sag_pdw_fs_tse` classified as non-fat-suppressed |
| `build_submission` validated its own constant fallback | the safety net that guarantees a file *raised* |
| SNGP refreshed its covariance in place between two head calls | backward failed with a version-counter error |
| `ConfidenceGate` marked hard-boolean thresholds learnable | parameters that can never receive gradient |
| `KneeOntology` assigned an undeclared attribute under `slots=True` | constructing it raised; nothing had instantiated it |
| `normalise_text` claimed to strip the casefold combining dot | it did not |
| LID checked CJK before kana | every Japanese report labelled Chinese |
| `vars()` on a `slots=True` dataclass | fold writer raised *after* computing the split |
| manifest/label merge collided on `PatientID` | became `PatientID_x`; the fold builder lost its group column |
| optimiser never stepped when `len(loader) < accum_steps` | short folds and debug runs trained nothing |
| the no-weights exit path validated its own constant fallback | second instance of the same safety-net bug |
| `shuffled_label` used a fixed tolerance | false-alarmed on any small eval; a blocking audit that cries wolf gets ignored |

---

## Licence and attribution

Model weights carry their upstream licences; every candidate checkpoint's
licence, redistribution rights and compatibility with the winners'
weight-publication obligation are verified before it enters an ensemble.

A competition AUC does not establish a safe clinical operating point. See
[`docs/DESIGN.md`](docs/DESIGN.md) §10 for what deployment would actually
require.
