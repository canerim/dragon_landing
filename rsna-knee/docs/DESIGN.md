# KAIROS — a geometry-aware, report-supervised, adaptive-compute system for multi-label knee MRI

**Knee Abnormality Inference with Report-Optimised Supervision**
RSNA Knee Abnormality Detection (Kaggle, 2026) · methods specification

---

## 0. Abstract

We describe a system for detecting twelve clinically defined abnormalities on
knee MRI examinations acquired at sixteen centres across five continents, each
paired with its original radiology report in one of approximately twelve
languages. The evaluation metric is the macro-average ROC-AUC over the twelve
targets.

The system rests on six claims, each of which is implemented, testable, and
falsifiable on out-of-fold data:

1. **The task is a domain-generalisation problem wearing a classification
   costume.** Sixteen centres, unconstrained protocols and a private test set
   whose site mix is unknown mean that empirical risk minimisation optimises
   the wrong functional. We optimise a distributionally-robust objective over
   site × language × scanner groups, and we measure the site AUC gap as a
   first-class metric alongside macro-AUC.

2. **Physical geometry, not array indices.** Slice order is recovered by
   projecting `ImagePositionPatient` onto the slice normal; positional
   encoding is a Fourier feature of *millimetres*; attention carries a metric
   distance bias; the state-space aggregator's discretisation step is scaled by
   the actual inter-slice gap. A model that indexes anatomy by array position
   learns a site-specific protocol, not anatomy.

3. **Twelve pathologies deserve twelve queries.** Each target owns a learned
   query that pools its own slice evidence and its own cross-sequence
   evidence, refined by a top-2 mixture of experts whose routing is initialised
   from the anatomical mechanism taxonomy and regularised on the Poincaré ball.

4. **The report is privileged information, never a test-time shortcut.** It
   enters through soft-target contrastive pretraining, optimal-transport
   phrase→slice grounding, four-state weak labels and cross-fitted
   distillation — all of which shape the *representation*. No branch of the
   deployed model is conditioned on text. That is a deliberate scoping
   decision, not an omission: the test set carries no reports, so a
   report-conditioned branch can only pay off through the representation
   (already covered) or as a KD teacher — and a teacher that reads the finding
   out of the report emits logits the image-only student cannot reproduce, so
   distilling them degenerates to label smoothing. `ReportShortcutRegulariser`
   and the blocking `shuffled_report` audit remain available for a variant that
   does build such a branch; they are not scheduled, and the objective
   validator now checks *output* preconditions so that a term the model cannot
   feed is a startup error rather than a silent no-op.

5. **The metric should be optimised directly, but only after the classifier
   exists.** We use the min-max margin formulation of the AUC surrogate, whose
   equivalence to the pairwise squared hinge we verify numerically, driven by a
   proximal min-max optimiser; plus a two-way partial AUC and a momentum
   memory queue for the labels where a minibatch contains no positive.

6. **Compute should follow pathology.** A coarse pass at 0.70 mm scores every
   slice; a per-(study, label) confidence-and-uncertainty gate decides whether
   to escalate; a Gumbel top-k selector chooses which slices; a fine pass at
   0.35 mm runs only there. Thresholds are read off a measured runtime–AUC
   Pareto frontier, and a closed-loop governor guarantees the notebook finishes.

---

## 1. Problem structure

### 1.1 What the metric implies

The score is

$$
\mathcal{M} \;=\; \frac{1}{12}\sum_{l=1}^{12} \mathrm{AUC}_l .
$$

Three consequences follow immediately and shape every design decision below.

**Rare labels are worth as much as common ones.** If `Fracture` has 3 %
prevalence and `Effusion` has 35 %, a 0.01 gain on `Fracture` and a 0.01 gain on
`Effusion` are worth exactly the same. But the *variance* of the `Fracture` AUC
estimate is roughly $\sqrt{35/3} \approx 3.4\times$ larger. So the labels that
matter most per unit of measurable improvement are precisely the ones on which
we can least tell whether we improved. Everything downstream — fold
construction, the memory-queue ranking loss, the shrinkage in ensemble
weighting, the DeLong tests — exists to buy statistical power on the rare tail.

**Calibration is free and thresholds are irrelevant.** AUC is invariant under
any strictly increasing transform. We still calibrate, because distillation,
report gating, adaptive compute and the clinical deliverable all need
probabilities — but we use *monotone* calibrators (temperature, beta) so that
calibration provably cannot move the leaderboard, and we keep isotonic
regression off by default because it can.

**Model selection must be multi-objective.** We select on
$(\mathcal{M},\ \min_l \mathrm{AUC}_l,\ \mathrm{sd}_k \mathcal{M}_k,\
\Delta_{\text{site}},\ t_{\text{infer}})$ — macro-AUC, worst-label AUC,
fold standard deviation, the best-minus-worst site AUC gap, and measured
runtime. A model that wins on macro-AUC while losing 0.05 on the worst label
is a model that will move on the private split, in an unknown direction.

### 1.2 The label geometry

The twelve targets carry two independent structures that a flat 12-way head
throws away.

*Mechanism* (a tree):

```
knee
├── ligament ────── ACL, MCL
├── meniscus ────── Medial Meniscus, Lateral Meniscus
├── degenerative ── Medial OA, Lateral OA, PF OA
├── inflammatory ── Effusion, Synovitis, Baker's
└── osseous ─────── Contusion, Fracture
```

*Compartment* (a partition): medial / lateral / patellofemoral / central
(intercondylar notch) / posterior (popliteal fossa) / global.

The mechanism tree is embedded on the Poincaré ball
(`models/hyperbolic.py`) because hyperbolic space embeds trees with distortion
that does not grow with depth, whereas any Euclidean embedding's does. The
resulting geodesic kernel does three jobs: it seeds the MoE routing bias, it
supplies the graph term in the contrastive soft target, and it defines the
shrinkage neighbourhoods for per-label ensemble weights, so `Fracture` borrows
`Contusion`'s weights instead of borrowing the uniform prior.

The compartment partition biases the label-query attention logits and drives
the ontology's compartment resolver, which is what stops a report sentence
about the *lateral* meniscus from producing a `Medial Meniscus` weak label.

### 1.3 What the multi-site design actually costs

Denote by $\mathcal{D}_k$ the distribution at centre $k$ and by $\pi$ the
mixture weights. Training minimises $\sum_k \hat\pi_k \mathcal{R}_k$; the
private leaderboard evaluates $\sum_k \pi^{\text{test}}_k \mathcal{R}_k$ with
$\pi^{\text{test}}$ unknown. The gap is bounded by

$$
\Big|\textstyle\sum_k (\pi^{\text{test}}_k - \hat\pi_k)\mathcal{R}_k\Big|
\;\le\; \|\pi^{\text{test}} - \hat\pi\|_1 \cdot \max_k \mathcal{R}_k ,
$$

so the only quantity we control is $\max_k \mathcal{R}_k$ — the worst-group
risk. That is the formal argument for Group-DRO here, and it is why we report
$\Delta_{\text{site}}$ as a headline metric rather than as a diagnostic.

---

## 2. Data engineering

### 2.1 Slice ordering is a geometry problem

`InstanceNumber` is a transmission index. Interleaved acquisitions, re-sent
series and multi-echo sequences all break it, and the breakage is silent. The
only reliable ordering uses the slice normal

$$
\mathbf{n} \;=\; \hat{\mathbf r} \times \hat{\mathbf c},
\qquad
z_i \;=\; \langle \mathbf{p}_i, \mathbf{n}\rangle ,
$$

with $\hat{\mathbf r}, \hat{\mathbf c}$ the Gram–Schmidt-orthogonalised triples
of `ImageOrientationPatient` and $\mathbf{p}_i$ = `ImagePositionPatient`.
Re-orthogonalisation is not pedantry: several vendors emit orientations with a
fraction of a degree of skew, harmless for display and cumulative when the
affine is inverted.

We take the *median* normal over the series after sign-aligning to the first
(some vendors flip the in-plane vectors mid-series, and a naive median of
$+\mathbf n$ and $-\mathbf n$ is meaningless), then canonicalise direction so
slice 0 means the same anatomy at every centre, and finally canonicalise
laterality — a left knee is a mirror image of a right one, and training on both
without canonicalisation spends capacity learning a reflection symmetry that
can be handed over for free. **The mirror is recorded**, because `Medial OA` and
`Lateral OA` swap meaning under an unrecorded flip; a silent version of that bug
costs more AUC on those two labels than any architecture change wins back.

QC flags, all emitted into the manifest and all actionable: inconsistent
orientation, duplicate physical positions, irregular spacing (CV > 0.15), large
slice gap (> 2.5 × median), variable in-plane spacing, variable matrix size,
degenerate z-extent.

### 2.2 Resample to physical spacing, not to a pixel grid

Every study is resampled to a *metric* grid: 0.70 mm in-plane for the coarse
pass, 0.35 mm for the fine pass. A 3 mm meniscal root tear then occupies the
same number of pixels at all sixteen centres. Resampling to a fixed 256 × 256
grid instead makes the same lesion 6 px wide at one site and 14 px at another,
and the network must learn the scale nuisance from data it does not have.

### 2.3 Sequence taxonomy

`SeriesDescription` is free text authored independently at sixteen sites in
twelve languages: a hint, never a key. Series are mapped into a fifteen-element
closed vocabulary of (plane, weighting, fat-sat) triples using DICOM
acquisition parameters plus the description, with an explicit `unknown` bucket
rather than force-fitting. The plane comes from the *normal*, with a generous
30° tolerance — knee protocols routinely tilt the sagittal stack along the ACL,
and an oblique-sagittal series is far more useful labelled `SAGITTAL` than
discarded into `OBLIQUE`.

The (label × family) prior matrix biases cross-sequence attention logits. It is
a **learned bias initialised from the prior**, never a hard gate: atypical
protocols exist, and hard gating silently destroys recall at exactly the sites
that use them.

### 2.4 Studies that do not survive decoding

`00_build_manifest.py` decides usability from DICOM *headers*
(`stop_before_pixels=True`, which is what makes scanning a 16-site archive
affordable), while `load_study` additionally rejects on a pixel-decode failure.
A missing codec plugin or a truncated `PixelData` therefore passes QC and only
fails at train time, producing a `StudyRecord` with no series at all.

Such a record must not be treated as data. Its `series_mask` row is all-False,
so cross-sequence fusion masks every logit to $-\infty$, softmax returns a
uniform distribution over *padded* slots, and the head emits a finite logit
computed entirely from zeros. Scored against the study's real targets that is
pure padding gradient in training and a fabricated row in the OOF matrix.
`collate_studies` therefore emits a `study_valid` mask, NaNs the targets of
invalid studies — so every masked loss skips them by the same mechanism that
already handles unobserved labels — and `Trainer.predict` drops them from the
OOF with a logged count, because a systematic decode failure must look like a
systematic decode failure and not like a bad epoch.

### 2.5 Cross-validation as constrained optimisation

Plain stratified $k$-fold is inadequate on three counts simultaneously: it
cannot respect patient grouping, it handles multi-label marginals poorly, and it
ignores site/language balance entirely. We instead *optimise* the split.

Assign patient **groups** (indivisible) to folds to minimise

$$
J(\pi)=
\lambda_{\text{mar}}\!\sum_{k,l}\!\frac{w_l\,(p_{kl}-p_l)^2}{p_l(1-p_l)+\varepsilon}
+\lambda_{\text{co}}\!\sum_k\!\big\lVert C_k-\bar C\big\rVert_F^2
+\lambda_{\text{cov}}\!\sum_{k,c}\!\mathrm{KL}(q_{kc}\|q_c)
+\lambda_{\text{sz}}\!\sum_k (n_k-\bar n)^2 .
$$

The Bernoulli variance in the first denominator turns it into a chi-square-like
statistic, so terms are comparable across labels whose prevalence differs by an
order of magnitude; $w_l=(p_{\max}/p_l)^{1/2}$ additionally up-weights the rare
tail.

Optimisation is two-phase: group-level iterative stratification
(Sechidis et al., 2011) as a seed — $O(GL)$, gets the marginals close — then
simulated annealing over move and swap proposals with Metropolis acceptance.
The result is written **once** and treated as immutable; `fold_hash` is stored
in every checkpoint manifest, and `Trainer.load` refuses a checkpoint whose fold
hash disagrees. An OOF matrix that cannot be traced to a split is not safe to
ensemble, and this is enforced in code rather than in a README.

Measured effect on synthetic cohorts matched to the expected prevalence
profile: worst-case relative prevalence deviation on the rarest label drops from
> 100 % (random grouped split) to < 40 %, which is the difference between being
able and unable to resolve a 0.004 AUC effect.

---

## 3. Architecture

### 3.1 Why 2.5D and not 3D

CVPR 2026's *Revisiting 2D Foundation Models for Scalable 3D Medical Image
Classification* benchmarks twelve volumetric classification tasks and concludes
that properly-adapted 2D foundation models **beat** native 3D architectures,
that general-purpose SSL backbones match medical-specific ones once adaptation
is right, and that the adaptation mechanism dominates the backbone choice. That
is consistent with every recent RSNA-Kaggle result: 2022 cervical spine, 2023
abdominal trauma and 2024 lumbar spine were all won by 2.5D encoder + sequence
head designs.

We therefore use a strong pretrained 2D encoder per slice, a sequence model
across slices, and keep a genuine volumetric branch **only** for ensemble
diversity — measured by OOF error correlation, not assumed.

### 3.2 Forward pass

```
study
  └─ for each series s (variable count, missing families allowed)
       ├─ 2.5D stacks (5 adjacent slices) at 0.70 mm
       ├─ pretrained 2D backbone           → h_{s,i}
       ├─ + PhysicalPositionalEncoding(z)  → metric Fourier features
       ├─ AcquisitionFiLM(context_s)       → protocol-conditioned
       ├─ SliceTransformer ⊕ SelectiveScan → contextual tokens
       └─ LabelQueryPool                   → v_{l,s}, attention a_{l,s,·}
  ├─ CrossSequenceFusion (missing-masked)  → z_l
  ├─ LabelExpertRouter (top-2 MoE)         → z_l refined
  ├─ + Poincaré ontology query prior
  ├─ SNGPHead                              → coarse logits + epistemic variance
  ├─ ConfidenceGate(logits, variance)      → which labels escalate
  ├─ GumbelTopKSelector(a_{l,s,·})         → which slices, dilated to windows
  ├─ fine backbone on selected windows at 0.35 mm
  └─ gated fusion(coarse, fine)            → final logits
```

### 3.3 Physical positional encoding

$$
\phi(z)=\big[\cos(2\pi\omega_k z),\ \sin(2\pi\omega_k z)\big]_{k=1}^{K/2},
\qquad
\omega_k=\omega_{\min}\Big(\tfrac{\omega_{\max}}{\omega_{\min}}\Big)^{\frac{k-1}{K/2-1}},
$$

with periods log-spaced from 240 mm (the whole knee) to 2 mm (a meniscal root).
An MLP on a raw scalar coordinate has a strong low-frequency spectral bias and
cannot represent "slice 14 differs from slice 15" — which is exactly the
resolution at which knee pathology lives. Because $z$ is in millimetres, the
same anatomy receives the same code at a site scanning 3 mm slices and one
scanning 0.8 mm slices.

The same substitution is applied to rotary attention: RoPE driven by the metric
coordinate rather than the token index, so attention is translation-equivariant
in *physical* space and a 6 mm gap in the middle of a stack is represented
honestly instead of being papered over.

### 3.4 Metric distance bias (ALiBi in millimetres)

$$
A_{ij} \mathrel{+}= b_{\text{bucket}(|z_i - z_j|)} ,
$$

a learned per-head bias over logarithmically bucketed metric distance,
initialised monotone decreasing. At step 0 this is a soft locality prior; the
model is free to unlearn it, which it does for `Effusion` (a global finding) and
does not for `Medial Meniscus` (confined to 2–3 contiguous slices).

### 3.5 Selective state-space aggregation with physical Δ

$$
h_i=\bar A_i \odot h_{i-1}+\bar B_i x_i,\qquad
\bar A_i=\exp(\Delta_i A),\qquad
\Delta_i=\mathrm{softplus}(w^\top h_i+b)\cdot\frac{\delta z_i}{\overline{\delta z}} .
$$

This is the one place where an SSM is *strictly more natural* than attention:
the continuous-time formulation already models a signal sampled at irregular
intervals, so a variable slice gap is handled exactly rather than approximated.
Bidirectional, because pathology at slice $i$ is confirmed by context on both
sides. Linear in the number of slices, which also makes it the aggregator of
choice for the efficiency-track student.

### 3.6 Label queries and cross-sequence fusion

$$
a_{l,s,i}=\operatorname*{softmax}_i\!\big(q_l^\top W_s h_{s,i}+\pi_{l,s}\big),
\qquad v_{l,s}=\sum_i a_{l,s,i}h_{s,i},
$$

$$
\alpha_{l,s}=\operatorname*{softmax}_s\!\big(q_l^\top U v_{l,s}+\rho_{l,\text{fam}(s)}+m_s\big),
\qquad z_l=\sum_s \alpha_{l,s}v_{l,s},
$$

with $m_s=-\infty$ for absent sequences. An **attention entropy floor**
$\mathrm{ReLU}(\log 3 - H(a_{l,s,\cdot}))$ prevents the classic degenerate
solution where a query latches onto a fixed slice index, achieves a good
training loss, and generalises terribly.

Sequence availability is *not* missing-at-random — sites that skip the axial
series also tend not to dictate "synovitis" — so presence is used only as an
attention mask, never as a feature, and `sequence_dropout` (15 %) forces a
sensible answer from any subset. That is also what makes inference-time
sequence skipping safe.

### 3.7 Mixture of experts over label features

Top-2 routing with Switch-style load balancing plus a router z-loss. The label
groups want genuinely different features, but a hard partition throws away the
real cross-talk (ACL tears co-occur with lateral contusions from the same pivot
shift). The router bias is initialised from the mechanism taxonomy, which halves
the warm-up and removes most of the run-to-run variance a randomly-initialised
router introduces.

### 3.8 Distance-aware uncertainty (SNGP)

The final layer is a random-Fourier-feature GP,

$$
\Phi(h)=\sqrt{\tfrac{2}{M}}\cos(Wh+b),\qquad
\Sigma^{-1}=I+\sum_i p_i(1-p_i)\Phi_i\Phi_i^\top ,
$$

with the mean-field correction $f/\sqrt{1+\tfrac{\pi}{8}v}$. Distance-awareness
only holds if the feature map is bi-Lipschitz, which is enforced by a **soft**
spectral norm (rescale only when $\sigma_{\max} > c$, leaving the layer
untouched otherwise, so the effective learning rate is not throttled). Omitting
the spectral constraint degrades SNGP to an ordinary head with a decorative
variance — the most common way of getting SNGP wrong, and one that produces
uncertainties that look plausible and mean nothing.

The variance has two jobs: it drives the adaptive-compute gate, and it is the
OOD term in the ensemble's uncertainty triple
$U_l=\alpha H(\bar p_l)+\beta\operatorname{Var}_m(p_{l,m})+\gamma d_{\text{OOD}}$.

---

## 4. Objectives

### 4.1 Optimising the metric directly

The squared-hinge AUC surrogate $\mathbb E[(m-h(x^+)+h(x^-))^2]$ has an
$O(n^+n^-)$ pairwise form. Its min-max reformulation collapses to per-example
terms:

$$
\min_{\theta,a,b}\max_{\alpha\ge 0}\;
\underbrace{(1-p)\,\mathbb E\big[(h-a)^2\mathbb 1_{y=1}\big]}_{A_1}
+\underbrace{p\,\mathbb E\big[(h-b)^2\mathbb 1_{y=0}\big]}_{A_2}
+\underbrace{2\alpha\big(p(1-p)m+\mathbb E[p\,h\,\mathbb 1_{y=0}-(1-p)h\,\mathbb 1_{y=1}]\big)-p(1-p)\alpha^2}_{A_3}.
$$

At the saddle point this equals $p(1-p)\,\mathbb E[(m-h(x^+)+h(x^-))^2]$
exactly. **The expectations are unconditional (whole-batch with class
indicators), not class-conditional** — this is not stylistic. Using conditional
means weights the two variance terms by $(1-p)$ and $p$ instead of both by
$p(1-p)$, and the identity fails by a factor of ~6. Our test suite verifies the
identity end-to-end through `AUCMarginLoss.forward`, and it is how that bug was
caught during development.

$A_1$ and $A_2$ are *variance* terms: they pull positive scores together and
negative scores together without pushing either towards 0 or 1. That is why
AUC-M keeps improving ranking long after BCE has saturated.

The inner maximisation is concave with closed-form optimum
$\alpha^\star = m + \mathbb E[h\mid y{=}0] - \mathbb E[h\mid y{=}1]$; we keep
$\alpha$ as a parameter under gradient *ascent* rather than substituting the
closed form, because the closed form is optimal only for population moments and
is badly noisy on a minibatch.

**Who optimises what.** `PESG` owns the saddle-point block $(a,b,\alpha)$ and
nothing else: descent on $(a,b)$ with a proximal anchor
$\tfrac{\gamma}{2}\|\cdot-\cdot_{\text{ref}}\|^2$ reset each epoch, ascent on
$\alpha$, and projection onto $a,b\in[0,1]$, $\alpha\ge 0$. Without the anchor
the ascent and descent chase each other with a period of a few hundred steps.
$\theta$ stays with AdamW for the whole run, because $\theta$ appears only in
the minimisation — there is nothing about it that needs a min-max optimiser.

Splitting by *role* rather than by *stage* is deliberate, and it repairs three
failures of the obvious alternative (swap the whole optimiser for PESG during
the ranking stage), each of which is silent. First, PESG then ran at a fixed
step size, so the cosine decay and the stage's `lr_scale=0.3` were ignored for
the entire stage. Second, it was rebuilt at the next stage boundary, discarding
both its momentum and its proximal reference. Third, and worst, `auc_margin` is
scheduled in S4 as well as S3, but the stage test keyed on the string
`"ranking"` — so from the S3/S4 boundary onward $\alpha$ accumulated a gradient
that nothing applied. A frozen $\alpha$ turns $A_3$ into a fixed linear penalty
and the objective quietly stops being the min-max surrogate. PESG now steps
exactly on the steps where the term has non-zero weight, and rides the same
learning-rate schedule as everything else.

**Two-way partial AUC.** Nobody triages a knee MRI worklist at 70 % FPR. We add
a low-weight term restricted to $\mathrm{FPR}\le\beta$, $\mathrm{TPR}\ge\alpha$,
i.e. the hardest negatives against the hardest positives. Hard truncation is
piecewise constant, so selection uses entropy-regularised soft top-k: the
solution of

$$
\max_{w\in[0,1]^n,\ \mathbf 1^\top w=k}\ \langle w,s\rangle
-\tau\sum_i\big[w_i\log w_i+(1-w_i)\log(1-w_i)\big]
$$

is $w_i=\sigma((s_i-\nu)/\tau)$ with $\nu$ found by bisection. We differentiate
through the **closed form** using the implicit function theorem,
$\partial\nu/\partial s_j = w_j(1-w_j)/\sum_i w_i(1-w_i)$, rather than
unrolling the bisection — exact, $O(1)$ memory, and independent of the
iteration count.

**Memory-queue pairwise ranking.** At 0.8 % prevalence a 32-study batch contains
a positive roughly one time in four; on the other three batches the AUC
surrogate produces *no gradient at all* for that label. Cranking the sampler
until every batch has a positive distorts the joint label distribution
(oversampling `Fracture` silently oversamples `Contusion`). Instead we keep a
detached MoCo-style queue of the last 512 scores per label per class and form
pairs against it, so the effective pair count per step goes from $O(1)$ to
$O(Q)$. Enabled only after epoch ~5, when staleness is small and the surrogate
matters.

### 4.2 Supervised terms

**Asymmetric loss** with $\gamma^-=4$, $\gamma^+=0$, probability shift $m=0.05$.
For a 1 %-prevalence label this removes ~95 % of negatives from the gradient
after a few epochs — the difference between a usable and a dead rare-label head.
NaN-masked throughout, so a missing label costs nothing.

**Pattern-rarity reweighted BCE.** Per-study weight
$w_i=\hat P(\mathbf y_i)^{-\kappa}$ under the independence factorisation, clipped
to $[1/4, 4]$. Because the true joint is more concentrated than the product, this
over-weights genuinely unusual label combinations, which is where twelve
independent sigmoids are most wrong.

**Gaussian copula composite likelihood.** Latent $\mathbf z\sim\mathcal N(0,\Sigma)$
with $y_l=\mathbb 1[z_l>\Phi^{-1}(1-\pi_l)]$; $\Sigma$ low-rank-plus-diagonal
normalised to unit diagonal, so positive-definiteness is structural. The exact
$L=12$ orthant probability has no closed form, so we sum the 66 exact bivariate
NLLs (Varin & Vidoni, 2005) — consistent for $\Sigma$, cheap, and critically it
**does not change the marginals**, so it can be added to an AUC-optimised system
without disturbing the per-label ranking. Bivariate normal CDFs come from
Drezner's integral evaluated by 24-node Gauss–Legendre quadrature (Golub–Welsch),
accurate to ~1e-10 and differentiable in $(h,k,\rho)$.

### 4.3 Report supervision

**Soft-target contrastive.** In a batch of 32 knee studies several describe
genuinely similar pathology; vanilla InfoNCE labels them mutual negatives and
teaches the encoder to separate clinically identical cases. The target is

$$
s^\ast_{ij}=\alpha\,\mathbb 1[i=j]+\beta\,J(\mathbf y_i,\mathbf y_j)
+\gamma\,\cos(\mathbf g_i,\mathbf g_j),
$$

Jaccard over weak label sets (NaN-safe; empty union → 0, not 1) plus a clinical
concept-graph term. The loss is cross-entropy against the row-normalised
$s^\ast$ — knowledge distillation with clinical similarity as teacher.

**Optimal-transport phrase→slice grounding.** With no bounding boxes, the
natural object is a transport plan between report phrases and slices, cost
$C_{ij}=1-\langle u_i,v_j\rangle$. Two departures from textbook Sinkhorn are
essential:

*Unbalanced.* Not every phrase has a visual correlate ("clinical history: pain")
and not every slice has a described finding. Forcing $T\mathbf 1=\mu$ therefore
*creates* spurious alignments. KL marginal relaxation of strength $\tau$
(Chizat et al., 2018) damps the Sinkhorn update to

$$
f\leftarrow\Big(\tfrac{\mu}{Kg}\Big)^{\frac{\tau}{\tau+\varepsilon}},\qquad
g\leftarrow\Big(\tfrac{\nu}{K^\top f}\Big)^{\frac{\tau}{\tau+\varepsilon}},
$$

letting mass be destroyed where nothing matches. At $\tau=0.5,\varepsilon=0.05$
about a third of phrases go untransported, matching the fraction of a knee
report that is history, technique or comparison.

*Log-domain.* $e^{-C/\varepsilon}$ underflows fp16 immediately at
$\varepsilon=0.05$; all iterations run on potentials.

Gradients use the **envelope theorem**: at the optimum $\partial\mathrm{OT}/
\partial C = T^\star$, so the plan is detached and we back-propagate only
through $\langle T^\star_{\text{detach}}, C\rangle$ — cheaper and far more
stable than unrolling, exact up to the entropic bias.

An anatomical-prior cost bump discourages grounding a PF-OA phrase onto a
coronal T1 slice (a *bump*, not a mask), and a bounded Jensen–Shannon term ties
the plan's per-label marginal to the model's own slice attention. The whole term
is gated on extraction confidence: ungated, it will happily ground a *negated*
finding onto a slice, which is worse than no supervision.

**No report-conditioned prediction branch — and why.** The obvious next step
is a model that reads the report at training time and is distilled into an
image-only student. We do not build one, and the reason is worth stating
because the omission would otherwise look like an oversight.

A report-conditioned branch has exactly two paths to score. The first is the
*representation*: aligning the image encoder to report semantics. That path is
already taken, by the soft-target contrastive term and the OT grounding above —
both of which shape the encoder without ever putting text in the inference
graph. The second is as a *KD teacher*. That path does not work here: the
report states the finding, so a report-conditioned teacher's logits are close
to a noisy copy of the labels, and distilling them into an image-only student
reduces to label smoothing on the same targets the supervised term already
uses. There is no dark knowledge for the student to recover, because the
teacher's advantage is information the student can never observe. The teacher
we actually distil from in S4 is the cross-fitted OOF ensemble of image-only
models, whose logits *are* reproducible from pixels.

The guard rail for anyone who does build such a branch is implemented and
tested: `ReportShortcutRegulariser` penalises

$$
\mathbb E\big[\mathrm{ReLU}(\mathcal C(z^{\text{shuf}})-\mathcal C(z^{\text{true}})+\delta)\big]
+\eta\,\mathbb E\big[\mathrm{KL}(\sigma(z^{\text{shuf}})\|\sigma(z^{\text{img}}))\big],
$$

$\mathcal C$ = negative predictive entropy: a shuffled report must not make the
model *more* certain, and under a shuffled report the prediction must fall back
to the image-only one. `shuffled_report_audit` is the blocking evaluation-time
counterpart. Neither is in the shipped curriculum. The objective registry
declares the model outputs each term reads, and `validate_schedule` refuses to
start a run that schedules a term the model cannot feed — which is how this
particular gap was found: the term had been scheduled at weight 0.5 for the
whole of S1 and had been returning `None` every step.

### 4.4 Multilingual report parsing

RadGraph, CheXbert and NegBio are English *and* chest-specific: their vocabulary
has "consolidation" and not "meniscal extrusion", and their negation cues are
English. We port the *method*, not the models.

A curated lexicon covers each of the twelve labels in English, Turkish, Spanish,
Portuguese, French, German, Italian, Dutch, Polish, Russian, Chinese and
Japanese, with accent-folded matching. Three details that a naive port gets
wrong:

* **Turkish casing.** The dotless `ı`/dotted `i` pair means Python's default
  `.lower()` turns `MENİSKÜS` into `meni̇sküs` with a combining dot, which then
  fails to match `menisküs`. We NFKC-normalise *after* case folding and strip
  combining marks from the match key.
* **Post-posed negation.** NegEx implements a left-context scope rule. Turkish
  (`… izlenmemektedir`) and Japanese (`… 認めない`) negate *after* the concept, so
  a left-only rule systematically mislabels both languages as positive.
* **Report structure.** Reports are bullet lists and semicolon-chained clauses,
  not prose. An off-the-shelf sentence splitter merges a whole findings section
  into one "sentence", destroying negation scope and every weak label derived
  from it.

Output is four-state (positive / negative / uncertain / not-mentioned) with a
confidence that downstream losses gate on, plus a compartment resolver so that
"posterior horn of the *medial* meniscus" cannot produce a `Lateral Meniscus`
weak label. Resolution across mentions is POSITIVE > UNCERTAIN > NEGATIVE:
the negations are usually the templated checklist ("ACL intact, PCL intact, …"),
high-volume and low-information, and letting it outvote a single explicit
positive builds a weak labeller with 0.99 specificity and 0.4 sensitivity.

**The lexicon is a starting point, not a deliverable.** `scripts/02_parse_reports.py
--audit` dumps the highest-frequency unmatched sentences per language; the
lexicon is grown from what the corpus actually says.

### 4.5 Robustness

**Group-DRO** with exponentiated-gradient ascent on group weights,
$q_k^{(t+1)}\propto q_k^{(t)}\exp(\eta_q\hat{\mathcal R}_k)$, over site ×
language × scanner. Two fixes are required at this data scale and both are
implemented: a group with 30 studies has a risk estimate noisy enough to capture
all the weight, so risks are shrunk towards the mean by their own standard error
$\hat\sigma_k/\sqrt{n_k}$; and worst-group risk is otherwise dominated by *label
difficulty* rather than domain shift, so risks are standardised per label before
the max.

**$\chi^2$-DRO** as the smooth complement: $\min_\eta \sqrt{1+2\rho}\,\|(\ell-\eta)_+\|_2+\eta$,
whose induced weights are *linear* in the excess loss rather than 0/1. That
single difference is why it tolerates label noise where CVaR does not — a
mislabelled study gets a large but finite weight instead of the entire budget.

**IRMv1** over *acquisition* environments — the (site, scanner, field-strength)
buckets stamped into `StudyRecord.env_index` — asking that the optimal rescaling
of the logits be the same at every centre. A feature needing a different gain at
one site is a feature about that site. Scoped honestly: it is applied to the
image classifier, scheduled in S4 only, and we have no ablation of our own
isolating its contribution; the justification is Arjovsky et al.'s plus the fact
that the test-set site distribution differs from training. `04_train.py` resolves
the environment column from `site` → `scanner_proxy` → `manufacturer` →
`field_strength_bucket` and **disables both `group_dro` and `irm`, recording the
fact in the run manifest, when only one environment is resolvable** — a
single-group DRO is the mean and a single-environment IRM is identically zero,
so running them would cost compute, log a plausible number and change nothing.

Schedule: ERM for the first third (you cannot robustify a model that has not
learnt the task), then a linear ramp to `0.3·GroupDRO + 0.1·χ²-DRO`.

### 4.6 Multi-task gradient conflict

Twelve labels share one trunk and their gradients conflict: the update that
helps `Effusion` (large bright fluid, low-frequency) hurts `Medial Meniscus`
(a 2 mm dark line). We default to **Aligned-MTL**: form $G\in\mathbb R^{L\times P}$,
eigendecompose the small Gram matrix $GG^\top$, and apply
$B=\sigma_{\min}U\Sigma^{-1}U^\top$. Of the three implemented (PCGrad, CAGrad,
Aligned-MTL) it is the only one whose fixed point does not depend on the
arbitrary relative scaling of the twelve losses — and our losses have wildly
different natural scales. Surgery is applied only to the fusion + router + head
block: memory is $L\times P$ floats, and early-backbone gradient conflict is
empirically negligible.

Three implementation details decide whether this helps or silently stops
training.

*The spectrum must be truncated, not clamped.* `Medial OA` and `Lateral OA`
produce near-collinear gradients constantly, so $GG^\top$ is rank deficient and
its smallest singular value is numerically zero. The $\sigma_{\min}$ prefactor
would then scale the **entire** update to zero: the loss plateaus and nothing in
the logs says why. We discard directions below $10^{-6}\sigma_{\max}$ and take
$\sigma_{\min}$ over the retained spectrum — the pseudo-inverse on the row
space, which loses nothing because discarded directions contribute to no $g_l$.
An all-zero $G$ falls back to the plain sum, not to $0$. Tasks whose gradient is
numerically zero on a batch (a rare label with no positive) are dropped before
combination and surgery is skipped below two active tasks.

*It has to compose, not overwrite.* The twelve per-label terms are one part of a
larger objective. `prepare`/`apply_` add
$\Delta=\mathcal S(G)-\sum_l g_l$ to `.grad`, leaving
$g_{\text{other}}+\mathcal S(G)$ — exact whatever else is in the loss. The
per-label decomposition must sum to the term it replaces *exactly*, which is why
`AsymmetricLoss`/`AUCMarginLoss` expose a `label_contrib` reduction (dividing by
the total valid count, not the per-label count) and why `compute_losses`
evaluates only the decomposition and takes its sum as the scalar — calling the
term twice would decay `AUCMarginLoss`'s prevalence EMA twice per step.

*Order matters for memory.* $G$ is measured **before** the main `backward`,
while the graph is alive but `.grad` is still empty, and applied after. Since
the inputs are restricted to the fusion/head block, autograd traverses only the
tail. Running `loss.backward(retain_graph=True)` first instead would pin the
entire backbone activation stack for the duration. Measured cost of surgery in
the shipped configuration: none detectable (20.15 s/step off, 19.77 s/step on).

---

## 5. Adaptive computation

A study is 3–7 series of 20–50 slices. Processing everything at 0.35 mm costs
~8× the 0.70 mm pass, and the extra resolution matters for perhaps 10 % of
slices. Uniform compute wastes ~7/8 of the fine budget — which is the budget the
nine-hour limit and the efficiency prize are made of.

**Which labels escalate.**
$\texttt{run\_fine}_l=\mathbb 1[U_l>\tau_U \lor p_l\in(\tau_{\text{lo}},\tau_{\text{hi}})]$,
per (study, label). Thresholds are not hand-picked; they are read off the
runtime–AUC Pareto frontier computed on OOF with a *measured* cost model.

**Which slices.** Gumbel top-k sampling without replacement is an exact sample
from the Plackett–Luce distribution over ordered $k$-subsets (Kool et al., 2019);
gradients come from a temperature-annealed relaxation of the same quantity.
Selected slices are dilated to contiguous windows — a tear on slice $i$ is
almost always visible on $i\pm1$, and the fine encoder's 2.5D stem needs the
neighbours anyway, so dilating here means the scheduler knows the true decode
cost instead of discovering it later.

Exploration is not optional. A selector trained purely on its own scores has an
obvious degenerate optimum: always pick the same indices, get a decent loss,
never discover the pathology is elsewhere. 25 % of the budget is sampled
uniformly during training.

**Budget in the objective.** Selected fraction is penalised against a target
during training. A hard cap applied at inference to a model trained without one
silently truncates exactly the slices the model relies on.

**The governor.** Kaggle gives nine wall-clock hours for an unknown number of
test studies; the hidden set may be 3× the public one. A fixed policy either
wastes the budget or blows it. A proportional controller tracks realised
throughput and adjusts the escalation rate so the projection lands inside a
budget with 12 % reserved. Asymmetric gains (back off fast at 0.6, ramp slowly
at 0.15) and no integral term — an integral term oscillates when study sizes are
heavy-tailed, which they are. **A submission that produces no CSV scores
nothing**, so the governor degrades to coarse-only rather than overrun.

---

## 6. Validation, ensembling, calibration

### 6.1 Statistics that let you tell 0.002 from noise

**DeLong** (Sun & Xu's $O(n\log n)$ midrank form) gives the exact non-parametric
(co)variance of empirical AUCs:
$\mathrm{Var}(\hat A)=S_{10}/m+S_{01}/n$. For *paired* models the same structural
components give the covariance, so "is A better than B on `Fracture`" is a
z-test with an exact variance instead of a bootstrap. The covariance term is the
whole point: two models scored on the same studies are strongly positively
correlated, and the naive unpaired SE overstates the uncertainty enough to hide
real improvements.

**Cluster bootstrap** at the *patient* level, optionally stratified by site.
Resampling studies independently understates variance, sometimes by 2×. The
macro-AUC CI is bootstrapped directly rather than assembled from per-label CIs,
because per-label errors are correlated through the shared patient sample and
ignoring that gives an interval too wide by roughly $\sqrt{L}$.

**Debiased, equal-mass ECE.** Equal-width bins are badly biased when predictions
concentrate near 0 — for a 1 %-prevalence label 14 of 15 equal-width bins are
empty. Per-bin debiasing subtracts the sampling variance $\bar p(1-\bar p)/n_b$
that a perfectly calibrated model would still produce.

### 6.2 The audit suite

Ten audits, each with a known way of destroying a private score while leaving
OOF healthy or improved. Five are **blocking**: `shuffled_label`
(macro-AUC must be 0.5 ± 0.03 under permutation — failure means the harness
mis-aligns predictions and labels, the most catastrophic and most easily missed
bug in the pipeline), `shuffled_report` (image share
$\mathcal S=(A_{\text{img}}-0.5)/(A_{\text{true}}-0.5)\ge 0.5$),
`duplicate_hash`, `embedding_neighbour` (cosine > 0.995 across folds with
different group ids), and the fold-hash consistency check inside the trainer.
Non-blocking but reported: `metadata_only`, `sequence_description_only`,
`text_only`, `fold_prevalence`, `prediction_site_gap`.

An audit that only warns is an audit that gets ignored, so
`scripts/05_oof_eval.py` exits non-zero on a blocking failure and the submission
script will not run without a passing audit record.

### 6.3 Per-label ensemble weights without overfitting the OOF

Twelve labels × $M$ models is $12(M-1)$ parameters fitted on an OOF matrix whose
effective sample size for the rarest label may be 40 positives. Unconstrained
per-label search is a reliable way to gain 0.004 on OOF and lose 0.003 on the
private split — the most common late-stage self-inflicted wound in
medical-imaging competitions. Three mechanisms:

1. **Anchored mirror descent on the simplex.** Exponentiated gradient on a
   sigmoid-smoothed AUC surrogate (the exact AUC is piecewise constant in $w$,
   so its gradient is zero almost everywhere), with an entropic anchor to the
   uniform weights. $\lambda\to\infty$ recovers the simple average.
2. **Hierarchical James–Stein shrinkage along the ontology.**
   $\tilde w_l=(1-\kappa_l)w_l+\kappa_l\bar w_{g(l)}$ with
   $\kappa_l=\sigma^2_l/(\sigma^2_l+n^+_l\tau^2)$. `Fracture` borrows
   `Contusion`, a far better prior than uniform.
3. **Nested leave-one-fold-out.** Weights for fold $k$ are fitted on folds
   $\ne k$ and the reported gain is measured on $k$. **Decision rule: if the
   nested weighted macro does not beat the nested uniform macro by more than one
   bootstrap SE, ship the uniform average.**

**Fit and deployment must combine in the same space.** The weights above are a
per-label simplex fitted on **rank-transformed** OOF predictions, so
`combine_members` — the single function both `06_ensemble.py` and the
submission notebook call — rank-transforms each member before applying them.
Combining raw probabilities at test time, which the notebook used to do, would
optimise one objective and ship another; and since macro-AUC sees only the
ordering, rank averaging is the right combiner anyway, being invariant to the
fact that a member trained with AUC-margin and one trained with ASL do not put
their probability mass in the same place. The ranks are midranks, matching the
convention `roc_auc` already uses: ordinal ranks break ties in argsort order,
which is not a monotone function of the input, so a *single* model's AUC would
move under a transform meant to leave it alone — worth ~2 · 10⁻⁴ on a column
with the tie structure a failed-study fallback produces. The combined ranks are
finally mapped back onto the members' probability scale through a strictly
increasing per-label map, so the file still carries interpretable probabilities
at exactly the AUC of the ranking.

Also provided: Caruana greedy selection with replacement, and a Bayesian
bootstrap that propagates "which model is better for `Fracture`" uncertainty
into the weights. Rank averaging is useless for distillation (ranks are not
probabilities), so the KD teacher always uses probability averaging.

### 6.4 Calibration and conformal risk control

Monotone calibrators only (Newton temperature, three-parameter beta) so
calibration provably cannot move the leaderboard. Conformal risk control
(Angelopoulos et al., 2023) chooses the largest $\hat\lambda$ with
$\hat R_n(\lambda)\le\alpha-\frac{B-\alpha}{n}$, guaranteeing
$\mathbb E[R(\hat\lambda)]\le\alpha$ under exchangeability alone — no
distributional assumption, no asymptotics, no requirement that the model be
calibrated. Default risk is 1 − sensitivity, the quantity a triage deployment
cares about. **Mondrian** stratification by site, because exchangeability holds
far better within a centre than across; strata with < 30 positives fall back to
the global threshold, since a guarantee computed from eight positives is not a
guarantee.

---

## 7. The efficiency track

The efficiency score is
$E \propto \frac{\max(0,A_{\max}-A_{\text{sub}})}{\max(0,A_{\max}-A_{\text{base}})}+\frac{t_{\text{sub}}}{t_{\max}}$:
runtime enters **linearly** while accuracy enters through a normalised deficit.
Near the frontier, trading a little AUC for a lot of runtime is almost always
correct — and the only way to know where "near" is, is to measure.

Total notebook time is
$t=t_{\text{decode}}+t_{\text{preproc}}+t_{\text{load}}+t_{\text{infer}}+t_{\text{post}}$,
so a fast GPU model can still lose on serial DICOM decoding. Highest-return
optimisations, in order: pre-built manifest, batched/GPU DICOM decode
(nvJPEG2000 is ~5× on the HTJ2K studies), whole-volume caching, coarse-to-fine
top-k, dynamic resolution, sequence skipping, BF16, then — only after profiling
confirms the model is the bottleneck — structured channel pruning and INT8.

**The student.** ConvNeXt-T or EfficientNetV2-S, 0.70 mm all-slice coarse
encoder, label-specific top-k, 0.35 mm shared fine encoder, two-layer sequence
model, one checkpoint, ≤ 2 TTA. Loss:
$\mathcal L=\mathcal L_{\text{gold}}+0.5\mathcal L_{\text{logit KD}}+0.1\mathcal L_{\text{feature KD}}+0.05\mathcal L_{\text{attention KD}}$
with **decoupled** KD (target and non-target branches weighted independently)
and **cross-fitted teacher logits** — from the study's OOF teacher prediction,
never from a teacher trained on that study, or the student learns the teacher's
overfit confidence.

Attention KD is the most valuable component for the rare osseous labels: it
transfers *where the teacher looks*, which is what the student cannot recover
from logits alone.

The student sees no ranking losses. Its teacher's logits already encode the
ranking, and adding AUC-M on top makes it trade teacher agreement for its own
noisy ranking estimate — measurably worse on every label with < 150 positives.

---

### Test-time augmentation is off by default

The notebook has a correct TTA path and does not use it. Two reasons, both
learned the hard way.

The view used to be built with `batch.__class__(**batch.__dict__)`, and
`StudyBatch` is `@dataclass(slots=True)` — it has no `__dict__`. That raised on
the first study and was swallowed by the surrounding `except Exception: pass`,
so TTA had been a silent no-op: it looked enabled, cost nothing, and gained
nothing. It is now built with `dataclasses.replace`, and a failure is counted
and logged once rather than swallowed.

The view itself was a left-right mirror, averaged into the original logits
**without permuting the labels** — the exact corruption `MEDIAL_LATERAL_SWAP`
exists to prevent during training, applied at submission time to `Medial
Meniscus`, `Lateral Meniscus`, `Medial OA` and `Lateral OA`. And even permuted
it is out of distribution: laterality is canonicalised in the loader and
`horizontal_flip_prob` is 0, so the network has never seen a mirrored knee. The
mirror is now correctly permuted and gated behind `KAIROS_TTA_FLIP`, for weights
actually trained with flips.

What remains is a gamma view, inside the training distribution and
label-preserving. It is still off by default (`MAX_TTA=1`), because a second
forward pass doubles inference cost and the governor pays for that by
throttling the fine pass — trading a designed, measurable coarse-to-fine gain
for an unmeasured augmentation-averaging one. Raise `KAIROS_MAX_TTA` when an
OOF number says the trade is worth it.

## 8. The final ensemble

Diversity is measured, not assumed: members are selected on OOF error
correlation, and a member whose residuals correlate > 0.9 with an existing
member is dropped regardless of its solo score.

| # | backbone | aggregator | resolution | distinct inductive bias |
|---|----------|-----------|------------|--------------------------|
| 1 | ConvNeXt-S (IN-22k) | transformer | 0.70 → 0.35 mm | modern CNN locality |
| 2 | Swin-S | transformer | 0.70 → 0.35 mm | hierarchical window attention |
| 3 | DINOv2/v3 ViT-B + LoRA | SSM | 0.70 → 0.35 mm | SSL features, linear-time slices |
| 4 | MedSigLIP-400M (frozen + plugins) | transformer | 0.55 mm single-scale | medical VLP features |
| 5 | 3D ResNet / Video-Swin | native 3D | isotropic 0.8 mm crop | true volumetric context |
| 6 | rare-label specialist | transformer | 0.35 mm, osseous crop | Fracture/Contusion recall |

Ten seeds of one backbone is not an ensemble.

**Candidate pretrained weights**, all offline-packageable as Kaggle datasets and
all to be ablated against ImageNet init on the *same* folds and budget before
earning a slot: `timm` ConvNeXt/Swin/EfficientNetV2; DINOv2 and DINOv3 (the
latter's Gram-anchored patch features and its released ConvNeXt distillations);
MedSigLIP-400M (medically-tuned SigLIP vision tower, 448 px, open weights);
OrthoFoundation (DINOv3 backbone, 1.2 M unlabeled knee X-ray/MRI, if weights are
released in time); Triad (3D MRI FM, 131 k volumes); MedicalNet / Models Genesis
for the 3D branch; RadImageNet for a radiology-domain 2D init; TotalSegmentator
MRI or an nnU-Net knee model as a *localiser only*, never as a classifier.

Domain-specific pretraining is a hypothesis, not a guarantee — chest-pretrained
weights in particular have a long history of not transferring to
musculoskeletal MRI. Every checkpoint's licence, redistribution rights and
compatibility with the winners' weight-publication obligation are verified in
week 1, not week 11.

---

## 9. Execution order

The ordering below is by *what has to be true before the next thing is
meaningful*, which is not the same as by importance.

| week | deliverable | gate to pass |
|------|-------------|--------------|
| 1 | DICOM manifest + QC + geometric ordering | < 0.5 % decode failures; ordering verified on 50 hand-checked series |
| 1–2 | fold artefact + hash + 2.5D baseline | rare-label prevalence deviation < 40 %; blocking audits pass |
| 2 | **measured Kaggle runtime** on a dummy submission | end-to-end < 9 h with headroom |
| 3 | report parser + text-only baseline | unmatched-sentence audit reviewed per language |
| 3–4 | label queries + cross-sequence fusion | macro-AUC ↑ **and** ≥ 4 hard labels ↑ |
| 4–5 | image–report pretraining | image-only OOF ↑ at equal budget; `report_reliance` > 0 |
| 5–6 | coarse-to-fine adaptive | equal AUC at lower runtime, or ↑ AUC at equal runtime |
| 6–7 | 3D / Video-Swin diversity branch | OOF residual correlation < 0.9 with members 1–3 |
| 7–8 | ranking fine-tune (AUC-M, pAUC) | macro-AUC ↑ with worst-label AUC not ↓ |
| 8–9 | robustness stage | site AUC gap ↓ without macro-AUC ↓ |
| 9–10 | ensemble + nested weight check | nested weighted > nested uniform + 1 SE, else ship uniform |
| 10 | efficiency student | ≥ 90 % of teacher macro-AUC at ≤ 25 % runtime |
| 11 | offline notebook hardening, freeze | three clean commit runs inside the limit |

Priority zero, before any of it: a flawless patient-level five-fold split, a
reliable DICOM pipeline, OOF predictions, and a **measured** Kaggle inference
time. Every hour spent on architecture before those four exist is an hour spent
on a number you cannot trust.

---

## 10. Reproducibility and the winners' deliverable

Stored per run: git commit, full config, **fold manifest hash**, package lock,
seed, backbone weight source and licence, train/val UID lists, OOF matrix,
checkpoint SHA-256, preprocessing schema version, validation report, inference
runtime profile, submission SHA-256. `Trainer.load` refuses a checkpoint whose
fold hash or label order disagrees with the current run — the failure mode it
prevents (silently ensembling across split versions) produces an OOF score that
is optimistic by an amount nobody can later reconstruct.

CLAIM (2024 update) and TRIPOD+AI are used as the reporting checklists for the
method description, since the winners' obligations include a public model, a
video and a method write-up.

**On clinical use.** A competition AUC does not establish a safe operating
point. Deployment requires PACS integration, site-specific external validation,
a prospective silent trial, calibration-drift monitoring, human-in-the-loop
review, abstention, audit logging, versioned preprocessing, scanner/protocol
drift alerting, post-operative and paediatric fallbacks, per-label evidence
display, and prospectively determined clinical thresholds. The conformal risk
controller in §6.4 is the part of this system that is actually deployment-shaped;
the leaderboard score is not.

---

## 11. What is different from a conventional strong solution

A conventional strong entry — patient-level folds, 2.5D ConvNeXt/Swin,
attention pooling, ASL, a diverse ensemble, per-label weights, a distilled
student — is a good plan and would place respectably. The differences here, in
descending order of expected effect:

1. **Metric-direct ranking optimisation** (AUC min-max + pAUC + memory queue)
   instead of BCE and hope, on labels where BCE's gradient is 95 % easy
   negatives.
2. **Physical-geometry conditioning throughout** (metric positional encoding,
   metric rotary attention, metric ALiBi, physical Δ in the SSM, metric
   resampling) instead of index-based encodings that memorise site protocols.
3. **Twelve label queries + MoE + ontology prior** instead of one pooled vector
   and twelve linear heads, with Aligned-MTL surgery on the shared fusion/head
   block so the twelve gradients are whitened by their own spectrum rather than
   summed — the update no longer depends on the arbitrary relative scale of the
   twelve losses.
4. **Report as privileged information, confined to the representation**,
   including OT grounding without boxes, instead of either ignoring reports or
   letting them leak into a prediction path that cannot exist at test time.
5. **Distributional robustness with the corrections that make it work at this
   scale** (SE shrinkage, per-label standardisation) instead of vanilla
   Group-DRO, which at 16 sites of unequal size chases the smallest one.
6. **Adaptive compute as a trained, budgeted, closed-loop system** instead of a
   fixed resolution — this is where the efficiency prize is, and it also buys
   main-track accuracy by spending the saved budget on ensemble members.
7. **Statistics that can resolve the effects being chased**: DeLong paired
   tests, patient-cluster bootstrap, nested ensemble validation, and a blocking
   audit suite — so that the last three weeks are spent on real improvements
   rather than on noise.

None of these is a guarantee. Each is measurable on out-of-fold data before it
costs a submission, which is the property that matters.
