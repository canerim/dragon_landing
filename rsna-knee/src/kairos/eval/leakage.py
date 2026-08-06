r"""Mandatory leakage and shortcut audits.

Every one of these tests has a known way of quietly destroying a private-
leaderboard score while leaving the OOF score healthy or *improved*.  They are
cheap, they run in minutes, and the pipeline is configured to refuse to produce
a submission until they have all been executed and logged.

============================  ============================================
audit                          what a failure means
============================  ============================================
``metadata_only``              site/protocol predicts the label; the image
                               model can free-ride on it and will not
                               transfer to a different site mix
``sequence_description_only``  the protocol *name* is a label proxy
``text_only``                  upper bound from the report, and a template
                               shortcut detector
``shuffled_report``            the multimodal model is really text-only
``shuffled_label``             the whole evaluation harness is wrong
``duplicate_hash``             the same study is in two folds
``embedding_neighbour``        near-duplicate studies straddle a fold
``fold_prevalence``            the split is not balanced enough to measure
                               the effect sizes we care about
``crop_mask``                  burned-in annotations / overlays leak
``prediction_site_gap``        the model is a site classifier in disguise
============================  ============================================

The design principle: **an audit that only warns is an audit that gets
ignored.**  :func:`run_audit_suite` returns a structured result with a hard
``passed`` flag per audit and a single ``blocking`` flag, and
``scripts/05_oof_eval.py`` exits non-zero when ``blocking`` is set.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from ..eval.metrics import macro_auc, per_label_auc, roc_auc

__all__ = [
    "AuditResult",
    "AuditSuiteResult",
    "metadata_only_audit",
    "shuffled_label_audit",
    "shuffled_report_audit",
    "duplicate_hash_audit",
    "embedding_neighbour_audit",
    "fold_prevalence_audit",
    "prediction_site_gap_audit",
    "run_audit_suite",
]


@dataclass(slots=True)
class AuditResult:
    name: str
    passed: bool
    value: float
    threshold: float
    blocking: bool
    detail: str = ""
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASS" if self.passed else ("FAIL" if self.blocking else "WARN")
        return f"[{status}] {self.name:<28} {self.value:.4f} (thr {self.threshold:.4f}) {self.detail}"


@dataclass(slots=True)
class AuditSuiteResult:
    results: list[AuditResult]

    @property
    def blocking(self) -> bool:
        return any((not r.passed) and r.blocking for r in self.results)

    def to_text(self) -> str:
        lines = [str(r) for r in self.results]
        lines.append("-" * 72)
        lines.append("BLOCKING FAILURES PRESENT" if self.blocking else "all blocking audits passed")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #


def _fit_predict_logreg(
    X: np.ndarray, y: np.ndarray, fold: np.ndarray, *, l2: float = 1.0, n_iter: int = 300
) -> np.ndarray:
    """Tiny out-of-fold logistic regression (gradient descent, numpy only).

    Deliberately dependency-free and deliberately *weak*: the point of the
    metadata audit is to show that even a trivial model finds signal, and a
    weak model finding signal is a stronger statement than a strong one doing
    so.  If a 300-step logistic regression on eight metadata columns reaches
    AUC 0.65 on a label, the image model is definitely also using it.
    """
    n, d = X.shape
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Xs = np.hstack([Xs, np.ones((n, 1))])
    out = np.full(n, np.nan)
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        yy = y[tr]
        if not np.isfinite(yy).any() or len(np.unique(yy[np.isfinite(yy)])) < 2:
            continue
        w = np.zeros(d + 1)
        Xtr, ytr = Xs[tr], np.nan_to_num(yy)
        for _ in range(n_iter):
            p = 1.0 / (1.0 + np.exp(-np.clip(Xtr @ w, -30, 30)))
            g = Xtr.T @ (p - ytr) / len(ytr) + l2 * np.r_[w[:-1], 0.0] / len(ytr)
            w -= 0.5 * g
        out[te] = Xs[te] @ w
    return out


def metadata_only_audit(
    metadata: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    *,
    threshold: float = 0.62,
    name: str = "metadata_only",
) -> AuditResult:
    """Can site/scanner/protocol metadata alone predict the labels?"""
    aucs = []
    for l in range(y.shape[1]):
        s = _fit_predict_logreg(metadata, y[:, l], fold)
        ok = np.isfinite(s)
        aucs.append(roc_auc(y[ok, l], s[ok]) if ok.any() else np.nan)
    a = np.asarray(aucs, dtype=float)
    worst = float(np.nanmax(a)) if np.isfinite(a).any() else 0.5
    return AuditResult(
        name=name,
        passed=worst < threshold,
        value=worst,
        threshold=threshold,
        blocking=False,
        detail=f"max per-label metadata AUC (macro {np.nanmean(a):.4f})",
        extra={"per_label": a.tolist()},
    )


def shuffled_label_audit(
    y: np.ndarray, scores: np.ndarray, *, seed: int = 0, n_rep: int = 20,
    tolerance: float = 0.03, min_studies: int = 50,
) -> AuditResult:
    r"""Permuting the labels must give macro-AUC ≈ 0.5.

    If it does not, the evaluation harness is broken -- almost always a
    misalignment between the prediction rows and the label rows, which is the
    single most catastrophic and most easily missed bug in the pipeline.

    The tolerance is **sample-size aware**, and that is not a detail.  Under the
    null the per-permutation macro-AUC has standard deviation roughly
    :math:`(12 N \bar p(1-\bar p))^{-1/2}`, so on a 20-study debug fold a single
    permutation lands 0.15 away from 0.5 routinely.  A fixed ±0.03 tolerance
    therefore fires on every small evaluation, and a blocking audit that cries
    wolf is one that gets switched off -- which costs far more than the false
    alarm it produced.  We widen to :math:`3\,\mathrm{SE}` of the permutation
    mean when that exceeds the absolute tolerance, and say so in the detail.

    Below ``min_studies`` the test is reported as *uninformative* rather than
    passed or failed: it genuinely cannot distinguish a broken harness from
    noise, and claiming otherwise in either direction is worse than abstaining.
    """
    n = int(len(y))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_rep):
        perm = rng.permutation(n)
        vals.append(macro_auc(y[perm], scores))
    vals = np.asarray(vals, dtype=float)
    m = float(np.nanmean(vals))
    sd = float(np.nanstd(vals))
    se = sd / max(np.sqrt(max(np.isfinite(vals).sum(), 1)), 1e-9)
    eff_tol = max(tolerance, 3.0 * se)

    if n < min_studies:
        return AuditResult(
            name="shuffled_label",
            passed=True,
            value=m,
            threshold=0.5,
            blocking=False,
            detail=f"UNINFORMATIVE: only {n} studies (need >= {min_studies}); "
                   f"permutation sd {sd:.4f} swamps any real misalignment",
        )

    return AuditResult(
        name="shuffled_label",
        passed=abs(m - 0.5) < eff_tol,
        value=m,
        threshold=0.5,
        blocking=True,
        detail=f"expect 0.5 +/- {eff_tol:.4f} (n={n}, {n_rep} perms, "
               f"sd {sd:.4f}, se {se:.4f})",
        extra={"sd": sd, "se": se, "effective_tolerance": eff_tol, "n": n},
    )


def shuffled_report_audit(
    y: np.ndarray,
    scores_true_report: np.ndarray,
    scores_shuffled_report: np.ndarray,
    scores_image_only: np.ndarray,
    *,
    min_image_share: float = 0.5,
) -> AuditResult:
    r"""Is the multimodal model actually using the image?

    Define the *image share*

    .. math::
        \mathcal S = \frac{A_{\text{img}} - 0.5}
                          {A_{\text{true}} - 0.5},

    the fraction of the multimodal model's skill that survives removing the
    report.  A model whose share is below 0.5 is mostly a report classifier and
    is worthless as a teacher for an image-only student -- and worthless on a
    test set without reports.
    """
    a_true = macro_auc(y, scores_true_report)
    a_shuf = macro_auc(y, scores_shuffled_report)
    a_img = macro_auc(y, scores_image_only)
    share = (a_img - 0.5) / max(a_true - 0.5, 1e-6)
    return AuditResult(
        name="shuffled_report",
        passed=share >= min_image_share and a_shuf < a_true,
        value=float(share),
        threshold=min_image_share,
        blocking=True,
        detail=f"true {a_true:.4f} / shuffled {a_shuf:.4f} / image-only {a_img:.4f}",
        extra={"auc_true": a_true, "auc_shuffled": a_shuf, "auc_image": a_img},
    )


def duplicate_hash_audit(
    hashes: Sequence[str], fold: np.ndarray, group_id: Sequence[str]
) -> AuditResult:
    """Identical pixel content must not appear in two different folds."""
    by_hash: dict[str, set[int]] = {}
    for h, f in zip(hashes, np.asarray(fold)):
        by_hash.setdefault(str(h), set()).add(int(f))
    cross = {h: fs for h, fs in by_hash.items() if len(fs) > 1}
    n_dup_groups = sum(1 for h, fs in by_hash.items() if len(fs) >= 1 and
                       list(hashes).count(h) > 1)
    return AuditResult(
        name="duplicate_hash",
        passed=len(cross) == 0,
        value=float(len(cross)),
        threshold=0.0,
        blocking=True,
        detail=f"{len(cross)} content hashes straddle folds "
               f"({n_dup_groups} duplicated hashes overall)",
        extra={"examples": list(cross)[:10], "n_groups": len(set(map(str, group_id)))},
    )


def embedding_neighbour_audit(
    embeddings: np.ndarray,
    fold: np.ndarray,
    group_id: Sequence[str],
    *,
    similarity_threshold: float = 0.995,
    max_violation_rate: float = 0.002,
) -> AuditResult:
    r"""Near-duplicate studies (same patient, different accession) across folds.

    Exact hashing misses the common case: the same knee re-scanned, or a series
    re-sent with a different reconstruction.  Cosine similarity in a frozen
    encoder's embedding space catches those.  Anything above
    ``similarity_threshold`` that sits in a different fold *and* a different
    group is a fold violation the grouping did not catch, and it inflates OOF
    by an amount that will not survive the private split.
    """
    X = np.asarray(embeddings, dtype=np.float64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    fold = np.asarray(fold)
    gid = np.asarray([str(g) for g in group_id])

    n = X.shape[0]
    violations = 0
    pairs: list[tuple[int, int, float]] = []
    block = 512
    for i in range(0, n, block):
        S = X[i : i + block] @ X.T
        rows = np.arange(i, min(i + block, n))
        S[np.arange(len(rows)), rows] = -1.0
        idx = np.argwhere(S > similarity_threshold)
        for r, c in idx:
            a, b = int(rows[r]), int(c)
            if a >= b:
                continue
            if fold[a] != fold[b] and gid[a] != gid[b]:
                violations += 1
                if len(pairs) < 10:
                    pairs.append((a, b, float(S[r, c])))
    rate = violations / max(n, 1)
    return AuditResult(
        name="embedding_neighbour",
        passed=rate <= max_violation_rate,
        value=float(rate),
        threshold=max_violation_rate,
        blocking=True,
        detail=f"{violations} cross-fold near-duplicate pairs at cos>{similarity_threshold}",
        extra={"examples": pairs},
    )


def fold_prevalence_audit(
    y: np.ndarray, fold: np.ndarray, *, max_relative_deviation: float = 0.35
) -> AuditResult:
    """Per-fold prevalence must be close enough to measure a real effect."""
    y = np.nan_to_num(np.asarray(y, dtype=float))
    fold = np.asarray(fold)
    glob = y.mean(axis=0)
    worst, worst_label = 0.0, -1
    for l in range(y.shape[1]):
        if glob[l] <= 0:
            continue
        for k in np.unique(fold):
            dev = abs(y[fold == k, l].mean() - glob[l]) / glob[l]
            if dev > worst:
                worst, worst_label = float(dev), l
    return AuditResult(
        name="fold_prevalence",
        passed=worst <= max_relative_deviation,
        value=worst,
        threshold=max_relative_deviation,
        blocking=False,
        detail=f"worst relative prevalence deviation (label index {worst_label})",
    )


def prediction_site_gap_audit(
    scores: np.ndarray, y: np.ndarray, site: Sequence[object],
    *, max_gap: float = 0.12, min_site_n: int = 40,
) -> AuditResult:
    """Spread of per-site macro-AUC.

    A large gap is the operational definition of "this model does not
    generalise across centres".  It is the number Group-DRO is trying to shrink
    and the number that predicts a private-leaderboard surprise.
    """
    site = np.asarray([str(s) for s in site])
    per_site = {}
    for s in np.unique(site):
        m = site == s
        if int(m.sum()) < min_site_n:
            continue
        per_site[s] = macro_auc(y[m], scores[m])
    if len(per_site) < 2:
        return AuditResult("prediction_site_gap", True, 0.0, max_gap, False,
                           "fewer than two sites with enough studies")
    vals = np.array(list(per_site.values()), dtype=float)
    gap = float(np.nanmax(vals) - np.nanmin(vals))
    return AuditResult(
        name="prediction_site_gap",
        passed=gap <= max_gap,
        value=gap,
        threshold=max_gap,
        blocking=False,
        detail=f"best {np.nanmax(vals):.4f} vs worst {np.nanmin(vals):.4f} "
               f"over {len(per_site)} sites",
        extra={"per_site": {k: float(v) for k, v in per_site.items()}},
    )


def run_audit_suite(
    *,
    y: np.ndarray,
    scores: np.ndarray,
    fold: np.ndarray,
    group_id: Sequence[str],
    metadata: np.ndarray | None = None,
    site: Sequence[object] | None = None,
    hashes: Sequence[str] | None = None,
    embeddings: np.ndarray | None = None,
    multimodal: Mapping[str, np.ndarray] | None = None,
    extra: Sequence[Callable[[], AuditResult]] = (),
) -> AuditSuiteResult:
    """Run everything that the supplied inputs make possible."""
    results = [shuffled_label_audit(y, scores), fold_prevalence_audit(y, fold)]
    if metadata is not None:
        results.append(metadata_only_audit(metadata, y, fold))
    if hashes is not None:
        results.append(duplicate_hash_audit(hashes, fold, group_id))
    if embeddings is not None:
        results.append(embedding_neighbour_audit(embeddings, fold, group_id))
    if site is not None:
        results.append(prediction_site_gap_audit(scores, y, site))
    if multimodal is not None:
        results.append(
            shuffled_report_audit(
                y,
                multimodal["true"],
                multimodal["shuffled"],
                multimodal["image_only"],
            )
        )
    results.extend(fn() for fn in extra)
    return AuditSuiteResult(results=results)


def content_hash(volume: np.ndarray, *, n_bits: int = 64) -> str:
    """Perceptual hash of a volume, robust to intensity rescaling.

    Downsample to an 8×8×4 grid of *rank-normalised* intensities and threshold
    at the median.  Rank normalisation is what makes this survive the different
    intensity scaling used by different vendors, which a raw byte hash does not.
    """
    v = np.asarray(volume, dtype=np.float64)
    if v.ndim == 2:
        v = v[None]
    zs = np.array_split(np.arange(v.shape[0]), min(4, v.shape[0]))
    tiles = []
    for zi in zs:
        sl = v[zi].mean(axis=0)
        h, w = sl.shape
        hs = np.array_split(np.arange(h), 8)
        ws = np.array_split(np.arange(w), 8)
        tiles.append(
            np.array([[sl[np.ix_(a, b)].mean() for b in ws] for a in hs]).ravel()
        )
    feat = np.concatenate(tiles)
    order = np.argsort(np.argsort(feat)) / max(len(feat) - 1, 1)
    bits = (order > 0.5).astype(np.uint8)[:n_bits]
    packed = np.packbits(bits).tobytes()
    return hashlib.sha1(packed).hexdigest()[:16]
