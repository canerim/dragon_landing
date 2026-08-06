r"""Calibration and distribution-free risk control.

ROC-AUC is invariant to any strictly increasing transform of the scores, so
temperature scaling *cannot* change the leaderboard.  Calibration is
nonetheless load-bearing in four places, and each needs a different tool:

1. **Distillation.** A student trained against uncalibrated teacher
   probabilities inherits the teacher's overconfidence and its ranking gets
   *worse*.  → per-label temperature / beta calibration.
2. **The report gate.** Deciding whether to trust a text signal requires
   comparing two probabilities on the same scale.  → per-label calibration.
3. **The adaptive-compute gate.** "Is this case borderline?" is a statement
   about a probability, not a logit.  → per-label calibration.
4. **Deployment and the winners' deliverable.** A model whose 0.9 means 0.9 is
   the difference between a demo and a tool. → conformal risk control.

Implemented:

``TemperatureScaler``
    Per-label :math:`T_l` fitted by Newton's method on the NLL.  Newton rather
    than LBFGS because the 1-D NLL in :math:`\log T` is smooth and strictly
    convex, so Newton converges in ~5 iterations exactly and needs no
    line-search bookkeeping.

``BetaCalibrator``
    Three-parameter :math:`p \mapsto \sigma(a\log p - b\log(1-p) + c)`
    (Kull et al., 2017).  Strictly more expressive than temperature scaling and
    still monotone -- so, unlike isotonic regression, it *provably cannot change
    the AUC*.  That property is why it is the default here: we can calibrate the
    submission without any risk of moving the leaderboard score.

``IsotonicCalibrator``
    Pool-adjacent-violators.  Non-parametric and unbeatable in-sample, but it
    ties scores together and therefore **does** change AUC (usually downwards
    on held-out data for rare labels).  Provided, off by default, with the
    warning in code.

``ConformalRiskController``
    Distribution-free control of a monotone risk (Angelopoulos et al., 2023).
    Chooses the largest threshold :math:`\hat\lambda` such that the empirical
    risk on a calibration set, with the finite-sample correction

    .. math::
        \hat R_n(\lambda) \;\le\; \alpha - \frac{B - \alpha}{n},

    guarantees :math:`\mathbb E[R(\hat\lambda)] \le \alpha` on a new exchangeable
    study.  Applied per label to give an operating point with a *guaranteed*
    expected false-negative rate -- the number a clinical reader actually needs
    and the one the challenge's clinical framing asks for.

``MondrianConformal``
    The same, stratified by site/scanner, because exchangeability holds *within*
    a centre far better than across centres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "TemperatureScaler",
    "BetaCalibrator",
    "IsotonicCalibrator",
    "ConformalRiskController",
    "MondrianConformal",
    "reliability_curve",
]

_EPS = 1e-12


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)


class TemperatureScaler:
    """Per-label temperature, fitted by Newton on :math:`\\log T`."""

    def __init__(self, num_labels: int) -> None:
        self.log_t = np.zeros(num_labels)

    def fit(self, logits: np.ndarray, y: np.ndarray, *, n_iter: int = 25) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        for l in range(logits.shape[1]):
            ok = np.isfinite(y[:, l]) & np.isfinite(logits[:, l])
            z, t = logits[ok, l], y[ok, l]
            if z.size < 10 or t.max() == t.min():
                continue
            u = 0.0
            for _ in range(n_iter):
                s = np.exp(-u)
                p = _sigmoid(z * s)
                r = p - t
                # dL/du  and  d2L/du2  for L = BCE(sigmoid(z e^{-u}), t)
                g = float(np.mean(-r * z * s))
                h = float(np.mean(p * (1 - p) * (z * s) ** 2 + r * z * s))
                if abs(h) < 1e-12:
                    break
                step = g / h
                u -= np.clip(step, -1.0, 1.0)
                if abs(step) < 1e-8:
                    break
            self.log_t[l] = u
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, dtype=np.float64) * np.exp(-self.log_t)[None, :]

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return _sigmoid(self.transform(logits))

    @property
    def temperature(self) -> np.ndarray:
        return np.exp(self.log_t)


class BetaCalibrator:
    r"""Monotone three-parameter beta calibration.

    Fitted as a logistic regression on the two features
    :math:`(\log p, -\log(1-p))`; monotonicity is guaranteed by projecting
    :math:`a, b` to be non-negative after each step, which is a cheap and
    faithful way to keep the AUC-preservation property that motivates using
    this over isotonic.
    """

    def __init__(self, num_labels: int) -> None:
        self.a = np.ones(num_labels)
        self.b = np.ones(num_labels)
        self.c = np.zeros(num_labels)

    def fit(self, p: np.ndarray, y: np.ndarray, *, lr: float = 0.1,
            n_iter: int = 400) -> "BetaCalibrator":
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
        y = np.asarray(y, dtype=np.float64)
        for l in range(p.shape[1]):
            ok = np.isfinite(y[:, l])
            if ok.sum() < 20 or y[ok, l].max() == y[ok, l].min():
                continue
            x1 = np.log(p[ok, l])
            x2 = -np.log1p(-p[ok, l])
            t = y[ok, l]
            a, b, c = 1.0, 1.0, 0.0
            for _ in range(n_iter):
                z = a * x1 + b * x2 + c
                r = _sigmoid(z) - t
                a -= lr * float(np.mean(r * x1))
                b -= lr * float(np.mean(r * x2))
                c -= lr * float(np.mean(r))
                a, b = max(a, 0.0), max(b, 0.0)
            self.a[l], self.b[l], self.c[l] = a, b, c
        return self

    def predict_proba(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
        z = self.a[None, :] * np.log(p) + self.b[None, :] * (-np.log1p(-p)) + self.c[None, :]
        return _sigmoid(z)


class IsotonicCalibrator:
    """PAV isotonic regression.  **Changes AUC.**  Use only when you have
    verified on nested folds that it does not hurt the label in question."""

    def __init__(self, num_labels: int) -> None:
        self.knots: list[tuple[np.ndarray, np.ndarray] | None] = [None] * num_labels

    @staticmethod
    def _pav(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(x, kind="mergesort")
        xs, ys = x[order], y[order]
        vals = list(ys.astype(np.float64))
        wts = [1.0] * len(vals)
        i = 0
        while i < len(vals) - 1:
            if vals[i] <= vals[i + 1]:
                i += 1
                continue
            w = wts[i] + wts[i + 1]
            v = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / w
            vals[i : i + 2] = [v]
            wts[i : i + 2] = [w]
            if i > 0:
                i -= 1
        out, k = np.empty(len(ys)), 0
        for v, w in zip(vals, wts):
            n = int(round(w))
            out[k : k + n] = v
            k += n
        return xs, out

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        p = np.asarray(p, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        for l in range(p.shape[1]):
            ok = np.isfinite(y[:, l])
            if ok.sum() < 20:
                continue
            self.knots[l] = self._pav(p[ok, l], y[ok, l])
        return self

    def predict_proba(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        out = p.copy()
        for l, kn in enumerate(self.knots):
            if kn is None:
                continue
            xs, ys = kn
            out[:, l] = np.interp(p[:, l], xs, ys)
        return out


# --------------------------------------------------------------------------- #
# Conformal risk control                                                       #
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ConformalRiskController:
    r"""Per-label threshold with a guaranteed expected risk.

    ``risk`` is any function of ``(y, p, lambda)`` that is **non-increasing in
    lambda** and bounded by ``B``.  The default is the false-negative rate
    :math:`R(\lambda) = \mathbb E[\,y(1 - \mathbb 1[p \ge \lambda])\,]/\mathbb E[y]`,
    i.e. 1 − sensitivity, which is what a screening/triage deployment cares
    about.

    The guarantee is *marginal over the calibration draw* and requires only
    exchangeability between calibration and test studies -- no distributional
    assumption, no calibration of the model itself, no asymptotics.
    """

    alpha: float = 0.10
    bound: float = 1.0
    n_grid: int = 500
    lambdas: np.ndarray | None = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "ConformalRiskController":
        p = np.asarray(p, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        L = p.shape[1]
        out = np.zeros(L)
        grid = np.linspace(0.0, 1.0, self.n_grid)
        for l in range(L):
            ok = np.isfinite(y[:, l]) & np.isfinite(p[:, l])
            yy, pp = y[ok, l], p[ok, l]
            n = int((yy > 0.5).sum())
            if n == 0:
                out[l] = 0.0
                continue
            # Empirical risk over the grid, with the finite-sample correction.
            fn = np.array([float(((pp < g) & (yy > 0.5)).sum()) / n for g in grid])
            corrected = (n * fn + self.bound) / (n + 1)
            feasible = np.flatnonzero(corrected <= self.alpha)
            out[l] = float(grid[feasible.max()]) if feasible.size else 0.0
        self.lambdas = out
        return self

    def predict_set(self, p: np.ndarray) -> np.ndarray:
        """Boolean ``(N, L)``: which labels are *asserted* at the guaranteed risk."""
        if self.lambdas is None:
            raise RuntimeError("call fit() first")
        return np.asarray(p, dtype=np.float64) >= self.lambdas[None, :]

    def empirical_risk(self, p: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Realised risk per label -- report this next to ``alpha``."""
        if self.lambdas is None:
            raise RuntimeError("call fit() first")
        p = np.asarray(p, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        out = np.full(p.shape[1], np.nan)
        for l in range(p.shape[1]):
            ok = np.isfinite(y[:, l])
            pos = ok & (y[:, l] > 0.5)
            if not pos.any():
                continue
            out[l] = float((p[pos, l] < self.lambdas[l]).mean())
        return out


class MondrianConformal:
    """Conformal risk control stratified by a categorical taxonomy (site/scanner).

    Exchangeability is a much safer assumption within a centre than across
    centres, so a single global threshold delivers its guarantee *on average*
    while being systematically too loose at one site and too tight at another.
    Strata with fewer than ``min_per_stratum`` positives fall back to the global
    threshold -- a guarantee computed from eight positives is not a guarantee.
    """

    def __init__(self, alpha: float = 0.10, *, min_per_stratum: int = 30) -> None:
        self.alpha = alpha
        self.min_per_stratum = min_per_stratum
        self.global_ = ConformalRiskController(alpha=alpha)
        self.by_stratum: dict[object, ConformalRiskController] = {}

    def fit(self, p: np.ndarray, y: np.ndarray, stratum: np.ndarray) -> "MondrianConformal":
        self.global_.fit(p, y)
        for s in np.unique(stratum):
            m = stratum == s
            if int((np.nan_to_num(y[m]) > 0.5).sum()) < self.min_per_stratum:
                continue
            self.by_stratum[s] = ConformalRiskController(alpha=self.alpha).fit(p[m], y[m])
        return self

    def thresholds_for(self, stratum_value) -> np.ndarray:
        c = self.by_stratum.get(stratum_value, self.global_)
        assert c.lambdas is not None
        return c.lambdas

    def predict_set(self, p: np.ndarray, stratum: np.ndarray) -> np.ndarray:
        out = np.zeros_like(p, dtype=bool)
        for s in np.unique(stratum):
            m = stratum == s
            out[m] = p[m] >= self.thresholds_for(s)[None, :]
        return out


def reliability_curve(
    y: np.ndarray, p: np.ndarray, *, n_bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-mass reliability curve: ``(mean_pred, empirical_rate, count)``."""
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if y.size == 0:
        return np.array([]), np.array([]), np.array([])
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    mp, er, cn = [], [], []
    for i in range(len(edges) - 1):
        m = (p >= edges[i]) & (p < edges[i + 1]) if i < len(edges) - 2 else (p >= edges[i])
        if not m.any():
            continue
        mp.append(float(p[m].mean()))
        er.append(float(y[m].mean()))
        cn.append(int(m.sum()))
    return np.array(mp), np.array(er), np.array(cn)
