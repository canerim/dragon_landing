r"""Torch ``Dataset``, class-aware batch sampler and ``DataLoader`` wiring.

The sampler is the part that deserves an explanation.

A naive ``WeightedRandomSampler`` on a multi-label problem is wrong in a way
that is easy to miss: up-weighting a study because it is positive for
``Fracture`` also up-weights every *other* label that study carries.  Fracture
co-occurs with Contusion, so oversampling Fracture 4× silently oversamples
Contusion, shifts its effective prevalence, and moves its decision threshold —
on a metric where each label counts equally, that trade is not obviously
positive and is never measured.

:class:`ClassAwareBatchSampler` instead builds each batch to *contain* a quota
of rare positives while leaving the rest of the batch drawn from the natural
distribution:

* every batch gets at least ``min_positives_per_rare_label`` studies for each
  label below ``rare_threshold`` prevalence, drawn from that label's positive
  pool;
* the quota is filled greedily in order of *increasing* prevalence, so a study
  that happens to satisfy two quotas at once counts for both and the batch does
  not over-fill;
* the remainder is sampled uniformly, preserving the joint distribution;
* a per-epoch repeat cap (``max_repeat``) stops the same 40 fracture studies
  from appearing eight times an epoch, which is memorisation rather than
  balance.

The result is an epoch whose *marginal* label distribution is close to the true
one, but where the AUC surrogate always has pairs to work with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import numpy as np

from ..constants import NUM_TARGETS
from .dataset import StudyRecord, collate_studies
from .transforms import MEDIAL_LATERAL_SWAP, AugmentConfig, SeriesAugmenter

__all__ = [
    "StudyDataset",
    "ClassAwareBatchSampler",
    "build_dataloader",
    "SamplerConfig",
]


class StudyDataset:
    """Maps an index to a preprocessed :class:`StudyRecord`.

    ``load_fn`` is injected rather than hard-coded so the same class serves
    three cases without branching: real DICOM directories, a pre-built tensor
    cache (much faster, and what you should use after week 2), and synthetic
    records in tests.
    """

    def __init__(
        self,
        study_uids: Sequence[str],
        *,
        load_fn: Callable[[str], StudyRecord],
        labels: np.ndarray | None = None,
        augment: AugmentConfig | None = None,
        group_id: Sequence[str] | None = None,
        env_id: Sequence[int] | None = None,
        weak_labels: np.ndarray | None = None,
        weak_confidence: np.ndarray | None = None,
        teacher_logits: np.ndarray | None = None,
    ) -> None:
        self.uids = [str(u) for u in study_uids]
        self.load_fn = load_fn
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.augmenter = SeriesAugmenter(augment) if augment is not None else None
        self.group_id = list(group_id) if group_id is not None else list(self.uids)
        self.env_id = np.asarray(env_id) if env_id is not None else np.zeros(len(self.uids), int)

        # Auxiliary per-study supervision, aligned row-for-row with ``uids``.
        # These are what make ``weak_label`` and ``kd`` computable at all; every
        # run before them had to pass ``--disable weak_label,kd``.
        def _aux(a, name):
            if a is None:
                return None
            arr = np.asarray(a, dtype=np.float32)
            if arr.shape != (len(self.uids), NUM_TARGETS):
                raise ValueError(
                    f"{name} must be ({len(self.uids)}, {NUM_TARGETS}), got {arr.shape}"
                )
            return arr

        self.weak_labels = _aux(weak_labels, "weak_labels")
        self.weak_confidence = _aux(weak_confidence, "weak_confidence")
        self.teacher_logits = _aux(teacher_logits, "teacher_logits")

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int) -> StudyRecord:
        rec = self.load_fn(self.uids[i])
        if self.labels is not None and rec.labels is None:
            rec.labels = self.labels[i].copy()
        # Stamp the cohort-level indices so Group-DRO and IRM actually receive
        # them.  Without this the terms are registered, scheduled, and inert.
        if rec.group_index is None:
            rec.group_index = int(self.env_id[i])
        if rec.env_index is None:
            rec.env_index = int(self.env_id[i])
        for name in ("weak_labels", "weak_confidence", "teacher_logits"):
            arr = getattr(self, name)
            if arr is not None and getattr(rec, name) is None:
                setattr(rec, name, arr[i].copy())
        if self.augmenter is not None:
            # The mirror decision must be made ONCE per study.  Calling the
            # augmenter per series lets it flip series 0 and not series 1, and
            # permutes the medial/lateral labels once per flipped series -- so
            # with an even number of flips the labels come back unpermuted while
            # half the pixels are mirrored.  Sampling the flip here and passing
            # it down keeps pixels and labels in agreement by construction.
            flip = self.augmenter.sample_flip()
            for i, s in enumerate(rec.series):
                s.pixels, new_labels = self.augmenter(
                    s.pixels,
                    labels=rec.labels if i == 0 else None,
                    force_flip=flip,
                )
                if new_labels is not None:
                    rec.labels = np.asarray(new_labels, dtype=np.float32)
            if flip:
                # The mirror swaps medial and lateral, so *every* per-label
                # vector attached to the study has to be permuted, not just the
                # gold labels: a weak label or a teacher logit left unpermuted
                # would supervise the mirrored pixels with the unmirrored side.
                for name in ("weak_labels", "weak_confidence", "teacher_logits"):
                    v = getattr(rec, name)
                    if v is not None:
                        setattr(rec, name, np.asarray(v)[list(MEDIAL_LATERAL_SWAP)])
        return rec


@dataclass(slots=True)
class SamplerConfig:
    batch_size: int = 8
    rare_threshold: float = 0.10
    min_positives_per_rare_label: int = 1
    max_repeat: float = 3.0
    shuffle: bool = True
    drop_last: bool = True
    seed: int = 0


class ClassAwareBatchSampler:
    """Yields lists of dataset indices with a rare-positive quota per batch."""

    def __init__(self, labels: np.ndarray, cfg: SamplerConfig | None = None) -> None:
        self.cfg = cfg or SamplerConfig()
        y = np.nan_to_num(np.asarray(labels, dtype=np.float64))
        if y.ndim != 2:
            raise ValueError(f"labels must be (N, L), got {y.shape}")
        self.y = y
        self.n = y.shape[0]
        self.prevalence = y.mean(axis=0)
        self.rare = np.flatnonzero(
            (self.prevalence > 0) & (self.prevalence < self.cfg.rare_threshold)
        )
        # Rarest first: a study satisfying two quotas should be spent on the
        # scarcer one.
        self.rare = self.rare[np.argsort(self.prevalence[self.rare])]
        self.pools = {int(l): np.flatnonzero(y[:, l] > 0.5) for l in self.rare}
        self.rng = np.random.default_rng(self.cfg.seed)

    def __len__(self) -> int:
        b = self.cfg.batch_size
        return self.n // b if self.cfg.drop_last else math.ceil(self.n / b)

    def __iter__(self) -> Iterator[list[int]]:
        cfg = self.cfg
        order = self.rng.permutation(self.n) if cfg.shuffle else np.arange(self.n)
        cap = max(1, int(cfg.max_repeat))
        used = np.zeros(self.n, dtype=np.int32)
        cursor = 0

        for _ in range(len(self)):
            batch: list[int] = []

            for l in self.rare:
                pool = self.pools[int(l)]
                if pool.size == 0:
                    continue
                # "One study can satisfy several quotas at once" is expressed
                # by counting what the batch *already holds*, not by a
                # `covered` set.  A set marks a label fully satisfied after a
                # single co-occurring pick, which is only correct at quota 1;
                # at quota 2 the second positive is never drawn, and because
                # the short-circuit also skipped this block the have/need
                # accounting below was dead code that always saw have == 0.
                have = sum(1 for i in batch if self.y[i, l] > 0.5)
                need = cfg.min_positives_per_rare_label - have
                for _ in range(max(need, 0)):
                    eligible = pool[used[pool] < cap]
                    if eligible.size == 0:
                        eligible = pool  # cap reached everywhere: allow reuse
                    pick = int(self.rng.choice(eligible))
                    if pick in batch:
                        continue
                    batch.append(pick)
                    used[pick] += 1
                if len(batch) >= cfg.batch_size:
                    break

            while len(batch) < cfg.batch_size:
                if cursor >= self.n:
                    order = self.rng.permutation(self.n)
                    cursor = 0
                idx = int(order[cursor])
                cursor += 1
                if idx not in batch:
                    batch.append(idx)
                    used[idx] += 1

            yield batch[: cfg.batch_size]

    def epoch_prevalence(self) -> np.ndarray:
        """Realised label prevalence over one epoch of this sampler.

        Log this next to the true prevalence.  If a rare label's sampled
        prevalence is 6× its true one, you are not balancing, you are
        memorising, and the OOF AUC for its *co-occurring* labels will drift.
        """
        counts = np.zeros(self.y.shape[1])
        n = 0
        for batch in self:
            counts += self.y[batch].sum(axis=0)
            n += len(batch)
        return counts / max(n, 1)


def _seed_worker_rng(worker_id: int) -> None:
    """Give every DataLoader worker its own augmentation stream.

    ``SeriesAugmenter`` binds ``np.random.default_rng(seed)`` once, in the
    parent process.  ``fork`` copies that Generator's state into every worker
    and torch's own per-worker seeding only touches ``torch`` and the global
    ``random``/``np.random`` module state -- it cannot reach a Generator object
    living on the dataset.  The result was that all workers replayed an
    *identical* augmentation stream: with the shipped ``--workers 4``, four
    different studies in the same step received the same rotation, scale, bias
    field, noise draw and slice-dropout mask, dividing the effective
    augmentation diversity by four with nothing in the logs to show for it.

    Reseeding from ``torch.initial_seed()`` keeps the run reproducible: torch
    derives that per worker from the epoch's base seed.
    """
    import torch

    info = torch.utils.data.get_worker_info()
    ds = getattr(info, "dataset", None)
    ds = getattr(ds, "inner", ds)  # build_dataloader wraps the StudyDataset
    aug = getattr(ds, "augmenter", None)
    if aug is not None:
        aug.rng = np.random.default_rng([torch.initial_seed() % (2**32), worker_id])


def build_dataloader(
    dataset: StudyDataset,
    *,
    sampler_cfg: SamplerConfig | None = None,
    num_workers: int = 4,
    device: str = "cpu",
    balanced: bool = True,
    pin_memory: bool | None = None,
):
    """Wrap a :class:`StudyDataset` in a torch ``DataLoader``.

    ``collate_fn`` is :func:`kairos.data.dataset.collate_studies`, which pads
    the ragged (series, slice) dimensions and emits the boolean masks the model
    requires.  Batching happens in the sampler, so ``batch_size`` is left at
    ``None`` on the loader itself.
    """
    import torch
    from torch.utils.data import DataLoader

    cfg = sampler_cfg or SamplerConfig()
    if pin_memory is None:
        pin_memory = device != "cpu"

    class _Wrap(torch.utils.data.Dataset):
        #: Named, not just closed over, so ``_seed_worker_rng`` can reach the
        #: real dataset (and its augmenter) from inside a worker process.
        inner = dataset

        def __len__(self):
            return len(dataset)

        def __getitem__(self, i):
            return dataset[i]

    common = dict(
        num_workers=num_workers,
        collate_fn=lambda recs: collate_studies(recs, device="cpu"),
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker_rng if num_workers > 0 else None,
    )

    if balanced and dataset.labels is not None:
        return DataLoader(_Wrap(), batch_sampler=ClassAwareBatchSampler(
            dataset.labels, cfg), **common)

    return DataLoader(
        _Wrap(),
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        drop_last=cfg.drop_last,
        **common,
    )


def infer_prevalence(labels: np.ndarray) -> np.ndarray:
    """NaN-aware per-label prevalence, floored away from 0 and 1."""
    y = np.asarray(labels, dtype=np.float64)
    valid = np.isfinite(y)
    n = valid.sum(axis=0)
    p = np.where(n > 0, np.nansum(np.where(valid, y, 0.0), axis=0) / np.maximum(n, 1), 0.0)
    return np.clip(p, 1.0 / max(len(y), 1), 1.0 - 1.0 / max(len(y), 1))[:NUM_TARGETS]
