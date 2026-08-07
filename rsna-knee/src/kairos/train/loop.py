r"""The training loop: composes the curriculum, the losses and the optimisers.

Deliberate design choices, each of which fixes a specific failure:

* **The loss registry is data, not code.**  Every term is registered with the
  signature ``(outputs, batch, state) -> Tensor``; the loop multiplies by the
  schedule weight and sums.  Adding a term never touches the loop, and a term
  with weight 0 is skipped entirely rather than computed and multiplied by
  zero -- which matters because the OT and copula terms are the expensive ones
  and they are inactive for most of training.

* **Two optimisers, split by role rather than by stage.**  AdamW owns
  :math:`\theta` for the whole run; :class:`~kairos.optim.pesg.PESG` owns the
  AUC min-max auxiliaries :math:`(a,b,\alpha)` and steps them exactly on the
  steps where ``auc_margin`` has a non-zero weight.  :math:`\theta` appears
  only in the minimisation, so AdamW is correct for it -- but :math:`\alpha`
  must be *ascended* and projected onto :math:`\alpha\ge 0`, and running AdamW
  on it is the single most common way to get "AUC-M didn't help".  Splitting by
  role rather than swapping the whole optimiser at a stage boundary also means
  no momentum is thrown away, no proximal reference is lost, and the learning
  rate schedule applies to both.

* **EMA is the model you evaluate.**  With rare labels and a small dataset the
  raw weights bounce; the EMA is reliably 0.002–0.005 macro-AUC better and
  costs one extra copy of the parameters.

* **Gradient surgery is applied to a slice of the graph, not all of it.**  See
  :class:`~kairos.optim.pesg.GradientSurgery` for why.

* **Everything that could differ between runs is recorded.**  The run manifest
  carries the git commit, the config, the *fold hash*, package versions, the
  seed, and the checkpoint SHA.  An OOF matrix that cannot be traced to a fold
  hash is not usable for ensembling and the loop refuses to write one.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..constants import NUM_TARGETS, TARGETS
from ..optim.pesg import ASAM, PESG, GradientSurgery, cosine_with_warmup
from .curriculum import CurriculumPlan, LossSchedule

__all__ = ["TrainConfig", "Trainer", "EMA", "RunManifest"]


# --------------------------------------------------------------------------- #


class EMA:
    r"""Exponential moving average of the parameters, with warmup.

    The shadow is initialised **from the current weights**, not from zero, so
    the running average

    .. math::
        \bar\theta_t = \delta^t\,\theta_0
                     + (1-\delta)\sum_{i=1}^{t}\delta^{t-i}\theta_i

    already has coefficients summing to one and needs **no bias correction**.
    An earlier revision applied Adam-style debiasing on top of that: at step 1
    with :math:`\delta = 0.999` it divided by :math:`10^{-3}`, i.e. it scaled
    every weight by a thousand.  The symptom was an EMA evaluation that looked
    like a broken model while the raw weights were fine -- and since the EMA is
    what we score and what we ship, the whole run was worthless.  A test now
    pins the invariant that an EMA of constant weights returns those weights.

    What bias correction was reaching for is real, though: at
    :math:`\delta=0.9999` the average is dominated by the initialisation for
    ~10k steps.  The fix is a warmup on the decay itself (timm's ModelEmaV2),

    .. math:: \delta_t = \min\Big(\delta,\ \frac{1+t}{10+t}\Big),

    which starts near 0.1 and anneals to the target, so the EMA tracks closely
    early and smooths heavily late.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, *,
                 warmup: bool = True) -> None:
        self.decay = decay
        self.warmup = warmup
        self.step = 0
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    def _decay_at(self, step: int, decay: float | None = None) -> float:
        d = self.decay if decay is None else decay
        if not self.warmup:
            return d
        return min(d, (1.0 + step) / (10.0 + step))

    @torch.no_grad()
    def update(self, model: nn.Module, decay: float | None = None) -> None:
        self.step += 1
        d = self._decay_at(self.step, decay)
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))

    def state_dict_for_checkpoint(self) -> dict:
        """The averaged weights, in the model's own key layout."""
        return {k: v.clone() for k, v in self.shadow.items()}

    def state_dict(self) -> dict:
        return {"decay": self.decay, "step": self.step, "warmup": self.warmup,
                "shadow": self.shadow}

    def load_state_dict(self, sd: Mapping) -> None:
        self.decay = sd["decay"]
        self.step = sd["step"]
        self.warmup = sd.get("warmup", True)
        self.shadow = {k: v.clone() for k, v in sd["shadow"].items()}


@dataclass(slots=True)
class TrainConfig:
    fold: int = 0
    seed: int = 20261022
    backbone_lr: float = 2e-5
    head_lr: float = 4e-4
    weight_decay: float = 0.02
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    accum_steps: int = 1
    amp_dtype: str = "bf16"  # "bf16" | "fp16" | "fp32"
    use_ema: bool = True
    use_asam: bool = False
    asam_rho: float = 0.5
    #: Step size for the AUC min-max auxiliaries ``(a, b, alpha)``, which PESG
    #: owns.  Deliberately larger than ``head_lr``: they are 36 scalars with a
    #: closed-form optimum that moves every time theta does, and a slow alpha
    #: is what makes the min-max surrogate lag behind the classifier.
    pesg_lr: float = 3e-3
    pesg_gamma: float = 500.0
    #: "none" | "aligned" | "pcgrad" | "cagrad".  Default "aligned", matching
    #: what docs/DESIGN.md §4.6 describes: measured cost in the shipped
    #: configuration is not detectable (19.77 vs 20.15 s/step), because the
    #: per-task gradients are measured *before* the main backward and only over
    #: the fusion/head block.
    gradient_surgery: str = "aligned"
    log_every: int = 50
    eval_every_epochs: int = 1
    out_dir: str = "runs/kairos"
    max_grad_norm_warn: float = 50.0


@dataclass(slots=True)
class RunManifest:
    """Everything needed to reproduce and to safely ensemble a run."""

    run_id: str
    git_commit: str
    config: dict
    fold_hash: str
    fold: int
    seed: int
    package_versions: dict
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    checkpoint_sha256: str | None = None
    best_macro_auc: float | None = None
    best_worst_label_auc: float | None = None
    #: Objective terms deliberately switched off for this run.  Recorded so a
    #: later comparison cannot mistake "we never trained that term" for "that
    #: term did not help".
    disabled_terms: list[str] = field(default_factory=list)
    notes: str = ""

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True, default=str))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #


class Trainer:
    """Curriculum-driven trainer for :class:`~kairos.models.system.KairosModel`."""

    def __init__(
        self,
        model: nn.Module,
        plan: CurriculumPlan,
        cfg: TrainConfig,
        *,
        loss_terms: Mapping[str, Callable],
        fold_hash: str,
        device: str | torch.device = "cpu",
        sample_batch: object | None = None,
        disabled_terms: Sequence[str] = (),
        validate: bool = True,
    ) -> None:
        self.model = model
        self.plan = plan
        self.cfg = cfg
        self.schedule = LossSchedule(plan)
        self.loss_terms = dict(loss_terms)
        self.device = torch.device(device)
        self.fold_hash = fold_hash
        self.disabled_terms = set(disabled_terms)

        if not fold_hash:
            raise ValueError(
                "fold_hash is required: an OOF matrix that cannot be traced to a "
                "split is not safe to ensemble"
            )

        # Refuse to start when the curriculum schedules an objective that
        # cannot be computed.  See train/objectives.py for why this is fatal
        # rather than a warning: the run would otherwise optimise a strictly
        # smaller objective and report a perfectly healthy loss curve.
        if validate:
            from .objectives import validate_schedule

            filtered = {
                k: v for k, v in self.loss_terms.items() if k not in self.disabled_terms
            }
            # One no-grad forward so the validator can check *output*
            # preconditions too, not just batch fields.  Failures here are
            # swallowed on purpose: a forward that cannot run will raise a much
            # clearer error at step 1 than a validation message would, and we
            # do not want a shape quirk in the caller's sample batch to block
            # construction.
            sample_outputs = None
            if sample_batch is not None:
                was_training = model.training
                model.eval()
                try:
                    with torch.no_grad():
                        sample_outputs = model(sample_batch, update_precision=False)
                except Exception:  # noqa: BLE001 - see comment above
                    sample_outputs = None
                finally:
                    model.train(was_training)
            problems = validate_schedule(
                filtered, self.schedule, sample_batch=sample_batch,
                sample_outputs=sample_outputs, strict=False,
            )
            hard = [p for p in problems if p.split("'")[1] not in self.disabled_terms]
            if hard:
                from .objectives import MissingObjectiveError

                raise MissingObjectiveError(
                    "the curriculum schedules objectives that cannot be computed:\n"
                    "  - " + "\n  - ".join(hard)
                    + "\n\nEither supply the missing batch fields, or pass the term "
                    "names in `disabled_terms` so the omission is recorded in the "
                    "run manifest instead of being silent."
                )

        groups = model.parameter_groups(
            backbone_lr=cfg.backbone_lr,
            head_lr=cfg.head_lr,
            weight_decay=cfg.weight_decay,
        )
        # Loss modules own real trainable state -- the SSL decoder and
        # projector, the copula's correlation factors, the chi-square dual
        # variable, the contrastive temperature, the KD projector.  None of it
        # is reachable from ``model.parameters()``, so an earlier revision left
        # all 40 of those tensors in no optimizer at all: stage S0 ran its
        # masked-token objective every step and never updated the decoder that
        # computes it.  They are added here, split by dimensionality so that a
        # temperature or a dual variable is not weight-decayed towards zero.
        # The saddle-point block is collected by *identity*, from the modules
        # that declare it, never from a per-tensor marker attribute.  Those do
        # not survive a device move: nn.Module._apply rebuilds every Parameter
        # when the shallow-copy check fails (it does for cpu->cuda, though not
        # for a dtype cast), and every attribute set on the old object is
        # dropped.  Routing on such a marker worked on CPU, passed its tests on
        # CPU, and on GPU quietly put alpha in AdamW.
        minimax, ascent = [], []
        for term in self.loss_terms.values():
            mod = getattr(term, "module", None)
            if hasattr(mod, "minimax_parameters"):
                desc, asc = mod.minimax_parameters()
                minimax.extend(desc)
                minimax.extend(asc)
                ascent.extend(asc)
        minimax_ids = {id(p) for p in minimax}

        obj_decay, obj_no_decay = [], []
        for name, p in self._objective_parameters():
            if id(p) in minimax_ids:
                continue  # owned by PESG, see below
            (obj_no_decay if p.ndim <= 1 else obj_decay).append(p)
        if obj_decay:
            groups.append({"params": obj_decay, "lr": cfg.head_lr,
                           "weight_decay": cfg.weight_decay})
        if obj_no_decay:
            groups.append({"params": obj_no_decay, "lr": cfg.head_lr,
                           "weight_decay": 0.0})
        self.n_objective_params = len(obj_decay) + len(obj_no_decay)

        self.opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
        self.base_lrs = [g["lr"] for g in self.opt.param_groups]

        # PESG owns the min-max auxiliaries (a, b, alpha) and nothing else.
        #
        # The earlier design swapped the *whole* optimiser for PESG during the
        # ranking stage.  Three things went wrong with that and all three are
        # silent: (i) PESG ran at a fixed lr, so the cosine decay and the
        # stage's lr_scale=0.3 were ignored for the entire stage; (ii) it was
        # rebuilt from ``self.pesg = None`` on the next stage boundary, throwing
        # away both its momentum and its proximal reference; (iii) it applied a
        # single SGD-momentum step size to a backbone whose lr was tuned for
        # Adam, 15x smaller.
        #
        # Restricting PESG to the saddle-point block removes all three.  It is
        # also where the min-max structure actually is: theta appears only in
        # the minimisation, so ordinary AdamW descent on theta is correct, while
        # alpha genuinely needs ascent-and-project and (a, b) genuinely need the
        # proximal anchor that keeps them from chasing a moving theta.
        self.minimax_params = minimax
        self.pesg: PESG | None = (
            PESG(minimax, lr=cfg.pesg_lr, gamma=cfg.pesg_gamma, weight_decay=0.0,
                 ascent_params=ascent)
            if minimax else None
        )
        self.pesg_base_lrs = (
            [g["lr"] for g in self.pesg.param_groups] if self.pesg else []
        )
        self._clip_params = [
            p for g in self.opt.param_groups for p in g["params"] if p.requires_grad
        ]
        self.asam = ASAM(self.opt, model, rho=cfg.asam_rho) if cfg.use_asam else None
        self.ema = EMA(model, decay=plan.stages[0].ema_decay) if cfg.use_ema else None

        self.surgery: GradientSurgery | None = None
        if cfg.gradient_surgery != "none":
            self.surgery = GradientSurgery(
                list(self._surgery_params()), mode=cfg.gradient_surgery
            )

        self.step = 0
        self.epoch = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------ #

    def _objective_parameters(self):
        """Yield ``(name, param)`` for every trainable loss-module parameter."""
        seen = set()
        for term_name, term in self.loss_terms.items():
            mod = getattr(term, "module", None)
            if not isinstance(mod, nn.Module):
                continue
            for pname, p in mod.named_parameters():
                if p.requires_grad and id(p) not in seen:
                    seen.add(id(p))
                    yield f"{term_name}.{pname}", p

    def _surgery_params(self) -> Iterable[nn.Parameter]:
        """Only the fusion + router + head: see GradientSurgery's docstring."""
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith(("fusion.", "router.", "head.", "linear_head.", "pool.")):
                yield p

    def _amp(self):
        if self.cfg.amp_dtype == "fp32" or self.device.type != "cuda":
            return contextlib.nullcontext()
        dt = torch.bfloat16 if self.cfg.amp_dtype == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dt)

    def _set_lr(self) -> float:
        """Warmup + cosine + stage scale, applied to *every* optimiser.

        PESG used to be left out of this loop, so the min-max block ran at a
        constant step size while theta decayed -- the auxiliaries then keep
        moving at full speed while the classifier freezes, and the ranking term
        slowly drifts away from the classifier it is supposed to be ranking.
        """
        stage, _ = self.plan.stage_at(self.step)
        warm = int(self.plan.total_steps * self.cfg.warmup_frac)
        mult = cosine_with_warmup(self.step, total=self.plan.total_steps, warmup=warm)
        mult *= stage.lr_scale
        for g, base in zip(self.opt.param_groups, self.base_lrs):
            g["lr"] = base * mult
        if self.pesg is not None:
            for g, base in zip(self.pesg.param_groups, self.pesg_base_lrs):
                g["lr"] = base * mult
        return mult

    def _pesg_wanted(self, weights: Mapping[str, float]) -> bool:
        """PESG steps exactly when the min-max term is live.

        Keyed on the *weight*, not on the stage name: ``auc_margin`` is on in
        both S3 and S4, and the old ``'ranking' in stage.name`` test silently
        stopped stepping alpha at the S3/S4 boundary while the term stayed in
        the loss.
        """
        return (
            self.pesg is not None
            and weights.get("auc_margin", 0.0) > 0.0
            and "auc_margin" not in self.disabled_terms
        )

    # ------------------------------------------------------------------ #

    def compute_losses(
        self, outputs: dict, batch, weights: Mapping[str, float],
        *, per_label: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None]:
        r"""Weighted sum of the scheduled terms.

        When ``per_label`` is set, also returns ``parts``: an ``(L,)`` tensor
        with :math:`\text{parts}_l = \sum_t w_t\, c_{t,l}`, where :math:`c_t` is
        term *t*'s exact additive decomposition over labels.  By construction
        ``parts.sum()`` equals the contribution those terms make to ``total``,
        which is precisely what
        :meth:`~kairos.optim.pesg.GradientSurgery.correct_` needs in order to
        cancel their plain-sum gradient and substitute the aligned one.

        The decomposition is accumulated *inside this loop*, not recomputed by
        the caller, so a term that was skipped here (weight zero, inapplicable
        to this batch, or non-finite) is skipped there too.  Recomputing it
        outside would let the two drift apart on exactly the batches where it
        matters -- and the resulting gradient error is silent.

        When a term has a decomposition we evaluate *only* the decomposition
        and take its sum as the scalar, rather than calling the term twice.
        Two reasons, and the second is the one that bites: it halves the work,
        and the terms are not all pure.  :class:`AUCMarginLoss` updates its
        prevalence EMA inside ``forward``, so a second call would decay that
        buffer twice per step -- a slow, invisible drift in the very quantity
        the min-max weights depend on.
        """
        # Anchored to a graph-connected tensor so that a batch in which every
        # term happens to be inapplicable still yields a backward-able zero.
        # A bare ``torch.zeros(())`` makes ``loss.backward()`` raise "does not
        # require grad", which turns a benign degenerate batch into a crash.
        anchor = outputs.get("logits")
        total = (
            anchor.sum() * 0.0 if torch.is_tensor(anchor) and anchor.requires_grad
            else torch.zeros((), device=self.device)
        )
        logs: dict[str, float] = {}
        parts: torch.Tensor | None = None
        state = {"step": self.step, "epoch": self.epoch, "weights": weights}
        for name, w in weights.items():
            if w == 0.0 or name in self.disabled_terms:
                continue
            fn = self.loss_terms.get(name)
            if fn is None:
                continue
            decomposed = per_label and getattr(fn, "per_label", None) is not None
            c = fn.per_label(outputs, batch, state) if decomposed else None
            if c is not None:
                val = c.sum()
            else:
                decomposed = False
                val = fn(outputs, batch, state)
            if val is None:
                continue
            if isinstance(val, dict):
                val = val["total"]
            if not torch.isfinite(val):
                logs[f"loss/{name}"] = float("nan")
                continue
            total = total + w * val
            logs[f"loss/{name}"] = float(val.detach())
            if decomposed:
                parts = w * c if parts is None else parts + w * c

        logs["loss/total"] = float(total.detach())
        return total, logs, parts

    def _clip(self) -> torch.Tensor:
        """Clip everything AdamW owns -- the model *and* the loss modules.

        Clipping only ``model.parameters()`` leaves the SSL decoder, the copula
        factors and the chi-square dual unclipped, and those are exactly the
        tensors whose gradients spike early in training.
        """
        return torch.nn.utils.clip_grad_norm_(self._clip_params, self.cfg.grad_clip)

    def _project_minimax(self) -> None:
        for term in self.loss_terms.values():
            mod = getattr(term, "module", None)
            if hasattr(mod, "project"):
                mod.project()

    def _optimizer_step(self, batch, weights, stage) -> dict[str, float]:
        """One optimiser step: clip, step (PESG + ASAM/AdamW), project, EMA."""
        logs: dict[str, float] = {}
        gn = self._clip()
        logs["grad_norm"] = float(gn)
        minimax_ok = all(
            p.grad is None or torch.isfinite(p.grad).all() for p in self.minimax_params
        )
        if not torch.isfinite(gn) or not minimax_ok:
            # A non-finite gradient must not be applied.  Dropping the step is
            # strictly better than poisoning every parameter with NaN, and the
            # counter makes the event visible instead of silent.
            logs["grad_nonfinite"] = 1.0
            self.opt.zero_grad(set_to_none=True)
            if self.pesg is not None:
                self.pesg.zero_grad(set_to_none=True)
            return logs
        if float(gn) > self.cfg.max_grad_norm_warn:
            logs["grad_norm_spike"] = 1.0

        # -- min-max block first, from the unperturbed gradient -------------- #
        if self._pesg_wanted(weights):
            self.pesg.step()
            self._project_minimax()
            logs["pesg/active"] = 1.0
        if self.pesg is not None:
            # Always cleared, stepped or not: a stale alpha gradient carried
            # into the next accumulation cycle would be applied twice.
            self.pesg.zero_grad(set_to_none=True)

        # -- theta ----------------------------------------------------------- #
        if self.asam is not None:
            self.asam.ascent_step()  # perturbs, then zeroes AdamW's grads
            if self.pesg is not None:
                self.pesg.zero_grad(set_to_none=True)  # ascent_step misses these
            with self._amp():
                out2 = self.model(batch, update_precision=False)
                l2, _, _ = self.compute_losses(out2, batch, weights)
            # NOT divided by accum_steps: this is one fresh microbatch used as
            # an unbiased estimate of the perturbed gradient, not one term of an
            # accumulation.  Dividing made the descent step accum_steps times
            # too small -- which looks like "ASAM just trains slower" and is
            # very hard to spot.  The perturbation itself is still built from
            # the fully accumulated gradient, which is the part that matters.
            l2.backward()
            gn2 = self._clip()  # the *descent* gradient is the one that is applied
            logs["asam/grad_norm"] = float(gn2)
            if torch.isfinite(gn2):
                self.asam.descent_step()
            else:
                self.asam.restore()
                self.opt.zero_grad(set_to_none=True)
                logs["asam/nonfinite"] = 1.0
            if self.cfg.accum_steps > 1:
                logs["asam/microbatch_only"] = 1.0
            if self.pesg is not None:
                self.pesg.zero_grad(set_to_none=True)
        else:
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

        if self.ema is not None:
            self.ema.update(self.model, decay=stage.ema_decay)
        self.step += 1
        return logs

    def train_epoch(self, loader) -> dict[str, float]:
        self.model.train()
        stage, _ = self.plan.stage_at(self.step)
        if hasattr(self.model, "cfg"):
            self.model.cfg.enable_fine_pass = stage.enable_fine_pass
        # Freezing must be *reversed* when the stage ends, or a single frozen
        # stage silently freezes the backbone for the rest of the run.
        want_frozen = bool(stage.freeze_backbone)
        if want_frozen != getattr(self, "_backbone_frozen", False):
            for p in self.model.backbone.parameters():
                p.requires_grad_(not want_frozen)
            self._backbone_frozen = want_frozen

        agg: dict[str, float] = {}
        n = 0
        pending = False
        t0 = time.time()

        for i, batch in enumerate(loader):
            batch = _to_device(batch, self.device)
            weights = self.schedule(self.step)
            lr_mult = self._set_lr()

            with self._amp():
                outputs = self.model(batch, update_precision=True)
                loss, logs, parts = self.compute_losses(
                    outputs, batch, weights, per_label=self.surgery is not None
                )
                loss = loss / self.cfg.accum_steps

            # Surgery has to happen *per microbatch*: by the time the
            # accumulation cycle closes in _optimizer_step, only the last
            # microbatch's graph would still be alive.  It is measured *before*
            # backward (cheap -- see GradientSurgery.prepare) and applied
            # *after*, once .grad holds the plain sum it has to correct.  The
            # 1/accum_steps scaling matches the division applied to ``loss``
            # above, so the cancellation stays exact under accumulation.
            delta = None
            if self.surgery is not None and parts is not None and parts.requires_grad:
                delta, slogs = self.surgery.prepare(
                    list(parts / self.cfg.accum_steps)
                )
                logs.update(slogs)

            loss.backward()
            if delta is not None:
                self.surgery.apply_(delta)

            pending = True
            if (i + 1) % self.cfg.accum_steps == 0:
                logs.update(self._optimizer_step(batch, weights, stage))
                pending = False

            logs["lr_mult"] = lr_mult
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1

        # Flush a partial accumulation cycle.  Without this, an epoch whose
        # batch count is not a multiple of accum_steps discards its trailing
        # gradients -- and when the loader is *shorter* than accum_steps (a
        # small fold, a debug run, the last shard of a sharded dataset) the
        # optimiser never steps at all and the run silently trains nothing.
        if pending and n:
            tail = self._optimizer_step(batch, weights, stage)
            for k, v in tail.items():
                agg[k] = agg.get(k, 0.0) + v
            agg["accum_flush"] = agg.get("accum_flush", 0.0) + 1

        self.epoch += 1
        out = {k: v / max(n, 1) for k, v in agg.items()}
        out["epoch_seconds"] = time.time() - t0
        out["stage"] = stage.name
        if self.pesg is not None:
            self.pesg.update_reference()
        self.history.append(out)
        return out

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict(self, loader, *, use_ema: bool = True, run_fine: bool | None = None
                ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Return ``(logits, targets, study_uids)`` for a loader."""
        backup = None
        if use_ema and self.ema is not None:
            backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.ema.copy_to(self.model)
        self.model.eval()

        zs, ys, uids = [], [], []
        n_dropped = 0
        for batch in loader:
            batch = _to_device(batch, self.device)
            with self._amp():
                out = self.model(batch, run_fine=run_fine, update_precision=False)
            # A study with no usable series produces a logit from zero padding.
            # Scoring that against its real targets does not measure the model;
            # it measures the head's bias.  Drop it, and count it, because a
            # systematic decode failure must not look like a bad epoch.
            keep = batch.study_valid
            sel = (keep.detach().cpu().numpy().astype(bool) if torch.is_tensor(keep)
                   else np.ones(out["logits"].shape[0], dtype=bool))
            n_dropped += int((~sel).sum())
            zs.append(out["logits"].float().cpu().numpy()[sel])
            if batch.targets is not None:
                ys.append(batch.targets.float().cpu().numpy()[sel])
            if batch.study_uid is not None:
                uids.extend([u for u, k in zip(batch.study_uid, sel) if k])

        if backup is not None:
            self.model.load_state_dict(backup)
        # An attribute rather than a history entry: history is one dict per
        # *epoch* and a partial record there would corrupt every consumer of
        # history.json.  04_train.py prints this after each evaluation.
        self.n_unusable_predicted = n_dropped

        z = np.concatenate(zs) if zs else np.zeros((0, NUM_TARGETS))
        y = np.concatenate(ys) if ys else np.zeros((0, NUM_TARGETS))
        return z, y, uids

    # ------------------------------------------------------------------ #

    def save(self, path: str | Path, manifest: RunManifest) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The checkpoint must contain the weights that produced the reported
        # score.  ``predict`` evaluates the EMA, so storing the raw weights
        # under "model" ships a model that was never measured -- the OOF matrix
        # and the checkpoint would describe different networks, and the
        # ensemble weights fitted on the former would be applied to the latter.
        raw = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        scored = raw
        if self.ema is not None:
            backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.ema.copy_to(self.model)
            scored = {k: v.detach().cpu().clone()
                      for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(backup)

        payload = {
            "model": scored,
            "model_raw": raw,
            "scored_weights": "ema" if self.ema is not None else "raw",
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "optimizer": self.opt.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "config": asdict(self.cfg),
            "fold_hash": self.fold_hash,
            "targets": list(TARGETS),
            "disabled_terms": sorted(self.disabled_terms),
            "model_config": _model_config_dict(self.model),
        }
        torch.save(payload, path)
        sha = _sha256(path)
        manifest.checkpoint_sha256 = sha
        manifest.disabled_terms = sorted(self.disabled_terms)
        manifest.finished_at = time.time()
        manifest.write(path.with_suffix(".manifest.json"))
        return sha

    def load(self, path: str | Path, *, strict: bool = True) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("fold_hash") != self.fold_hash:
            raise ValueError(
                "fold_hash mismatch between checkpoint and trainer: refusing to load.\n"
                f"  checkpoint: {payload.get('fold_hash')}\n  trainer:    {self.fold_hash}"
            )
        if list(payload.get("targets", TARGETS)) != list(TARGETS):
            raise ValueError("checkpoint was trained with a different label order")
        self.model.load_state_dict(payload["model"], strict=strict)
        if self.ema is not None and payload.get("ema"):
            self.ema.load_state_dict(payload["ema"])
        self.step = int(payload.get("step", 0))
        self.epoch = int(payload.get("epoch", 0))


def _to_device(batch, device):
    for f in batch.__slots__ if hasattr(batch, "__slots__") else []:
        v = getattr(batch, f, None)
        if torch.is_tensor(v):
            setattr(batch, f, v.to(device, non_blocking=True))
    return batch


def bf16_available() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def suggest_accum(target_studies: int, per_device: int, world_size: int = 1) -> int:
    """Gradient accumulation so the effective batch matches the recipe.

    Effective batch size drives the AUC surrogate's variance far more than it
    drives the supervised loss: at 16 studies/batch the rarest label sees a
    positive in one batch out of eight.  Keep the effective batch at 48–64
    studies even if that means 8 accumulation steps.

    **Caveat with ASAM.**  ASAM's second (descent) pass re-runs the model on one
    microbatch, so with ``accum_steps > 1`` the gradient that is actually
    applied comes from ``per_device`` studies, not from the 48–64 this function
    is sizing for -- only the sharpness *perturbation* uses the full
    accumulation.  Running the second pass over every microbatch would restore
    it at another full backward per microbatch, which is not worth it; the
    trade is deliberate, and ``_optimizer_step`` logs ``asam/microbatch_only``
    so a run made under it is identifiable afterwards.  If the AUC-surrogate
    variance is the thing you are buying, prefer ``--asam`` with ``--accum 1``
    and a larger per-device batch.
    """
    return max(1, math.ceil(target_studies / max(per_device * world_size, 1)))


def _model_config_dict(model: nn.Module) -> dict:
    """Serialise the model config so a checkpoint can be reloaded standalone.

    The Kaggle notebook reads this to rebuild the architecture without needing
    the training config file: a checkpoint that cannot describe its own
    architecture is a checkpoint that will be loaded with the wrong one.
    """
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        return {}
    from dataclasses import asdict, is_dataclass

    if not is_dataclass(cfg):
        return {}
    out = asdict(cfg)
    # BackboneSpec is nested; asdict already flattened it into a plain dict.
    return out
