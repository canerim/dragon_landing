r"""The training loop: composes the curriculum, the losses and the optimisers.

Deliberate design choices, each of which fixes a specific failure:

* **The loss registry is data, not code.**  Every term is registered with the
  signature ``(outputs, batch, state) -> Tensor``; the loop multiplies by the
  schedule weight and sums.  Adding a term never touches the loop, and a term
  with weight 0 is skipped entirely rather than computed and multiplied by
  zero -- which matters because the OT and copula terms are the expensive ones
  and they are inactive for most of training.

* **Two optimisers, not one.**  The min-max block (backbone + heads +
  :math:`(a,b,\alpha)`) is driven by :class:`~kairos.optim.pesg.PESG` during the
  ranking stage, and by AdamW everywhere else.  Running AdamW on
  :math:`\alpha` is the single most common way to get "AUC-M didn't help".

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
    """Exponential moving average of the parameters, with bias correction.

    The bias correction (dividing by :math:`1-\\delta^t`) matters for the first
    few hundred steps: without it the EMA starts at the *initialisation* and an
    early evaluation reports a randomly-initialised model, which reads as a
    training bug.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.step = 0
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module, decay: float | None = None) -> None:
        d = self.decay if decay is None else decay
        self.step += 1
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        bias = 1.0 - self.decay**max(self.step, 1)
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_((v / bias).to(sd[k].dtype))

    def state_dict(self) -> dict:
        return {"decay": self.decay, "step": self.step, "shadow": self.shadow}

    def load_state_dict(self, sd: Mapping) -> None:
        self.decay = sd["decay"]
        self.step = sd["step"]
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
    use_pesg_in_ranking: bool = True
    pesg_lr: float = 3e-4
    pesg_gamma: float = 500.0
    gradient_surgery: str = "none"  # "none" | "aligned" | "pcgrad" | "cagrad"
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
            problems = validate_schedule(
                filtered, self.schedule, sample_batch=sample_batch, strict=False
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
        self.opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
        self.base_lrs = [g["lr"] for g in self.opt.param_groups]

        self.pesg: PESG | None = None
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
        stage, _ = self.plan.stage_at(self.step)
        warm = int(self.plan.total_steps * self.cfg.warmup_frac)
        mult = cosine_with_warmup(self.step, total=self.plan.total_steps, warmup=warm)
        mult *= stage.lr_scale
        for g, base in zip(self.opt.param_groups, self.base_lrs):
            g["lr"] = base * mult
        return mult

    def _maybe_switch_to_pesg(self, stage_name: str) -> None:
        wants = self.cfg.use_pesg_in_ranking and "ranking" in stage_name
        if wants and self.pesg is None:
            params = [p for p in self.model.parameters() if p.requires_grad]
            for term in self.loss_terms.values():
                mod = getattr(term, "module", None)
                if isinstance(mod, nn.Module):
                    params += [p for p in mod.parameters() if p.requires_grad]
            self.pesg = PESG(
                params,
                lr=self.cfg.pesg_lr,
                gamma=self.cfg.pesg_gamma,
                weight_decay=self.cfg.weight_decay,
            )
        elif not wants:
            self.pesg = None

    # ------------------------------------------------------------------ #

    def compute_losses(
        self, outputs: dict, batch, weights: Mapping[str, float]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total = torch.zeros((), device=self.device)
        logs: dict[str, float] = {}
        state = {"step": self.step, "epoch": self.epoch, "weights": weights}
        for name, w in weights.items():
            if w == 0.0 or name in self.disabled_terms:
                continue
            fn = self.loss_terms.get(name)
            if fn is None:
                continue
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
        logs["loss/total"] = float(total.detach())
        return total, logs

    def train_epoch(self, loader) -> dict[str, float]:
        self.model.train()
        stage, _ = self.plan.stage_at(self.step)
        self._maybe_switch_to_pesg(stage.name)
        if hasattr(self.model, "cfg"):
            self.model.cfg.enable_fine_pass = stage.enable_fine_pass
        if stage.freeze_backbone:
            for p in self.model.backbone.parameters():
                p.requires_grad_(False)

        agg: dict[str, float] = {}
        n = 0
        t0 = time.time()

        for i, batch in enumerate(loader):
            batch = _to_device(batch, self.device)
            weights = self.schedule(self.step)
            lr_mult = self._set_lr()

            with self._amp():
                outputs = self.model(batch, update_precision=True)
                loss, logs = self.compute_losses(outputs, batch, weights)
                loss = loss / self.cfg.accum_steps

            loss.backward()

            if (i + 1) % self.cfg.accum_steps == 0:
                gn = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
                logs["grad_norm"] = float(gn)
                if float(gn) > self.cfg.max_grad_norm_warn:
                    logs["grad_norm_spike"] = 1.0

                if self.pesg is not None:
                    self.pesg.step()
                    self.pesg.zero_grad(set_to_none=True)
                    for term in self.loss_terms.values():
                        mod = getattr(term, "module", None)
                        if hasattr(mod, "project"):
                            mod.project()
                elif self.asam is not None:
                    self.asam.ascent_step()
                    with self._amp():
                        out2 = self.model(batch, update_precision=False)
                        l2, _ = self.compute_losses(out2, batch, weights)
                    (l2 / self.cfg.accum_steps).backward()
                    self.asam.descent_step()
                else:
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)

                if self.ema is not None:
                    self.ema.update(self.model, decay=stage.ema_decay)
                self.step += 1

            logs["lr_mult"] = lr_mult
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1

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
        for batch in loader:
            batch = _to_device(batch, self.device)
            with self._amp():
                out = self.model(batch, run_fine=run_fine, update_precision=False)
            zs.append(out["logits"].float().cpu().numpy())
            if batch.targets is not None:
                ys.append(batch.targets.float().cpu().numpy())
            if batch.study_uid is not None:
                uids.extend(batch.study_uid)

        if backup is not None:
            self.model.load_state_dict(backup)

        z = np.concatenate(zs) if zs else np.zeros((0, NUM_TARGETS))
        y = np.concatenate(ys) if ys else np.zeros((0, NUM_TARGETS))
        return z, y, uids

    # ------------------------------------------------------------------ #

    def save(self, path: str | Path, manifest: RunManifest) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.state_dict(),
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
