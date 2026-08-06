"""KAIROS -- Knee Abnormality Inference with Report-Optimised Supervision.

A multimodal, geometry-aware, adaptive-compute system for the RSNA Knee
Abnormality Detection challenge.  See ``docs/DESIGN.md`` for the method and
``docs/PLAYBOOK.md`` for the week-by-week execution plan.

The numpy-only surface (folds, metrics, ensembling, calibration, submission)
imports without torch so it can run inside a lightweight validation job; the
torch-dependent surface is lazily imported.
"""

from .constants import NUM_TARGETS, SUBMISSION_COLUMNS, TARGETS, Target

__version__ = "1.0.0"

__all__ = ["TARGETS", "NUM_TARGETS", "SUBMISSION_COLUMNS", "Target", "__version__"]
