"""Loss functions.  Torch-dependent; imported lazily so the numpy-only parts of
the package (folds, metrics, ensembling, submission) work without torch."""

__all__ = [
    "AUCMarginLoss", "PartialAUCLoss", "PairwiseRankQueue", "soft_topk_weights",
    "AsymmetricLoss", "SoftJaccardBCE", "GaussianCopulaNLL",
    "PhraseSliceOT", "sinkhorn_log", "unbalanced_sinkhorn_log", "sinkhorn_divergence",
    "SoftContrastive", "ReportDistillation", "ReportShortcutRegulariser",
    "GroupDRO", "CVaRLoss", "ChiSquareDRO", "IRMPenalty",
]


def __getattr__(name):
    import importlib
    for mod in ("auc", "supervised", "ot", "multimodal", "robust"):
        m = importlib.import_module(f".{mod}", __name__)
        if hasattr(m, name):
            return getattr(m, name)
    raise AttributeError(name)
