from .folds import FoldResult, FoldSpec, assignment_hash, fold_report, make_folds

__all__ = [
    "FoldResult", "FoldSpec", "assignment_hash", "fold_report", "make_folds",
    # Torch/pydicom-dependent; lazily imported below.
    "SeriesRecord", "StudyRecord", "load_study", "collate_studies",
    "build_inference_batch", "robust_normalise", "resample_to_spacing",
    "stack_neighbours", "select_slices", "foreground_mask",
    "classify_series", "infer_weighting", "infer_fat_sat",
]


def __getattr__(name):
    import importlib
    for mod in ("dataset", "sequence_taxonomy"):
        m = importlib.import_module(f".{mod}", __name__)
        if hasattr(m, name):
            return getattr(m, name)
    raise AttributeError(name)
