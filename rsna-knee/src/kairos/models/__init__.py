"""Model components.  Torch-dependent; see ``losses/__init__.py`` for the same
lazy-import rationale."""

__all__ = [
    "KairosConfig", "KairosModel", "StudyBatch", "BackboneSpec", "build_backbone",
    "SliceTransformer", "SelectiveScanAggregator", "LabelQueryPool",
    "CrossSequenceFusion", "LabelExpertRouter", "LabelHeadBank",
    "SNGPHead", "EvidentialHead", "GumbelTopKSelector", "ConfidenceGate",
    "PonderHalting", "OntologyEmbedding", "PoincareBall",
    "PhysicalPositionalEncoding", "AcquisitionFiLM", "AcquisitionContext",
    "build_context_vector",
]


def __getattr__(name):
    import importlib
    for mod in ("system", "backbones", "aggregator", "label_queries", "heads",
                "adaptive", "hyperbolic", "encoding"):
        m = importlib.import_module(f".{mod}", __name__)
        if hasattr(m, name):
            return getattr(m, name)
    raise AttributeError(name)
