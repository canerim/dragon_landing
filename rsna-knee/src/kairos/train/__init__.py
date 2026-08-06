from .curriculum import CurriculumPlan, LossSchedule, Stage, default_plan, student_plan

__all__ = [
    "CurriculumPlan", "LossSchedule", "Stage", "default_plan", "student_plan",
    "Trainer", "TrainConfig", "EMA", "RunManifest",
]


def __getattr__(name):
    import importlib
    m = importlib.import_module(".loop", __name__)
    if hasattr(m, name):
        return getattr(m, name)
    raise AttributeError(name)
