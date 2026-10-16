"""统一流水线兼容入口。"""

from .problem_pipeline import (
    ProblemBasedExperimentResult,
    load_yaml_config,
    run_problem_based_experiment,
)

__all__ = [
    "ProblemBasedExperimentResult",
    "load_yaml_config",
    "run_problem_based_experiment",
]
