"""旧导入路径兼容层；正式模型为problem_based_methods。"""

from .problem_based_methods import (
    ProblemBasedArtifacts,
    ProblemBasedConfig,
    ProblemBasedMethods,
    build_problem_based_parameters,
)

BaselineArtifacts = ProblemBasedArtifacts
BaselineConfig = ProblemBasedConfig
MeanVarianceBaseline = ProblemBasedMethods

__all__ = [
    "BaselineArtifacts",
    "BaselineConfig",
    "MeanVarianceBaseline",
    "ProblemBasedArtifacts",
    "ProblemBasedConfig",
    "ProblemBasedMethods",
    "build_problem_based_parameters",
]
