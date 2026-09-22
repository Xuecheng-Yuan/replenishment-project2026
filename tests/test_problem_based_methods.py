"""problem_based_methods公式与防泄漏接口测试。"""

import inspect

import pandas as pd

from replenishment.problem_based_methods import (
    ProblemBasedConfig,
    ProblemBasedMethods,
    build_problem_based_parameters,
)


def test_protection_period_formula_uses_lead_plus_review() -> None:
    """L=1、r=1时，均值和方差都按2期保护期聚合。"""

    statistics = pd.DataFrame(
        {"item_id": [1], "mean_demand": [10.0], "variance_demand": [4.0]}
    )
    parameters = build_problem_based_parameters(
        statistics,
        safety_factor=1.0,
        lead_time=1,
        review_period=1,
    )

    # ceil(2*10 + sqrt(2*4)) = ceil(22.828...)=23
    assert parameters.loc[0, "s"] == 23.0
    assert parameters.loc[0, "S"] == 23.0


def test_fit_only_accepts_train_demand() -> None:
    """拟合接口不允许传入test，避免用test选择安全系数。"""

    parameters = list(inspect.signature(ProblemBasedMethods.fit).parameters)
    assert parameters == ["self", "train_demand"]


def test_train_is_split_then_full_history_is_refit() -> None:
    """前段用于估计、后段用于校准，z冻结后再用完整train重估。"""

    train = pd.DataFrame(
        {
            "item_id": [1, 2],
            **{
                f"period_{period}": [10 + period % 2, 20 + period % 3]
                for period in range(1, 11)
            },
        }
    )
    model = ProblemBasedMethods(
        ProblemBasedConfig(validation_periods=4, z_min=0, z_max=1, z_step=0.5)
    ).fit(train)
    artifacts = model.get_artifacts()

    assert artifacts.fit_periods == 6
    assert artifacts.validation_periods == 4
    assert len(artifacts.parameters) == 2
    assert (artifacts.parameters["s"] == artifacts.parameters["S"]).all()
