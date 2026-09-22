"""旧导入路径兼容层；库存参数由problem_based_methods统一构造。"""

from __future__ import annotations

import pandas as pd

from .problem_based_methods import build_problem_based_parameters


def build_mean_variance_base_stock(
    statistics: pd.DataFrame,
    *,
    safety_factor: float,
    lead_time: int,
    integer_quantity: bool = True,
) -> pd.DataFrame:
    """兼容旧函数名，按当前固定复查周期 ``r=1`` 构造参数。"""

    prepared = statistics.copy()
    if "variance_demand" not in prepared and "std_demand" in prepared:
        prepared["variance_demand"] = prepared["std_demand"] ** 2
    return build_problem_based_parameters(
        prepared,
        safety_factor=safety_factor,
        lead_time=lead_time,
        review_period=1,
        integer_quantity=integer_quantity,
    )


def validate_policy_parameters(parameters: pd.DataFrame) -> None:
    """检查策略参数满足 ``S >= s >= 0``。"""

    required = {"item_id", "s", "S"}
    if not required.issubset(parameters.columns):
        raise ValueError(f"策略参数缺少字段：{sorted(required - set(parameters.columns))}")
    numeric = parameters[["s", "S"]].apply(pd.to_numeric, errors="coerce")
    if parameters["item_id"].isna().any() or parameters["item_id"].duplicated().any():
        raise ValueError("item_id不能为空或重复。")
    if numeric.isna().any().any() or (numeric < 0).any().any():
        raise ValueError("s和S必须为非负数且不能缺失。")
    if (numeric["S"] < numeric["s"]).any():
        raise ValueError("策略参数必须满足S >= s。")
