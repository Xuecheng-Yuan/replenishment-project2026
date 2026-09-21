"""补货策略参数的构造与约束检查。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_mean_variance_base_stock(
    statistics: pd.DataFrame,
    *,
    safety_factor: float,
    lead_time: int,
    integer_quantity: bool = True,
) -> pd.DataFrame:
    """根据均值–方差公式生成每个 SKU 的静态 base-stock 水平。

    公式为 ``B_i = mean_i * L + z * std_i * sqrt(L)``。
    当前没有固定订货成本，因此令 ``s_i = S_i = B_i``。
    """

    required = {"item_id", "mean_demand", "std_demand"}
    missing = required - set(statistics.columns)
    if missing:
        raise ValueError(f"需求统计量缺少字段：{sorted(missing)}")
    if safety_factor < 0:
        raise ValueError("安全系数 safety_factor 不能为负数。")
    if lead_time < 1:
        raise ValueError("提前期 lead_time 必须至少为 1。")

    level = (
        statistics["mean_demand"] * lead_time
        + safety_factor * statistics["std_demand"] * np.sqrt(lead_time)
    ).clip(lower=0)
    if integer_quantity:
        level = np.ceil(level)

    parameters = pd.DataFrame(
        {
            "item_id": statistics["item_id"].to_numpy(),
            "s": level.to_numpy(dtype=float),
            "S": level.to_numpy(dtype=float),
            "safety_factor": float(safety_factor),
        }
    )
    validate_policy_parameters(parameters)
    return parameters


def validate_policy_parameters(parameters: pd.DataFrame) -> None:
    """检查 (s, S) 策略参数是否完整且满足 ``S >= s >= 0``。"""

    required = {"item_id", "s", "S"}
    missing = required - set(parameters.columns)
    if missing:
        raise ValueError(f"策略参数缺少字段：{sorted(missing)}")
    if parameters["item_id"].isna().any() or parameters["item_id"].duplicated().any():
        raise ValueError("策略参数中的 item_id 不能为空或重复。")

    numeric = parameters[["s", "S"]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("s 或 S 存在缺失值或非数值。")
    if (numeric < 0).any().any():
        raise ValueError("s 和 S 不能为负数。")
    if (numeric["S"] < numeric["s"]).any():
        raise ValueError("策略参数必须满足 S >= s。")

