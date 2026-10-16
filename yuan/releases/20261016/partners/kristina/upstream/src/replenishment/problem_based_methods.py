"""基于库存问题结构的周期复查 ``(s, S)`` 基准方法。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .problem_evaluation import InventoryEvaluator


@dataclass(frozen=True)
class ProblemBasedConfig:
    """problem-based baseline 的全部参数。"""

    target_service_level: float = 0.95
    lead_time: int = 1
    review_period: int = 1
    planning_horizon: int = 4
    validation_periods: int = 18
    z_min: float = 0.0
    z_max: float = 4.0
    z_step: float = 0.05
    integer_quantity: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.target_service_level <= 1:
            raise ValueError("目标服务率必须位于 (0, 1]。")
        if self.lead_time != 1 or self.review_period != 1:
            raise ValueError("当前研究固定使用 lead_time=1、review_period=1。")
        if self.planning_horizon != 4:
            raise ValueError("当前研究固定使用4期滚动规划。")
        if self.validation_periods < 4:
            raise ValueError("验证窗口至少需要4期。")
        if self.z_min < 0 or self.z_max < self.z_min or self.z_step <= 0:
            raise ValueError("安全系数 z 的搜索范围不合法。")


@dataclass(frozen=True)
class ProblemBasedArtifacts:
    """完成 train 内校准后冻结的模型产物。"""

    parameters: pd.DataFrame
    selected_z: float
    validation_summary: dict[str, object]
    fit_periods: int
    validation_periods: int
    feasible_on_validation: bool


class ProblemBasedMethods:
    """正态需求假设下、每期复查的 ``s=S`` 滚动补货基准。"""

    def __init__(self, config: ProblemBasedConfig | None = None) -> None:
        self.config = config or ProblemBasedConfig()
        self.artifacts_: ProblemBasedArtifacts | None = None

    def fit(self, train_demand: pd.DataFrame) -> "ProblemBasedMethods":
        """只使用 train 估计需求分布并选择安全系数。"""

        period_columns = self._validate_train_demand(train_demand)
        split = len(period_columns) - self.config.validation_periods
        if split < 2:
            raise ValueError("train 前段至少需要两个时期估计均值和方差。")

        fit_columns = period_columns[:split]
        validation_columns = period_columns[split:]
        statistics = self._statistics(train_demand, fit_columns)
        validation_demand = train_demand[["item_id", *validation_columns]].copy()
        evaluator = InventoryEvaluator(
            lead_time=self.config.lead_time,
            planning_horizon=self.config.planning_horizon,
        )

        candidates: list[tuple[float, pd.DataFrame, dict[str, object], bool]] = []
        full_window_name = (
            "first_4" if len(validation_columns) == 4 else f"full_{len(validation_columns)}"
        )
        for z in self._z_candidates():
            parameters = build_problem_based_parameters(
                statistics,
                safety_factor=float(z),
                lead_time=self.config.lead_time,
                review_period=self.config.review_period,
                integer_quantity=self.config.integer_quantity,
            )
            item_info = parameters[["item_id", "S"]].rename(
                columns={"S": "initial_inventory"}
            )
            result = evaluator.evaluate(validation_demand, item_info, parameters)
            first_four = result.summary["windows"]["first_4"]
            full_window = result.summary["windows"][full_window_name]
            feasible = (
                float(first_four["no_stockout_service_rate"])
                >= self.config.target_service_level
                and float(full_window["no_stockout_service_rate"])
                >= self.config.target_service_level
            )
            candidates.append((float(z), parameters, result.summary, feasible))

        feasible_candidates = [candidate for candidate in candidates if candidate[3]]
        if feasible_candidates:
            selected = min(
                feasible_candidates,
                key=lambda candidate: (
                    float(
                        candidate[2]["windows"][full_window_name]
                        ["inventory_demand_ratio"]
                    ),
                    float(candidate[2]["windows"]["first_4"]["inventory_demand_ratio"]),
                    candidate[0],
                ),
            )
        else:
            selected = max(
                candidates,
                key=lambda candidate: (
                    min(
                        float(candidate[2]["windows"]["first_4"]["no_stockout_service_rate"]),
                        float(
                            candidate[2]["windows"][full_window_name]
                            ["no_stockout_service_rate"]
                        ),
                    ),
                    -float(
                        candidate[2]["windows"][full_window_name]
                        ["inventory_demand_ratio"]
                    ),
                ),
            )

        selected_z, _, validation_summary, feasible = selected
        final_statistics = self._statistics(train_demand, period_columns)
        final_parameters = build_problem_based_parameters(
            final_statistics,
            safety_factor=selected_z,
            lead_time=self.config.lead_time,
            review_period=self.config.review_period,
            integer_quantity=self.config.integer_quantity,
        )
        self.artifacts_ = ProblemBasedArtifacts(
            parameters=final_parameters,
            selected_z=selected_z,
            validation_summary=validation_summary,
            fit_periods=len(fit_columns),
            validation_periods=len(validation_columns),
            feasible_on_validation=feasible,
        )
        return self

    def get_artifacts(self) -> ProblemBasedArtifacts:
        """返回冻结产物的副本。"""

        if self.artifacts_ is None:
            raise RuntimeError("problem_based_methods 尚未拟合。")
        return ProblemBasedArtifacts(
            parameters=self.artifacts_.parameters.copy(),
            selected_z=self.artifacts_.selected_z,
            validation_summary=dict(self.artifacts_.validation_summary),
            fit_periods=self.artifacts_.fit_periods,
            validation_periods=self.artifacts_.validation_periods,
            feasible_on_validation=self.artifacts_.feasible_on_validation,
        )

    def get_summary(self) -> dict[str, object]:
        """生成可以直接写入 JSON 的模型说明。"""

        artifacts = self.get_artifacts()
        protection_period = self.config.lead_time + self.config.review_period
        return {
            "name": "problem_based_methods",
            "objective": "minimize total ending inventory / total demand",
            "binding_constraint": "no-stockout SKU-period rate >= 95%",
            "formula": (
                "s_i = S_i = ceil((L+r)*mean_i + "
                "z*sqrt((L+r)*variance_i)); L=1, r=1"
            ),
            "protection_periods": protection_period,
            "selected_z": artifacts.selected_z,
            "fit_periods": artifacts.fit_periods,
            "validation_periods": artifacts.validation_periods,
            "feasible_on_validation": artifacts.feasible_on_validation,
            "validation": artifacts.validation_summary,
            "config": asdict(self.config),
        }

    def _z_candidates(self) -> np.ndarray:
        """生成包含上界的安全系数网格。"""

        stop = self.config.z_max + self.config.z_step / 2
        return np.arange(self.config.z_min, stop, self.config.z_step)

    @staticmethod
    def _statistics(demand: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """按 SKU 计算历史需求均值和样本方差。"""

        values = demand[columns].to_numpy(dtype=float)
        return pd.DataFrame(
            {
                "item_id": demand["item_id"].to_numpy(),
                "mean_demand": values.mean(axis=1),
                "variance_demand": values.var(axis=1, ddof=1),
            }
        )

    @staticmethod
    def _validate_train_demand(train_demand: pd.DataFrame) -> list[str]:
        """检查 train 宽表并返回按时期排序的需求列。"""

        if "item_id" not in train_demand.columns:
            raise ValueError("train_demand 缺少 item_id。")
        if train_demand["item_id"].isna().any() or train_demand["item_id"].duplicated().any():
            raise ValueError("train_demand 的 item_id 不能为空或重复。")
        columns = [column for column in train_demand if str(column).startswith("period_")]
        try:
            columns = sorted(columns, key=lambda column: int(str(column).split("_")[-1]))
        except ValueError as exc:
            raise ValueError("训练时期必须使用 period_1、period_2 等命名。") from exc
        values = train_demand[columns].apply(pd.to_numeric, errors="coerce")
        if len(columns) < 6 or values.isna().any().any() or (values < 0).any().any():
            raise ValueError("train 至少需要6期非负、无缺失需求。")
        return columns


def build_problem_based_parameters(
    statistics: pd.DataFrame,
    *,
    safety_factor: float,
    lead_time: int,
    review_period: int,
    integer_quantity: bool = True,
) -> pd.DataFrame:
    """根据图片中的保护期需求公式构造 ``s=S``。"""

    required = {"item_id", "mean_demand", "variance_demand"}
    if not required.issubset(statistics.columns):
        raise ValueError(f"需求统计量缺少字段：{sorted(required - set(statistics.columns))}")
    if safety_factor < 0 or lead_time < 1 or review_period < 1:
        raise ValueError("安全系数、提前期和复查周期必须合法。")

    protection_period = lead_time + review_period
    expected_demand = protection_period * statistics["mean_demand"]
    variance = protection_period * statistics["variance_demand"]
    level = (expected_demand + safety_factor * np.sqrt(variance.clip(lower=0))).clip(lower=0)
    if integer_quantity:
        level = np.ceil(level)

    return pd.DataFrame(
        {
            "item_id": statistics["item_id"].to_numpy(),
            "mean_demand": statistics["mean_demand"].to_numpy(dtype=float),
            "variance_demand": statistics["variance_demand"].to_numpy(dtype=float),
            "s": level.to_numpy(dtype=float),
            "S": level.to_numpy(dtype=float),
            "safety_factor": float(safety_factor),
        }
    )
