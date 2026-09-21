"""仅使用训练集拟合的均值–方差 baseline。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .evaluation import InventoryEvaluator
from .optimization import build_mean_variance_base_stock


@dataclass(frozen=True)
class BaselineConfig:
    """baseline 的全部可调参数。

    ``validation_periods=18`` 表示 60 期 train 的前 42 期用于估计，
    后 18 期仅用于选择安全系数。任何 test 字段都不属于该配置或拟合接口。
    """

    target_service_level: float = 0.95
    lead_time: int = 1
    validation_periods: int = 18
    z_min: float = 0.0
    z_max: float = 4.0
    z_step: float = 0.05
    integer_quantity: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.target_service_level <= 1:
            raise ValueError("目标服务水平必须位于 (0, 1]。")
        if self.lead_time < 1:
            raise ValueError("提前期必须至少为 1。")
        if self.validation_periods < 1:
            raise ValueError("验证期数量必须至少为 1。")
        if self.z_min < 0 or self.z_max < self.z_min or self.z_step <= 0:
            raise ValueError("安全系数 z 的搜索范围不合法。")


@dataclass(frozen=True)
class BaselineArtifacts:
    """训练完成并冻结的 baseline 产物。"""

    parameters: pd.DataFrame
    train_statistics: pd.DataFrame
    selected_z: float
    validation_summary: dict[str, object]
    fit_periods: int
    validation_periods: int


class MeanVarianceBaseline:
    """每个 SKU 一个静态 base-stock、全体 SKU 共用安全系数的 baseline。"""

    def __init__(self, config: BaselineConfig | None = None) -> None:
        self.config = config or BaselineConfig()
        self.artifacts_: BaselineArtifacts | None = None

    def fit(self, train_demand: pd.DataFrame) -> "MeanVarianceBaseline":
        """只接收 train 需求并完成估计、验证校准和最终重估。

        接口刻意不接受 test 数据、test 初始库存或 test 指标，避免调参阶段发生泄漏。
        """

        period_columns = self._validate_train_demand(train_demand)
        split_index = len(period_columns) - self.config.validation_periods
        if split_index < 2:
            raise ValueError("训练窗口至少需要两个时期，且必须为验证窗口保留时期。")

        fit_columns = period_columns[:split_index]
        validation_columns = period_columns[split_index:]

        # 只用 train 的前段估计候选目标库存。
        calibration_statistics = self._statistics(train_demand, fit_columns)
        validation_demand = train_demand[["item_id", *validation_columns]].copy()
        evaluator = InventoryEvaluator(lead_time=self.config.lead_time)

        selected_z = self.config.z_max
        selected_summary: dict[str, object] | None = None

        # 从小到大搜索 z，选择第一个使 train 内部验证期达到目标服务水平的候选。
        for candidate_z in self._z_candidates():
            candidate_parameters = build_mean_variance_base_stock(
                calibration_statistics,
                safety_factor=float(candidate_z),
                lead_time=self.config.lead_time,
                integer_quantity=self.config.integer_quantity,
            )

            # train 内部验证从候选目标库存开始，以排除任意初始库存的干扰。
            validation_initial_inventory = candidate_parameters[["item_id", "S"]].rename(
                columns={"S": "initial_inventory"}
            )
            validation_result = evaluator.evaluate(
                validation_demand,
                validation_initial_inventory,
                candidate_parameters,
            )
            selected_z = float(candidate_z)
            selected_summary = validation_result.summary
            if (
                float(validation_result.summary["overall_service_level"])
                >= self.config.target_service_level
            ):
                break

        if selected_summary is None:  # 防御性检查；正常 z 网格至少包含一个候选。
            raise RuntimeError("安全系数搜索没有产生任何候选结果。")

        # z 已锁定后，用完整 60 期 train 重估 SKU 均值和标准差。
        # 此处仍未读取 test，因此最终策略参数不可能依赖 test 的需求分布。
        final_statistics = self._statistics(train_demand, period_columns)
        final_parameters = build_mean_variance_base_stock(
            final_statistics,
            safety_factor=selected_z,
            lead_time=self.config.lead_time,
            integer_quantity=self.config.integer_quantity,
        )

        self.artifacts_ = BaselineArtifacts(
            parameters=final_parameters,
            train_statistics=final_statistics,
            selected_z=selected_z,
            validation_summary=selected_summary,
            fit_periods=len(fit_columns),
            validation_periods=len(validation_columns),
        )
        return self

    def get_artifacts(self) -> BaselineArtifacts:
        """返回冻结的训练产物副本。"""

        if self.artifacts_ is None:
            raise RuntimeError("baseline 尚未拟合。")
        return BaselineArtifacts(
            parameters=self.artifacts_.parameters.copy(),
            train_statistics=self.artifacts_.train_statistics.copy(),
            selected_z=self.artifacts_.selected_z,
            validation_summary=dict(self.artifacts_.validation_summary),
            fit_periods=self.artifacts_.fit_periods,
            validation_periods=self.artifacts_.validation_periods,
        )

    def get_summary(self) -> dict[str, object]:
        """返回可写入 JSON 的模型摘要。"""

        artifacts = self.get_artifacts()
        return {
            "model": "mean_variance_base_stock",
            "formula": "B_i = ceil(mean_i * L + z * std_i * sqrt(L)); s_i = S_i = B_i",
            "selected_z": artifacts.selected_z,
            "fit_periods": artifacts.fit_periods,
            "validation_periods": artifacts.validation_periods,
            "validation": artifacts.validation_summary,
            "config": asdict(self.config),
        }

    def _z_candidates(self) -> np.ndarray:
        """生成包含端点的安全系数搜索网格。"""

        count = int(np.floor((self.config.z_max - self.config.z_min) / self.config.z_step))
        candidates = self.config.z_min + np.arange(count + 1) * self.config.z_step
        if candidates[-1] < self.config.z_max:
            candidates = np.append(candidates, self.config.z_max)
        return candidates

    @staticmethod
    def _statistics(demand: pd.DataFrame, period_columns: list[str]) -> pd.DataFrame:
        """按 SKU 计算给定历史窗口的均值、方差和标准差。"""

        values = demand[period_columns].to_numpy(dtype=float)
        return pd.DataFrame(
            {
                "item_id": demand["item_id"].to_numpy(),
                "mean_demand": values.mean(axis=1),
                "variance_demand": values.var(axis=1, ddof=1),
                "std_demand": values.std(axis=1, ddof=1),
            }
        )

    @staticmethod
    def _validate_train_demand(train_demand: pd.DataFrame) -> list[str]:
        """验证训练宽表，并返回按时间排序的时期列。"""

        if "item_id" not in train_demand.columns:
            raise ValueError("train_demand 缺少 item_id。")
        if train_demand["item_id"].isna().any() or train_demand["item_id"].duplicated().any():
            raise ValueError("train_demand 的 item_id 不能为空或重复。")

        period_columns = [
            column for column in train_demand.columns if str(column).startswith("period_")
        ]
        try:
            period_columns = sorted(
                period_columns, key=lambda column: int(str(column).split("_")[-1])
            )
        except ValueError as exc:
            raise ValueError("训练时期必须使用 period_1、period_2 等命名。") from exc
        if len(period_columns) < 3:
            raise ValueError("baseline 至少需要三个训练时期。")

        values = train_demand[period_columns].apply(pd.to_numeric, errors="coerce")
        if values.isna().any().any():
            raise ValueError("训练需求存在缺失值或非数值。")
        if (values < 0).any().any():
            raise ValueError("训练需求不能为负数。")
        return period_columns
