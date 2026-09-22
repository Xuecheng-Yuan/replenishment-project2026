"""4期滚动补货仿真，以及两种服务率和库存需求比计算。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvaluationResult:
    """一次库存仿真的SKU指标、汇总指标和可选轨迹。"""

    per_item: pd.DataFrame
    summary: dict[str, object]
    trajectory: pd.DataFrame | None = None


class InventoryEvaluator:
    """按“到货—需求—下单”顺序执行4期滚动计划。"""

    def __init__(self, lead_time: int = 1, planning_horizon: int = 4) -> None:
        if lead_time != 1:
            raise ValueError("当前基准固定使用1期提前期。")
        if planning_horizon != 4:
            raise ValueError("当前基准固定使用4期滚动规划。")
        self.lead_time = lead_time
        self.planning_horizon = planning_horizon

    def evaluate(
        self,
        demand: pd.DataFrame,
        item_info: pd.DataFrame,
        parameters: pd.DataFrame,
        *,
        record_trajectory: bool = False,
    ) -> EvaluationResult:
        """冻结参数后仿真；每次生成4期计划，但只执行第一期订单。"""

        data, period_columns = self._align_inputs(demand, item_info, parameters)
        demand_values = data[period_columns].to_numpy(dtype=float)
        n_items, n_periods = demand_values.shape
        item_ids = data["item_id"].to_numpy()
        categories = data["category"].to_numpy() if "category" in data else None

        on_hand = data["initial_inventory"].to_numpy(dtype=float, copy=True)
        reorder_point = data["s"].to_numpy(dtype=float)
        order_up_to = data["S"].to_numpy(dtype=float)
        mean_demand = data["mean_demand"].to_numpy(dtype=float)
        pipeline = np.zeros((self.lead_time, n_items), dtype=float)

        sales = np.zeros_like(demand_values)
        lost_sales = np.zeros_like(demand_values)
        ending_inventory = np.zeros_like(demand_values)
        orders = np.zeros_like(demand_values)
        trajectory_parts: list[pd.DataFrame] = []

        for period in range(n_periods):
            # 上期订单先到货，本期需求随后发生。
            arrival = pipeline[0].copy()
            on_hand += arrival
            pipeline[0] = 0.0
            opening_inventory = on_hand.copy()
            period_demand = demand_values[:, period]
            period_sales = np.minimum(on_hand, period_demand)
            period_lost = period_demand - period_sales
            on_hand -= period_sales

            # 生成 q_t,...,q_{t+3}，但真实系统只执行 q_t。
            plan = self.plan_orders(
                on_hand=on_hand,
                pipeline=pipeline,
                reorder_point=reorder_point,
                order_up_to=order_up_to,
                expected_demand=mean_demand,
            )
            first_order = plan[:, 0]
            pipeline[-1] += first_order

            sales[:, period] = period_sales
            lost_sales[:, period] = period_lost
            ending_inventory[:, period] = on_hand
            orders[:, period] = first_order

            if record_trajectory:
                part = pd.DataFrame(
                    {
                        "item_id": item_ids,
                        "period": period + 1,
                        "opening_inventory": opening_inventory,
                        "arrival": arrival,
                        "demand": period_demand,
                        "sales": period_sales,
                        "lost_sales": period_lost,
                        "ending_inventory": on_hand.copy(),
                        "executed_order": first_order,
                    }
                )
                for step in range(self.planning_horizon):
                    part[f"planned_order_{step + 1}"] = plan[:, step]
                trajectory_parts.append(part)

        window_lengths = [min(4, n_periods)]
        if n_periods not in window_lengths:
            window_lengths.append(n_periods)

        summaries: dict[str, dict[str, object]] = {}
        per_item_parts: list[pd.DataFrame] = []
        for length in window_lengths:
            name = "first_4" if length == 4 else f"full_{length}"
            summaries[name] = self._aggregate_metrics(
                demand_values[:, :length],
                sales[:, :length],
                lost_sales[:, :length],
                ending_inventory[:, :length],
                orders[:, :length],
                categories,
            )
            per_item_parts.append(
                self._per_item_metrics(
                    name,
                    item_ids,
                    categories,
                    demand_values[:, :length],
                    sales[:, :length],
                    lost_sales[:, :length],
                    ending_inventory[:, :length],
                    orders[:, :length],
                )
            )

        return EvaluationResult(
            per_item=pd.concat(per_item_parts, ignore_index=True),
            summary={
                "objective_direction": "min",
                "binding_service_metric": "no_stockout_service_rate",
                "windows": summaries,
            },
            trajectory=(pd.concat(trajectory_parts, ignore_index=True) if trajectory_parts else None),
        )

    def plan_orders(
        self,
        *,
        on_hand: np.ndarray,
        pipeline: np.ndarray,
        reorder_point: np.ndarray,
        order_up_to: np.ndarray,
        expected_demand: np.ndarray,
    ) -> np.ndarray:
        """用期望需求向前规划4步；调用方只能执行返回值第一列。"""

        projected_on_hand = np.asarray(on_hand, dtype=float).copy()
        projected_pipeline = np.asarray(pipeline, dtype=float).copy()
        plan = np.zeros((len(projected_on_hand), self.planning_horizon), dtype=float)

        for step in range(self.planning_horizon):
            inventory_position = projected_on_hand + projected_pipeline.sum(axis=0)
            order = np.where(
                inventory_position <= reorder_point,
                np.maximum(0.0, order_up_to - inventory_position),
                0.0,
            )
            plan[:, step] = order
            projected_pipeline[-1] += order

            # 推演到下一决策点，未来需求只能使用 train 估计的期望值。
            projected_on_hand += projected_pipeline[0]
            projected_pipeline[0] = 0.0
            projected_on_hand = np.maximum(0.0, projected_on_hand - expected_demand)

        return plan

    @staticmethod
    def _aggregate_metrics(
        demand: np.ndarray,
        sales: np.ndarray,
        lost_sales: np.ndarray,
        ending_inventory: np.ndarray,
        orders: np.ndarray,
        categories: np.ndarray | None,
    ) -> dict[str, object]:
        """先汇总分子分母，再计算总体与类别指标。"""

        total_demand = float(demand.sum())
        total_inventory = float(ending_inventory.sum())
        no_stockout_rate = float((lost_sales <= 1e-9).mean())
        result: dict[str, object] = {
            "n_items": int(demand.shape[0]),
            "n_periods": int(demand.shape[1]),
            "inventory_demand_ratio": (
                total_inventory / total_demand if total_demand > 0 else float("inf")
            ),
            "positive_inventory_service_rate": float((ending_inventory > 0).mean()),
            "no_stockout_service_rate": no_stockout_rate,
            "constraint_met": no_stockout_rate >= 0.95,
            "total_ending_inventory": total_inventory,
            "total_demand": total_demand,
            "total_sales": float(sales.sum()),
            "total_lost_sales": float(lost_sales.sum()),
            "total_order_quantity": float(orders.sum()),
        }
        if categories is not None:
            category_metrics: dict[str, dict[str, float]] = {}
            category_text = categories.astype(str)
            for category in sorted(set(category_text)):
                mask = category_text == category
                category_demand = demand[mask]
                category_inventory = ending_inventory[mask]
                category_lost = lost_sales[mask]
                denominator = float(category_demand.sum())
                category_metrics[category] = {
                    "inventory_demand_ratio": (
                        float(category_inventory.sum()) / denominator
                        if denominator > 0
                        else float("inf")
                    ),
                    "positive_inventory_service_rate": float(
                        (category_inventory > 0).mean()
                    ),
                    "no_stockout_service_rate": float((category_lost <= 1e-9).mean()),
                }
            result["category_metrics"] = category_metrics
        return result

    @staticmethod
    def _per_item_metrics(
        window: str,
        item_ids: np.ndarray,
        categories: np.ndarray | None,
        demand: np.ndarray,
        sales: np.ndarray,
        lost_sales: np.ndarray,
        ending_inventory: np.ndarray,
        orders: np.ndarray,
    ) -> pd.DataFrame:
        """生成同一CSV中的4期与完整窗口SKU明细。"""

        total_demand = demand.sum(axis=1)
        total_inventory = ending_inventory.sum(axis=1)
        ratio = np.divide(
            total_inventory,
            total_demand,
            out=np.where(total_inventory > 0, np.inf, 0.0),
            where=total_demand > 0,
        )
        frame = pd.DataFrame(
            {
                "window": window,
                "item_id": item_ids,
                "inventory_demand_ratio": ratio,
                "positive_inventory_service_rate": (ending_inventory > 0).mean(axis=1),
                "no_stockout_service_rate": (lost_sales <= 1e-9).mean(axis=1),
                "total_ending_inventory": total_inventory,
                "total_demand": total_demand,
                "total_sales": sales.sum(axis=1),
                "total_lost_sales": lost_sales.sum(axis=1),
                "total_order_quantity": orders.sum(axis=1),
            }
        )
        if categories is not None:
            frame.insert(2, "category", categories)
        return frame

    @staticmethod
    def _align_inputs(
        demand: pd.DataFrame,
        item_info: pd.DataFrame,
        parameters: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        """按 item_id 对齐三类输入并检查数值边界。"""

        period_columns = [column for column in demand if str(column).startswith("period_")]
        period_columns = sorted(period_columns, key=lambda column: int(column.split("_")[-1]))
        required_parameters = {"item_id", "mean_demand", "s", "S"}
        if not required_parameters.issubset(parameters.columns):
            raise ValueError("策略参数缺少 item_id、mean_demand、s 或 S。")
        if "initial_inventory" not in item_info.columns:
            raise ValueError("item_info 缺少 initial_inventory。")

        info_columns = ["item_id", "initial_inventory"]
        if "category" in item_info.columns:
            info_columns.append("category")
        data = (
            demand[["item_id", *period_columns]]
            .merge(item_info[info_columns], on="item_id", validate="one_to_one")
            .merge(
                parameters[["item_id", "mean_demand", "s", "S"]],
                on="item_id",
                validate="one_to_one",
            )
        )
        numeric = [column for column in data if column not in {"item_id", "category"}]
        data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
        if data[numeric].isna().any().any() or (data[numeric] < 0).any().any():
            raise ValueError("需求、库存和策略参数必须非负且不能缺失。")
        if (data["S"] < data["s"]).any():
            raise ValueError("策略参数必须满足 S >= s。")
        return data, period_columns
