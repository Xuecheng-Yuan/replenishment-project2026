"""统一的补货策略仿真与服务水平、周转率计算。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .optimization import validate_policy_parameters


@dataclass(frozen=True)
class EvaluationResult:
    """一次库存仿真的明细与汇总结果。"""

    per_item: pd.DataFrame
    summary: dict[str, object]
    trajectory: pd.DataFrame | None = None


class InventoryEvaluator:
    """按“先到货、再需求、最后下单”的顺序模拟 lost-sales 库存系统。"""

    def __init__(self, lead_time: int = 1) -> None:
        if lead_time < 1:
            raise ValueError("提前期 lead_time 必须至少为 1。")
        self.lead_time = lead_time

    def evaluate(
        self,
        demand: pd.DataFrame,
        item_info: pd.DataFrame,
        parameters: pd.DataFrame,
        *,
        record_trajectory: bool = False,
    ) -> EvaluationResult:
        """在已经冻结的策略参数上评估需求，不执行任何调参。"""

        data, period_columns = self._align_inputs(demand, item_info, parameters)
        demand_values = data[period_columns].to_numpy(dtype=float)
        item_ids = data["item_id"].to_numpy()

        # 测试第一期的可用库存完全来自 test 文件中的 initial_inventory。
        # 模型不能在看见 test 需求之前补写一笔“期初订单”。
        on_hand = data["initial_inventory"].to_numpy(dtype=float, copy=True)
        reorder_point = data["s"].to_numpy(dtype=float)
        order_up_to = data["S"].to_numpy(dtype=float)
        pipeline = np.zeros((self.lead_time, len(data)), dtype=float)

        total_demand = demand_values.sum(axis=1)
        total_sales = np.zeros(len(data), dtype=float)
        ending_inventory_sum = np.zeros(len(data), dtype=float)
        order_count = np.zeros(len(data), dtype=int)
        total_order_quantity = np.zeros(len(data), dtype=float)
        system_inventory_by_period: list[float] = []
        trajectory_parts: list[pd.DataFrame] = []

        for period_index, period_column in enumerate(period_columns):
            # 第一步：本期期初接收此前已经下达、且到达提前期的订单。
            arrival = pipeline[0].copy()
            on_hand += arrival
            if self.lead_time > 1:
                pipeline[:-1] = pipeline[1:]
            pipeline[-1] = 0.0
            opening_inventory = on_hand.copy()

            # 第二步：需求发生。库存不足的部分作为 lost sales，不结转欠单。
            period_demand = demand_values[:, period_index]
            sales = np.minimum(on_hand, period_demand)
            lost_sales = period_demand - sales
            on_hand -= sales

            # 第三步：根据期末库存位置执行 (s, S) 补货。
            inventory_position = on_hand + pipeline.sum(axis=0)
            order_quantity = np.where(
                inventory_position <= reorder_point,
                np.maximum(0.0, order_up_to - inventory_position),
                0.0,
            )
            pipeline[-1] += order_quantity

            total_sales += sales
            ending_inventory_sum += on_hand
            order_count += order_quantity > 0
            total_order_quantity += order_quantity
            system_inventory_by_period.append(float(on_hand.sum()))

            if record_trajectory:
                trajectory_parts.append(
                    pd.DataFrame(
                        {
                            "item_id": item_ids,
                            "period": period_index + 1,
                            "period_name": period_column,
                            "opening_inventory": opening_inventory,
                            "arrival": arrival,
                            "demand": period_demand,
                            "sales": sales,
                            "lost_sales": lost_sales,
                            "ending_inventory": on_hand.copy(),
                            "inventory_position": inventory_position,
                            "order_quantity": order_quantity,
                        }
                    )
                )

        n_periods = len(period_columns)
        average_inventory = ending_inventory_sum / n_periods
        item_service_level = np.divide(
            total_sales,
            total_demand,
            out=np.ones_like(total_sales),
            where=total_demand > 0,
        )
        item_turnover = np.divide(
            total_sales,
            average_inventory,
            out=np.where(total_sales > 0, np.inf, 0.0),
            where=average_inventory > 0,
        )

        per_item = pd.DataFrame(
            {
                "item_id": item_ids,
                "service_level": item_service_level,
                "fill_rate": item_service_level,
                "average_ending_inventory": average_inventory,
                "turnover": item_turnover,
                "total_demand": total_demand,
                "total_sales": total_sales,
                "total_lost_sales": total_demand - total_sales,
                "order_count": order_count,
                "total_order_quantity": total_order_quantity,
            }
        )
        if "category" in data.columns:
            # 把原始分类写进同一份 SKU 明细，避免额外生成分组明细文件。
            per_item.insert(1, "category", data["category"].to_numpy())

        total_demand_all = float(total_demand.sum())
        total_sales_all = float(total_sales.sum())
        average_total_inventory = float(np.mean(system_inventory_by_period))
        overall_service_level = (
            total_sales_all / total_demand_all if total_demand_all > 0 else 1.0
        )
        overall_turnover = (
            total_sales_all / average_total_inventory
            if average_total_inventory > 0
            else (float("inf") if total_sales_all > 0 else 0.0)
        )

        summary: dict[str, object] = {
            "n_items": len(data),
            "n_periods": n_periods,
            "overall_service_level": overall_service_level,
            "overall_fill_rate": overall_service_level,
            "overall_turnover": overall_turnover,
            "average_total_ending_inventory": average_total_inventory,
            "total_demand": total_demand_all,
            "total_sales": total_sales_all,
            "total_lost_sales": total_demand_all - total_sales_all,
            "item_service_level_95_share": float((item_service_level >= 0.95).mean()),
        }
        if "category" in per_item.columns:
            # 分组指标沿用总体指标的聚合定义，不对 SKU 指标做简单平均。
            summary["category_metrics"] = self._category_metrics(per_item)

        trajectory = (
            pd.concat(trajectory_parts, ignore_index=True) if trajectory_parts else None
        )
        return EvaluationResult(per_item=per_item, summary=summary, trajectory=trajectory)

    def evaluate_dynamic(
        self,
        demand: pd.DataFrame,
        item_info: pd.DataFrame,
        order_up_to: pd.DataFrame,
    ) -> EvaluationResult:
        """评价随时间变化的 base-stock；仅支持当前研究的1期提前期。"""

        if self.lead_time != 1:
            raise ValueError("动态 base-stock 当前只支持 lead_time=1。")
        period_columns = self._period_columns(demand)
        required = {"item_id", *period_columns}
        if not required.issubset(order_up_to.columns):
            raise ValueError("动态目标库存缺少 item_id 或时期列。")
        if order_up_to["item_id"].duplicated().any():
            raise ValueError("动态目标库存中的 item_id 不能重复。")

        targets = order_up_to[["item_id", *period_columns]].rename(
            columns={column: f"target_{column}" for column in period_columns}
        )
        info_columns = ["item_id", "initial_inventory"]
        if "category" in item_info.columns:
            info_columns.append("category")
        data = (
            demand[["item_id", *period_columns]]
            .merge(item_info[info_columns], on="item_id", validate="one_to_one")
            .merge(targets, on="item_id", validate="one_to_one")
        )
        numeric_columns = [column for column in data if column not in {"item_id", "category"}]
        data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
        if data[numeric_columns].isna().any().any() or (data[numeric_columns] < 0).any().any():
            raise ValueError("动态评价输入必须是非负数且不能缺失。")

        demands = data[period_columns].to_numpy(dtype=float)
        target_values = data[[f"target_{column}" for column in period_columns]].to_numpy(
            dtype=float
        )
        on_hand = data["initial_inventory"].to_numpy(dtype=float, copy=True)
        arrivals = np.zeros(len(data), dtype=float)
        total_demand = demands.sum(axis=1)
        total_sales = np.zeros(len(data), dtype=float)
        inventory_sum = np.zeros(len(data), dtype=float)
        order_count = np.zeros(len(data), dtype=int)
        total_order_quantity = np.zeros(len(data), dtype=float)

        for period in range(demands.shape[1]):
            on_hand += arrivals
            sales = np.minimum(on_hand, demands[:, period])
            on_hand -= sales
            arrivals = np.maximum(0.0, target_values[:, period] - on_hand)
            total_sales += sales
            inventory_sum += on_hand
            order_count += arrivals > 0
            total_order_quantity += arrivals

        average_inventory = inventory_sum / demands.shape[1]
        service_level = np.divide(
            total_sales, total_demand, out=np.ones_like(total_sales), where=total_demand > 0
        )
        turnover = np.divide(
            total_sales,
            average_inventory,
            out=np.where(total_sales > 0, np.inf, 0.0),
            where=average_inventory > 0,
        )
        per_item = pd.DataFrame(
            {
                "item_id": data["item_id"].to_numpy(),
                "service_level": service_level,
                "fill_rate": service_level,
                "average_ending_inventory": average_inventory,
                "turnover": turnover,
                "total_demand": total_demand,
                "total_sales": total_sales,
                "total_lost_sales": total_demand - total_sales,
                "order_count": order_count,
                "total_order_quantity": total_order_quantity,
            }
        )
        if "category" in data.columns:
            per_item.insert(1, "category", data["category"].to_numpy())

        total_demand_all = float(total_demand.sum())
        total_sales_all = float(total_sales.sum())
        average_total_inventory = float(inventory_sum.sum() / demands.shape[1])
        summary: dict[str, object] = {
            "n_items": len(data),
            "n_periods": demands.shape[1],
            "overall_service_level": total_sales_all / total_demand_all if total_demand_all else 1.0,
            "overall_fill_rate": total_sales_all / total_demand_all if total_demand_all else 1.0,
            "overall_turnover": (
                total_sales_all / average_total_inventory
                if average_total_inventory > 0
                else (float("inf") if total_sales_all > 0 else 0.0)
            ),
            "average_total_ending_inventory": average_total_inventory,
            "total_demand": total_demand_all,
            "total_sales": total_sales_all,
            "total_lost_sales": total_demand_all - total_sales_all,
            "item_service_level_95_share": float((service_level >= 0.95).mean()),
        }
        if "category" in per_item.columns:
            summary["category_metrics"] = self._category_metrics(per_item)
        return EvaluationResult(per_item=per_item, summary=summary)

    @staticmethod
    def _category_metrics(per_item: pd.DataFrame) -> dict[str, dict[str, float | int]]:
        """按商品类别计算需求加权服务率和测试窗口周转率。"""

        metrics: dict[str, dict[str, float | int]] = {}
        for category, group in per_item.groupby("category", sort=True, dropna=False):
            total_demand = float(group["total_demand"].sum())
            total_sales = float(group["total_sales"].sum())
            average_total_inventory = float(group["average_ending_inventory"].sum())
            service_level = total_sales / total_demand if total_demand > 0 else 1.0
            turnover = (
                total_sales / average_total_inventory
                if average_total_inventory > 0
                else (float("inf") if total_sales > 0 else 0.0)
            )
            metrics[str(category)] = {
                "n_items": int(len(group)),
                "service_level": service_level,
                "fill_rate": service_level,
                "turnover": turnover,
                "average_total_ending_inventory": average_total_inventory,
                "total_demand": total_demand,
                "total_sales": total_sales,
                "total_lost_sales": total_demand - total_sales,
                "item_service_level_95_share": float(
                    (group["service_level"] >= 0.95).mean()
                ),
            }
        return metrics

    @staticmethod
    def _period_columns(demand: pd.DataFrame) -> list[str]:
        """提取并按数字顺序排列 period_t 字段。"""

        columns = [column for column in demand.columns if str(column).startswith("period_")]
        try:
            columns = sorted(columns, key=lambda column: int(str(column).split("_")[-1]))
        except ValueError as exc:
            raise ValueError("时期列必须使用 period_1、period_2 等命名。") from exc
        if not columns:
            raise ValueError("需求数据中没有 period_t 字段。")
        return columns

    def _align_inputs(
        self,
        demand: pd.DataFrame,
        item_info: pd.DataFrame,
        parameters: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        """按 item_id 一对一合并输入，并阻止静默丢失 SKU。"""

        validate_policy_parameters(parameters)
        period_columns = self._period_columns(demand)

        required_frames = {
            "demand": (demand, {"item_id", *period_columns}),
            "item_info": (item_info, {"item_id", "initial_inventory"}),
        }
        for name, (frame, required_columns) in required_frames.items():
            missing = required_columns - set(frame.columns)
            if missing:
                raise ValueError(f"{name} 缺少字段：{sorted(missing)}")
            if frame["item_id"].isna().any() or frame["item_id"].duplicated().any():
                raise ValueError(f"{name} 的 item_id 不能为空或重复。")

        demand_items = set(demand["item_id"])
        if demand_items != set(item_info["item_id"]):
            raise ValueError("需求数据与 initial_inventory 的 SKU 集合不一致。")
        if demand_items != set(parameters["item_id"]):
            raise ValueError("需求数据与策略参数的 SKU 集合不一致。")

        item_info_columns = ["item_id", "initial_inventory"]
        if "category" in item_info.columns:
            if item_info["category"].isna().any():
                raise ValueError("item_info 的 category 不能为空。")
            item_info_columns.append("category")

        data = demand[["item_id", *period_columns]].merge(
            item_info[item_info_columns],
            on="item_id",
            how="left",
            validate="one_to_one",
        ).merge(
            parameters[["item_id", "s", "S"]],
            on="item_id",
            how="left",
            validate="one_to_one",
        )

        numeric_columns = [*period_columns, "initial_inventory", "s", "S"]
        data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
        if data[numeric_columns].isna().any().any():
            raise ValueError("需求、库存或策略参数中存在缺失值或非数值。")
        if (data[[*period_columns, "initial_inventory", "s", "S"]] < 0).any().any():
            raise ValueError("需求、库存和策略参数不能为负数。")
        return data, period_columns
