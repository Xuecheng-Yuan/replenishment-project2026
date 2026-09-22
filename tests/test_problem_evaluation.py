"""滚动库存时序、两种服务率和目标指标测试。"""

import pandas as pd
import pytest

from replenishment.problem_evaluation import InventoryEvaluator


def _parameters(level: float, mean: float = 8.0) -> pd.DataFrame:
    return pd.DataFrame(
        {"item_id": [1], "mean_demand": [mean], "s": [level], "S": [level]}
    )


def test_order_arrives_next_period_and_only_first_plan_is_executed() -> None:
    """本期末下单，下期初到货；后三个计划订单不能提前进入库存。"""

    demand = pd.DataFrame(
        {"item_id": [1], "period_1": [8], "period_2": [8], "period_3": [8], "period_4": [8]}
    )
    item_info = pd.DataFrame({"item_id": [1], "initial_inventory": [10]})
    result = InventoryEvaluator().evaluate(
        demand, item_info, _parameters(10), record_trajectory=True
    )

    assert result.trajectory is not None
    assert result.trajectory["arrival"].tolist() == [0.0, 8.0, 8.0, 8.0]
    assert result.trajectory["executed_order"].tolist() == [8.0, 8.0, 8.0, 8.0]
    assert result.trajectory["planned_order_1"].equals(
        result.trajectory["executed_order"]
    )


def test_two_service_rates_treat_exact_depletion_differently() -> None:
    """库存恰好归零时属于无缺货，但不属于严格正库存。"""

    demand = pd.DataFrame(
        {"item_id": [1], "period_1": [8], "period_2": [9], "period_3": [8], "period_4": [8]}
    )
    item_info = pd.DataFrame({"item_id": [1], "initial_inventory": [8]})
    metrics = InventoryEvaluator().evaluate(demand, item_info, _parameters(8)).summary[
        "windows"
    ]["first_4"]

    assert metrics["positive_inventory_service_rate"] == 0.0
    assert metrics["no_stockout_service_rate"] == pytest.approx(0.75)
    assert metrics["constraint_met"] is False


def test_inventory_demand_ratio_uses_total_ending_inventory() -> None:
    """目标必须是所有SKU-时期总期末库存除以同期总需求。"""

    demand = pd.DataFrame(
        {"item_id": [1], "period_1": [8], "period_2": [8], "period_3": [8], "period_4": [8]}
    )
    item_info = pd.DataFrame({"item_id": [1], "initial_inventory": [10]})
    metrics = InventoryEvaluator().evaluate(demand, item_info, _parameters(10)).summary[
        "windows"
    ]["first_4"]

    assert metrics["total_ending_inventory"] == 8.0
    assert metrics["total_demand"] == 32.0
    assert metrics["inventory_demand_ratio"] == 0.25
