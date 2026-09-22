"""兼容入口下的库存时序和指标回归测试。"""

import pandas as pd

from replenishment.evaluation import InventoryEvaluator


def test_evaluator_keeps_arrival_demand_order_sequence() -> None:
    """第一期无到货，本期订单只能在下一期初到达。"""

    demand = pd.DataFrame(
        {
            "item_id": [1],
            "period_1": [8],
            "period_2": [8],
            "period_3": [8],
            "period_4": [8],
        }
    )
    item_info = pd.DataFrame({"item_id": [1], "initial_inventory": [10]})
    parameters = pd.DataFrame(
        {"item_id": [1], "mean_demand": [8], "s": [10], "S": [10]}
    )
    result = InventoryEvaluator().evaluate(
        demand, item_info, parameters, record_trajectory=True
    )

    assert result.trajectory is not None
    assert result.trajectory["arrival"].tolist() == [0.0, 8.0, 8.0, 8.0]
    metrics = result.summary["windows"]["first_4"]
    assert metrics["no_stockout_service_rate"] == 1.0
    assert metrics["positive_inventory_service_rate"] == 1.0
    assert metrics["inventory_demand_ratio"] == 0.25


def test_category_service_is_counted_by_sku_period() -> None:
    """分类服务率按无缺货SKU-时期计数，不按需求量加权。"""

    demand = pd.DataFrame(
        {
            "item_id": [1, 2],
            "period_1": [100, 10],
            "period_2": [0, 0],
            "period_3": [0, 0],
            "period_4": [0, 0],
        }
    )
    item_info = pd.DataFrame(
        {
            "item_id": [1, 2],
            "category": ["爆品", "长尾品"],
            "initial_inventory": [50, 10],
        }
    )
    parameters = pd.DataFrame(
        {
            "item_id": [1, 2],
            "mean_demand": [0, 0],
            "s": [0, 0],
            "S": [0, 0],
        }
    )
    metrics = InventoryEvaluator().evaluate(demand, item_info, parameters).summary[
        "windows"
    ]["first_4"]

    assert metrics["no_stockout_service_rate"] == 7 / 8
    assert metrics["category_metrics"]["爆品"]["no_stockout_service_rate"] == 0.75
    assert metrics["category_metrics"]["长尾品"]["no_stockout_service_rate"] == 1.0
