"""库存流转、服务水平和周转率测试。"""

import pandas as pd
import pytest

from replenishment.evaluation import InventoryEvaluator


def test_evaluator_follows_arrival_demand_order_sequence() -> None:
    """第一期无模型补货到货，第一期期末订单在第二期初到达。"""

    demand = pd.DataFrame(
        {"item_id": [1], "period_1": [8], "period_2": [8], "period_3": [8]}
    )
    item_info = pd.DataFrame({"item_id": [1], "initial_inventory": [10]})
    parameters = pd.DataFrame({"item_id": [1], "s": [10], "S": [10]})

    result = InventoryEvaluator(lead_time=1).evaluate(
        demand,
        item_info,
        parameters,
        record_trajectory=True,
    )

    assert result.summary["overall_service_level"] == 1.0
    assert result.summary["total_sales"] == 24.0
    assert result.summary["average_total_ending_inventory"] == 2.0
    assert result.summary["overall_turnover"] == 12.0
    assert result.trajectory is not None
    assert result.trajectory["arrival"].tolist() == [0.0, 8.0, 8.0]


def test_service_level_is_demand_weighted() -> None:
    """总体服务水平必须等于总销售/总需求，而不是 SKU 服务水平简单平均。"""

    demand = pd.DataFrame(
        {"item_id": [1, 2], "period_1": [100, 10], "period_2": [0, 0]}
    )
    item_info = pd.DataFrame(
        {
            "item_id": [1, 2],
            "category": ["爆品", "长尾品"],
            "initial_inventory": [50, 10],
        }
    )
    parameters = pd.DataFrame({"item_id": [1, 2], "s": [0, 0], "S": [0, 0]})

    result = InventoryEvaluator().evaluate(demand, item_info, parameters)
    assert result.summary["overall_service_level"] == pytest.approx(60 / 110)
    assert result.per_item["category"].tolist() == ["爆品", "长尾品"]
    assert result.summary["category_metrics"]["爆品"]["service_level"] == 0.5
    assert result.summary["category_metrics"]["长尾品"]["service_level"] == 1.0
    assert result.summary["category_metrics"]["长尾品"]["turnover"] == float("inf")
