import numpy as np
import pandas as pd

from replenishment.evaluation import InventoryEvaluator
from replenishment.time_series import (
    build_targets,
    cold_start_forecast_path,
)


def test_category_safety_factors_create_integer_targets() -> None:
    training = np.tile(np.arange(100.0, 112.0), (2, 1))
    episode = np.array([[10.0, 11.0], [10.0, 11.0]])
    categories = np.array(["爆品", "长尾品"])
    path = cold_start_forecast_path(training, episode, categories, {"name": "naive"})

    targets = build_targets(path, categories, {"爆品": 1.0, "长尾品": 2.0})

    assert targets.shape == episode.shape
    assert np.all(targets == np.ceil(targets))
    assert np.all(targets[1] >= targets[0])


def test_cold_start_does_not_carry_training_level_or_see_future() -> None:
    training = np.tile(np.arange(100.0, 112.0), (2, 1))
    episode = np.array([[10.0, 11.0, 12.0], [10.0, 11.0, 12.0]])
    changed = episode.copy()
    changed[:, 2] = 9999.0
    categories = np.array(["爆品", "长尾品"])

    original = cold_start_forecast_path(training, episode, categories, {"name": "naive"})
    modified = cold_start_forecast_path(training, changed, categories, {"name": "naive"})

    assert np.all(original.forecasts[:, 0] < 20.0)
    np.testing.assert_allclose(original.forecasts[:, :2], modified.forecasts[:, :2])


def test_dynamic_evaluator_uses_test_initial_inventory() -> None:
    demand = pd.DataFrame({"item_id": [1], "period_1": [5.0], "period_2": [5.0]})
    item_info = pd.DataFrame(
        {"item_id": [1], "category": ["爆品"], "initial_inventory": [3.0]}
    )
    targets = pd.DataFrame({"item_id": [1], "period_1": [5.0], "period_2": [5.0]})

    result = InventoryEvaluator(lead_time=1).evaluate_dynamic(demand, item_info, targets)

    # 第1期只能卖初始库存3；期末订单第2期才到货，因此总销量为8。
    assert result.summary["total_sales"] == 8.0
    assert result.summary["overall_service_level"] == 0.8
