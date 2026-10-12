import unittest
import numpy as np
import pandas as pd
from common.evaluation import InventoryEvaluator
from common.trajectories import combine_model_trajectories
from common.trajectories import simulate_item_trajectory
from models.baseline import BaselineConfig
from models.baseline import MeanVarianceBaseline

class InventoryEvaluatorTest(unittest.TestCase):

    def test_model_trajectories_use_shared_periods_and_demand(self) -> None:
        first, _ = simulate_item_trajectory(np.array([4.0, 5.0]), 5.0, 5.0, 5.0)
        second, _ = simulate_item_trajectory(np.array([4.0, 5.0]), 7.0, 7.0, 7.0)
        combined = combine_model_trajectories({'baseline': first, 'model_b': second}, 1)
        self.assertEqual(set(combined['model']), {'baseline', 'model_b'})
        self.assertEqual(len(combined), 4)

    def test_item_trajectory_matches_replenishment_timing(self) -> None:
        trajectory, summary = simulate_item_trajectory(demands=np.array([10.0, 10.0]), initial_inventory=10.0, s=10.0, S=10.0, lead_time=1)
        self.assertEqual(trajectory.loc[0, 'order_quantity'], 10.0)
        self.assertEqual(trajectory.loc[1, 'arrival'], 10.0)
        self.assertEqual(summary['fill_rate'], 1.0)

    def test_constant_demand_has_full_fill_rate(self) -> None:
        demand = pd.DataFrame({'item_id': [1], 'period_1': [10], 'period_2': [10]})
        info = pd.DataFrame({'item_id': [1], 'initial_inventory': [10]})
        parameters = pd.DataFrame({'item_id': [1], 's': [10], 'S': [10]})
        result = InventoryEvaluator(lead_time=1).evaluate(demand, info, parameters)
        self.assertEqual(result.per_sku.loc[0, 'fill_rate'], 1.0)
        self.assertEqual(result.summary['total_fill_rate'], 1.0)
        self.assertEqual(result.per_sku.loc[0, 'average_inventory'], 0.0)

    def test_fill_rate_uses_units_not_stockout_periods(self) -> None:
        demand = pd.DataFrame({'item_id': [1], 'period_1': [10], 'period_2': [10]})
        info = pd.DataFrame({'item_id': [1], 'initial_inventory': [5]})
        parameters = pd.DataFrame({'item_id': [1], 's': [5], 'S': [5]})
        result = InventoryEvaluator().evaluate(demand, info, parameters)
        self.assertEqual(result.per_sku.loc[0, 'fill_rate'], 0.5)

    def test_baseline_produces_s_equal_to_S(self) -> None:
        demand = pd.DataFrame({'item_id': [1, 2], 'period_1': [5, 10], 'period_2': [6, 11], 'period_3': [5, 9], 'period_4': [6, 10]})
        config = BaselineConfig(validation_fraction=0.5, z_step=0.5)
        parameters = MeanVarianceBaseline(config).fit(demand).get_parameters()
        self.assertTrue(np.array_equal(parameters['s'], parameters['S']))

    def test_dynamic_base_stock_uses_next_arrival(self) -> None:
        demand = pd.DataFrame({'item_id': [1], 'period_1': [5], 'period_2': [9]})
        info = pd.DataFrame({'item_id': [1], 'initial_inventory': [5]})
        targets = pd.DataFrame({'item_id': [1], 'period_1': [9], 'period_2': [9]})
        result = InventoryEvaluator().evaluate_dynamic_base_stock(demand, info, targets)
        self.assertEqual(result.summary['total_fill_rate'], 1.0)
        self.assertEqual(result.per_sku.loc[0, 'total_order_quantity'], 18.0)
if __name__ == '__main__':
    unittest.main()
