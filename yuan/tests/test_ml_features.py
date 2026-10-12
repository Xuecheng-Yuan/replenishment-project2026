import unittest
import numpy as np
from common.features import build_supervised_data

class MachineLearningFeatureTest(unittest.TestCase):

    def test_current_period_x_is_not_used(self) -> None:
        demand = np.arange(1, 15, dtype=float)[None, :]
        x = np.zeros((1, 14, 20), dtype=float)
        categories = np.array(['A'])
        original = build_supervised_data(demand, x, categories, np.array([12])).features
        x[:, 12, :] = 999.0
        changed = build_supervised_data(demand, x, categories, np.array([12])).features
        np.testing.assert_array_equal(original, changed)

    def test_previous_period_x_is_used(self) -> None:
        demand = np.arange(1, 15, dtype=float)[None, :]
        x = np.zeros((1, 14, 20), dtype=float)
        categories = np.array(['A'])
        original = build_supervised_data(demand, x, categories, np.array([12])).features
        x[:, 11, :] = 1.0
        changed = build_supervised_data(demand, x, categories, np.array([12])).features
        self.assertFalse(np.array_equal(original, changed))

    def test_next_unobserved_period_can_be_built(self) -> None:
        demand = np.arange(1, 15, dtype=float)[None, :]
        x = np.zeros((1, 14, 20), dtype=float)
        data = build_supervised_data(demand, x, np.array(['A']), np.array([14]), include_target=False)
        self.assertEqual(data.features.shape[0], 1)
        self.assertIsNone(data.target)
if __name__ == '__main__':
    unittest.main()
