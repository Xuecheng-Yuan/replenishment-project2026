import unittest
import numpy as np
from common.ready_evaluation import ReadyRateEvaluator

class ReadyRateTests(unittest.TestCase):

    def test_zero_inventory_not_ready(self):
        e = ReadyRateEvaluator(np.array([[2.0, 2.0]]), np.array([1]), cover=0)
        m = e.evaluate(np.array([[2.0, 2.0]]), lambda t, i: np.array([2.0]))
        self.assertEqual(m['fill_rate'], 1)
        self.assertEqual(m['mean_ready_rate'], 0)
        self.assertFalse(m['all_skus_feasible'])

    def test_shared_capacity(self):
        e = ReadyRateEvaluator(np.ones((2, 3)), np.array([1, 2]), cover=0)
        q = e.order_up_to(np.array([10.0, 10.0]), np.zeros(2))
        self.assertAlmostEqual(q.sum(), e.capacity)
        with self.assertRaises(ValueError):
            e.evaluate(np.ones((2, 1)), lambda t, i: np.array([10.0, 10.0]))

    def test_average_inventory_turnover(self):
        e = ReadyRateEvaluator(np.ones((1, 3)), np.array([1]), cover=2)
        m = e.evaluate(np.ones((1, 2)), lambda t, i: np.array([1.0]))
        self.assertEqual(m['mean_ready_rate'], 1)
        self.assertEqual(m['average_total_inventory'], 2)
        self.assertEqual(m['turnover'], 1)
if __name__ == '__main__':
    unittest.main()
