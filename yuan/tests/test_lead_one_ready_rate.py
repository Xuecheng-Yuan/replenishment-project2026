import unittest
import numpy as np
from common.lead_one_evaluation import LeadOneReadyRateEvaluator

class LeadOneTests(unittest.TestCase):

    def test_cannot_serve_current_demand_with_new_order(self):
        e = LeadOneReadyRateEvaluator(np.array([[10.0, 10.0]]), np.array([1]), cover=0)
        m = e.evaluate_forecasts(np.array([[10.0, 10.0]]), np.array([[10.0, 10.0]]))
        self.assertEqual(m['fill_rate'], 0.5)
        self.assertEqual(m['mean_ready_rate'], 0)

    def test_perfect_forecast_can_fill_but_not_be_ready(self):
        e = LeadOneReadyRateEvaluator(np.array([[10.0, 10.0]]), np.array([1]), cover=1)
        m = e.evaluate_forecasts(np.array([[10.0, 10.0, 10.0]]), np.array([[10.0, 10.0, 10.0]]))
        self.assertEqual(m['fill_rate'], 1)
        self.assertEqual(m['mean_ready_rate'], 0)

    def test_no_order_when_remaining_stock_covers_forecast(self):
        e = LeadOneReadyRateEvaluator(np.array([[10.0, 10.0]]), np.array([1]), cover=10)
        m = e.evaluate_forecasts(np.array([[10.0, 10.0]]), np.array([[10.0, 10.0]]))
        self.assertEqual(m['average_total_order'], 0)
        self.assertEqual(m['mean_ready_rate'], 1)
if __name__ == '__main__':
    unittest.main()
