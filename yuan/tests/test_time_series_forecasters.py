import unittest
import numpy as np
from models.exponential_smoothing import exponential_smoothing_path
from models.moving_average import moving_average_path
from models.naive import naive_path

class TimeSeriesForecastersTest(unittest.TestCase):

    def setUp(self) -> None:
        self.history = np.array([[1.0, 2.0, 3.0, 4.0]])
        self.future = np.array([[5.0, 6.0]])

    def test_naive_is_strictly_one_step_ahead(self) -> None:
        path = naive_path(self.history, self.future)
        np.testing.assert_array_equal(path.forecasts, [[4.0, 5.0, 6.0]])

    def test_moving_average_does_not_use_current_future_value(self) -> None:
        path = moving_average_path(self.history, self.future, window=2)
        np.testing.assert_array_equal(path.forecasts, [[3.5, 4.5, 5.5]])

    def test_exponential_smoothing_returns_extra_next_forecast(self) -> None:
        path = exponential_smoothing_path(self.history, self.future, alpha=0.8, beta=0.2, damping=1.0)
        self.assertEqual(path.forecasts.shape, (1, 3))
        self.assertEqual(path.residual_scale.shape, (1,))
        self.assertTrue((path.forecasts >= 0).all())
if __name__ == '__main__':
    unittest.main()
