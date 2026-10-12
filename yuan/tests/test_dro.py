import unittest
import numpy as np
from models.dro_moment import fit_robust_buffer
from models.dro_moment import rolling_moving_average_residuals

class DistributionallyRobustTest(unittest.TestCase):

    def setUp(self) -> None:
        self.demand = np.asarray([[10, 11, 9, 12, 10, 13, 11, 14, 12, 15, 13, 16, 14, 17, 15, 18], [20, 19, 21, 18, 22, 17, 23, 16, 24, 15, 25, 14, 26, 13, 27, 12]], dtype=float)

    def test_rolling_residuals_do_not_use_current_demand(self) -> None:
        forecasts, _ = rolling_moving_average_residuals(self.demand, window=4)
        expected = self.demand[:, :4].mean(axis=1)
        np.testing.assert_allclose(forecasts[:, 0], expected)

    def test_all_dro_methods_satisfy_fitted_robust_bound(self) -> None:
        cases = (('moment', 1.0), ('wasserstein', 0.01), ('chi_square', 0.05))
        for method, parameter in cases:
            with self.subTest(method=method):
                model = fit_robust_buffer(self.demand, method, parameter, window=4)
                self.assertLessEqual(model.robust_constraint_gap, 1e-06)
                self.assertTrue(np.isfinite(model.safety_buffer).all())

    def test_more_ambiguity_does_not_reduce_wasserstein_buffer(self) -> None:
        small = fit_robust_buffer(self.demand, 'wasserstein', 0.005, window=4)
        large = fit_robust_buffer(self.demand, 'wasserstein', 0.05, window=4)
        self.assertGreaterEqual(large.buffer_multiplier, small.buffer_multiplier)
if __name__ == '__main__':
    unittest.main()
