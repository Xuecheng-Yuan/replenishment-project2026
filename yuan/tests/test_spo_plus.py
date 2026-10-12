import unittest
import numpy as np
from models.spo_linear import SPOPlusLinear
from models.direct_lightgbm import direct_spo_rank_grad_hess
from common.policy_state import action_levels
from models.spo_linear import action_costs
from models.spo_linear import action_targets
from common.policy_state import append_current_inventory
from common.policy_state import augment_policy_features
from common.policy_state import behavior_ending_inventory
from models.xgboost_spo import hybrid_grad_hess
from common.policy_state import rolling_reference
from common.policy_state import rollout_action_costs
from common.policy_state import rollout_inventory_aware_policy

class SPOPlusTests(unittest.TestCase):

    def test_actions_and_costs(self):
        targets = action_targets(np.array([10.0, 20.0]), np.array([2.0, 4.0]), np.array([0.0, 1.0]))
        np.testing.assert_array_equal(targets, [[10.0, 12.0], [20.0, 24.0]])
        costs = action_costs(np.array([11.0, 18.0]), targets, shortage_penalty=3.0)
        np.testing.assert_array_equal(costs, [[3.0, 1.0], [2.0, 6.0]])

    def test_linear_spo_learns_separable_actions(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(800, 2)).astype(np.float32)
        best = (x[:, 0] > 0).astype(int)
        costs = np.ones((len(x), 2), dtype=np.float32)
        costs[np.arange(len(x)), best] = 0.0
        model = SPOPlusLinear(epochs=18, batch_size=64, learning_rate=0.2, ridge=0.0001)
        model.fit(x, costs)
        accuracy = np.mean(model.predict_actions(x) == best)
        self.assertGreater(accuracy, 0.94)

    def test_hybrid_gradient_matches_finite_difference(self):
        true_costs = np.array([[0.3, 1.2, 2.0]], dtype=float)
        predicted = np.array([[0.7, 0.4, 1.1]], dtype=float)
        alpha = 0.6
        temperature = 0.8
        gradient, hessian = hybrid_grad_hess(true_costs, predicted, alpha=alpha, temperature=temperature)

        def loss(values):
            logits = (true_costs - 2.0 * values) / temperature
            smooth_max = temperature * (np.log(np.exp(logits - logits.max()).sum()) + logits.max())
            oracle = int(np.argmin(true_costs[0]))
            smooth_spo = smooth_max + 2.0 * values[0, oracle] - true_costs[0, oracle]
            mse = np.square(values - true_costs).sum() / (2.0 * true_costs.shape[1])
            return (1.0 - alpha) * mse + alpha * smooth_spo
        numerical = np.zeros_like(predicted)
        epsilon = 1e-05
        for column in range(predicted.shape[1]):
            upper = predicted.copy()
            lower = predicted.copy()
            upper[0, column] += epsilon
            lower[0, column] -= epsilon
            numerical[0, column] = (loss(upper) - loss(lower)) / (2.0 * epsilon)
        np.testing.assert_allclose(gradient, numerical, rtol=0.0002, atol=0.0002)
        self.assertTrue((hessian > 0).all())

    def test_direct_policy_rollout_costs(self):
        reference = rolling_reference(np.array([[2.0, 4.0, 6.0]]), window=2)
        np.testing.assert_allclose(reference, [[2.0, 3.0, 5.0]])
        np.testing.assert_array_equal(action_levels(np.array([5.0]), np.array([1.0, 2.0])), [[5.0, 10.0]])
        costs = rollout_action_costs(np.array([1.0]), np.array([[6.0]]), np.array([[5.0]]), np.array([1.0, 2.0]), shortage_penalty=3.0)
        np.testing.assert_array_equal(costs, [[3.0, 4.0]])

    def test_direct_policy_features_include_inventory_state(self):
        features = augment_policy_features(np.ones((2, 3)), reference=np.array([2.0, 4.0]), opening_inventory=np.array([4.0, 8.0]), ending_inventory=np.array([1.0, 2.0]), current_demand=np.array([3.0, 6.0]), current_lost_sales=np.array([0.0, 0.0]))
        self.assertEqual(features.shape, (2, 11))
        np.testing.assert_allclose(features[:, -3:], [[2.0, 0.5, 1.5]] * 2)

    def test_predict_then_optimize_rollout_uses_current_ending_inventory(self):
        demand = np.array([[3.0, 4.0]])
        base_features = np.zeros((3, 2), dtype=np.float32)
        observed_inventory = []

        def selector(_block, features):
            observed_inventory.append(float(features[0, -1]))
            return (np.array([0]), np.array([5.0]))
        rollout = rollout_inventory_aware_policy(demand, np.array([6.0]), base_features, selector, include_pre_period_decision=False)
        np.testing.assert_allclose(observed_inventory, [3.0, 1.0])
        np.testing.assert_allclose(rollout.ending_inventory, [[3.0, 1.0]])
        self.assertEqual(append_current_inventory(np.ones((1, 2)), [3.0]).shape, (1, 3))

    def test_behavior_inventory_is_historical_and_nonnegative(self):
        ending = behavior_ending_inventory(np.array([[2.0, 10.0, 1.0]]), np.array([3.0]), reference_window=2, behavior_multiplier=2.0)
        np.testing.assert_allclose(ending, [[1.0, 0.0, 11.0]])

    def test_direct_spo_rank_gradient_matches_finite_difference(self):
        actual = np.array([[0.3, 1.2, 2.0]], dtype=float)
        predicted = np.array([[0.7, 0.4, 1.1]], dtype=float)
        spo_weight, rank_weight = (0.4, 0.35)
        spo_temperature, rank_temperature = (0.8, 1.2)
        gradient, hessian = direct_spo_rank_grad_hess(actual, predicted, spo_weight=spo_weight, rank_weight=rank_weight, spo_temperature=spo_temperature, rank_temperature=rank_temperature)

        def loss(values):
            oracle = int(np.argmin(actual[0]))
            mse = np.square(values - actual).sum() / (2.0 * actual.shape[1])
            logits = (actual - 2.0 * values) / spo_temperature
            smooth_max = spo_temperature * (np.log(np.exp(logits - logits.max()).sum()) + logits.max())
            smooth_spo = smooth_max + 2.0 * values[0, oracle] - actual[0, oracle]
            regret = actual[0] - actual[0, oracle]
            weights = regret / regret.sum()
            pairwise = np.sum(weights * np.logaddexp(0.0, (values[0, oracle] - values[0]) / rank_temperature))
            return (1.0 - spo_weight - rank_weight) * mse + spo_weight * smooth_spo + rank_weight * pairwise
        numerical = np.zeros_like(predicted)
        epsilon = 1e-05
        for column in range(predicted.shape[1]):
            upper, lower = (predicted.copy(), predicted.copy())
            upper[0, column] += epsilon
            lower[0, column] -= epsilon
            numerical[0, column] = (loss(upper) - loss(lower)) / (2.0 * epsilon)
        np.testing.assert_allclose(gradient, numerical, rtol=0.0002, atol=0.0002)
        self.assertTrue((hessian > 0).all())
if __name__ == '__main__':
    unittest.main()
