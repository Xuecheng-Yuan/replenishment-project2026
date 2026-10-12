"""Check causal timing, true simulator agreement and service loss gradients."""
import unittest
import numpy as np
import pandas as pd
from common.evaluation import InventoryEvaluator
try:
    import torch
except ModuleNotFoundError:
    torch = None
if torch is not None:
    from models.service_policy import numpy_targets, rollout, service_loss

class FixedPolicy(torch.nn.Module if torch is not None else object):

    def __init__(self, target):
        super().__init__()
        self.target = torch.nn.Parameter(torch.tensor(float(target)))

    def forward(self, features, inventory, reference, indices):
        return self.target.expand_as(inventory)

@unittest.skipIf(torch is None, 'Optional PyTorch policy experiment requires torch')
class ServicePolicyTests(unittest.TestCase):

    def test_no_first_test_order_and_evaluator_agreement(self):
        policy = FixedPolicy(10)
        demand = np.array([[6.0, 14.0, 7.0]])
        initial, targets = numpy_targets(policy, np.zeros((4, 1, 1)), np.ones((4, 1)), demand, np.array([4.0]), pre_period_decision=False)
        result = InventoryEvaluator().evaluate_dynamic_base_stock(pd.DataFrame({'item_id': [1], 'period_1': [6.0], 'period_2': [14.0], 'period_3': [7.0]}), pd.DataFrame({'item_id': [1], 'initial_inventory': initial}), pd.DataFrame({'item_id': [1], **{f'period_{i + 1}': targets[:, i] for i in range(3)}}))
        self.assertEqual(initial[0], 4)
        self.assertAlmostEqual(result.summary['total_fill_rate'], 21 / 27)
        self.assertAlmostEqual(result.summary['average_total_inventory'], 1)

    def test_rollout_gradient_reduces_shortage_when_infeasible(self):
        policy = FixedPolicy(8)
        demand = torch.tensor([[10.0], [10.0]])
        inv, lost, _ = rollout(policy, torch.zeros(2, 1, 1), torch.ones(2, 1), demand, torch.zeros(1), torch.arange(1))
        loss, gap = service_loss(inv, lost, demand.sum(), multiplier=3.0, rho=100.0)
        loss.backward()
        self.assertGreater(float(gap.detach()), 0)
        self.assertLess(float(policy.target.grad), 0)

    def test_feasible_inventory_gradient_reduces_excess_stock(self):
        policy = FixedPolicy(12)
        demand = torch.tensor([[10.0], [10.0]])
        inv, lost, _ = rollout(policy, torch.zeros(2, 1, 1), torch.ones(2, 1), demand, torch.zeros(1), torch.arange(1))
        loss, gap = service_loss(inv, lost, demand.sum(), multiplier=3.0, rho=100.0)
        loss.backward()
        self.assertLess(float(gap.detach()), 0)
        self.assertGreater(float(policy.target.grad), 0)

    def test_empirical_loss_matches_inventory_and_fill_definitions(self):
        loss, gap = service_loss(torch.tensor(20.0), torch.tensor(10.0), torch.tensor(100.0), multiplier=2.0, rho=100.0)
        self.assertAlmostEqual(float(gap), 0.05)
        self.assertAlmostEqual(float(loss), 0.2 + 2 * 0.05 + 50 * 0.05 ** 2)
if __name__ == '__main__':
    unittest.main()
