import unittest
import torch
from models.dfl_improved import DecisionConfig, order, ready_surrogate

class ImprovedTests(unittest.TestCase):

    def test_no_order_when_stock_covers_target(self):
        mu = torch.log1p(torch.tensor([10.0, 20.0]))
        q = order(mu, torch.ones(2) * 0.1, torch.tensor([100.0, 200.0]), torch.tensor([10.0, 20.0]), torch.tensor(100.0), None, DecisionConfig(), 42, optimize=False)
        self.assertTrue(torch.equal(q, torch.zeros(2)))

    def test_projection_and_gradients(self):
        mu = torch.log1p(torch.tensor([10.0, 20.0])).requires_grad_(True)
        q = order(mu, torch.ones(2) * 0.1, torch.zeros(2), torch.tensor([10.0, 20.0]), torch.tensor(15.0), None, DecisionConfig(steps=3), 42, create_graph=True)
        self.assertLessEqual(q.sum().item(), 15.00001)
        self.assertTrue((q >= 0).all())
        q[0].backward()
        self.assertTrue(torch.isfinite(mu.grad).all())
        self.assertGreater(mu.grad.abs().sum().item(), 0)

    def test_shortage_has_service_gradient(self):
        available = torch.tensor([9.0], requires_grad=True)
        ready_surrogate(available, torch.tensor([10.0]), torch.tensor([10.0])).sum().backward()
        self.assertGreater(available.grad.item(), 0)

    def test_full_unroll_multiple_periods_has_finite_gradients(self):
        mu = torch.log1p(torch.tensor([1.0, 10.0, 100.0])).requires_grad_(True)
        inv = torch.zeros(3)
        scale = torch.tensor([1.0, 10.0, 100.0])
        for t in range(7):
            q = order(mu, torch.ones(3) * 0.2, inv, scale, torch.tensor(100.0), None, DecisionConfig(safety_z=4.0, steps=20), 42 + t, True)
            inv = (inv + q - scale).clamp_min(0)
        inv.sum().backward()
        self.assertTrue(torch.isfinite(mu.grad).all())
if __name__ == '__main__':
    unittest.main()
