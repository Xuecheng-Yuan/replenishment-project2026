"""Task-only outer gradient and frozen rollout update guard on tiny inputs."""
import unittest
import numpy as np
import torch
from models import online_learning as online
from models import turnover_dfl as dfl

class TurnoverTaskTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rng=np.random.default_rng(42)
        self.raw=dict(demand=rng.uniform(5,15,(3,88)),features=rng.normal(size=(75,3,38)).astype(np.float32),item_ids=np.arange(3))
        self.data,self.cfg,self.e,self.model,_,_=online.setup(self.raw,60)

    def test_outer_loss_has_finite_task_gradient_without_nll(self):
        warm=dfl.states(self.model,self.data,15,self.e,None,42)
        loss,_=dfl.window_loss(self.model,self.data,8,15,*warm[8],self.e,None,1000.,42,nll_weight=0.)
        grads=torch.autograd.grad(loss,tuple(self.model.parameters()),allow_unused=True)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(g).all() for g in grads if g is not None))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads if g is not None),0.)

    def test_frozen_evaluation_keeps_network_parameters_unchanged(self):
        state={k:v.clone() for k,v in self.model.state_dict().items()}
        metrics,_=online.evaluate(self.model,self.data,self.cfg,self.e,None,64,'frozen',with_updates=False)
        self.assertEqual(metrics['lead_time'],1)
        self.assertTrue(all(torch.equal(v,self.model.state_dict()[k]) for k,v in state.items()))

if __name__=='__main__':unittest.main()
