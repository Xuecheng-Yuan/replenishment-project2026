"""新目录增加的动作网格与训练信息边界合同。"""
import unittest
from unittest.mock import patch
import numpy as np
from tests.test_core import fixture
from replenishment.training.action_grid import refine_grid
from replenishment.training.decision import targets, policy
from replenishment.runtime import training_data


class ReleaseTests(unittest.TestCase):
    def test_explicit_grid_changes_real_target(self):
        result=targets('spo_linear',np.array([10.]),np.array([2.]),[0.,.3125,4.])
        np.testing.assert_array_equal(result,[[10.,10.625,18.]])
        direct=targets('direct_xgboost',np.array([10.]),np.array([2.]),[0.,1.3125,4.])
        np.testing.assert_array_equal(direct,[[0.,13.125,40.]])

    def test_refinement_keeps_sentinels(self):
        actions=np.arange(0,4.01,.5)
        costs=np.ones((20,9));costs[:,2:4]=0
        grid=refine_grid(actions,costs,np.ones_like(costs))
        self.assertEqual(len(np.unique(grid)),9)
        self.assertEqual(grid[0],0)
        self.assertEqual(grid[-1],4)
        np.testing.assert_array_equal(grid,refine_grid(actions,costs,np.ones_like(costs)))

    def test_runtime_uses_saved_grid_and_current_inventory(self):
        class Model:
            def predict_actions(self,x): return np.full(len(x),1)
        bundle=dict(centers=np.full((2,3),10.),sigma=np.ones(2),scale=np.ones(2),
                    features=np.zeros((3,2,59)),actions=[0.,.3125,4.])
        decide=policy('spo_linear',Model(),bundle,0,2.)
        np.testing.assert_allclose(decide(0,np.array([0.,20.])),[12.3125,0.])

    def test_training_guard_does_not_mutate_real_test(self):
        raw=fixture();before=raw['test_demand'].copy()
        with patch('replenishment.runtime.data',return_value=raw): masked=training_data()
        self.assertTrue((masked['test_demand']==0).all())
        self.assertTrue((masked['test_x']==0).all())
        np.testing.assert_array_equal(raw['test_demand'],before)


if __name__=='__main__': unittest.main()
