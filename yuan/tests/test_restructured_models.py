"""New checks: model factories actually fit, and refactored MPC works."""
import importlib
import unittest
import numpy as np
from models.adaptive_mpc import GroupForecaster,PolicyConfig,ScenarioPolicy,simulate,project_orders

class RefactoredTests(unittest.TestCase):
    def test_every_ml_file_builds_and_fits_its_own_estimator(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(80,6));y=np.abs(10+x[:,0]+.2*x[:,1])
        params={'linear_regression':{},'ridge':{'alpha':1},'lasso':{'alpha':.01},
            'elastic_net':{'alpha':.01,'l1_ratio':.5},'svr':{'C':1.,'epsilon':.05},
            'random_forest':{'n_estimators':3,'max_depth':2},
            'xgboost':{'n_estimators':3,'max_depth':2},
            'lightgbm':{'n_estimators':3,'num_leaves':5},
            'catboost':{'iterations':3,'depth':2}}
        for name,p in params.items():
            with self.subTest(model=name):
                model=importlib.import_module('models.'+name).fit(x,y,p,1)
                prediction=model.predict(x[:4])
                self.assertEqual(prediction.shape,(4,))
                self.assertTrue(np.isfinite(prediction).all())

    def test_mpc_capacity_and_inventory_on_small_known_example(self):
        mean=np.array([10.,20.,30.]);forecast=np.repeat(mean[:,None,None],4,1)
        forecast=np.repeat(forecast,3,2)
        p=ScenarioPolicy(forecast,np.zeros((3,8)),np.zeros(3,dtype=int),mean,np.ones(3),1.05*mean.sum(),PolicyConfig(horizon=3,steps=2,buffer_factor=0.))
        demand=np.repeat(mean[:,None],4,1)
        result=simulate(demand,p,np.repeat(mean[:,None],8,1),np.arange(3))
        self.assertEqual(result['lead_time'],1)
        self.assertTrue(np.isfinite(result['turnover']))
        self.assertLessEqual(result['fill_rate'],1.)

    def test_mpc_projection_is_nonnegative_and_within_capacity(self):
        q=project_orders(np.array([[2.,3.],[4.,5.]]),np.array([10.,20.]),60.)
        self.assertTrue((q>=0).all())
        self.assertTrue(((q*np.array([10.,20.])[:,None]).sum(0)<=60+1e-8).all())

if __name__=='__main__':unittest.main()
