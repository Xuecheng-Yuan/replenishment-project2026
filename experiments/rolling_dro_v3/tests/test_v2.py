"""Mathematical and causal checks for the revised inventory model."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from model_v2 import (Forecaster, make_forecasts, robust_fill_level,
                      robust_fill_lp_check, simulate)


class RevisedModelTests(unittest.TestCase):
    def test_robust_root_matches_independent_lp(self):
        rng=np.random.default_rng(62)
        for beta in (.7,.95,.99):
            for radius in (0.,.1,.5):
                samples=rng.lognormal(-.1,.5,25)
                z=robust_fill_level(samples,beta,radius)
                lp=robust_fill_lp_check(samples,beta,radius)
                self.assertAlmostEqual(z,lp,places=7)
                lo=np.maximum(0,samples-radius); hi=samples+radius
                worst=np.maximum(np.maximum(lo-z,0)-(1-beta)*lo,
                                 np.maximum(hi-z,0)-(1-beta)*hi).mean()
                self.assertLessEqual(worst,1e-8)

    def test_radius_monotonic_and_degenerate_sample(self):
        samples=np.array([0.,.5,1.,2.,3.])
        levels=[robust_fill_level(samples,.95,r) for r in (0,.01,.1,.5)]
        self.assertTrue(np.all(np.diff(levels)>=0))
        self.assertAlmostEqual(robust_fill_level(np.array([10.]),.95,0),9.5,places=7)

    def test_future_demand_cannot_change_current_order_or_forecast(self):
        rng=np.random.default_rng(5)
        train=np.maximum(1,np.arange(30)[:,None]+rng.normal(10,2,(30,4)))
        test=np.maximum(1,rng.normal(20,4,(12,4)))
        model=Forecaster(train,np.array([0,0,1,1]),1,3,0)
        changed=test.copy(); changed[6:]+=1000
        fa=make_forecasts(model,test); fb=make_forecasts(model,changed)
        np.testing.assert_allclose(fa[:6],fb[:6])
        ra=simulate(test,np.ones(4)*20,fa,np.ones(4)*1.2,keep=True)
        rb=simulate(changed,np.ones(4)*20,fb,np.ones(4)*1.2,keep=True)
        np.testing.assert_array_equal(ra['orders'][:6],rb['orders'][:6])

    def test_lead_time_lost_sales_conservation_and_four_plans(self):
        y=np.array([[8.],[9.],[3.],[5.]])
        fc=np.full((4,4,1),10.)
        r=simulate(y,np.array([2.]),fc,np.array([1.]),keep=True)
        self.assertEqual(r['sold'][0,0],2.)
        self.assertEqual(r['orders'][0,0],10.)
        self.assertEqual(r['commencing'][1,0],10.)
        np.testing.assert_array_equal(r['commencing'][1:],r['ending'][:-1]+r['orders'][:-1])
        np.testing.assert_array_equal(r['commencing'],r['sold']+r['ending'])
        self.assertTrue(np.all((y-r['sold'])*r['ending']==0))
        self.assertEqual(r['plans'].shape,(4,4,1))
        np.testing.assert_array_equal(r['plans'][:,0],r['orders'])
        np.testing.assert_array_equal(r['orders'],np.ceil(r['orders']))

    def test_four_period_decomposition_without_coupling(self):
        y=np.full((3,1),10.); fc=np.ones((3,4,1))*10
        changed=fc.copy(); changed[:,1:]*=100
        a=simulate(y,np.array([10.]),fc,np.array([1.2]),keep=True)
        b=simulate(y,np.array([10.]),changed,np.array([1.2]),keep=True)
        # No capacity/setup costs: distant target cannot require stocking early.
        np.testing.assert_array_equal(a['orders'],b['orders'])
        self.assertFalse(np.array_equal(a['plans'][:,1:],b['plans'][:,1:]))


if __name__=='__main__': unittest.main()
