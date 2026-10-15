import unittest
import numpy as np
from extend_with_ro import box

class RobustCounterpartTests(unittest.TestCase):
    def test_box_nested_and_nonnegative(self):
        samples=[0,0,1,2,8]
        lo0,hi0=box(samples,.5);lo1,hi1=box(samples,3)
        self.assertGreaterEqual(lo1,0)
        self.assertLessEqual(lo1,lo0);self.assertGreaterEqual(hi1,hi0)

    def test_fill_constraint_equivalent_to_endpoint(self):
        beta=.95; upper=15.7; target=beta*upper
        demand=np.linspace(0,upper,1001)
        self.assertTrue(np.all(np.maximum(demand-target,0)<=(1-beta)*demand+1e-12))
        self.assertGreater(max(upper-(target-.01),0),(1-beta)*upper)

    def test_strict_positive_integer_guard(self):
        for upper in (0,1,1.1,19.999,20):
            target=np.floor(upper)+1
            self.assertGreater(target,upper)
            self.assertLessEqual(target-1,upper)

if __name__=='__main__':unittest.main()
