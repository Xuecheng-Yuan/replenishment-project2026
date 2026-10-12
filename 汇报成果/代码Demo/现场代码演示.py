from pathlib import Path
import numpy as np
from model_v2 import robust_fill_level, robust_fill_lp_check
samples = np.load(Path(__file__).with_name("真实爆品残差.npz"))["samples"]
beta, radius = 0.9580576277958447, 0.1
z = robust_fill_level(samples, beta, radius)
print(f"Real training residuals: {len(samples)}")
print(f"DRO base-stock multiplier: {z:.8f}")
small = samples[:200]
a = robust_fill_level(small, beta, radius)
b = robust_fill_lp_check(small, beta, radius)
print(f"Bisection/LP absolute gap: {abs(a-b):.3e}")
assert abs(a-b) < 1e-8
