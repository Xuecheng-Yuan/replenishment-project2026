"""只用拟合段的动作成本分布细化网格，不观察验证/测试需求选范围。"""
import numpy as np


def refine_grid(actions, costs, orders):
    actions = np.asarray(actions, float)
    # 并列最优按权重均分，防止argmin总偏向第一个动作。
    best = np.isclose(costs, costs.min(1, keepdims=True), atol=1e-10, rtol=0)
    active = ((orders > 0) & best).any(1)
    weights = (best / best.sum(1)[:, None])[active].sum(0)
    if weights.sum() == 0:
        return actions.copy()
    cdf = np.cumsum(weights / weights.sum())
    low = actions[min(np.searchsorted(cdf, .1), len(actions)-1)]
    high = actions[min(np.searchsorted(cdf, .9), len(actions)-1)]
    if high-low < .5:
        low, high = max(0., low-.25), min(4., high+.25)
    # 始终保留0和4哨兵，不根据测试表现删除极端动作。
    refined = np.r_[0., np.linspace(low, high, 9)[1:-1], 4.]
    assert len(np.unique(refined)) == 9
    return refined
