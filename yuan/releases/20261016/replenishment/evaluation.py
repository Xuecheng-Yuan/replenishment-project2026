"""唯一评价器：服务水平与Fill Rate分别计算，不混用约束。"""
import numpy as np
import pandas as pd

from .runtime import ROOT, save

LOW, HIGH = .95, .96


def key(m):
    """先接近服务区间，再最小化库存需求比；与最新版实验的排序一致。"""
    r = m['ready_rate']
    gap = max(LOW - r, 0, r - HIGH)
    return int(gap > 1e-12), gap, m['inventory_demand_ratio'], abs(r - (LOW + HIGH) / 2)


def metrics(d, a, b, s, q):
    """b为期末库存，s为实际销量。全零需求的比值返回None，而不是伪造0。"""
    n, periods = d.shape
    demand, sales, stock = float(d.sum()), float(s.sum()), float(b.sum())
    rates = (b > 0).mean(1)
    r = float((b > 0).mean())
    return dict(ready_rate=r, fill_rate=sales / demand if demand else None,
                inventory_demand_ratio=stock / demand if demand else None,
                average_ending_total_inventory=stock / periods,
                average_inventory_per_sku_period=stock / (n * periods),
                ready_passing_skus=int((rates >= LOW).sum()),
                minimum_sku_ready_rate=float(rates.min()), maximum_sku_ready_rate=float(rates.max()),
                r=sales / stock if stock else None,
                ending_inventory_units=stock, demand_units=demand, sales_units=sales,
                lost_units=float((d - s).sum()), order_units=float(q.sum()),
                final_stock=float(b[:, -1].sum()), final_pipeline=float(q[:, -1].sum()),
                interval_feasible=bool(LOW - 1e-12 <= r <= HIGH + 1e-12), periods=periods, skus=n)


def validate(d, initial, a, b, s, q):
    """检查库存守恒、一期到货、非负库存、真实销量和整数订单。"""
    if any(v.shape != d.shape or not np.isfinite(v).all() or (v < 0).any()
           for v in (a, b, s, q)):
        raise ValueError('轨迹形状、非负性或有限性错误。')
    np.testing.assert_array_equal(a[:, 0], initial)
    np.testing.assert_array_equal(a[:, 1:], b[:, :-1] + q[:, :-1])
    np.testing.assert_array_equal(s, np.minimum(a, d))
    np.testing.assert_array_equal(a, s + b)
    np.testing.assert_array_equal(q, np.ceil(q))


def simulate(d, initial, policy, score_start=0):
    """先收上期订单，再满足当期需求，最后决策；当期订单不能救当期缺货。"""
    d, initial = np.asarray(d, float), np.asarray(initial, float)
    if d.ndim != 2 or not 0 <= score_start < d.shape[1]:
        raise ValueError('需求需为二维，评分起点应在回放区间内。')
    if initial.shape != (len(d),) or not np.isfinite(initial).all() or (initial < 0).any():
        raise ValueError('初始库存无效。')
    if not np.isfinite(d).all() or (d < 0).any():
        raise ValueError('需求无效。')
    arrays = [np.empty_like(d) for _ in range(4)]
    a, b, s, q = arrays
    inv, pending = initial.copy(), np.zeros(len(d))
    for t in range(d.shape[1]):
        a[:, t] = inv + pending
        s[:, t] = np.minimum(a[:, t], d[:, t])
        inv = a[:, t] - s[:, t]
        b[:, t] = inv
        request = np.asarray(policy(t, inv.copy()), float)
        if request.shape != inv.shape or not np.isfinite(request).all():
            raise ValueError('策略订单须为每SKU一个有限数。')
        pending = np.ceil(np.maximum(request, 0))
        q[:, t] = pending
    validate(d, initial, *arrays)
    return metrics(d[:, score_start:], *(v[:, score_start:] for v in arrays)), arrays


def target_policy(p, sigma, z, low=None):
    """期末订货至下一期目标；保留low接口，但本次baseline采用s=S。"""
    def decide(t, b):
        uncertainty = sigma[:, t + 1] if sigma.ndim == 2 else sigma
        high = np.maximum(p[:, t + 1] + z * uncertainty, 0)
        q = np.maximum(high - b, 0)
        return q if low is None else np.where(b < np.maximum(p[:, t + 1] + (z - low) * uncertainty, 0), q, 0)
    return decide


def calibrate_target(d, initial, p, sigma, start=30, low=None):
    """验证期一维标定；保留原实验的有限次二分搜索，不声称全局最优。"""
    records = []
    def run(z):
        m, _ = simulate(d, initial, target_policy(p, sigma, z, low), start)
        records.append(dict(z=float(z), **m))
        return m
    lo, hi = -8., 16.
    ml, mh = run(lo), run(hi)
    if ml['ready_rate'] <= LOW <= mh['ready_rate']:
        for _ in range(24):
            mid = (lo + hi) / 2
            if run(mid)['ready_rate'] >= LOW:
                hi = mid
            else:
                lo = mid
        run(hi + 1e-5)
    best = min(records, key=key)
    return best['z'], best, records


class InventoryEvaluator:
    """各方案可直接调用的评价类；逐SKU指标落盘，终端只展示汇总。"""

    def evaluate(self, name, d, initial, policy, item_ids, metadata=None, split='test'):
        m, arrays = simulate(d, initial, policy)
        a, b, s, q = arrays
        out = ROOT / 'results' / name / split
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / 'trajectory.npz', demand=d, available=a, ending=b,
                            sales=s, orders=q, initial=initial, item_ids=item_ids)
        total = d.sum(1)
        fr = np.divide(s.sum(1), total, out=np.full(len(d), np.nan), where=total > 0)
        pd.DataFrame(dict(item_id=item_ids, service_level=(b > 0).mean(1), fill_rate=fr,
                          average_ending_inventory=b.mean(1))).to_csv(
                              out / 'sku_metrics.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame([dict(period=t + 1, **metrics(d[:, t:t + 1], *(v[:, t:t + 1] for v in arrays)))
                      for t in range(d.shape[1])]).to_csv(out / 'period_metrics.csv', index=False, encoding='utf-8-sig')
        row = dict(model=name, split=split, **m, metadata=metadata or {})
        save(out / 'summary.json', row)
        fr = '无定义' if m['fill_rate'] is None else f'{m["fill_rate"]:.4%}'
        ratio = '无定义' if m['inventory_demand_ratio'] is None else f'{m["inventory_demand_ratio"]:.6f}'
        print(f'{name}: 服务水平={m["ready_rate"]:.4%}, Fill Rate={fr}, J={ratio}')
        return row
