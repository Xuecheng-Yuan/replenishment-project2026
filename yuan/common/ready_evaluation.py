"""common.ready_evaluation: actual model/processing implementation, not a legacy runner alias.

Refactored mechanically from the existing project. See matching docs for
information timing, historical results, variants and limitations.
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path(__file__).resolve().parents[1]
if str(HERE) in sys.path:sys.path.remove(str(HERE))
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
PROJECT_ROOT=ROOT
DATA_DIR=ROOT/'data'
OUTPUT_DIR=ROOT/'results/generated'
from pathlib import Path
import numpy as np
import pandas as pd

class ReadyRateEvaluator:

    def __init__(self, training_demand, item_ids, capacity_multiplier=1.05, cover=1.25):
        self.mean = np.mean(training_demand, axis=1)
        self.std = np.std(training_demand, axis=1, ddof=1)
        self.initial = cover * self.mean
        self.capacity = capacity_multiplier * self.mean.sum()
        self.weights = 1 / np.maximum(self.mean, 0.5)
        self.weights /= self.weights.mean()
        self.item_ids = item_ids

    def evaluate(self, demand, policy, output=None):
        """policy(t, inventory) cannot receive current demand. Demand is [SKU,T]."""
        inventory = self.initial.copy()
        orders, sales, stocks = ([], [], [])
        for t in range(demand.shape[1]):
            q = np.asarray(policy(t, inventory.copy()), dtype=float)
            if q.shape != inventory.shape or not np.isfinite(q).all():
                raise ValueError('Invalid order vector')
            if q.min() < -1e-06 or q.sum() > self.capacity * (1 + 2e-06):
                raise ValueError('Negative orders or shared capacity violated')
            q = np.maximum(q, 0)
            available = inventory + q
            sold = np.minimum(available, demand[:, t])
            inventory = available - sold
            orders.append(q)
            sales.append(sold)
            stocks.append(inventory.copy())
        orders, sales, stocks = map(lambda a: np.stack(a, axis=1), (orders, sales, stocks))
        h = demand.shape[1]
        ready = (stocks > 1e-08).mean(axis=1)
        average = stocks.mean(axis=1)
        metrics = dict(mean_ready_rate=float(ready.mean()), sku_service_pass_rate=float((ready >= 0.95).mean()), minimum_ready_rate=float(ready.min()), all_skus_feasible=bool((ready >= 0.95).all()), fill_rate=float(sales.sum() / demand.sum()), turnover=float(sales.sum() / max(average.sum(), 1e-08)), weighted_turnover=float(self.weights @ sales.sum(axis=1) / max(self.weights @ average, 1e-08)), average_total_inventory=float(average.sum()), average_total_order=float(orders.sum(axis=0).mean()), capacity=float(self.capacity), periods=h)
        if output is not None:
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(dict(item_id=self.item_ids, ready_rate=ready, fill_rate=sales.sum(axis=1) / np.maximum(demand.sum(axis=1), 1e-08), average_inventory=average, turnover=sales.sum(axis=1) / np.maximum(average, 1e-08))).to_csv(output / 'sku_metrics.csv', index=False)
            np.savez_compressed(output / 'trajectory.npz', item_ids=self.item_ids, demand=demand, orders=orders, sales=sales, end_inventory=stocks, initial_inventory=self.initial, capacity=self.capacity)
        return metrics

    def order_up_to(self, target, inventory):
        q = np.maximum(target - inventory, 0)
        return q * min(1.0, self.capacity / max(q.sum(), 1e-08))

def selection_key(metrics):
    """Feasibility first; if infeasible maximize passing SKUs, then mean readiness."""
    if metrics['all_skus_feasible']:
        return (1, metrics['weighted_turnover'], 0, 0)
    return (0, metrics['sku_service_pass_rate'], metrics['mean_ready_rate'], metrics['weighted_turnover'])
