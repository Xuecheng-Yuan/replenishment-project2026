"""common.lead_one_evaluation: actual model/processing implementation, not a legacy runner alias.

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
from common.ready_evaluation import ReadyRateEvaluator

class LeadOneReadyRateEvaluator(ReadyRateEvaluator):

    def evaluate_forecasts(self, demand, forecasts, safety_z=0.0, output=None):
        """Forecast column t targets demand t using only information through t-1.

        Initial pending orders are zero. After observing demand t, place an
        order for t+1 using forecast column t+1. Last-period orders are zero:
        they would arrive outside the evaluated horizon. No rounding buffer.
        """
        demand = np.asarray(demand, dtype=float)
        forecasts = np.asarray(forecasts, dtype=float)
        if forecasts.shape != demand.shape or not np.isfinite(forecasts).all():
            raise ValueError('Forecast shape/values invalid')
        if (forecasts < 0).any() or safety_z < 0:
            raise ValueError('Negative forecast/safety factor')
        n, h = demand.shape
        stocks = np.zeros_like(demand)
        sales = np.zeros_like(demand)
        arrivals = np.zeros_like(demand)
        orders = np.zeros_like(demand)
        inventory = self.initial.copy()
        pending = np.zeros(n)
        targets = forecasts + safety_z * self.std[:, None]
        for t in range(h):
            arrivals[:, t] = pending
            available = inventory + pending
            sales[:, t] = np.minimum(available, demand[:, t])
            inventory = available - sales[:, t]
            stocks[:, t] = inventory
            pending = self.order_up_to(targets[:, t + 1], inventory) if t + 1 < h else np.zeros(n)
            orders[:, t] = pending
        metrics = super().evaluate(demand, lambda t, i: arrivals[:, t], output)
        metrics.update(lead_time=1, safety_z=float(safety_z), average_total_order=float(orders.sum(axis=0).mean()), average_order_active_period=float(orders[:, :-1].sum(axis=0).mean()), test_wape=float(np.abs(forecasts - demand).sum() / demand.sum()))
        if output is not None:
            path = Path(output)
            np.savez_compressed(path / 'trajectory.npz', item_ids=self.item_ids, demand=demand, forecast=forecasts, targets=targets, placed_orders=orders, arrivals=arrivals, sales=sales, end_inventory=stocks, initial_inventory=self.initial, initial_pending=np.zeros(n), capacity=self.capacity, safety_z=safety_z, lead_time=1)
        return metrics
