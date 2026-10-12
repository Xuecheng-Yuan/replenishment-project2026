"""models.exponential_smoothing: actual model/processing implementation, not a legacy runner alias.

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
import numpy as np

def exponential_smoothing_path(history: np.ndarray, future: np.ndarray, alpha: float, beta: float=0.0, damping: float=1.0) -> ForecastPath:
    """Vectorized Holt exponential smoothing with optional damping.

    ``beta=0`` gives simple exponential smoothing. The same smoothing
    coefficients are shared by all SKUs so tuning remains stable with only
    42 pre-validation observations per item.
    """
    history, future = _validate_arrays(history, future)
    if not 0 < alpha <= 1 or not 0 <= beta <= 1 or (not 0 < damping <= 1):
        raise ValueError('alpha、beta 或 damping 不合法。')
    level = history[:, 0].astype(float, copy=True)
    if beta > 0 and history.shape[1] > 1:
        trend = history[:, 1] - history[:, 0]
    else:
        trend = np.zeros(len(history), dtype=float)
    errors: list[np.ndarray] = []
    for index in range(1, history.shape[1]):
        forecast = level + damping * trend
        actual = history[:, index]
        errors.append(actual - forecast)
        previous_level = level
        level = alpha * actual + (1 - alpha) * forecast
        if beta > 0:
            trend = beta * (level - previous_level) + (1 - beta) * damping * trend
    forecasts: list[np.ndarray] = []
    for index in range(future.shape[1] + 1):
        forecasts.append(np.clip(level + damping * trend, 0, None))
        if index == future.shape[1]:
            break
        actual = future[:, index]
        previous_level = level
        forecast = level + damping * trend
        level = alpha * actual + (1 - alpha) * forecast
        if beta > 0:
            trend = beta * (level - previous_level) + (1 - beta) * damping * trend
    return ForecastPath(np.column_stack(forecasts), _error_scale(np.column_stack(errors)))
from common.ts_support import ForecastPath, _error_scale, _validate_arrays

if __name__=='__main__':
    from common.cli import start
    start('exponential_smoothing', globals(), default_protocol='legacy')
