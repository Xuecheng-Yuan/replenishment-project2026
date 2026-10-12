"""models.moving_average: actual model/processing implementation, not a legacy runner alias.

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

def moving_average_path(history: np.ndarray, future: np.ndarray, window: int) -> ForecastPath:
    history, future = _validate_arrays(history, future)
    if window < 1 or window > history.shape[1]:
        raise ValueError('window 必须在 1 和历史期数之间。')
    combined = np.concatenate([history, future], axis=1)
    start = history.shape[1]
    forecasts = np.column_stack([combined[:, index - window:index].mean(axis=1) for index in range(start, combined.shape[1] + 1)])
    fitted = np.column_stack([history[:, index - window:index].mean(axis=1) for index in range(window, history.shape[1])])
    errors = history[:, window:] - fitted
    return ForecastPath(np.clip(forecasts, 0, None), _error_scale(errors))
from common.ts_support import ForecastPath, _error_scale, _validate_arrays

if __name__=='__main__':
    from common.cli import start
    start('moving_average', globals(), default_protocol='legacy')
