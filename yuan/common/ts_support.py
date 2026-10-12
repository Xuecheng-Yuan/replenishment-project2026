"""common.ts_support: actual model/processing implementation, not a legacy runner alias.

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
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ForecastPath:
    """Forecasts for each observed future period plus one following period."""
    forecasts: np.ndarray
    residual_scale: np.ndarray

def _error_scale(errors: np.ndarray) -> np.ndarray:
    if errors.shape[1] < 2:
        scale = np.mean(np.abs(errors), axis=1)
    else:
        scale = np.std(errors, axis=1, ddof=1)
    return np.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)

def _validate_arrays(history: np.ndarray, future: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    history = np.asarray(history, dtype=float)
    future = np.asarray(future, dtype=float)
    if history.ndim != 2 or future.ndim != 2 or history.shape[0] != future.shape[0]:
        raise ValueError('history 和 future 必须是 SKU 行数相同的二维数组。')
    if history.shape[1] < 2 or not np.isfinite(history).all() or (not np.isfinite(future).all()):
        raise ValueError('时间序列必须有至少两个历史期且不能含缺失或无穷值。')
    if (history < 0).any() or (future < 0).any():
        raise ValueError('需求不能为负数。')
    return (history, future)
