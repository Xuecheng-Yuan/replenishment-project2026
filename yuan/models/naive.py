"""models.naive: actual model/processing implementation, not a legacy runner alias.

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

def naive_path(history: np.ndarray, future: np.ndarray) -> ForecastPath:
    history, future = _validate_arrays(history, future)
    forecasts = np.empty((len(history), future.shape[1] + 1), dtype=float)
    forecasts[:, 0] = history[:, -1]
    forecasts[:, 1:] = future
    errors = history[:, 1:] - history[:, :-1]
    return ForecastPath(np.clip(forecasts, 0, None), _error_scale(errors))
from common.ts_support import ForecastPath, _error_scale, _validate_arrays

if __name__=='__main__':
    from common.cli import start
    start('naive', globals(), default_protocol='legacy')
