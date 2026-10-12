"""models.arima: actual model/processing implementation, not a legacy runner alias.

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
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable
import numpy as np

def arima_paths(train: np.ndarray, test: np.ndarray, validation_start: int, n_jobs: int | None=None, candidate_orders: Iterable[tuple[int, int, int]] | None=None) -> tuple[ForecastPath, ForecastPath, list[tuple[int, int, int]], int]:
    """Fit per-SKU ARIMA orders in parallel and return validation/test paths."""
    train, test = _validate_arrays(train, test)
    if not 3 <= validation_start < train.shape[1]:
        raise ValueError('validation_start 不合法。')
    orders = tuple(candidate_orders or ((0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 0), (0, 1, 2)))
    jobs = n_jobs or max(1, min(8, (os.cpu_count() or 2) - 1))
    payloads = [(train[index], test[index], validation_start, orders) for index in range(train.shape[0])]
    if jobs == 1:
        rows = map(_arima_worker, payloads)
    else:
        executor = ProcessPoolExecutor(max_workers=jobs)
        rows = executor.map(_arima_worker, payloads, chunksize=16)
    validation_forecasts: list[np.ndarray] = []
    validation_scales: list[float] = []
    test_forecasts: list[np.ndarray] = []
    test_scales: list[float] = []
    selected_orders: list[tuple[int, int, int]] = []
    fallback_count = 0
    try:
        for validation_pred, validation_scale, test_pred, test_scale, order, fallback in rows:
            validation_forecasts.append(validation_pred)
            validation_scales.append(validation_scale)
            test_forecasts.append(test_pred)
            test_scales.append(test_scale)
            selected_orders.append(order)
            fallback_count += int(fallback)
    finally:
        if jobs != 1:
            executor.shutdown()
    return (ForecastPath(np.vstack(validation_forecasts), np.asarray(validation_scales)), ForecastPath(np.vstack(test_forecasts), np.asarray(test_scales)), selected_orders, fallback_count)

def _arima_worker(payload: tuple[np.ndarray, np.ndarray, int, tuple[tuple[int, int, int], ...]]) -> tuple[np.ndarray, float, np.ndarray, float, tuple[int, int, int], bool]:
    series, test, validation_start, orders = payload
    fit_series = np.asarray(series[:validation_start], dtype=float)
    validation_actual = np.asarray(series[validation_start:], dtype=float)
    try:
        selected_order, validation_result = _best_arima(fit_series, orders)
        validation_scale = _result_scale(validation_result, fit_series)
        validation_predictions = _rolling_arima(validation_result, validation_actual)
        _, test_result = _best_arima(np.asarray(series, dtype=float), (selected_order,))
        test_scale = _result_scale(test_result, np.asarray(series, dtype=float))
        test_predictions = _rolling_arima(test_result, np.asarray(test, dtype=float))
        return (validation_predictions, validation_scale, test_predictions, test_scale, selected_order, False)
    except Exception:
        validation_path = naive_path(fit_series[None, :], validation_actual[None, :])
        test_path = naive_path(series[None, :], test[None, :])
        return (validation_path.forecasts[0], float(validation_path.residual_scale[0]), test_path.forecasts[0], float(test_path.residual_scale[0]), (0, 1, 0), True)

def _best_arima(series: np.ndarray, orders: Iterable[tuple[int, int, int]]):
    from statsmodels.tsa.arima.model import ARIMA
    best_order: tuple[int, int, int] | None = None
    best_result = None
    best_aic = float('inf')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for order in orders:
            try:
                result = ARIMA(series, order=order, trend='t', enforce_stationarity=False, enforce_invertibility=False).fit(method_kwargs={'warn_convergence': False})
            except Exception:
                continue
            if np.isfinite(result.aic) and result.aic < best_aic:
                best_order, best_result, best_aic = (order, result, float(result.aic))
    if best_result is None or best_order is None:
        raise RuntimeError('ARIMA 拟合失败。')
    return (best_order, best_result)

def _rolling_arima(result, future: np.ndarray) -> np.ndarray:
    predictions: list[float] = []
    current = result
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for actual in [*future, None]:
            predictions.append(max(0.0, float(np.asarray(current.forecast(1))[0])))
            if actual is not None:
                current = current.append([float(actual)], refit=False)
    return np.asarray(predictions, dtype=float)

def _result_scale(result, series: np.ndarray) -> float:
    residuals = np.asarray(result.resid, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size > 4:
        residuals = residuals[2:]
    if residuals.size < 2:
        residuals = np.diff(series)
    return float(_error_scale(residuals[None, :])[0])
from common.ts_support import ForecastPath, _error_scale, _validate_arrays
from models.naive import naive_path

if __name__=='__main__':
    from common.cli import start
    start('arima', globals(), default_protocol='legacy')
