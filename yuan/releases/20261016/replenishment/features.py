"""严格滞后的59维特征；不使用目标期未知的x。"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
X_COLUMNS = [f'x{i}' for i in range(1, 21)]
DEMAND_LAGS = (1, 2, 3, 6, 12)
ROLLING_WINDOWS = (3, 6, 12)

@dataclass(frozen=True)
class SupervisedData:
    features: np.ndarray
    target: np.ndarray | None
    feature_names: list[str]
    target_periods: np.ndarray

def build_supervised_data(
    demand: np.ndarray,
    x: np.ndarray,
    categories: np.ndarray,
    target_periods: np.ndarray,
    include_target: bool=True
) -> SupervisedData:
    """按“周期优先、SKU次之”排列监督样本，共59列。

    目标是第t期时，所有需求和匿名x只取t−1及更早。
    include_target=False允许为下一个尚未观测的周期构造输入。
    """
    demand = np.asarray(demand, dtype=float)
    x = np.asarray(x, dtype=float)
    target_periods = np.asarray(target_periods, dtype=int)
    if demand.ndim != 2 or x.ndim != 3 or x.shape[:2] != demand.shape:
        raise ValueError('demand 必须是二维，x 必须是对应的三维数组。')
    if len(categories) != demand.shape[0]:
        raise ValueError('categories 行数与SKU数不一致。')
    if target_periods.size == 0 or target_periods.min() < max(DEMAND_LAGS):
        raise ValueError('目标周期必须至少有12期历史。')
    maximum = demand.shape[1] - 1 if include_target else demand.shape[1]
    if target_periods.max() > maximum:
        raise ValueError('目标周期超出可构造范围。')
    category_values = sorted({str(value) for value in categories})
    category_map = {value: index for index, value in enumerate(category_values)}
    category_code = np.asarray([category_map[str(value)] for value in categories], dtype=float)
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for period in target_periods:
        lag_values = [demand[:, period - lag] for lag in DEMAND_LAGS]
        rolling_means = [demand[:, period - window:period].mean(axis=1) for window in ROLLING_WINDOWS]
        rolling_stds = [demand[:, period - window:period].std(axis=1, ddof=1) for window in ROLLING_WINDOWS]
        lag_1, lag_2, lag_3 = (lag_values[0], lag_values[1], lag_values[2])
        mean_3, mean_6 = (rolling_means[0], rolling_means[1])
        x_lag_1 = x[:, period - 1, :]
        x_lag_2 = x[:, period - 2, :]
        block = np.column_stack([
            *lag_values,
            *rolling_means,
            *rolling_stds,
            lag_1 - lag_2,
            lag_1 - lag_3,
            (lag_1 + 1.0) / (mean_6 + 1.0),
            (lag_1 - lag_3) / 2.0,
            np.log1p(lag_1),
            np.log1p(mean_3),
            x_lag_1,
            x_lag_1 - x_lag_2,
            category_code,
            np.full(demand.shape[0], period + 1, dtype=float)
        ])
        rows.append(block)
        if include_target:
            targets.append(demand[:, period])
    names = [f'demand_lag_{lag}' for lag in DEMAND_LAGS]
    names += [f'demand_mean_{window}' for window in ROLLING_WINDOWS]
    names += [f'demand_std_{window}' for window in ROLLING_WINDOWS]
    names += ['demand_diff_1', 'demand_diff_3', 'lag1_to_mean6', 'recent_slope']
    names += ['log_demand_lag_1', 'log_demand_mean_3']
    names += [f'{column}_lag_1' for column in X_COLUMNS]
    names += [f'{column}_change_1' for column in X_COLUMNS]
    names += ['category_code', 'period_index']
    return SupervisedData(
        features=np.vstack(rows).astype(np.float32),
        target=np.concatenate(targets).astype(np.float32) if include_target else None,
        feature_names=names,
        target_periods=target_periods
    )

def common_features(raw):
    d = np.concatenate([raw['train_demand'], raw['test_demand']], axis=1)
    x = np.concatenate([raw['train_x'], raw['test_x']], axis=1)
    f = build_supervised_data(d, x, raw['categories'], np.arange(12, 89), include_target=False)
    X = f.features.reshape(77, len(d), -1)
    X[48:, :, -1] = np.arange(1, 30)[:, None]
    return (d, X, f.feature_names)

def normalized_features(X, scale):
    f = X.copy()
    f[:, :, :13] /= scale[None, :, None]
    f[:, :, 14] /= scale[None, :]
    return f
