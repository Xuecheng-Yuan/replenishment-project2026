"""只使用历史需求的简洁滚动时间序列模型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForecastPath:
    """未来各期的一步预测，以及同一时点可获得的误差尺度。"""

    forecasts: np.ndarray
    error_scales: np.ndarray


def cold_start_forecast_path(
    training: np.ndarray,
    episode: np.ndarray,
    categories: np.ndarray,
    model: dict[str, object],
    *,
    error_window: int = 12,
) -> ForecastPath:
    """把 episode 视为独立窗口，观察本期后才预测下一期。

    train 只提供增长先验和误差比例，绝不把最后一期的需求水平传入新窗口。
    返回的第 t 列是在观察 episode 第 t 期后形成的下一期预测。
    """

    training = _as_matrix(training, "training")
    episode = _as_matrix(episode, "episode")
    categories = np.asarray(categories)
    if len(training) != len(episode) or len(categories) != len(training):
        raise ValueError("training、episode 和 categories 的 SKU 数必须一致。")
    if training.shape[1] < 12 or error_window < 1:
        raise ValueError("训练历史至少需要12期，error_window 必须为正整数。")

    absolute_errors, actual_history = _training_errors(training, categories, model)
    forecasts: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    observed = np.empty((len(training), 0), dtype=float)
    growth_prior = _category_growth_prior(training, categories)

    for period in range(episode.shape[1]):
        actual = episode[:, period]
        if forecasts:
            absolute_errors.append(np.abs(actual - forecasts[-1]))
            actual_history.append(actual)
        observed = np.column_stack([observed, actual])

        prediction = (
            observed[:, -1] * growth_prior
            if observed.shape[1] == 1
            else _forecast_next(observed, categories, model)
        )
        recent_errors = absolute_errors[-error_window:]
        recent_actual = actual_history[-error_window:]
        relative_wape = np.sum(recent_errors, axis=0) / np.maximum(
            np.sum(recent_actual, axis=0), 1.0
        )
        forecasts.append(np.clip(prediction, 0.0, None))
        scales.append(relative_wape * np.maximum(prediction, 1.0))

    return ForecastPath(np.column_stack(forecasts), np.column_stack(scales))


def build_targets(
    path: ForecastPath,
    categories: np.ndarray,
    safety_factors: dict[str, float],
) -> np.ndarray:
    """把每期观察后形成的下一期预测转换为整数 base-stock。"""

    categories = np.asarray(categories).astype(str)
    missing = set(categories) - set(safety_factors)
    if missing:
        raise ValueError(f"安全系数缺少类别：{sorted(missing)}")
    factors = np.array([safety_factors[value] for value in categories], dtype=float)
    if (factors < 0).any():
        raise ValueError("安全系数不能为负数。")
    target = path.forecasts + factors[:, None] * path.error_scales
    return np.ceil(np.clip(target, 0.0, None))


def simulate_dynamic_policy(
    demand: np.ndarray,
    initial_inventory: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """快速模拟提前期为1的动态 base-stock，用于 train 内部调参。"""

    demand = _as_matrix(demand, "demand")
    targets = _as_matrix(targets, "targets")
    on_hand = np.asarray(initial_inventory, dtype=float).copy()
    if demand.shape != targets.shape or len(on_hand) != len(demand):
        raise ValueError("需求、目标库存和初始库存的维度不一致。")

    arrivals = np.zeros(len(demand), dtype=float)
    total_sales = 0.0
    inventory_sum = 0.0
    for period in range(demand.shape[1]):
        on_hand += arrivals
        sales = np.minimum(on_hand, demand[:, period])
        on_hand -= sales
        arrivals = np.maximum(0.0, targets[:, period] - on_hand)
        total_sales += float(sales.sum())
        inventory_sum += float(on_hand.sum())

    total_demand = float(demand.sum())
    average_inventory = inventory_sum / demand.shape[1]
    return {
        "service_level": total_sales / total_demand if total_demand else 1.0,
        "turnover": total_sales / average_inventory if average_inventory else float("inf"),
        "total_demand": total_demand,
        "total_sales": total_sales,
        "inventory_sum": inventory_sum,
    }


def _forecast_next(
    history: np.ndarray,
    categories: np.ndarray,
    model: dict[str, object],
) -> np.ndarray:
    """用一个小而透明的模型集合预测下一期。"""

    name = str(model["name"])
    window = min(int(model.get("window", history.shape[1])), history.shape[1])
    recent = history[:, -window:]

    if name == "naive":
        forecast = history[:, -1]
    elif name == "moving_average":
        forecast = recent.mean(axis=1)
    elif name == "drift":
        forecast = recent[:, -1] + (recent[:, -1] - recent[:, 0]) / max(window - 1, 1)
    elif name == "linear_trend":
        forecast = _linear_next(recent)
    elif name == "ses":
        forecast = _holt_next(history, float(model["alpha"]), 0.0, 1.0)
    elif name == "damped_holt":
        forecast = _holt_next(
            history,
            float(model["alpha"]),
            float(model["beta"]),
            float(model["damping"]),
        )
    elif name == "arima_110":
        differences = np.diff(recent, axis=1)
        numerator = (differences[:, 1:] * differences[:, :-1]).sum(axis=1)
        denominator = (differences[:, :-1] ** 2).sum(axis=1)
        phi = np.divide(numerator, denominator, out=np.zeros(len(history)), where=denominator > 0)
        forecast = history[:, -1] + np.clip(phi, -0.98, 0.98) * differences[:, -1]
    elif name == "hierarchical_trend":
        forecast = _hierarchical_next(history, categories, window, int(model["share_window"]))
    else:
        raise ValueError(f"未知时间序列模型：{name}")
    return np.clip(np.asarray(forecast, dtype=float), 0.0, None)


def _training_errors(
    history: np.ndarray,
    categories: np.ndarray,
    model: dict[str, object],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """生成 train 内严格的一步预测误差，供动态安全库存使用。"""

    errors: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    for period in range(12, history.shape[1]):
        prediction = _forecast_next(history[:, :period], categories, model)
        errors.append(np.abs(history[:, period] - prediction))
        actuals.append(history[:, period])
    return errors, actuals


def _category_growth_prior(history: np.ndarray, categories: np.ndarray) -> np.ndarray:
    """从 train 学习分类中位增长率，不携带需求绝对水平。"""

    prior = np.ones(len(history), dtype=float)
    for category in np.unique(categories):
        mask = categories == category
        totals = history[mask].sum(axis=0)
        ratios = totals[1:] / np.maximum(totals[:-1], 1.0)
        prior[mask] = float(np.median(ratios))
    return prior


def _linear_next(values: np.ndarray) -> np.ndarray:
    """用闭式最小二乘拟合局部线性趋势。"""

    time = np.arange(values.shape[1], dtype=float)
    centered = time - time.mean()
    slope = ((values - values.mean(axis=1, keepdims=True)) @ centered) / (centered @ centered)
    return values.mean(axis=1) + slope * (values.shape[1] - time.mean())


def _holt_next(history: np.ndarray, alpha: float, beta: float, damping: float) -> np.ndarray:
    """向量化的简单指数平滑/Holt阻尼趋势。"""

    level = history[:, 0].copy()
    trend = history[:, 1] - history[:, 0] if beta > 0 else np.zeros(len(history))
    for period in range(1, history.shape[1]):
        previous_level = level
        fitted = level + damping * trend
        level = alpha * history[:, period] + (1.0 - alpha) * fitted
        if beta > 0:
            trend = beta * (level - previous_level) + (1.0 - beta) * damping * trend
    return level + damping * trend


def _hierarchical_next(
    history: np.ndarray,
    categories: np.ndarray,
    trend_window: int,
    share_window: int,
) -> np.ndarray:
    """先预测类别总量，再按近期SKU份额分配，保证预测可加总。"""

    forecast = np.zeros(len(history), dtype=float)
    for category in np.unique(categories):
        mask = categories == category
        group = history[mask]
        aggregate = group.sum(axis=0, keepdims=True)
        group_total = float(_linear_next(aggregate[:, -trend_window:])[0])
        recent_sales = group[:, -min(share_window, group.shape[1]) :].sum(axis=1)
        if recent_sales.sum() == 0:
            shares = np.full(len(group), 1.0 / len(group))
        else:
            shares = recent_sales / recent_sales.sum()
        forecast[mask] = max(0.0, group_total) * shares
    return forecast


def _as_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError(f"{name} 必须是非负、有限的二维数组。")
    return matrix
