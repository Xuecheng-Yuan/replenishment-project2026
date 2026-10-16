"""五种历史预测和九种ML预测，统一由训练期验证选择补货尺度。"""
import numpy as np
from ..features import common_features, normalized_features
GRID = {
    'linear_regression': [{}],
    'ridge': [{'alpha': 1.0}, {'alpha': 100.0}],
    'lasso': [{'alpha': 0.001}, {'alpha': 0.01}],
    'elastic_net': [{'alpha': 0.001, 'l1_ratio': 0.3}, {'alpha': 0.01, 'l1_ratio': 0.7}],
    'svr': [{'C': 1.0, 'epsilon': 0.05}, {'C': 10.0, 'epsilon': 0.1}],
    'random_forest': [
        {'n_estimators': 80, 'max_depth': 16, 'min_samples_leaf': 3},
        {'n_estimators': 120, 'max_depth': 24, 'min_samples_leaf': 3}
    ],
    'xgboost': [
        {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.9},
        {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.05, 'subsample': 0.9}
    ],
    'lightgbm': [
        {'n_estimators': 250, 'num_leaves': 31, 'learning_rate': 0.05, 'min_child_samples': 30},
        {'n_estimators': 350, 'num_leaves': 63, 'learning_rate': 0.05, 'min_child_samples': 30}
    ],
    'catboost': [
        {'iterations': 200, 'depth': 6, 'learning_rate': 0.05},
        {'iterations': 300, 'depth': 8, 'learning_rate': 0.05}
    ]
}

def ml_fit(name, params, d, X, split):
    """仅用截止期之前的样本拟合；SVR和随机森林按固定种子限量抽样。"""
    scale = np.maximum(d[:, :split].mean(1), 0.5)
    f = normalized_features(X, scale)
    xx = f[:split - 12].reshape(-1, 59)
    yy = (np.log1p(d[:, 12:split]).T - np.log1p(scale)[None, :]).ravel()
    # 核SVR限制拟合样本，不限制评价的SKU数。
    if name == 'svr':
        limit = 8000 if split <= 42 else 12000
    elif name == 'random_forest':
        limit = 60000 if split <= 42 else 120000
    else:
        limit = len(yy)
    if len(yy) > limit:
        ix = np.random.default_rng(42).choice(len(yy), limit, False)
        xx, yy = (xx[ix], yy[ix])
    model = build_regressor(name, params, 2)
    model.fit(xx, yy)
    return dict(estimator=model, scale=scale, split=split, sample_count=len(yy), parameters=params, name=name)

def ml_predict(pack, X):
    """将对数相对预测还原为需求量；不在预测阶段重新拟合。"""
    f = normalized_features(X, pack['scale'])
    pred = pack['estimator'].predict(f.reshape(-1, 59)).reshape(f.shape[:2])
    return np.expm1(np.clip(pred + np.log1p(pack['scale'])[None, :], 0, 20)).T

def ml_bundle(name, params, d, X, split):
    """最后12期的前缀样本外残差用于估计不确定性。"""
    pack = ml_fit(name, params, d, X, split)
    p = ml_predict(pack, X)
    cut = split - 12
    oof = ml_fit(name, params, d, X, cut)
    pred = ml_predict(oof, X[cut - 12:split - 12])
    errors = d[:, cut:split] - pred
    sigma = np.maximum(errors.std(1, ddof=1), 0.03 * pack['scale'])
    pack['residual_prefix'] = cut
    return (pack, p, sigma, errors)

def history_prediction(d, kind, params, split):
    """历史统计预测；目标期的预测只读取此前已经观测的需求。"""
    n, T = d.shape
    out = np.zeros((n, T + 1))
    if kind == 'baseline':
        out[:] = d[:, :split].mean(1)[:, None]
    elif kind in ['naive', 'moving_average']:
        w = 1 if kind == 'naive' else params.get('window', 12)
        for t in range(1, T + 1):
            out[:, t] = d[:, max(0, t - w):t].mean(1)
    elif kind == 'exponential_smoothing':
        alpha, beta = (params['alpha'], params['beta'])
        level = d[:, 0].copy()
        trend = d[:, 1] - d[:, 0] if beta else np.zeros(n)
        for t in range(1, T + 1):
            out[:, t] = np.maximum(level + 0.98 * trend, 0)
            if t < T:
                old = level
                level = alpha * d[:, t] + (1 - alpha) * out[:, t]
                trend = beta * (level - old) + (1 - beta) * 0.98 * trend
    elif kind == 'arima':
        order = params['p']
        delta = np.diff(d[:, :split], axis=1)
        if order == 0:
            coef = delta.mean(1)[:, None]
        else:
            y = delta[:, order:]
            design = np.stack([np.ones_like(y)] + [delta[:, order - k:-k] for k in range(1, order + 1)], 2)
            lhs = np.einsum('ntj,ntk->njk', design, design)
            lhs += np.eye(order + 1)[None] * 1e-06
            rhs = np.einsum('ntj,nt->nj', design, y)
            coef = np.linalg.solve(lhs, rhs[..., None])[..., 0]
            weight = np.abs(coef[:, 1:]).sum(1)
            coef[:, 1:] *= np.minimum(1, 0.98 / np.maximum(weight, 1e-09))[:, None]
        for t in range(max(order + 1, 1), T + 1):
            step = coef[:, 0].copy()
            for k in range(1, order + 1):
                step += coef[:, k] * (d[:, t - k] - d[:, t - k - 1])
            out[:, t] = np.maximum(d[:, t - 1] + step, 0)
        params = dict(params, coefficients=coef)
    p = out[:, 12:]
    errors = d[:, 12:split] - p[:, :split - 12]
    sigma = np.maximum(errors.std(1, ddof=1), 0.03 * np.maximum(d[:, :split].mean(1), 0.5))
    if kind == 'baseline':
        sigma = np.maximum(d[:, :split].std(1, ddof=1), sigma)
    return (dict(name=kind, parameters=params, split=split), p, sigma, errors)

def configurations(name):
    """与最新版实验一致的有限候选网格，不代表全局最优参数。"""
    if name in GRID:
        return GRID[name]
    if name in ['baseline', 'naive']:
        return [{}]
    if name == 'moving_average':
        return [{'window': w} for w in [3, 6, 12]]
    if name == 'exponential_smoothing':
        return [dict(alpha=a, beta=b) for a, b in [(0.25, 0), (0.5, 0.1), (0.8, 0.15)]]
    if name == 'arima':
        return [{'p': p, 'd': 1, 'q': 0, 'fit': 'conditional_least_squares'} for p in [0, 1, 2]]
    raise ValueError('未知预测模型')

def cv_scale(errors, actual, mean, prediction):
    """用前缀样本外相对残差计算风险尺度。"""
    old_prediction = actual - errors
    cv = (errors / np.maximum(old_prediction, np.maximum(0.1 * mean, 1)[:, None])).std(1, ddof=1)
    cv = np.maximum(cv, 0.015)
    return (np.maximum(prediction, 0.1 * mean[:, None]) * cv[:, None], cv)

def build_regressor(name, parameters, n_jobs=2):
    """动态加载真实模型构造器；未选择的可选库不必导入。"""
    from importlib import import_module
    return import_module('replenishment.models.' + name).build(parameters, n_jobs)


def predict(name, pack, raw, start=0):
    """已拟合预测器的滚动推理；AR系数固定，不在测试上重新估计。"""
    d, X, _ = common_features(raw)
    if name in GRID:
        return ml_predict(pack, X[start:])
    if name != 'arima':
        return history_prediction(d, name, dict(pack['parameters']), pack['split'])[1][:, start:]
    order = pack['parameters']['p']
    coef = pack['parameters']['coefficients']
    p = np.zeros((len(d), d.shape[1] + 1))
    for t in range(max(order + 1, 1), d.shape[1] + 1):
        step = coef[:, 0].copy()
        for k in range(1, order + 1):
            step += coef[:, k] * (d[:, t - k] - d[:, t - k - 1])
        p[:, t] = np.maximum(d[:, t - 1] + step, 0)
    return p[:, 12 + start:]
