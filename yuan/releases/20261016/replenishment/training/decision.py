"""五种SPO/直接决策方法共用库存状态与动作代价构造。"""
import json
import numpy as np
from ..runtime import ROOT, data
from ..evaluation import simulate, target_policy
from ..features import normalized_features
from .forecast import history_prediction, ml_fit, ml_predict
from ..models.spo_linear import SPOPlusLinear
from ..models.xgboost_spo import XGBSPOHybrid
from ..models.lightgbm_spo import LGBMSPOHybrid
from ..models.direct_lightgbm import LGBMDirectSPOPolicy
ACTIONS = np.arange(0, 4.01, 0.5)
FAMILIES = ['spo_linear', 'xgboost_spo', 'lightgbm_spo', 'direct_xgboost', 'direct_lightgbm']

def centers(name, d, X, split):
    """构造动作中心；树SPO使用前缀拟合的需求模型，其他使用MA6。"""
    if name in ['xgboost_spo', 'lightgbm_spo']:
        base = 'xgboost' if name == 'xgboost_spo' else 'lightgbm'
        selection = ROOT / 'models' / base / 'selection.json'
        # 主入口先训练基础模型；函数独立测试时可用确定的第一候选。
        if selection.exists():
            params = json.loads(selection.read_text())['parameters']
        else:
            from .forecast import GRID
            params = GRID[base][0]
        full = ml_fit(base, params, d, X, split)
        oof = ml_fit(base, params, d, X, split - 12)
        p = ml_predict(full, X)
        op = ml_predict(oof, X)
        errors = d[:, split - 12:split] - op[:, split - 24:split - 12]
        sigma = np.maximum(errors.std(1, ddof=1), 0.03 * full['scale'])
        return (p, op, sigma, full)
    pack, p, sigma, errors = history_prediction(d, 'moving_average', {'window': 6}, split)
    return (p, p, sigma, pack)

def augment(f, b, center, sigma, scale, direct=False):
    """附加当前真实期末库存等状态，所有库存来自策略回放。"""
    if not direct:
        return np.column_stack([f, b / scale, center / scale, sigma / scale]).astype(np.float32)
    return np.column_stack([
        f,
        b / scale,
        np.log1p(b),
        b / np.maximum(center, 1),
        center / scale,
        sigma / scale,
        (b <= 0).astype(float),
        f[:, 0] - f[:, 1],
        f[:, 5]
    ]).astype(np.float32)

def targets(name, center, sigma, actions=None):
    """有限动作对应的目标库存，不是直接把未来真实需求作为决策。"""
    actions = ACTIONS if actions is None else np.asarray(actions, float)
    return np.maximum(
        center[:, None] * actions[None, :] if name.startswith('direct_')
        else center[:, None] + sigma[:, None] * actions[None, :],
        0
    )

def build_training(name, d, X, split, penalty, actions=None, center_bundle=None, return_orders=False):
    """用训练段需求构造动作代价；直接决策采用四期代理代价。"""
    p, op, sigma, base = centers(name, d, X, split) if center_bundle is None else center_bundle
    actions = ACTIONS if actions is None else np.asarray(actions, float)
    scale = np.maximum(d[:, :split].mean(1), 0.5)
    f = normalized_features(X, scale)
    _, behavior, bsigma, _ = history_prediction(d, 'moving_average', {'window': 6}, split)
    _, arr = simulate(d[:, 12:split], data()['initial'], target_policy(behavior[:, :split - 11], bsigma, 2.0))
    end = arr[1]
    horizon = 4 if name.startswith('direct_') else 1
    Xs = []
    Cs = []
    Ts = []
    Qs = []
    for period in range(split - 12, split - horizon + 1):
        inv = end[:, period - 13]
        center = op[:, period - 12]
        targ = targets(name, center, sigma, actions)
        available = np.maximum(inv[:, None], targ)
        stock = np.maximum(available - d[:, period, None], 0)
        failure = (stock <= 0).astype(float)
        cost = stock / max(d[:, :split].mean(), 1) + penalty * failure
        for h in range(1, horizon):
            future = np.maximum(stock, (center + 2 * sigma)[:, None])
            stock = np.maximum(future - d[:, period + h, None], 0)
            cost += stock / max(d[:, :split].mean(), 1) + penalty * (stock <= 0)
        Xs.append(augment(f[period - 12], inv, center, sigma, scale, name.startswith('direct_')))
        Cs.append(cost / horizon)
        Ts.append(targ)
        Qs.append(np.maximum(targ - inv[:, None], 0))
    xx, cc, tt, qq = map(np.vstack, (Xs, Cs, Ts, Qs))
    limit = 30000 if split == 42 else 45000
    ix = np.random.default_rng(42).choice(len(xx), min(limit, len(xx)), False)
    result = (
        xx[ix],
        cc[ix],
        tt[ix],
        dict(
            base=base,
            centers=p,
            sigma=sigma,
            scale=scale,
            features=f,
            sample_count=len(ix),
            horizon=horizon,
            penalty=penalty,
            actions=actions
        )
    )
    return (*result, qq[ix]) if return_orders else result

def estimator(name, penalty):
    """每种决策学习方法的真实损失与模型参数。"""
    if name == 'spo_linear':
        return SPOPlusLinear(epochs=16, learning_rate=0.03, batch_size=2048, ridge=0.001)
    if name in ['xgboost_spo', 'direct_xgboost']:
        return XGBSPOHybrid(
            alpha=0.8 if name == 'xgboost_spo' else 1.0,
            n_estimators=70,
            max_depth=4,
            n_jobs=2
        )
    if name == 'lightgbm_spo':
        return LGBMSPOHybrid(alpha=0.8, n_estimators=80, num_leaves=31, n_jobs=2)
    return LGBMDirectSPOPolicy(spo_weight=0.5, rank_weight=0.4, n_estimators=80, num_leaves=31, n_jobs=2)

def policy(name, model, pack, start, z):
    """当天需求已发生，使用当前库存决定下一期到货的订单。"""

    def decide(t, b):
        k = start + t + 1
        center = pack['centers'][:, k]
        targ = targets(name, center, pack['sigma'], pack.get('actions'))
        f = augment(pack['features'][k], b, center, pack['sigma'], pack['scale'], name.startswith('direct_'))
        choice = model.predict_actions(
            f,
            targ
        ) if name in ['lightgbm_spo', 'direct_lightgbm'] else model.predict_actions(f)
        target = targ[np.arange(len(b)), choice] + z * pack['sigma']
        return np.maximum(target - b, 0)
    return decide
