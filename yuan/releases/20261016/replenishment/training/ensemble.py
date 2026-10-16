"""14个需求专家的四种融合，使用时间向前OOF；不是测试集挑最优模型。"""
import numpy as np
import pandas as pd
from ..runtime import ROOT, data, save
from ..evaluation import key, target_policy
from ..evaluation import simulate as rollout, calibrate_target as calibrate
from ..features import common_features
from . import forecast
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ..registry import FORECAST_MODELS as NAMES

def simulate(d, initial, p, sigma, z, score_start=0):
    return rollout(d, initial, target_policy(p, sigma, z), score_start)

def train_raw(raw):
    return dict(raw, test_demand=np.zeros_like(raw['test_demand']), test_x=np.zeros_like(raw['test_x']))

def projection(v):
    """逐行投影到非负、和为1的概率单纯形。"""
    u = np.sort(v, axis=-1)[..., ::-1]
    s = np.cumsum(u, axis=-1) - 1
    j = np.arange(1, v.shape[-1] + 1)
    rho = (u - s / j > 0).sum(-1) - 1
    theta = np.take_along_axis(s, rho[..., None], -1)[..., 0] / (rho + 1)
    return np.maximum(v - theta[..., None], 0)

def check_weights(w):
    assert np.isfinite(w).all() and (w >= -1e-12).all()
    np.testing.assert_allclose(w.sum(-1), 1, atol=1e-10)

def uncertainty(pred, actual, scale, mode):
    """二层模型仅用OOF误差估计每SKU不确定性。"""
    errors = actual - pred
    fixed = np.maximum(errors.std(1, ddof=1), 0.03 * scale)
    cv = (errors / np.maximum(pred, np.maximum(0.1 * scale, 1)[:, None])).std(1, ddof=1)
    cv = np.clip(cv, 0.015, 3)
    return (fixed, cv)

def risk_scale(pred, scale, fixed, cv, mode):
    """固定残差尺度，或随预测水平变化的相对尺度。"""
    return fixed if mode == 'fixed' else np.maximum(pred, 0.1 * scale[:, None]) * cv[:, None]

def global_weights(P, y):
    """用时间向前OOF预测学习非负、和为1的全局融合权重。"""
    G = np.einsum('ntm,ntk->mk', P, P) / y.size + 0.001 * np.eye(P.shape[2])
    b = np.einsum('ntm,nt->m', P, y) / y.size
    fun = lambda w: 0.5 * w @ G @ w - b @ w + 0.5 * np.mean(y * y)
    jac = lambda w: G @ w - b
    starts = [
        np.full(P.shape[2], 1 / P.shape[2]),
        np.eye(P.shape[2])[np.argmin(((P - y[..., None]) ** 2).mean((0, 1)))]
    ]
    fits = [minimize(
        fun,
        w,
        jac=jac,
        method='SLSQP',
        bounds=[(0, 1)] * len(w),
        constraints=[dict(type='eq', fun=lambda v: v.sum() - 1, jac=lambda v: np.ones_like(v))],
        options=dict(ftol=1e-12, maxiter=1000)
    ) for w in starts]
    assert all((f.success for f in fits)), [f.message for f in fits]
    assert abs(fits[0].fun - fits[1].fun) < 1e-05
    fit = min(fits, key=lambda f: f.fun)
    w = projection(fit.x)
    check_weights(w)
    return (
        w,
        dict(
            success=True,
            objective=float(fit.fun),
            two_start_objective_gap=abs(float(fits[0].fun - fits[1].fun)),
            iterations=int(fit.nit),
            weights=dict(zip(NAMES, w.tolist()))
        )
    )

def sku_weights(P, y, prior, tau, iterations=800):
    """逐SKU权重向全局权重收缩，FISTA求解并检查收敛。"""
    n = P.shape[1]
    G = np.einsum('ntm,ntk->nmk', P, P) / n
    b = np.einsum('ntm,nt->nm', P, y) / n
    reg = tau / n
    G += reg * np.eye(P.shape[2])[None]
    b += reg * prior[None]
    lip = np.linalg.eigvalsh(G)[:, -1].clip(1e-08)
    w = np.broadcast_to(prior, b.shape).copy()
    v = w.copy()
    moment = 1.0
    history = []
    for step in range(iterations):
        grad = np.einsum('nmk,nk->nm', G, v) - b
        new = projection(v - grad / lip[:, None])
        nm = (1 + np.sqrt(1 + 4 * moment * moment)) / 2
        v = new + (moment - 1) / nm * (new - w)
        w = new
        moment = nm
        if (step + 1) % 100 == 0:
            grad = np.einsum('nmk,nk->nm', G, w) - b
            res = np.max(np.abs(w - projection(w - grad)))
            history.append(dict(iteration=step + 1, maximum_projected_gradient=float(res)))
            if res < 1e-07:
                break
    check_weights(w)
    grad = np.einsum('nmk,nk->nm', G, w) - b
    res = np.max(np.abs(w - projection(w - grad)))
    assert res < 0.0001, ('Stacking solver did not converge', float(res))
    return (w, dict(tau=tau, iterations=step + 1, maximum_projected_gradient=float(res), history=history))

def grouping(raw):
    """仅用前42期训练数据做六组SKU分组，服务于硬路由收缩。"""
    d = raw['train_demand'][:, :42]
    mu = np.maximum(d.mean(1), 0.5)
    cv = d.std(1) / mu
    trend = (d[:, -6:].mean(1) - d[:, :6].mean(1)) / mu
    stats = np.column_stack([np.log1p(mu), cv, trend, (d == 0).mean(1), d[:, -3:].mean(1) / mu])
    x = raw['train_x'][:, :42].mean(1)
    xs = StandardScaler().fit_transform(x)
    pc = PCA(n_components=3, random_state=42).fit_transform(xs)
    f = StandardScaler().fit_transform(np.column_stack([stats, 0.5 * pc]))
    labels = KMeans(n_clusters=6, n_init=10, random_state=42).fit_predict(f)
    return (labels, f)

def fit_meta():
    """用19—42期OOF拟合二层，43—60期验证，绝不按测试挑SKU模型。"""
    raw = train_raw(data())
    N = len(raw['initial'])
    M = len(NAMES)
    base = ROOT / 'models/base'
    caches = [dict(np.load(base / (n + '.npz'))) for n in NAMES]
    oof = np.stack([c['oof'] for c in caches], 2)
    val = np.stack([c['validation_prediction'] for c in caches], 2)
    warm = np.stack([c['warm'] for c in caches], 2)
    assert oof.shape == (N, 24, M) and val.shape == (N, 49, M)
    actual = raw['train_demand'][:, 18:42]
    scale = np.maximum(raw['train_demand'][:, :42].mean(1), 0.5)
    P = oof / scale[:, None, None]
    y = actual / scale[:, None]
    prior, global_log = global_weights(P, y)
    labels, group_features = grouping(raw)
    pd.DataFrame(dict(item_id=raw['item_ids'], group=labels)).to_csv(
        ROOT / 'logs/sku_groups.csv',
        index=False,
        encoding='utf-8-sig'
    )
    candidates = []
    solver_logs = dict(global_stacking=global_log)
    candidates.append(dict(
        id='equal_mean',
        family='equal_mean',
        parameters={},
        weights=np.full((N, M), 1 / M)
    ))
    candidates.append(dict(
        id='global_stacking',
        family='global_stacking',
        parameters={},
        weights=np.broadcast_to(prior, (N, M)).copy()
    ))
    for tau in [6.0, 24.0, 96.0]:
        print('SKU WEIGHTS tau', tau, flush=True)
        w, log = sku_weights(P, y, prior, tau)
        cid = 'sku_stacking_tau' + str(int(tau))
        solver_logs[cid] = log
        candidates.append(dict(id=cid, family='sku_stacking', parameters=dict(tau=tau), weights=w))
    inv = []
    fail = []
    for j, name in enumerate(NAMES):
        fixed, cv = uncertainty(oof[:, :, j], actual, scale, 'fixed')
        path = np.column_stack([warm[:, :, j], oof[:, :, j], val[:, 30, j]])
        _, arrays = simulate(raw['train_demand'][:, 12:42], raw['initial'], path, fixed, 2.0, 6)
        inv.append(arrays[1][:, 6:].sum(1))
        fail.append((arrays[1][:, 6:] <= 0).mean(1))
    I = np.stack(inv, 1)
    fail = np.stack(fail, 1)
    costI = I / (actual.sum() / N)
    for penalty in [0.25, 1.0, 4.0, 16.0]:
        cost = costI + penalty * fail
        pooled = np.empty_like(cost)
        for g in np.unique(labels):
            pooled[labels == g] = cost[labels == g].mean(0)
        for tau in [0.0, 24.0]:
            alpha = 24 / (24 + tau)
            choice = (alpha * cost + (1 - alpha) * pooled).argmin(1)
            w = np.eye(M)[choice]
            candidates.append(dict(
                id=f'sku_router_lambda{penalty}_tau{tau}',
                family='sku_router',
                parameters=dict(penalty=penalty, tau=tau),
                weights=w
            ))
    search = []
    best = {}
    for ci, c in enumerate(candidates):
        w = c['weights']
        p = np.einsum('ntm,nm->nt', val, w)
        mp = np.einsum('ntm,nm->nt', oof, w)
        fixed, cv = uncertainty(mp, actual, scale, 'fixed')
        options = []
        for mode in ['fixed', 'relative']:
            sigma = risk_scale(p, scale, fixed, cv, mode)
            z, m, records = calibrate(raw['train_demand'][:, 12:60], raw['initial'], p, sigma, 30)
            options.append((key(m), mode, z, m))
            search.extend((dict(
                candidate=c['id'],
                family=c['family'],
                parameters=c['parameters'],
                scale_mode=mode,
                **r
            ) for r in records))
        _, mode, z, m = min(options, key=lambda q: q[0])
        c.update(scale_mode=mode, z=z, validation=m, fixed=fixed, cv=cv)
        print(
            'VALIDATED',
            c['id'],
            mode,
            'R',
            round(m['ready_rate'], 6),
            'J',
            round(m['inventory_demand_ratio'], 6),
            flush=True
        )
        if c['family'] not in best or key(m) < key(best[c['family']]['validation']):
            best[c['family']] = c
    out = ROOT / 'models/meta'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / 'all_candidate_weights.npz', **{c['id']: c['weights'] for c in candidates})
    selections = []
    for name, c in best.items():
        np.savez_compressed(
            out / (name + '.npz'),
            weights=c['weights'],
            fixed=c['fixed'],
            cv=c['cv'],
            z=c['z']
        )
        row = {k: v for k, v in c.items() if k not in ['weights', 'fixed', 'cv']}
        selections.append(row)
        w = c['weights']
        dom = w.argmax(1)
        frame = pd.DataFrame(dict(
            item_id=raw['item_ids'],
            dominant_model=np.array(NAMES)[dom],
            maximum_weight=w.max(1),
            group=labels
        ))
        for j, n in enumerate(NAMES):
            frame['weight_' + n] = w[:, j]
        frame.to_csv(out / (name + '_sku_weights.csv'), index=False, encoding='utf-8-sig')
    save(out / 'selected.json', selections)
    save(ROOT / 'logs/solver_logs.json', solver_logs)
    pd.DataFrame(search).to_csv(
        ROOT / 'logs/ensemble_validation_search.csv',
        index=False,
        encoding='utf-8-sig'
    )
    early = (P[:, :12] - y[:, :12, None]) ** 2
    late = (P[:, 12:] - y[:, 12:, None]) ** 2
    stability = dict(
        meta_periods=24,
        early_late_MSE_winner_switch_share=float((early.mean(1).argmin(1) != late.mean(1).argmin(1)).mean()),
        warning='Short-window best expert may be unstable; does not prove test improvement'
    )
    save(ROOT / 'logs/meta_stability.json', stability)
    return selections

def fit(name, raw, cut, end):
    """按时间前缀拟合单个基模型，输出下一块需要的预测。"""
    d, X, _ = common_features(raw)
    params = forecast.configurations(name)[0]
    if name in forecast.GRID:
        pack = forecast.ml_fit(name, params, d, X, cut)
        p = np.zeros((len(d), end - 11))
        if cut == 42:
            # 验证必须从13期预热，且需要61期的下一步特征来记录最后订单。
            p = forecast.ml_predict(pack, X[:end - 11])
        elif cut != 60:
            first = 0 if cut == 18 else cut - 12
            p[:, first:end - 12] = forecast.ml_predict(pack, X[first:end - 12])
    else:
        pack, p, _, _ = forecast.history_prediction(d, name, params, cut)
        p = p[:, :end - 11]
    assert np.isfinite(p).all() and (p >= 0).all()
    return (pack, p, params)
