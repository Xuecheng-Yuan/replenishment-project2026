"""四期自适应情景MPC；仅保留所需聚类函数，不导入鲁棒模型。"""
import time
import numpy as np
import pandas as pd
from ..runtime import ROOT, data, sha, freeze_selection
from ..evaluation import key, simulate
from ..features import normalized_features
from scipy.cluster.vq import kmeans2
import torch
import torch.nn.functional as F
torch.set_num_threads(2)

def feature_matrix(y, x):
    """训练需求统计量加20特征均值的前三个PCA分量。"""
    mu = np.maximum(y.mean(0), 1.0)
    cv = y.std(0) / mu
    demand = np.column_stack((
        np.log1p(mu),
        np.log1p(cv),
        (y == 0).mean(0),
        np.log1p(np.quantile(y, 0.95, axis=0) / mu),
        np.log1p(y[-8:].mean(0) / np.maximum(y[:8].mean(0), 1))
    ))
    xm = x.mean(0)
    z = (xm - xm.mean(0)) / np.maximum(xm.std(0), 1e-08)
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    pcs = z @ vt[:3].T
    a = (demand - demand.mean(0)) / np.maximum(demand.std(0), 1e-08)
    b = (pcs - pcs.mean(0)) / np.maximum(pcs.std(0), 1e-08)
    return (np.column_stack((a, b * 0.5)), demand, xm)

def cluster(y, x, k=6):
    """多初值三组聚类，仅用拟合前缀的数据。"""
    f, demand, xm = feature_matrix(y, x)
    best = None
    rng = np.random.default_rng(20260929)
    for _ in range(8):
        centers, labels = kmeans2(f, f[rng.choice(len(f), k, replace=False)], iter=60, minit='matrix')
        loss = ((f - centers[labels]) ** 2).sum()
        if best is None or loss < best[0]:
            best = (loss, labels, centers)
    return (best[1], demand, xm)

def fit_predict(raw, d, X, split, ci):
    """按训练前缀分组，拟合四个预测步长的LightGBM。"""
    from lightgbm import LGBMRegressor
    scale = np.maximum(d[:, :split].mean(1), 0.5)
    f = normalized_features(X, scale)
    labels = cluster(d[:, :split].T, raw['train_x'][:, :split].transpose(1, 0, 2), 3)[0]
    params = dict(
        n_estimators=120 if ci == 0 else 200,
        num_leaves=15 if ci == 0 else 31,
        learning_rate=0.06,
        min_child_samples=40,
        n_jobs=2,
        verbosity=-1,
        random_state=42
    )
    pred = np.zeros((77, len(d), 4))
    models = {}
    for g in np.unique(labels):
        mask = labels == g
        for h in range(4):
            length = split - 12 - h
            xx = f[:length, mask].reshape(-1, 59)
            y = (np.log1p(d[mask, 12 + h:split]).T - np.log1p(scale[mask])[None]).ravel()
            model = LGBMRegressor(objective='regression_l1', **params)
            model.fit(xx, y)
            out = model.predict(f[:, mask].reshape(-1, 59)).reshape(77, mask.sum())
            pred[:, mask, h] = np.expm1(np.clip(out + np.log1p(scale[mask])[None], 0, 20))
            models[f'{g}_{h}'] = model
    return (dict(models=models, scale=scale, groups=labels, parameters=params, split=split), pred)

def mpc_policy(pack, raw, start, z, rho, alpha):
    """四期滚动优化，只执行第一期订单；误差修正仅用已知需求。"""
    forecast = pack['forecast']
    base_sigma = pack['sigma']
    groups = pack['groups']
    n = len(base_sigma)
    scale = torch.tensor(pack['scale'], dtype=torch.float32)
    bias = np.zeros(n)
    adapt_sigma = base_sigma.copy()
    full_d = np.concatenate([raw['train_demand'], raw['test_demand']], 1)

    def decide(t, b):
        k = start + t
        actual = full_d[:, 12 + k]
        error = actual - forecast[k, :, 0]
        bias[:] = (1 - alpha) * bias + alpha * error
        adapt_sigma[:] = (1 - alpha) * adapt_sigma + alpha * np.abs(error)
        group_bias = np.zeros(n)
        for group in np.unique(groups):
            group_bias[groups == group] = bias[groups == group].mean()
        means = np.maximum(forecast[k + 1] + (0.5 * bias + 0.5 * group_bias)[:, None], 0).T
        g = torch.Generator().manual_seed(42 + 10000 + k)
        eps = torch.randn(8, 4, n, generator=g)
        common = torch.randn(
            8,
            4,
            len(np.unique(groups)),
            generator=g
        )[:, :, torch.tensor(groups, dtype=torch.long)]
        sigma = torch.tensor(adapt_sigma, dtype=torch.float32)
        mu = torch.tensor(means, dtype=torch.float32)
        scen = (mu[None] + sigma[None, None] * (eps + 0.2 * common) / np.sqrt(1.04)).clamp_min(0)
        inv = torch.tensor(b, dtype=torch.float32)
        expected = inv
        initial = []
        for h in range(4):
            q = (mu[h] + z * sigma - expected).clamp_min(0)
            initial.append(q)
            expected = (expected + q - mu[h]).clamp_min(0)
        u = (torch.stack(initial) / scale[None]).detach().requires_grad_(True)
        for _ in range(12):
            stock = inv[None].expand(8, -1)
            ends = []
            ready = []
            for h in range(4):
                a = stock + u[h].clamp_min(0) * scale
                delta = a - scen[:, h]
                stock = delta.clamp_min(0)
                ends.append(stock)
                ready.append(torch.sigmoid((delta - 1) / (0.05 * scale)))
            inventory = torch.stack(ends)
            r = torch.stack(ready).mean()
            cost = inventory.sum() / scen.sum().clamp_min(1) + rho * (F.relu(0.95 - r).square() + F.relu(r - 0.96).square())
            grad = torch.autograd.grad(cost, u)[0]
            u = (u - 0.02 * (n * 4 * grad).clamp(-4, 4)).clamp_min(0).detach().requires_grad_(True)
        return (u[0] * scale).detach().numpy()
    return decide

def train():
    from ..runtime import training_data
    raw = training_data()
    out = ROOT / 'models/adaptive_mpc'
    records = []
    options = []
    start = time.time()
    for ci, rho, alpha in [(0, 80.0, 0.1), (1, 240.0, 0.2)]:
        pack = dict(np.load(out / f'forecast_42_{ci}.npz'))

        def run(z):
            m, _ = simulate(
                raw['train_demand'][:, 12:60],
                raw['initial'],
                mpc_policy(pack, raw, 0, z, rho, alpha),
                30
            )
            records.append(dict(candidate=ci, rho=rho, alpha=alpha, z=float(z), **m))
            print('MPC CAL', ci, z, m['ready_rate'], flush=True)
        for z in [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
            run(z)
        for _ in range(2):
            local = [r for r in records if r['candidate'] == ci]
            best = min(local, key=key)
            zs = sorted(set((r['z'] for r in local)))
            j = zs.index(best['z'])
            lo = zs[max(0, j - 1)]
            hi = zs[min(len(zs) - 1, j + 1)]
            for z in np.linspace(lo, hi, 5):
                if all((abs(z - r['z']) > 1e-08 for r in local)):
                    run(z)
        val = min([r for r in records if r['candidate'] == ci], key=key)
        options.append(val)
    best = min(options, key=key)
    ci = best['candidate']
    pack = dict(np.load(out / f'forecast_60_{ci}.npz'))
    np.savez_compressed(out / 'policy.npz', **pack)
    freeze_selection(
        'adaptive_mpc',
        dict(
            validation={k: v for k, v in best.items() if k not in ['candidate', 'rho', 'alpha', 'z']},
            z=best['z'],
            rho=best['rho'],
            alpha=best['alpha'],
            candidate=ci,
            horizon=4,
            scenarios=8,
            inner_steps=12,
            group_count=3,
            training_status='trained',
            training_seconds=time.time() - start,
            fit_cutoff=42,
            refit_cutoff=60,
            final_model_sha256=sha(out / f'forecaster_60_{ci}.joblib'),
            policy_sha256=sha(out / 'policy.npz'),
            method='fresh grouped four-head LightGBM + adaptive four-period teacher-loss scenario MPC; no imposed capacity'
        )
    )
    pd.DataFrame(records).to_csv(out / 'validation_search.csv', index=False)
    print('TRAINED adaptive_mpc', best['ready_rate'], flush=True)
