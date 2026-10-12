"""models.adaptive_mpc: actual model/processing implementation, not a legacy runner alias.

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
OUT=ROOT/'results/generated/adaptive_mpc'
WORK=OUT
TABLES=OUT/'图表'
OLD=ROOT/'results/generated/ready_immediate'
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def history_features(demand, x, categories, origins, scale):
    """Zero-indexed origin t observes d[:t+1], x[t], and nothing after t."""
    rows = []
    for t in origins:
        if t < 12:
            raise ValueError('Need raw x and >=12 observations')
        y = demand[:, :t + 1]
        lag = [y[:, -j] / scale for j in (1, 2, 3, 6, 12)]
        means = [y[:, -j:].mean(1) / scale for j in (3, 6, 12)]
        stds = [y[:, -j:].std(1, ddof=1) / scale for j in (3, 6, 12)]
        now = x[:, t - 12]
        change = now - x[:, t - 13] if t > 12 else np.zeros_like(now)
        rows.append(np.column_stack([*lag, *means, *stds, lag[0] - lag[1], (lag[0] - lag[2]) / 2, np.log1p(scale), categories, now, change]))
    return np.vstack(rows).astype('float32')

class GroupForecaster:

    def __init__(self, cfg, horizon=3, seed=42):
        self.cfg, self.horizon, self.seed = (cfg, horizon, seed)

    def _regressor(self):
        if self.cfg['family'] == 'ridge':
            return make_pipeline(StandardScaler(), Ridge(alpha=self.cfg['alpha']))
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=self.cfg['trees'], num_leaves=self.cfg['leaves'], objective='regression', learning_rate=0.06, min_child_samples=80, reg_lambda=5.0, n_jobs=4, verbosity=-1, random_state=self.seed)

    def fit(self, demand, x, categories, end):
        self.scale = np.maximum(demand[:, :end].mean(1), 1.0)
        self.sigma = np.maximum(demand[:, :end].std(1, ddof=1), 1.0)
        train = demand[:, :end]
        descriptors = np.column_stack([np.log1p(self.scale), self.sigma / self.scale, (train == 0).mean(1), (train[:, -6:].mean(1) - train[:, :6].mean(1)) / self.scale])
        self.cluster_scaler = StandardScaler().fit(descriptors)
        self.clusterer = KMeans(n_clusters=self.cfg['groups'], n_init=10, random_state=self.seed)
        self.groups = self.clusterer.fit_predict(self.cluster_scaler.transform(descriptors))
        self.models = []
        for h in range(1, self.horizon + 1):
            origins = np.arange(12, end - h)
            features = history_features(demand, x, categories, origins, self.scale)
            target = (demand[:, origins + h] / self.scale[:, None]).T.ravel()
            group_rows = np.tile(self.groups, len(origins))
            global_model = self._regressor().fit(features, target)
            experts = {}
            if self.cfg['groups'] > 1:
                for g in range(self.cfg['groups']):
                    mask = group_rows == g
                    experts[g] = self._regressor().fit(features[mask], target[mask])
            self.models.append((global_model, experts))
        return self

    def predict(self, demand, x, categories, origins):
        features = history_features(demand, x, categories, origins, self.scale)
        n = len(self.scale)
        group_rows = np.tile(self.groups, len(origins))
        result = []
        for global_model, experts in self.models:
            y = global_model.predict(features)
            if experts:
                local = y.copy()
                for g, model in experts.items():
                    mask = group_rows == g
                    local[mask] = model.predict(features[mask])
                y = 0.5 * y + 0.5 * local
            result.append(np.maximum(y.reshape(len(origins), n).T * self.scale[:, None], 0.0))
        return np.stack(result, axis=-1)

def project_orders(v, scale, capacity, lower=None):
    """Exact projection in normalized u-space, constraints a@u<=C and u>=l."""
    v = np.asarray(v, float)
    if v.ndim == 1:
        return project_orders(v[:, None], scale, capacity, None if lower is None else lower[:, None])[:, 0]
    lower = np.zeros_like(v) if lower is None else np.broadcast_to(lower, v.shape)
    if np.any(lower < 0) or np.any(scale @ lower > capacity * (1 + 1e-12)):
        raise ValueError('Infeasible projection lower bounds')
    q = np.maximum(v, lower)
    active = scale @ q > capacity
    if active.any():
        vv, ll = (v[:, active], lower[:, active])
        lo = np.zeros(vv.shape[1])
        hi = np.maximum(((vv - ll) / scale[:, None]).max(0), 0.0) + 1.0
        for _ in range(45):
            mid = (lo + hi) / 2
            quantity = scale @ np.maximum(vv - scale[:, None] * mid, ll)
            over = quantity > capacity
            lo = np.where(over, mid, lo)
            hi = np.where(over, hi, mid)
        q[:, active] = np.maximum(vv - scale[:, None] * hi, ll)
    return q

@dataclass(frozen=True)
class PolicyConfig:
    horizon: int = 3
    quantile: float = 0.99
    individual_mix: float = 0.5
    buffer_factor: float = 1.0
    adaptive_gain: float = 0.5
    risk_mix: float = 0.5
    service_weight: float = 10.0
    shortage_weight: float = 10.0
    steps: int = 16
    step_size: float = 0.04
    scenarios: int = 8
    seed: int = 42

def calibrate_buffer(residual, groups, sigma, cfg):
    local = np.quantile(residual, cfg.quantile, axis=1)
    pooled = np.empty(len(groups))
    for g in np.unique(groups):
        pooled[groups == g] = np.quantile(residual[groups == g], cfg.quantile)
    return np.maximum((1 - cfg.individual_mix) * pooled + cfg.individual_mix * local, 0.0) * sigma * cfg.buffer_factor

class ScenarioPolicy:

    def __init__(self, forecast, residual, groups, scale, sigma, capacity, cfg):
        self.forecast, self.residual = (forecast, residual)
        self.scale, self.sigma, self.capacity, self.cfg = (scale, sigma, capacity, cfg)
        self.buffer = calibrate_buffer(residual, groups, sigma, cfg)
        self.failures = np.zeros(len(scale))
        self.observed = 0
        self.diagnostics = []

    def objective(self, u, inventory, scenarios, service_weight, gradient=False):
        a = self.scale
        y = inventory[None, :, None] / a[None, :, None] + np.cumsum(u[None, :, :] - scenarios / a[None, :, None], axis=2)
        b = self.cfg.shortage_weight + service_weight[None, :, None]
        cost = (np.maximum(y, 0.0) + b * np.maximum(-y, 0.0)).mean(axis=(1, 2))
        k = len(cost)
        tail = max(1, int(np.ceil(k * 0.25)))
        weights = np.full(k, (1 - self.cfg.risk_mix) / k)
        weights[np.argsort(cost)[-tail:]] += self.cfg.risk_mix / tail
        value = float(weights @ cost)
        if not gradient:
            return value
        gy = np.where(y > 0, 1.0, np.where(y < 0, -b, 0.0))
        gy = np.einsum('k,kih->ih', weights, gy) / u.shape[1]
        gu = np.cumsum(gy[:, ::-1], axis=1)[:, ::-1]
        return (value, gu)

    def order(self, t, inventory, remaining):
        cfg = self.cfg
        h = min(cfg.horizon, remaining)
        m = self.forecast[:, t, :h]
        debt = np.maximum(self.failures - 0.05 * self.observed, 0.0)
        service_weight = cfg.service_weight * (1 + cfg.adaptive_gain * debt)
        buffer = self.buffer + cfg.adaptive_gain * debt * self.sigma
        lower_first = np.maximum(m[:, 0] + buffer + 1e-06 * self.scale - inventory, 0.0)
        conflict = lower_first.sum() > self.capacity
        lower = np.zeros_like(m)
        if not conflict:
            lower[:, 0] = lower_first / self.scale
        rng = np.random.default_rng(cfg.seed + int(t) * 101)
        starts = rng.integers(0, self.residual.shape[1] - h + 1, size=cfg.scenarios)
        errors = np.stack([self.residual[:, j:j + h] for j in starts])
        scenarios = np.maximum(m[None, :, :] + errors * self.sigma[None, :, None], 0.0)
        warm = np.zeros_like(m)
        stock = inventory.copy()
        for j in range(h):
            warm[:, j] = np.maximum(m[:, j] + buffer - stock, 0.0) / self.scale
            warm[:, j] = project_orders(warm[:, j], self.scale, self.capacity, lower[:, j])
            stock = np.maximum(stock + self.scale * warm[:, j] - m[:, j], 0.0)
        u = warm.copy()
        best = u.copy()
        best_value = initial_value = self.objective(u, inventory, scenarios, service_weight)
        for step in range(cfg.steps):
            _, grad = self.objective(u, inventory, scenarios, service_weight, True)
            u = project_orders(u - cfg.step_size * grad / np.sqrt(step + 1), self.scale, self.capacity, lower)
            value = self.objective(u, inventory, scenarios, service_weight)
            if value < best_value:
                best, best_value = (u.copy(), value)
        q = self.scale * best[:, 0]
        self.diagnostics.append(dict(period=t, lower_conflict=bool(conflict), initial_objective=initial_value, best_objective=best_value, capacity_utilization=float(q.sum() / self.capacity), lower_request=float(lower_first.sum())))
        assert np.isfinite(q).all() and q.min() >= 0 and (q.sum() <= self.capacity * (1 + 1e-10))
        return q

    def observe(self, inventory):
        self.observed += 1
        self.failures += inventory <= 1e-08

def simulate(demand, policy, reference, item_ids, output=None):
    """Exact same cold start and lost-sales timing as lead-one benchmark."""
    n, h = demand.shape
    mean = reference.mean(1)
    initial = 1.25 * mean
    capacity = 1.05 * mean.sum()
    assert np.isclose(capacity, policy.capacity)
    inventory = initial.copy()
    pending = np.zeros(n)
    sales = np.zeros_like(demand)
    stocks = sales.copy()
    orders = sales.copy()
    arrivals = sales.copy()
    for t in range(h):
        arrivals[:, t] = pending
        available = inventory + pending
        sales[:, t] = np.minimum(available, demand[:, t])
        inventory = available - sales[:, t]
        stocks[:, t] = inventory
        policy.observe(inventory)
        pending = policy.order(t + 1, inventory, h - t - 1) if t + 1 < h else np.zeros(n)
        orders[:, t] = pending
    rr = (stocks > 1e-08).mean(1)
    average = stocks.mean(1)
    weights = 1 / np.maximum(mean, 0.5)
    weights /= weights.mean()
    metrics = dict(mean_ready_rate=float(rr.mean()), sku_service_pass_rate=float((rr >= 0.95).mean()), pass_skus=int((rr >= 0.95).sum()), minimum_ready_rate=float(rr.min()), all_skus_feasible=bool((rr >= 0.95).all()), fill_rate=float(sales.sum() / demand.sum()) if demand.sum() > 0 else 1.0, turnover=float(sales.sum() / max(average.sum(), 1e-08)), weighted_turnover=float(weights @ sales.sum(1) / max(weights @ average, 1e-08)), average_total_inventory=float(average.sum()), capacity=float(capacity), periods=h, lead_time=1, lower_conflict_periods=sum((r['lower_conflict'] for r in policy.diagnostics)), mean_capacity_utilization=float(orders[:, :-1].sum(0).mean() / capacity))
    previous = np.column_stack([initial, stocks[:, :-1]])
    assert np.allclose(previous + arrivals - sales, stocks, rtol=1e-10, atol=1e-06)
    assert np.array_equal(arrivals[:, 1:], orders[:, :-1]) and (not arrivals[:, 0].any())
    assert not orders[:, -1].any() and np.all(orders.sum(0) <= capacity * (1 + 1e-10))
    assert np.isfinite(stocks).all() and np.min(stocks) >= 0
    if output is not None:
        import pandas as pd
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(dict(item_id=item_ids, ready_rate=rr, fill_rate=sales.sum(1) / np.maximum(demand.sum(1), 1e-08), average_inventory=average, turnover=sales.sum(1) / np.maximum(average, 1e-08), service_pass=rr >= 0.95)).to_csv(output / 'sku_metrics.csv', index=False)
        np.savez_compressed(output / 'trajectory.npz', item_ids=item_ids, demand=demand, arrivals=arrivals, placed_orders=orders, sales=sales, end_inventory=stocks, initial_inventory=initial, capacity=capacity, forecast=policy.forecast)
    return metrics

def service_necessary_bound(demand, reference):
    h = demand.shape[1]
    allowed = int(np.floor(0.05 * h + 1e-10))
    required_sales = np.sort(demand, axis=1)[:, :h - allowed].sum()
    maximum_supply = 1.25 * reference.mean(1).sum() + (h - 1) * 1.05 * reference.mean(1).sum()
    return dict(required_sales_lower_bound=float(required_sales), maximum_supply=float(maximum_supply), aggregate_bound_impossible=bool(required_sales >= maximum_supply), initial_failure_skus=int((demand[:, 0] >= 1.25 * reference.mean(1)).sum()), allowed_failed_periods=allowed)

def selection_key(m):
    if m['all_skus_feasible']:
        return (1, m['weighted_turnover'], 0, 0)
    return (0, m['pass_skus'], m['mean_ready_rate'], m['weighted_turnover'])
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'replenishment-mpc-mpl'))
import hashlib
import itertools
import json
import time
from dataclasses import asdict, replace
import joblib
import pandas as pd

def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))

def forecast_metrics(p, actual):
    error = p - actual
    variance = np.square(actual - actual.mean()).sum()
    return dict(wape=float(np.abs(error).sum() / actual.sum()), rmse=float(np.sqrt(np.square(error).mean())), r2=float(1 - np.square(error).sum() / variance))

def score_policy(forecasts, residual, forecaster, cfg, actual, reference, ids, output=None):
    policy = ScenarioPolicy(forecasts, residual, forecaster.groups, np.maximum(reference.mean(1), 1.0), np.maximum(reference.std(1, ddof=1), 1.0), 1.05 * reference.mean(1).sum(), cfg)
    start = time.perf_counter()
    metrics = simulate(actual, policy, reference, ids, output)
    metrics.update(seconds=time.perf_counter() - start)
    return (metrics, policy)

def train_round1():
    OUT.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    raw = np.load(WORK / '数据/processed/panel.npz', allow_pickle=False)
    d, x, ids, cats = [raw[k] for k in ['demand', 'x_from_period_13', 'item_ids', 'categories']]
    train_d, train_x = (d[:, :60], x[:, :48])
    configs = [dict(family='ridge', groups=g, alpha=a) for g, a in [(1, 10.0), (4, 1.0), (4, 10.0), (4, 100.0), (8, 10.0)]]
    configs += [dict(family='lightgbm', groups=4, trees=trees, leaves=leaves) for trees, leaves in [(80, 15), (160, 31)]]
    forecasts = []
    predictor_rows = []
    for j, config in enumerate(configs):
        start = time.perf_counter()
        predictor = GroupForecaster(config).fit(train_d, train_x, cats, 32)
        calibration = predictor.predict(train_d, train_x, cats, np.arange(31, 43))
        residual = (train_d[:, 32:44] - calibration[:, :, 0]) / predictor.sigma[:, None]
        validation = predictor.predict(train_d, train_x, cats, np.arange(43, 59))
        metrics = forecast_metrics(validation[:, :, 0], train_d[:, 44:60])
        horizons = []
        for h in range(3):
            horizons.append(forecast_metrics(validation[:, :16 - h, h], train_d[:, 44 + h:60]))
        row = dict(candidate=j, **config, **metrics, horizon_metrics=horizons, seconds=time.perf_counter() - start)
        predictor_rows.append(row)
        forecasts.append((predictor, residual, validation))
        dump(TABLES / 'predictor_search.json', predictor_rows)
        print('FORECAST', j, config, 'WAPE', round(metrics['wape'], 4), 'seconds', round(row['seconds'], 1), flush=True)
    best_predictor = min(range(len(configs)), key=lambda j: predictor_rows[j]['wape'])
    predictor, residual, validation = forecasts[best_predictor]
    reference = train_d[:, :32]
    actual = train_d[:, 44:60]
    bound = service_necessary_bound(actual, reference)
    dump(TABLES / 'validation_feasibility.json', bound)
    print('VALIDATION NECESSARY BOUND', bound, flush=True)
    policy_rows = []
    best_cfg, best_metrics = (None, None)
    choices = list(itertools.product([1, 3], [0.25, 0.5, 1.0, 1.5], [0.95, 0.99], [0.0, 0.5]))
    for j, (h, factor, quantile, adaptive) in enumerate(choices):
        cfg = PolicyConfig(horizon=h, buffer_factor=factor, quantile=quantile, adaptive_gain=adaptive, steps=12)
        metrics, _ = score_policy(validation, residual, predictor, cfg, actual, reference, ids)
        policy_rows.append(dict(candidate=j, stage='coarse', config=asdict(cfg), **metrics))
        if best_metrics is None or selection_key(metrics) > selection_key(best_metrics):
            best_cfg, best_metrics = (cfg, metrics)
        dump(TABLES / 'policy_search.json', policy_rows)
        print('POLICY', j + 1, '/', len(choices), 'pass', metrics['pass_skus'], 'RR', round(metrics['mean_ready_rate'], 4), 'config', (h, factor, quantile, adaptive), flush=True)
    refinements = [replace(best_cfg, risk_mix=risk, service_weight=weight, steps=steps) for risk, weight, steps in itertools.product([0.0, 0.5, 1.0], [10.0, 30.0], [8, 24])]
    for cfg in refinements:
        metrics, _ = score_policy(validation, residual, predictor, cfg, actual, reference, ids)
        policy_rows.append(dict(candidate=len(policy_rows), stage='refine', config=asdict(cfg), **metrics))
        if selection_key(metrics) > selection_key(best_metrics):
            best_cfg, best_metrics = (cfg, metrics)
        dump(TABLES / 'policy_search.json', policy_rows)
        print('REFINE', len(policy_rows), 'pass', metrics['pass_skus'], 'RR', round(metrics['mean_ready_rate'], 4), flush=True)
    variants = {'new_selected': best_cfg, 'ablation_horizon_one': replace(best_cfg, horizon=1), 'ablation_no_adaptation': replace(best_cfg, adaptive_gain=0.0), 'ablation_no_inner_optimization': replace(best_cfg, steps=0), 'ablation_no_buffer': replace(best_cfg, buffer_factor=0.0, adaptive_gain=0.0), 'stability_seed_43': replace(best_cfg, seed=43), 'stability_seed_44': replace(best_cfg, seed=44), 'sensitivity_buffer_low': replace(best_cfg, buffer_factor=best_cfg.buffer_factor * 0.8), 'sensitivity_buffer_high': replace(best_cfg, buffer_factor=best_cfg.buffer_factor * 1.2), 'sensitivity_steps_double': replace(best_cfg, steps=max(best_cfg.steps * 2, 1))}
    selection = dict(predictor_config=configs[best_predictor], policy_config=asdict(best_cfg), validation_metrics=best_metrics, validation_feasibility=bound, predictor_candidates=len(configs), policy_candidates=len(policy_rows), selection_rule='forecast WAPE then policy feasibility/pass count/mean RR/weighted turnover', validation_fallback=not best_metrics['all_skus_feasible'], test_variants={k: asdict(v) for k, v in variants.items()}, test_not_used_in_this_selection=True, prior_test_exposure='same test already examined by earlier project runs')
    dump(TABLES / 'selection_frozen_before_test.json', selection)
    selection_hash = hashlib.sha256((TABLES / 'selection_frozen_before_test.json').read_bytes()).hexdigest()
    pd.DataFrame([{**r, 'config': json.dumps(r['config'])} for r in policy_rows]).to_csv(TABLES / 'policy_search.csv', index=False)
    final = GroupForecaster(configs[best_predictor]).fit(train_d, train_x, cats, 60)
    final_forecast = final.predict(d, x, cats, np.arange(59, 87))
    joblib.dump(final, OUT / 'forecaster.joblib')
    np.savez_compressed(OUT / 'calibration.npz', residual=residual, groups=final.groups)
    test_rows = []
    audits = []
    for name, cfg in variants.items():
        metrics, policy = score_policy(final_forecast, residual, final, cfg, d[:, 60:], train_d, ids, OUT / name)
        metrics.update(model=name, variant='new_experiment')
        test_rows.append(metrics)
        dump(OUT / name / 'policy_diagnostics.json', policy.diagnostics)
        dump(OUT / name / 'summary.json', dict(config=asdict(cfg), **metrics))
        audits.append(independent_audit(OUT / name, train_d, ids, metrics))
        print('TEST FIXED', name, 'pass', metrics['pass_skus'], 'FR', round(metrics['fill_rate'], 5), 'inventory', round(metrics['average_total_inventory']), 'weighted_turnover', round(metrics['weighted_turnover'], 3), flush=True)
    global_config = dict(configs[best_predictor], groups=1)
    global_cal = GroupForecaster(global_config).fit(train_d, train_x, cats, 32)
    cp = global_cal.predict(train_d, train_x, cats, np.arange(31, 43))
    global_residual = (train_d[:, 32:44] - cp[:, :, 0]) / global_cal.sigma[:, None]
    global_final = GroupForecaster(global_config).fit(train_d, train_x, cats, 60)
    gp = global_final.predict(d, x, cats, np.arange(59, 87))
    metrics, policy = score_policy(gp, global_residual, global_final, best_cfg, d[:, 60:], train_d, ids, OUT / 'ablation_no_groups')
    metrics.update(model='ablation_no_groups', variant='new_experiment')
    test_rows.append(metrics)
    audits.append(independent_audit(OUT / 'ablation_no_groups', train_d, ids, metrics))
    new_frame = pd.DataFrame(test_rows)
    old = pd.read_csv(ROOT / 'results/generated/lead_one/comparison.csv')
    assert np.allclose(old.capacity, 1.05 * train_d.mean(1).sum()) and (old.periods == 28).all()
    old['pass_skus'] = np.rint(old.sku_service_pass_rate * len(ids)).astype(int)
    full = pd.concat([new_frame, old], ignore_index=True)
    full.to_csv(TABLES / 'comparison.csv', index=False)
    dump(TABLES / '问题_1_结果.json', dict(selection=selection, test_results=test_rows, test_forecast=forecast_metrics(final_forecast[:, :, 0], d[:, 60:])))
    dump(TABLES / '灵敏度结果.json', {'fixed_variants': test_rows[5:], 'note': 'predeclared diagnostics, not used to reselect test winner'})
    audit_data = json.loads((TABLES / 'data_audit.json').read_text())
    identity = dict(academic_name='带共享容量和逐SKU服务校准的多期场景库存随机规划近似', canonical_model_family='有限场景凸分段线性随机规划', solver_algorithm='带下界的加权单纯形投影次梯度法')
    manifest = dict(data_preparation=audit_data, model_identity={'Q1': identity}, selection=selection, selection_sha256=selection_hash, test_results=test_rows, audits=audits, test_feasibility=service_necessary_bound(d[:, 60:], train_d), test_forecast=forecast_metrics(final_forecast[:, :, 0], d[:, 60:]), source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT / 'common/prepare.py']})
    dump(TABLES / '全部结果.json', manifest)
    dump(TABLES / 'code_manifest.json', manifest['source_hashes'])
    dump(WORK / '临时文件/问题_1_复核.json', audits)
    write_report(full, manifest, predictor_rows, policy_rows)
    assert (WORK / '计算结果.md').stat().st_size >= 1024
    assert (WORK / '建模报告.md').stat().st_size >= 1500
    assert (Path(__file__)).stat().st_size >= 500
    print('COMPLETE: input hashes, frozen selection, comparison, SKU outputs and independent audits', flush=True)

def independent_audit(directory, reference, ids, metrics):
    z = np.load(directory / 'trajectory.npz', allow_pickle=False)
    assert np.array_equal(z['item_ids'], ids)
    demand, q, arrivals, sales, stock = [z[k] for k in ['demand', 'placed_orders', 'arrivals', 'sales', 'end_inventory']]
    prev = np.column_stack([1.25 * reference.mean(1), stock[:, :-1]])
    assert np.array_equal(arrivals[:, 1:], q[:, :-1]) and (not arrivals[:, 0].any())
    assert not q[:, -1].any() and q.min() >= 0
    assert np.all(q.sum(0) <= 1.05 * reference.mean(1).sum() * (1 + 1e-10))
    assert np.allclose(sales, np.minimum(prev + arrivals, demand), rtol=0, atol=1e-06)
    assert np.allclose(stock, prev + arrivals - sales, rtol=0, atol=1e-06)
    rr = (stock > 1e-08).mean(1)
    assert int((rr >= 0.95).sum()) == metrics['pass_skus']
    assert np.isclose(sales.sum() / demand.sum(), metrics['fill_rate'])
    assert np.isclose(stock.mean(1).sum(), metrics['average_total_inventory'])
    weights = 1 / np.maximum(reference.mean(1), 0.5)
    weights /= weights.mean()
    assert np.isclose(weights @ sales.sum(1) / (weights @ stock.mean(1)), metrics['weighted_turnover'])
    return dict(model=directory.name, numerical_and_timing_checks=True, business_service_feasible=metrics['all_skus_feasible'], failed_skus=len(ids) - metrics['pass_skus'])

def write_report(frame, manifest, predictors, policies):
    selection = manifest['selection']
    m = manifest['test_results'][0]
    lines = ['# 自适应场景滚动补货：计算结果', '', '本实验整合商品分组、多时域预测、训练内残差校准、逐SKU服务反馈与容量约束场景优化。', '没有使用DFL微调；不是严格DRO；没有宣称方法原创。预测器训练和参数选择仅使用训练集内部时间留出。', '', '## 1. 训练、调参与冻结', '', '前32期拟合、33—44期校准残差、45—60期验证，最终前60期重拟合。', f'搜索了{len(predictors)}个预测器配置、{len(policies)}个策略配置。全部记录在图表文件夹。', '选中配置：', '', '```json', json.dumps(selection['predictor_config'], ensure_ascii=False), json.dumps(selection['policy_config'], ensure_ascii=False), '```', '', '## 2. 验证期可行性限制', '', json.dumps(selection['validation_feasibility'], ensure_ascii=False), '', f"验证选中方案是否逐SKU可行：{selection['validation_metrics']['all_skus_feasible']}。", '若必要供给界显示不可行，不能把参数搜索解释为已经达到了业务最优；降级选择规则和原始失败候选均保留。', '训练需求持续上升，测试需求在工作簿边界重置。新模型不读取未来需求或未来x，但既有测试已被前期研究查看过，不能称为完全未接触的最终外部验证。', '', '## 3. 同一期提前期、同初始库存、同容量比较', '', '达标要求每SKU Ready Rate≥95%；库存为平均总期末库存，加权周转非年化。新选中模型由验证确定，不能从下面测试消融中再挑赢家。', '', '|模型|版本|达标SKU|Fill Rate|平均库存（万件）|加权周转|全SKU可行|', '|---|---|---:|---:|---:|---:|---|']
    for _, r in frame.iterrows():
        lines.append(f'|{r.model}|{r.variant}|{int(r.pass_skus)}/6000|{r.fill_rate:.4%}|{r.average_total_inventory / 10000:.2f}|{r.weighted_turnover:.4f}|{bool(r.all_skus_feasible)}|')
    old_feasible = frame[(frame.variant != 'new_experiment') & frame.all_skus_feasible]
    best_old = old_feasible.sort_values('weighted_turnover', ascending=False).iloc[0]
    lines += ['', '## 4. 如实结论', '', f"验证选中模型测试达标{m['pass_skus']}/6000，Fill Rate={m['fill_rate']:.4%}，库存={m['average_total_inventory'] / 10000:.2f}万件，加权周转={m['weighted_turnover']:.4f}。", f'旧可行参考中加权周转最高为{best_old.model}（{best_old.weighted_turnover:.4f}）。', '只有新模型全部达标，且所选目标改善时，才能称为在这一测试口径下优于旧可行参考。消融中的更好结果仅是诊断，不替代预先冻结的主模型。', '服务反馈是使用截至当期已观察的库存状态执行固定规则，不是在测试上重新拟合预测器或搜索全局参数。', '', '## 5. 数值复核与局限', '', '独立复核了SKU对齐、订单非负、容量、上一期订单到货、期末守恒、销量、服务达标数、Fill Rate、平均库存及加权周转。', '规划代理允许负供需净余额，真实库存没有负值；固定步数近似优化不保证业务全局最优。首期硬下界可能与容量冲突，此时报告冲突并取消不可能同时满足的下界，不能隐藏。', '残差只有12个留出时期，尾部估计与分布重置风险仍在；不存在逐SKU样本外95%保证。', '反复查看同一个测试集不能形成独立证据，后续需新时间段或独立数据。', '', '## 6. 文件与复现', '', 'models/adaptive_mpc.py：模型和完整训练；tests/：小型逻辑测试。', '图表/predictor_search.json、policy_search.csv：真实搜索记录；selection_frozen_before_test.json：测试前冻结配置。', '图表/comparison.csv：旧20方案和固定新方案；results/各方案/sku_metrics.csv：6000个SKU指标；trajectory.npz：逐期库存、销量与订单。', '图表/全部结果.json保存规范输入哈希、代码哈希、独立审计与实验身份。数据预处理不改原数据，不读取旧标准化器。', '', '```bash', 'python -m models.adaptive_mpc --train --round all', '```', '']
    (WORK / '计算结果.md').write_text('\n'.join(lines))

def refine():
    data = np.load(WORK / '数据/processed/panel.npz', allow_pickle=False)
    d, x, ids, cats = [data[k] for k in ['demand', 'x_from_period_13', 'item_ids', 'categories']]
    train_d, train_x = (d[:, :60], x[:, :48])
    prior = json.loads((TABLES / 'selection_frozen_before_test.json').read_text())
    predictor = GroupForecaster(prior['predictor_config']).fit(train_d, train_x, cats, 32)
    residual = np.load(OUT / 'calibration.npz')['residual']
    validation = predictor.predict(train_d, train_x, cats, np.arange(43, 59))
    actual, reference = (train_d[:, 44:60], train_d[:, :32])
    base = PolicyConfig(**prior['policy_config'])
    configs = [replace(base, horizon=3, adaptive_gain=gain, buffer_factor=factor, individual_mix=mix, quantile=0.99, steps=16) for factor, gain, mix in itertools.product([1.5, 2.0, 3.0, 4.0], [0.25, 0.5, 1.0], [0.5, 1.0])]
    rows = []
    best = None
    best_cfg = None
    for j, cfg in enumerate(configs):
        metrics, _ = score_policy(validation, residual, predictor, cfg, actual, reference, ids)
        rows.append(dict(candidate=j, stage='full_component_extension', config=asdict(cfg), **metrics))
        if best is None or selection_key(metrics) > selection_key(best):
            best, best_cfg = (metrics, cfg)
        dump(TABLES / 'round2_search.json', rows)
        print('ROUND2', j + 1, '/', len(configs), 'pass', metrics['pass_skus'], 'RR', round(metrics['mean_ready_rate'], 4), 'factor/gain/mix', cfg.buffer_factor, cfg.adaptive_gain, cfg.individual_mix, flush=True)
    variants = {'full_components_selected': best_cfg, 'full_ablation_no_adaptation': replace(best_cfg, adaptive_gain=0.0), 'full_ablation_one_period': replace(best_cfg, horizon=1), 'full_ablation_no_optimization': replace(best_cfg, steps=0), 'full_ablation_group_only_buffer': replace(best_cfg, individual_mix=0.0), 'full_stability_seed_43': replace(best_cfg, seed=43), 'full_stability_seed_44': replace(best_cfg, seed=44), 'full_sensitivity_buffer_low': replace(best_cfg, buffer_factor=best_cfg.buffer_factor * 0.8), 'full_sensitivity_buffer_high': replace(best_cfg, buffer_factor=best_cfg.buffer_factor * 1.2)}
    selection = dict(config=asdict(best_cfg), validation_metrics=best, candidates=len(rows), rationale='round1 validation winner touched buffer search upper boundary; retain all proposed components', test_variants={k: asdict(v) for k, v in variants.items()}, test_exposure='round1 results already observed; development round, not independent confirmatory evidence', parameter_selection='training periods 45-60 only, same infeasible capacity stress')
    dump(TABLES / 'round2_selection_frozen_before_test.json', selection)
    pd.DataFrame([{**r, 'config': json.dumps(r['config'])} for r in rows]).to_csv(TABLES / 'round2_search.csv', index=False)
    final = joblib.load(OUT / 'forecaster.joblib')
    prediction = final.predict(d, x, cats, np.arange(59, 87))
    test = []
    audits = []
    for name, cfg in variants.items():
        metrics, policy = score_policy(prediction, residual, final, cfg, d[:, 60:], train_d, ids, OUT / name)
        metrics.update(model=name, variant='full_component_round2')
        test.append(metrics)
        audits.append(independent_audit(OUT / name, train_d, ids, metrics))
        dump(OUT / name / 'summary.json', dict(config=asdict(cfg), **metrics))
        dump(OUT / name / 'policy_diagnostics.json', policy.diagnostics)
        print('FULL TEST FIXED', name, 'pass', metrics['pass_skus'], 'FR', round(metrics['fill_rate'], 6), 'inventory', round(metrics['average_total_inventory']), 'weighted_turnover', round(metrics['weighted_turnover'], 3), flush=True)
    frame = pd.read_csv(TABLES / 'comparison.csv')
    frame = frame[frame.variant != 'full_component_round2']
    frame = pd.concat([frame, pd.DataFrame(test)], ignore_index=True)
    frame.to_csv(TABLES / 'comparison.csv', index=False)
    manifest = json.loads((TABLES / '全部结果.json').read_text())
    manifest['round2'] = dict(selection=selection, test_results=test, audits=audits)
    manifest['source_hashes']['程序/refine_full.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dump(TABLES / '全部结果.json', manifest)
    dump(TABLES / 'round2_results.json', manifest['round2'])
    dump(TABLES / 'code_manifest.json', manifest['source_hashes'])
    first_report = (WORK / '计算结果.md').read_text()
    marker = '\n## 7. 第二轮：保留全部组件的验证边界扩展'
    first_report = first_report.split(marker)[0]
    lines = [marker, '', '第一轮测试已被查看。本轮属于开发性扩展，不能视为新的独立测试。扩展依据是第一轮验证最优缓冲位于上边界，而不是从测试失败SKU拟合参数。', f'固定H=3且自适应权重为正，搜索{len(rows)}组，仍按原验证可行性优先规则选择。验证期容量不可行限制仍未消除。', '', '```json', json.dumps(asdict(best_cfg), ensure_ascii=False), '```', '', '|模型|达标SKU|Fill Rate|库存（万件）|加权周转|全SKU可行|', '|---|---:|---:|---:|---:|---|']
    for r in test:
        lines.append(f"|{r['model']}|{r['pass_skus']}/6000|{r['fill_rate']:.4%}|{r['average_total_inventory'] / 10000:.2f}|{r['weighted_turnover']:.4f}|{r['all_skus_feasible']}|")
    lines += ['', '主方案是full_components_selected，不能从测试消融中重新挑一个充当主方案。', '新增复现命令：在程序目录运行 `python -m models.adaptive_mpc --train --round 2`。', '全部SKU率和逐期轨迹仍在results/各方案中；第二轮搜索、冻结配置和结果在图表/round2_*。', '']
    (WORK / '计算结果.md').write_text(first_report + '\n'.join(lines))
    print('ROUND2 COMPLETE', flush=True)

def make_coeff(residual, groups, pooled_factor, individual_factor):
    """Only raise SKU upper buffers where individual errors warrant it."""
    pooled = np.zeros(len(groups))
    for group in np.unique(groups):
        pooled[groups == group] = np.quantile(residual[groups == group], 0.99)
    individual = residual.max(1)
    return np.maximum.reduce([np.zeros(len(groups)), pooled_factor * pooled, individual_factor * individual])

def run_policy(forecast, residual, coeff, final, reference, ids, cfg, actual, output=None):
    sigma = np.maximum(reference.std(1, ddof=1), 1.0)
    p = ScenarioPolicy(forecast, residual, final.groups, np.maximum(reference.mean(1), 1.0), sigma, 1.05 * reference.mean(1).sum(), cfg)
    p.buffer = coeff * sigma
    metrics = simulate(actual, p, reference, ids, output)
    return (metrics, p)

def train_round3():
    raw = np.load(WORK / '数据/processed/panel.npz', allow_pickle=False)
    d, x, ids, cats = [raw[k] for k in ['demand', 'x_from_period_13', 'item_ids', 'categories']]
    train_d, train_x = (d[:, :60], x[:, :48])
    first = json.loads((TABLES / 'selection_frozen_before_test.json').read_text())
    fit = GroupForecaster(first['predictor_config']).fit(train_d, train_x, cats, 32)
    cal_pred = fit.predict(train_d, train_x, cats, np.arange(31, 51))
    errors = (train_d[:, 32:52] - cal_pred[:, :, 0]) / fit.sigma[:, None]
    validation = fit.predict(train_d, train_x, cats, np.arange(51, 59))
    upper_needed = (train_d[:, 52:60] - validation[:, :, 0]) / fit.sigma[:, None]
    rows = []
    best = None
    coefficients = None
    for pooled in [1.0, 1.5]:
        for individual in [1.0, 1.25, 1.5, 2.0, 3.0]:
            coeff = make_coeff(errors, fit.groups, pooled, individual)
            coverage = (coeff[:, None] + 1e-06 > upper_needed).mean(1)
            row = dict(pooled_factor=pooled, individual_factor=individual, coverage_pass_skus=int((coverage >= 0.95).sum()), mean_coverage=float(coverage.mean()), minimum_coverage=float(coverage.min()), mean_normalized_buffer=float(coeff.mean()))
            rows.append(row)
            key = (row['coverage_pass_skus'], row['mean_coverage'], -row['mean_normalized_buffer'])
            if best is None or key > (best['coverage_pass_skus'], best['mean_coverage'], -best['mean_normalized_buffer']):
                best, coefficients = (row, coeff)
            print('TARGETED TRAIN VALIDATION', row, flush=True)
    cfg = replace(PolicyConfig(**first['policy_config']), horizon=3, adaptive_gain=0.25, steps=16)
    variants = {'targeted_selected': (cfg, coefficients), 'targeted_no_adaptation': (replace(cfg, adaptive_gain=0.0), coefficients), 'targeted_one_period': (replace(cfg, horizon=1), coefficients), 'targeted_no_optimization': (replace(cfg, steps=0), coefficients), 'targeted_seed43': (replace(cfg, seed=43), coefficients), 'targeted_seed44': (replace(cfg, seed=44), coefficients), 'targeted_buffer_low': (cfg, coefficients * 0.8), 'targeted_buffer_high': (cfg, coefficients * 1.2)}
    selection = dict(calibration_periods='33-52', validation_periods='53-60', best=best, policy_config=asdict(cfg), candidates=len(rows), criterion='per-SKU upper interval coverage>=95%; minimum normalized buffer, NOT realized Ready Rate', limitation='8 validation periods and 20 calibration periods cannot guarantee future per-SKU service', test_exposure='round1 and round2 test already examined; development only', fixed_test_variants={k: asdict(v[0]) for k, v in variants.items()})
    dump(TABLES / 'round3_selection_frozen_before_test.json', selection)
    dump(TABLES / 'round3_search.json', rows)
    pd.DataFrame(rows).to_csv(TABLES / 'round3_search.csv', index=False)
    final = joblib.load(OUT / 'forecaster.joblib')
    final_coeff = make_coeff(errors, final.groups, best['pooled_factor'], best['individual_factor'])
    forecast = final.predict(d, x, cats, np.arange(59, 87))
    ratios = {k: v[1] / np.maximum(coefficients, 1e-12) for k, v in variants.items()}
    np.savez_compressed(OUT / 'targeted_calibration.npz', residual=errors, coefficient=final_coeff, training_coefficient=coefficients, groups=final.groups)
    test = []
    audits = []
    for name, (config, _) in variants.items():
        coeff = final_coeff * ratios[name]
        coeff = np.where(coefficients > 0, coeff, final_coeff)
        metrics, policy = run_policy(forecast, errors, coeff, final, train_d, ids, config, d[:, 60:], OUT / name)
        metrics.update(model=name, variant='targeted_round3')
        test.append(metrics)
        audits.append(independent_audit(OUT / name, train_d, ids, metrics))
        dump(OUT / name / 'summary.json', dict(config=asdict(config), **metrics))
        dump(OUT / name / 'policy_diagnostics.json', policy.diagnostics)
        print('TARGETED TEST FIXED', name, metrics['pass_skus'], round(metrics['average_total_inventory']), round(metrics['weighted_turnover'], 3), flush=True)
    frame = pd.read_csv(TABLES / 'comparison.csv')
    frame = frame[frame.variant != 'targeted_round3']
    pd.concat([frame, pd.DataFrame(test)], ignore_index=True).to_csv(TABLES / 'comparison.csv', index=False)
    manifest = json.loads((TABLES / '全部结果.json').read_text())
    manifest['round3'] = dict(selection=selection, test_results=test, audits=audits)
    manifest['source_hashes']['程序/targeted_calibration.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dump(TABLES / '全部结果.json', manifest)
    dump(TABLES / 'round3_results.json', manifest['round3'])
    dump(TABLES / 'code_manifest.json', manifest['source_hashes'])
    report = (WORK / '计算结果.md').read_text()
    marker = '\n## 8. 第三轮：逐SKU针对性校准'
    report = report.split(marker)[0]
    lines = [marker, '', '本轮改动不是根据测试失败SKU修补，而是使用训练内33—52期误差，为尾部误差大的SKU增加缓冲。53—60期只用于选择区间覆盖与缓冲大小。', '这一选择标准是预测区间覆盖，不是容量约束下真实Ready Rate。冷启动容量验证已证明不可行，因此不能声称完成原业务目标的验证调优。', '前两轮测试已查看，本轮只能算开发性探索。所有测试变体一起预先冻结，没有从消融中重新挑主模型。', '', '选中校准参数：', '```json', json.dumps(best, ensure_ascii=False), '```', '', '|方案|达标SKU|Fill Rate|库存（万件）|加权周转|全SKU可行|', '|---|---:|---:|---:|---:|---|']
    for r in test:
        lines.append(f"|{r['model']}|{r['pass_skus']}/6000|{r['fill_rate']:.4%}|{r['average_total_inventory'] / 10000:.2f}|{r['weighted_turnover']:.4f}|{r['all_skus_feasible']}|")
    lines += ['', '主模型为targeted_selected。复现：在仓库根目录运行python -m models.adaptive_mpc --train --round all。', '']
    (WORK / '计算结果.md').write_text(report + '\n'.join(lines))
    print('ROUND3 COMPLETE', flush=True)

def train_round4():
    raw = np.load(WORK / '数据/processed/panel.npz', allow_pickle=False)
    d, x, ids, cats = [raw[k] for k in ['demand', 'x_from_period_13', 'item_ids', 'categories']]
    train_d, train_x = (d[:, :60], x[:, :48])
    first = json.loads((TABLES / 'selection_frozen_before_test.json').read_text())
    fit = GroupForecaster(first['predictor_config']).fit(train_d, train_x, cats, 32)
    cp = fit.predict(train_d, train_x, cats, np.arange(31, 51))
    raw_error = train_d[:, 32:52] - cp[:, :, 0]
    uncertainty = np.maximum(np.sqrt(np.square(raw_error).mean(1)), 1.0)
    residual = raw_error / uncertainty[:, None]
    vp = fit.predict(train_d, train_x, cats, np.arange(51, 59))
    needed = (train_d[:, 52:60] - vp[:, :, 0]) / uncertainty[:, None]
    rows = []
    best = None
    coeff = None
    for pooled in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for individual in [1.0, 1.5, 2.0]:
            c = make_coeff(residual, fit.groups, pooled, individual)
            coverage = (c[:, None] + 1e-06 > needed).mean(1)
            row = dict(pooled_factor=pooled, individual_factor=individual, coverage_pass_skus=int((coverage >= 0.95).sum()), mean_coverage=float(coverage.mean()), minimum_coverage=float(coverage.min()), total_buffer=float((c * uncertainty).sum()))
            rows.append(row)
            key = (row['coverage_pass_skus'], row['mean_coverage'], -row['total_buffer'])
            if best is None or key > (best['coverage_pass_skus'], best['mean_coverage'], -best['total_buffer']):
                best, coeff = (row, c)
            print('RESIDUAL SCALE TRAIN VALIDATION', row, flush=True)
    cfg = replace(PolicyConfig(**first['policy_config']), horizon=3, adaptive_gain=0.25, steps=16)
    variants = {'residual_scale_selected': (cfg, 1.0), 'residual_scale_no_adaptation': (replace(cfg, adaptive_gain=0.0), 1.0), 'residual_scale_one_period': (replace(cfg, horizon=1), 1.0), 'residual_scale_no_optimization': (replace(cfg, steps=0), 1.0), 'residual_scale_seed43': (replace(cfg, seed=43), 1.0), 'residual_scale_seed44': (replace(cfg, seed=44), 1.0), 'residual_scale_buffer_low': (cfg, 0.8), 'residual_scale_buffer_high': (cfg, 1.2)}
    selection = dict(best=best, candidates=len(rows), config=asdict(cfg), uncertainty='SKU RMS of temporal holdout forecast errors; frozen across final refit', calibration_periods='33-52', validation_periods='53-60', selection_target='uncapacitated upper-interval coverage, NOT realized Ready Rate', test_exposure='previous rounds observed: development evidence only', fixed_variants={k: dict(config=asdict(v[0]), buffer_multiplier=v[1]) for k, v in variants.items()})
    dump(TABLES / 'round4_search.json', rows)
    pd.DataFrame(rows).to_csv(TABLES / 'round4_search.csv', index=False)
    dump(TABLES / 'round4_selection_frozen_before_test.json', selection)
    final = joblib.load(OUT / 'forecaster.joblib')
    forecast = final.predict(d, x, cats, np.arange(59, 87))
    np.savez_compressed(OUT / 'residual_scale_calibration.npz', residual=residual, sigma=uncertainty, coeff=coeff, groups=fit.groups)
    test = []
    audits = []
    for name, (config, multiplier) in variants.items():
        policy = ScenarioPolicy(forecast, residual, fit.groups, np.maximum(train_d.mean(1), 1.0), uncertainty, 1.05 * train_d.mean(1).sum(), config)
        policy.buffer = coeff * uncertainty * multiplier
        m = simulate(d[:, 60:], policy, train_d, ids, OUT / name)
        m.update(model=name, variant='residual_scale_round4')
        test.append(m)
        audits.append(independent_audit(OUT / name, train_d, ids, m))
        dump(OUT / name / 'summary.json', dict(config=asdict(config), **m))
        dump(OUT / name / 'policy_diagnostics.json', policy.diagnostics)
        print('RESIDUAL SCALE TEST FIXED', name, m['pass_skus'], round(m['average_total_inventory']), round(m['weighted_turnover'], 3), flush=True)
    frame = pd.read_csv(TABLES / 'comparison.csv')
    frame = frame[frame.variant != 'residual_scale_round4']
    pd.concat([frame, pd.DataFrame(test)], ignore_index=True).to_csv(TABLES / 'comparison.csv', index=False)
    manifest = json.loads((TABLES / '全部结果.json').read_text())
    manifest['round4'] = dict(selection=selection, test_results=test, audits=audits)
    manifest['source_hashes']['程序/residual_scale.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dump(TABLES / '全部结果.json', manifest)
    dump(TABLES / 'round4_results.json', manifest['round4'])
    report = (WORK / '计算结果.md').read_text()
    marker = '\n## 9. 第四轮：预测残差尺度取代需求趋势尺度'
    report = report.split(marker)[0]
    lines = [marker, '', '前面版本将需求标准差当作误差尺度，并在60期重拟合后再扩大该尺度，会混入需求增长趋势。第四轮用33—52期预测残差RMS估计不确定性，最终重拟合时不重算需求标准差作为误差。', '这是一项明确的模型结构修订；不改变真实库存、初始库存、共享容量或服务评价。候选选择仍仅使用53—60期区间覆盖，不声称通过容量可行的业务验证。', '所有前轮测试已查看，第四轮仍为开发性结果，不是独立确认性证据。', '', '选中校准参数：', '```json', json.dumps(best, ensure_ascii=False), '```', '', '|方案|达标SKU|Fill Rate|库存（万件）|加权周转|全SKU可行|', '|---|---:|---:|---:|---:|---|']
    for r in test:
        lines.append(f"|{r['model']}|{r['pass_skus']}/6000|{r['fill_rate']:.4%}|{r['average_total_inventory'] / 10000:.2f}|{r['weighted_turnover']:.4f}|{r['all_skus_feasible']}|")
    lines += ['', '本轮主方案：residual_scale_selected。不能从测试消融中重新选主方案。', '复现：前三轮后运行residual_scale.py，再运行verify_and_render.py。', '']
    (WORK / '计算结果.md').write_text(report + '\n'.join(lines))
    print('ROUND4 COMPLETE', flush=True)

if __name__=='__main__':
    from common.cli import start
    start('adaptive_mpc', globals(), default_protocol='lead_one')
