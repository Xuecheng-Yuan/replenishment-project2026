"""models.online_learning: actual model/processing implementation, not a legacy runner alias.

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
OUT=ROOT/'results/generated/online_learning'
WORK=OUT
TABLES=OUT/'图表'
OLD=ROOT/'results/generated/ready_immediate'
import sys
import copy
import json
import time
from dataclasses import asdict
import numpy as np
import pandas as pd
import torch
from models.nn_scenario import ScaledDemandNet
from models import nn_scenario as p
from models.dfl_improved import DecisionConfig, order
from common.ready_evaluation import ReadyRateEvaluator, selection_key
DEVICE = torch.device('cpu')
DC = DecisionConfig(safety_z=4.0, radius=0.4)

def setup(raw, split, seed=42):
    p.set_seed(seed)
    d = raw['demand']
    e = ReadyRateEvaluator(d[:, :split], raw['item_ids'])
    f = raw['features'].copy()
    scale = np.maximum(e.mean, 0.5)
    f[:, :, :13] /= scale[None, :, None]
    f[:, :, 14] /= scale[None, :]
    fit = f[:split - 13]
    fm, fs = (fit.mean(axis=(0, 1)), np.maximum(fit.std(axis=(0, 1)), 1e-06))
    f = (f - fm) / fs
    data = p.PanelData(X=torch.tensor(f), D=torch.tensor(d[:, 13:].T, dtype=torch.float32), periods=list(range(14, 89)), sku_labels=raw['item_ids'].tolist(), feature_mean=torch.tensor(fm), feature_std=torch.tensor(fs))
    cfg = p.Config(train_periods=split - 13, test_periods=88 - split, n_features=f.shape[-1])
    model = ScaledDemandNet(data.N, data.F, np.log1p(scale))
    return (data, cfg, e, model, fm, fs)

class Learner:

    def __init__(self, initial, data, fit_count, mode, lr=0.0, alpha=0.0, seed=42):
        self.model = copy.deepcopy(initial)
        self.data = data
        self.mode, self.alpha = (mode, alpha)
        self.index = torch.arange(data.N)
        self.bias = torch.zeros(data.N)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) if lr else None
        self.rng = np.random.default_rng(seed)
        self.fit_count = fit_count
        self.updated_through = fit_count - 1
        self.logs = []
        self.clipped = 0

    def predict(self, k):
        assert self.updated_through < k, 'Model trained on its own target!'
        self.model.eval()
        with torch.no_grad():
            mu, sigma = self.model(self.data.X[k], self.index)
        corrected = mu + self.bias if self.alpha else mu
        return (mu, corrected, sigma)

    def observe(self, k, raw_mu):
        assert k == self.updated_through + 1
        losses = []
        if self.alpha:
            residual = torch.log1p(self.data.D[k]) - raw_mu
            self.bias = (1 - self.alpha) * self.bias + self.alpha * residual
            self.clipped += int((self.bias.abs() > 2).sum())
            self.bias = self.bias.clamp(-2, 2)
        if self.optimizer:
            self.model.train()
            permutation = self.rng.permutation(self.data.N)
            for start in range(0, self.data.N, 2048):
                sku = torch.tensor(permutation[start:start + 2048])
                mu, sig = self.model(self.data.X[k, sku], sku)
                fresh = p.gaussian_nll_log1p(mu, sig, self.data.D[k, sku])
                past_k = torch.tensor(self.rng.integers(0, k, size=1024))
                past_sku = torch.tensor(self.rng.integers(0, self.data.N, size=1024))
                pm, ps = self.model(self.data.X[past_k, past_sku], past_sku)
                replay = p.gaussian_nll_log1p(pm, ps, self.data.D[past_k, past_sku])
                loss = 0.75 * fresh + 0.25 * replay
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite online loss')
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0, error_if_nonfinite=True)
                self.optimizer.step()
                losses.append(float(loss.detach()))
        self.updated_through = k
        self.logs.append(dict(target_period=k + 14, mean_update_loss=float(np.mean(losses)) if losses else 0.0, mean_bias=float(self.bias.mean()), bias_clipped_count=self.clipped))

def validate_constraints(d, q, arr, sold, stock, initial, capacity):
    for a in (d, q, arr, sold, stock):
        assert np.isfinite(a).all() and a.min() >= -1e-07
    assert np.all(q.sum(axis=0) <= capacity * (1 + 2e-06))
    assert np.all(arr[:, 0] == 0)
    np.testing.assert_array_equal(arr[:, 1:], q[:, :-1])
    assert np.all(q[:, -1] == 0)
    previous = np.column_stack([initial, stock[:, :-1]])
    np.testing.assert_allclose(sold, np.minimum(previous + arr, d), rtol=1e-09, atol=1e-07)
    np.testing.assert_allclose(stock, previous + arr - sold, rtol=1e-09, atol=1e-07)

def evaluate(initial, data, cfg, e, basis, end_period, name, lr=0.0, alpha=0.0, seed=42, output=None, with_updates=True):
    start = cfg.train_periods
    stop = end_period - 13
    learner = Learner(initial, data, start, name, lr, alpha, seed)
    h, n = (stop - start, data.N)
    d = data.D[start:stop].numpy().T.astype(float)
    inv = e.initial.copy()
    pending = np.zeros(n)
    q, arr, stock, sold, fc, rawfc = [np.zeros((n, h)) for _ in range(6)]
    scale = torch.tensor(np.maximum(e.mean, 0.5), dtype=torch.float32)
    capacity = torch.tensor(e.capacity, dtype=torch.float32)
    daily = []
    current = learner.predict(start)
    t0 = time.perf_counter()
    for t, k in enumerate(range(start, stop)):
        raw_mu, mu, sig = current
        fc[:, t] = torch.expm1(mu.clamp(0, 20)).numpy()
        rawfc[:, t] = torch.expm1(raw_mu.clamp(0, 20)).numpy()
        arr[:, t] = pending
        available = inv + pending
        sold[:, t] = np.minimum(available, d[:, t])
        inv = available - sold[:, t]
        stock[:, t] = inv
        if with_updates:
            learner.observe(k, raw_mu)
        else:
            learner.updated_through = k
        if t + 1 < h:
            current = learner.predict(k + 1)
            _, next_mu, next_sig = current
            candidate = order(next_mu, next_sig, torch.tensor(inv, dtype=torch.float32), scale, capacity, basis, DC, seed + 10000 + t, False, True).detach().numpy().astype(float)
            pending = np.maximum(candidate, 0.0)
            pending *= min(1.0, e.capacity / max(pending.sum(), 1e-08))
        else:
            pending = np.zeros(n)
        q[:, t] = pending
        daily.append(dict(period=t + 1, raw_target_period=k + 14, forecast_total=float(fc[:, t].sum()), demand_total=float(d[:, t].sum()), wape=float(np.abs(fc[:, t] - d[:, t]).sum() / max(d[:, t].sum(), 1e-08)), end_inventory=float(inv.sum()), ordered=float(pending.sum()), ready_share=float((inv > 1e-08).mean()), mean_sigma=float(sig.mean())))
    validate_constraints(d, q, arr, sold, stock, e.initial, e.capacity)
    m = e.evaluate(d, lambda t, i: arr[:, t])
    m.update(model=name, learning_rate=lr, bias_alpha=alpha, seed=seed, test_wape=float(np.abs(fc - d).sum() / max(d.sum(), 1e-08)), raw_network_wape=float(np.abs(rawfc - d).sum() / max(d.sum(), 1e-08)), average_total_order=float(q.sum(axis=0).mean()), lead_time=1, elapsed_seconds=time.perf_counter() - t0, passing_skus=int(((stock > 1e-08).mean(axis=1) >= 0.95).sum()), bias_clipped_count=learner.clipped)
    if output:
        output.mkdir(parents=True, exist_ok=True)
        rr, av = ((stock > 1e-08).mean(axis=1), stock.mean(axis=1))
        pd.DataFrame(dict(item_id=e.item_ids, ready_rate=rr, fill_rate=sold.sum(axis=1) / np.maximum(d.sum(axis=1), 1e-08), average_inventory=av, turnover=sold.sum(axis=1) / np.maximum(av, 1e-08))).to_csv(output / 'sku_metrics.csv', index=False)
        pd.DataFrame(daily).to_csv(output / 'daily_metrics.csv', index=False)
        pd.DataFrame(learner.logs).to_csv(output / 'online_update_log.csv', index=False)
        np.savez_compressed(output / 'trajectory.npz', demand=d, forecast=fc, raw_forecast=rawfc, placed_orders=q, arrivals=arr, sales=sold, end_inventory=stock, initial_inventory=e.initial, initial_pending=np.zeros(n), item_ids=e.item_ids, weights=e.weights, capacity=e.capacity)
        torch.save(learner.model.state_dict(), output / 'final_state.pt')
        (output / 'summary.json').write_text(json.dumps(m, ensure_ascii=False, indent=2))
    return (m, (fc, q, learner.model.state_dict()))

def train_legacy():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    raw = np.load(OUT / '数据/processed/panel.npz', allow_pickle=False)
    data, cfg, e, model, fm, fs = setup(raw, 42)
    trace = dict(validation_pretrain=p.pretrain_forecaster(model, data, cfg, DEVICE))
    basis = p.compute_residual_factor_basis(model, data, cfg, DEVICE)
    grid = [('frozen', 0.0, 0.0)]
    grid += [('bias', 0.0, a) for a in (0.1, 0.3)]
    grid += [('online', lr, 0.0) for lr in (1e-05, 5e-05, 0.0002)]
    rows = []
    for name, lr, alpha in grid:
        m, _ = evaluate(model, data, cfg, e, basis, 60, name, lr, alpha)
        rows.append(m)
        print('validation', name, lr, alpha, 'pass', m['passing_skus'], 'WAPE', round(m['test_wape'], 4), flush=True)
    best_bias = max([r for r in rows if r['model'] == 'bias'], key=selection_key)
    best_net = max([r for r in rows if r['model'] == 'online'], key=selection_key)
    selected = [('frozen', 0.0, 0.0), ('bias', 0.0, best_bias['bias_alpha']), ('online', best_net['learning_rate'], 0.0), ('combined', best_net['learning_rate'], best_bias['bias_alpha'])]
    cm, _ = evaluate(model, data, cfg, e, basis, 60, *selected[-1])
    rows.append(cm)
    pd.DataFrame(rows).to_csv(OUT / '图表/validation_grid.csv', index=False)
    required = float(raw['demand'][:, 42:60].sum())
    supply = float(e.initial.sum() + 17 * e.capacity)
    selection = dict(variants=selected, decision_config=asdict(DC), rule='Within each mode: all-SKU feasible first; otherwise pass share, mean RR, weighted turnover', validation_split=42, validation_end=60, future_labels_for_selection=False, necessary_supply_check=dict(full_demand=required, max_available=supply, all_period_positive_inventory_possible=bool(supply > required)), diagnostics_frozen=[dict(seed=43), dict(seed=44), dict(lr_multiplier=0.8), dict(lr_multiplier=1.2)])
    (OUT / '图表/selection_frozen_before_test.json').write_text(json.dumps(selection, indent=2))
    data, cfg, e, model, fm, fs = setup(raw, 60)
    trace['final_pretrain'] = p.pretrain_forecaster(model, data, cfg, DEVICE)
    basis = p.compute_residual_factor_basis(model, data, cfg, DEVICE)
    torch.save(dict(state_dict=model.state_dict(), basis=basis, feature_mean=fm, feature_std=fs, training_mean=e.mean, initial=e.initial, capacity=e.capacity), OUT / 'initial_model.pt')
    results = []
    for name, lr, alpha in selected:
        m, _ = evaluate(model, data, cfg, e, basis, 88, name, lr, alpha, output=OUT / 'results' / name)
        results.append(m)
        print('TEST', name, 'pass', m['passing_skus'], 'inventory', round(m['average_total_inventory']), 'weighted', round(m['weighted_turnover'], 4), 'WAPE', round(m['test_wape'], 4), flush=True)
    diagnostics = []
    for name, lr, alpha in selected[2:]:
        for seed in (43, 44):
            tag = f'{name}_seed{seed}'
            m, _ = evaluate(model, data, cfg, e, basis, 88, tag, lr, alpha, seed, OUT / 'results' / tag)
            diagnostics.append(m)
    for multiplier in (0.8, 1.2):
        tag = f'online_lr{multiplier}'
        m, _ = evaluate(model, data, cfg, e, basis, 88, tag, best_net['learning_rate'] * multiplier, output=OUT / 'results' / tag)
        diagnostics.append(m)
    pd.DataFrame(results).to_csv(OUT / '图表/comparison.csv', index=False)
    (OUT / '图表/灵敏度结果.json').write_text(json.dumps(diagnostics, indent=2))
    (OUT / '图表/training_trace.json').write_text(json.dumps(trace, indent=2))
    result = dict(inherited_model='Original ScaledDemandNet + improved.order, lagged-x/lead1 adaptation', no_dfl_finetuning=True, original_decision_config=asdict(DC), selection=selection, results=results, data_preparation=json.loads((OUT / '图表/data_audit.json').read_text()), versions=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__))
    (OUT / '图表/问题_1_结果.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (OUT / '图表/全部结果.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))

if __name__=='__main__':
    from common.cli import start
    start('online_learning', globals(), default_protocol='lead_one')
