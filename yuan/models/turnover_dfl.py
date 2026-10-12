"""models.turnover_dfl: actual model/processing implementation, not a legacy runner alias.

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
OUT=ROOT/'results/generated/turnover_dfl'
WORK=OUT
TABLES=OUT/'图表'
OLD=ROOT/'results/generated/ready_immediate'
from models import online_learning as base
import sys
import copy
import json
from dataclasses import asdict
import numpy as np
import pandas as pd
import torch
from models import nn_scenario as p
from models.dfl_improved import order, ready_surrogate
from common.ready_evaluation import ReadyRateEvaluator
DC = base.DC

def rank(m):
    if m['all_skus_feasible']:
        return (1, m['turnover'], 0.0, 0.0)
    return (0, m['sku_service_pass_rate'], m['mean_ready_rate'], m['turnover'])

def states(model, data, count, e, basis, seed):
    """Detach warm states computed solely on observed training history."""
    idx = torch.arange(data.N)
    scale = torch.tensor(np.maximum(e.mean, 0.5), dtype=torch.float32)
    cap = torch.tensor(e.capacity, dtype=torch.float32)
    inv = torch.tensor(e.initial, dtype=torch.float32)
    pending = torch.zeros_like(inv)
    records = []
    model.eval()
    for k in range(count):
        records.append((inv.detach().clone(), pending.detach().clone()))
        inv = torch.relu(inv + pending - data.D[k])
        if k + 1 < count:
            with torch.no_grad():
                mu, sig = model(data.X[k + 1], idx)
            pending = order(mu, sig, inv, scale, cap, basis, DC, seed + 10000 + k, False).detach()
    return records

def window_loss(model, data, start, end, initial, pending, e, basis, lam, seed, nll_only=False, nll_weight=0.2):
    assert 0 <= start < end <= len(data.X)
    idx = torch.arange(data.N)
    if nll_only:
        terms = []
        for k in range(start, end):
            mu, sig = model(data.X[k], idx)
            terms.append(p.gaussian_nll_log1p(mu, sig, data.D[k]))
        forecast = torch.stack(terms).mean()
        return (nll_weight * forecast, dict(turn_loss=0.0, service_penalty=0.0, nll=float(forecast.detach()), soft_ready=0.0, true_sales=0.0, true_inventory_sum=0.0))
    scale = torch.tensor(np.maximum(e.mean, 0.5), dtype=torch.float32)
    cap = torch.tensor(e.capacity, dtype=torch.float32)
    inv = initial.clone()
    current_pending = pending.clone()
    total_sales = inv.new_tensor(0.0)
    total_inventory = inv.new_tensor(0.0)
    readiness = []
    nll = []
    for k in range(start, end):
        mu, sig = model(data.X[k], idx)
        nll.append(p.gaussian_nll_log1p(mu, sig, data.D[k]))
        available = inv + current_pending
        sold = torch.minimum(available, data.D[k])
        readiness.append(ready_surrogate(available, data.D[k], scale, DC.soft_width))
        inv = torch.relu(available - data.D[k])
        total_sales = total_sales + sold.sum()
        total_inventory = total_inventory + inv.sum()
        if k + 1 < end and (not nll_only):
            next_mu, next_sig = model(data.X[k + 1], idx)
            current_pending = order(next_mu, next_sig, inv, scale, cap, basis, DC, seed + 10000 + k, True)
        elif nll_only:
            current_pending = torch.zeros_like(inv)
    denominator = data.N * scale.mean()
    turn = torch.log(total_inventory / denominator + 1e-06) - torch.log(total_sales / denominator + 1e-06)
    rr = torch.stack(readiness).mean(dim=0)
    service = torch.relu(0.95 - rr).square().mean()
    forecast = torch.stack(nll).mean()
    loss = turn + lam * service + nll_weight * forecast
    return (loss, dict(turn_loss=float(turn.detach()), service_penalty=float(service.detach()), nll=float(forecast.detach()), soft_ready=float(rr.detach().mean()), true_sales=float(total_sales.detach()), true_inventory_sum=float(total_inventory.detach())))

def train_epoch(model, data, count, e, basis, lam, lr, epoch, optimizer=None, nll_only=False):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    warm = states(model, data, count, e, basis, 42)
    horizon = min(7, count)
    starts = np.unique(np.linspace(0, count - horizon, min(12, count - horizon + 1)).astype(int))
    np.random.default_rng(42 + epoch).shuffle(starts)
    model.train()
    logs = []
    infeasible_windows = 0
    for start in starts:
        end = int(start) + horizon
        assert end <= count, 'Training reached validation/test target!'
        inv, pending = warm[int(start)]
        supply = float(inv.sum() + pending.sum()) + (horizon - 1) * e.capacity
        if float(data.D[int(start):end].sum()) >= supply:
            infeasible_windows += 1
        loss, record = window_loss(model, data, int(start), end, inv, pending, e, basis, lam, 42, nll_only)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite task loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite gradient')
        optimizer.step()
        if not all((torch.isfinite(v).all() for v in model.parameters())):
            raise ValueError('Nonfinite trained parameter')
        logs.append(dict(**record, loss=float(loss.detach()), gradient_norm=float(norm), start_raw_period=int(start) + 14, end_raw_period=end + 13))
    record = dict(epoch=epoch, lambda_service=lam, learning_rate=lr, nll_only=nll_only, windows=len(logs), capacity_bound_infeasible_windows=infeasible_windows, max_training_raw_period=max((v['end_raw_period'] for v in logs)), **{key: float(np.mean([v[key] for v in logs])) for key in ('loss', 'turn_loss', 'service_penalty', 'nll', 'soft_ready', 'gradient_norm')})
    print('train', record, flush=True)
    return (optimizer, record)

def eval_model(model, data, cfg, e, basis, end, name, output=None, seed=42):
    m, _ = base.evaluate(model, data, cfg, e, basis, end, name, seed=seed, output=output, with_updates=False)
    return m

def refit(initial, data, cfg, e, basis, config, epochs, nll_only=False):
    model = copy.deepcopy(initial)
    optimizer = None
    trace = []
    for epoch in range(1, epochs + 1):
        optimizer, record = train_epoch(model, data, cfg.train_periods, e, basis, config['lambda_service'], config['learning_rate'], epoch, optimizer, nll_only)
        trace.append(record)
    return (model, trace)

def train_legacy():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    raw = np.load(OUT / '数据/processed/panel.npz', allow_pickle=False)
    operating = ReadyRateEvaluator(raw['demand'][:, :60], raw['item_ids'])
    data, cfg, _, initial, _, _ = base.setup(raw, 24)
    trace = dict(validation_pretrain=p.pretrain_forecaster(initial, data, cfg, torch.device('cpu')))
    basis = p.compute_residual_factor_basis(initial, data, cfg, torch.device('cpu'))
    baseline = eval_model(initial, data, cfg, operating, basis, 32, 'validation_frozen')
    rows = [dict(**baseline, lambda_service=0.0, epoch=0, finetune_learning_rate=0.0)]
    best_all = dict(lambda_service=100.0, learning_rate=1e-05, epoch=0)
    best_all_key = rank(baseline)
    best_positive = None
    best_positive_key = None
    train_trace = []
    for lam in (100.0, 1000.0, 10000.0):
        for lr in (1e-05, 5e-05):
            model = copy.deepcopy(initial)
            optimizer = None
            for epoch in range(1, 4):
                optimizer, record = train_epoch(model, data, cfg.train_periods, operating, basis, lam, lr, epoch, optimizer)
                train_trace.append(record)
                m = eval_model(model, data, cfg, operating, basis, 32, 'validation_dfl')
                rows.append(dict(**m, lambda_service=lam, epoch=epoch, finetune_learning_rate=lr))
                config = dict(lambda_service=lam, learning_rate=lr, epoch=epoch)
                key = rank(m)
                if key > best_all_key:
                    best_all_key = key
                    best_all = config
                if best_positive_key is None or key > best_positive_key:
                    best_positive_key = key
                    best_positive = config
                print('validation', lam, lr, epoch, 'pass', m['passing_skus'], 'turnover', m['turnover'], flush=True)
    pd.DataFrame(rows).to_csv(OUT / '图表/validation_grid.csv', index=False)
    selection = dict(overall=best_all, positive_dfl=best_positive, decision_config=asdict(DC), validation_split=24, validation_end=32, operating_parameters_from_training60=True, validation_demand=float(raw['demand'][:, 24:32].sum()), max_available=float(operating.initial.sum() + 7 * operating.capacity), selection_rule='feasible first, original total turnover; infeasible pass share/mean RR/total turnover', test_selection=False, diagnostics=['scene_seed43', 'scene_seed44', 'learning_rate_times0.8', 'learning_rate_times1.2'])
    (OUT / '图表/selection_frozen_before_test.json').write_text(json.dumps(selection, indent=2))
    data, cfg, e, initial, _, _ = base.setup(raw, 60)
    source = ROOT / 'results/generated/online_learning/initial_model.pt'
    ck = torch.load(source, map_location='cpu', weights_only=False)
    initial.load_state_dict(ck['state_dict'])
    basis = ck['basis']
    torch.save(ck, OUT / 'initial_model.pt')
    variants = [('frozen', best_positive, 0, False), ('dfl_selected', best_all, best_all['epoch'], False), ('dfl_positive_selected', best_positive, best_positive['epoch'], False), ('dfl_fixed3', best_positive, 3, False), ('nll_extra_training', best_positive, best_positive['epoch'], True)]
    results = []
    trained = {}
    for name, config, epochs, nll_only in variants:
        model, history = refit(initial, data, cfg, e, basis, config, epochs, nll_only)
        trace[name] = history
        trained[name] = model
        output = OUT / 'results' / name
        m = eval_model(model, data, cfg, e, basis, 88, name, output)
        m.update(lambda_service=config['lambda_service'], finetune_epochs=epochs, nll_only=nll_only, finetune_learning_rate=config['learning_rate'])
        (output / 'summary.json').write_text(json.dumps(m, indent=2))
        torch.save(model.state_dict(), output / 'model_state.pt')
        results.append(m)
        print('TEST', name, 'pass', m['passing_skus'], 'inventory', m['average_total_inventory'], 'turnover', m['turnover'], flush=True)
    diagnostics = []
    for seed in (43, 44):
        name = f'dfl_positive_seed{seed}'
        m = eval_model(trained['dfl_positive_selected'], data, cfg, e, basis, 88, name, OUT / 'results' / name, seed)
        diagnostics.append(m)
    for mult in (0.8, 1.2):
        config = {**best_positive, 'learning_rate': best_positive['learning_rate'] * mult}
        name = f'dfl_lr{mult}'
        model, history = refit(initial, data, cfg, e, basis, config, best_positive['epoch'])
        trace[name] = history
        m = eval_model(model, data, cfg, e, basis, 88, name, OUT / 'results' / name)
        m.update(finetune_learning_rate=config['learning_rate'])
        (OUT / 'results' / name / 'summary.json').write_text(json.dumps(m, indent=2))
        diagnostics.append(m)
    pd.DataFrame(results).to_csv(OUT / '图表/comparison.csv', index=False)
    (OUT / '图表/灵敏度结果.json').write_text(json.dumps(diagnostics, indent=2))
    trace['validation_task_training'] = train_trace
    (OUT / '图表/training_trace.json').write_text(json.dumps(trace, indent=2))
    result = dict(inherited_model='ScaledDemandNet + original finite-scene improved.order', method='custom first-order turnover and SKU-service DFL, not SPO+', selection=selection, results=results, data_preparation=json.loads((OUT / '图表/data_audit.json').read_text()), pretrained_source=str(source.relative_to(ROOT)), versions=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__))
    for name in ('问题_1_结果.json', '全部结果.json'):
        (OUT / '图表' / name).write_text(json.dumps(result, ensure_ascii=False, indent=2))

if __name__=='__main__':
    from common.cli import start
    start('turnover_dfl', globals(), default_protocol='lead_one')
