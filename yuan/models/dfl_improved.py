"""models.dfl_improved: actual model/processing implementation, not a legacy runner alias.

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
OUT=ROOT/'results/generated/dfl_improved'
WORK=OUT
TABLES=OUT/'图表'
OLD=ROOT/'results/generated/ready_immediate'
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from models import nn_scenario as p

@dataclass
class DecisionConfig:
    safety_z: float = 1.645
    radius: float = 0.2
    steps: int = 20
    step_size: float = 0.02
    service_rho: float = 30.0
    soft_width: float = 0.05
    num_scenarios: int = 8
    robust_mix: float = 0.65
    robust_tau: float = 0.08
    first_order: bool = True

def project_capacity(q, capacity):
    q = q.clamp_min(0)
    return q * capacity / torch.maximum(q.sum(), capacity)

def ready_surrogate(available, demand, scale, width=0.05):
    return torch.sigmoid((available - demand - 0.001 * scale) / (width * scale))

def order(mu, sigma, inventory, scale, capacity, basis, cfg, seed, create_graph=False, optimize=True):
    """Orders depend on prediction + observed inventory, never true current demand.

    Optimize SKU-scale-normalized quantities, not a near-saturated softmax.
    The initialization stays differentiable with respect to the forecaster.
    """
    target = torch.expm1((mu + cfg.safety_z * sigma).clamp(0, 20))
    gap = F.relu(target - inventory)
    q = project_capacity(gap, capacity)
    if not optimize or cfg.steps == 0:
        return q
    pcfg = p.Config(num_scenarios=cfg.num_scenarios)
    scenarios = p.make_scenarios(mu, sigma, basis, pcfg, cfg.radius, seed)
    u = q / scale
    if not create_graph:
        u = u.detach().requires_grad_(True)
    elif not u.requires_grad:
        u = u.requires_grad_(True)
    for _ in range(cfg.steps):
        available = inventory + scale * u
        ends = F.relu(available[None, :] - scenarios)
        ready = ready_surrogate(available[None, :], scenarios, scale, cfg.soft_width)
        service = (1 - cfg.robust_mix) * ready.mean(dim=0) + cfg.robust_mix * p.normalized_softmin(ready, cfg.robust_tau, dim=0)
        shortages = F.relu(scenarios - available[None, :])
        scenario_cost = (ends / scale).mean(dim=1) + 10 * (shortages / scale).mean(dim=1)
        smooth_max = cfg.robust_tau * torch.logsumexp(scenario_cost / cfg.robust_tau, dim=0)
        cost = (1 - cfg.robust_mix) * scenario_cost.mean() + cfg.robust_mix * smooth_max + cfg.service_rho * F.relu(0.95 - service).square().mean()
        full_unroll = create_graph and (not cfg.first_order)
        grad = torch.autograd.grad(cost, u, create_graph=full_unroll, retain_graph=create_graph)[0]
        if cfg.first_order:
            grad = grad.detach()
        direction = (u.numel() * grad).clamp(-5.0, 5.0)
        u = project_capacity(scale * (u - cfg.step_size * direction), capacity) / scale
        if not create_graph:
            u = u.detach().requires_grad_(True)
    return scale * u
import json, hashlib, random
from dataclasses import asdict
import numpy as np
import pandas as pd
from models.nn_scenario import setup
from common.ready_evaluation import selection_key

def evaluate(model, data, cfg, e, basis, dc, optimize, output=None):
    model.eval()
    idx = torch.arange(data.N)
    scale = torch.tensor(np.maximum(e.mean, 0.5), dtype=torch.float32)
    capacity = torch.tensor(e.capacity, dtype=torch.float32)
    predictions = []

    def policy(t, inventory):
        with torch.no_grad():
            mu, sigma = model(data.X[cfg.train_periods + t], idx)
            predictions.append(torch.expm1(mu.clamp(0, 20)).numpy())
        return order(mu, sigma, torch.tensor(inventory, dtype=torch.float32), scale, capacity, basis, dc, 42 + 10000 + t, False, optimize).detach().numpy()
    actual = data.raw_demand[:, cfg.train_periods:cfg.train_periods + cfg.test_periods]
    m = e.evaluate(actual, policy, output)
    m['test_wape'] = float(np.abs(np.stack(predictions, axis=1) - actual).sum() / actual.sum())
    return m

def state_snapshots(model, data, cfg, e, basis, dc):
    """Beginning-of-period states from the current policy on past fit demands."""
    model.eval()
    idx = torch.arange(data.N)
    scale = torch.tensor(np.maximum(e.mean, 0.5), dtype=torch.float32)
    capacity = torch.tensor(e.capacity, dtype=torch.float32)
    inv = torch.tensor(e.initial, dtype=torch.float32)
    states = []
    for t in range(cfg.train_periods):
        states.append(inv.detach().clone())
        with torch.no_grad():
            mu, sigma = model(data.X[t], idx)
        q = order(mu, sigma, inv, scale, capacity, basis, dc, 42 + t, False).detach()
        inv = F_relu(inv + q - data.D[t])
    return states

def F_relu(a):
    return torch.clamp(a, min=0)

def train_epoch(model, data, cfg, e, basis, dc, epoch):
    states = state_snapshots(model, data, cfg, e, basis, dc)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    idx = torch.arange(data.N)
    scale = torch.tensor(np.maximum(e.mean, 0.5), dtype=torch.float32)
    capacity = torch.tensor(e.capacity, dtype=torch.float32)
    starts = list(range(cfg.train_periods - 7 + 1))
    random.Random(42 + epoch).shuffle(starts)
    logs = []
    for start in starts[:10]:
        inv = states[start].clone()
        sales_sum = 0.0
        inventory_sum = 0.0
        readiness = []
        nll = []
        for t in range(start, start + 7):
            mu, sigma = model(data.X[t], idx)
            q = order(mu, sigma, inv, scale, capacity, basis, dc, 42 + 1000 * epoch + t, True)
            available = inv + q
            sales = torch.minimum(available, data.D[t])
            inv = F_relu(available - data.D[t])
            sales_sum = sales_sum + (sales / scale).mean()
            inventory_sum = inventory_sum + (inv / scale).mean()
            readiness.append(ready_surrogate(available, data.D[t], scale, dc.soft_width))
            nll.append(p.gaussian_nll_log1p(mu, sigma, data.D[t]))
        rr = torch.stack(readiness).mean(dim=0)
        turnover = sales_sum / (inventory_sum + 0.001)
        loss = torch.log(inventory_sum + 0.01) - torch.log(sales_sum + 0.01) + 100 * F_relu(0.95 - rr).square().mean() + 0.2 * torch.stack(nll).mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite DFL loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        logs.append([loss.item(), turnover.item(), rr.mean().item()])
    record = dict(epoch=epoch, loss=float(np.mean(logs, axis=0)[0]), training_turnover=float(np.mean(logs, axis=0)[1]), soft_ready=float(np.mean(logs, axis=0)[2]))
    print('DFL v2', record, flush=True)
    return record

def save_model(model, basis, fm, fs, dc, name, epochs, e):
    path = OUT / name
    path.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), basis=basis, feature_mean=fm, feature_std=fs, decision_config=asdict(dc), epochs=epochs, optimize=name != 'nn_inventory_gap', training_mean=e.mean, initial_inventory=e.initial, capacity=e.capacity, item_ids=e.item_ids), path / 'model.pt')

def train_legacy():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    raw = np.load(OLD / 'data.npz', allow_pickle=False)
    p.set_seed(42)
    data, cfg, e, model, fm, fs = setup(raw, 42)
    trace = dict(validation_pretrain=p.pretrain_forecaster(model, data, cfg, torch.device('cpu')))
    basis = p.compute_residual_factor_basis(model, data, cfg, torch.device('cpu'))
    grid = []
    choices = []
    for optimize in (False, True):
        for z in (0.0, 1.0, 1.645, 2.33, 3.0, 4.0):
            for radius in (0.0,) if not optimize else (0.0, 0.2, 0.4):
                dc = DecisionConfig(safety_z=z, radius=radius)
                m = evaluate(model, data, cfg, e, basis, dc, optimize)
                row = dict(optimize=optimize, epoch=0, **asdict(dc), **m)
                grid.append(row)
                choices.append((selection_key(m), dc, optimize))
                print('validation', optimize, z, radius, m['sku_service_pass_rate'], m['mean_ready_rate'], flush=True)
    _, pto_dc, _ = max([v for v in choices if not v[2]], key=lambda v: v[0])
    key, dc, _ = max([v for v in choices if v[2]], key=lambda v: v[0])
    best_epoch = 0
    trace['validation_dfl'] = []
    for epoch in range(1, 4):
        trace['validation_dfl'].append(train_epoch(model, data, cfg, e, basis, dc, epoch))
        m = evaluate(model, data, cfg, e, basis, dc, True)
        grid.append(dict(optimize=True, epoch=epoch, **asdict(dc), **m))
        print('validation epoch', epoch, m, flush=True)
        if selection_key(m) > key:
            key = selection_key(m)
            best_epoch = epoch
    pd.DataFrame(grid).to_csv(OUT / 'validation_grid.csv', index=False)
    selection = dict(optimized_config=asdict(dc), pto_config=asdict(pto_dc), selected_epoch=best_epoch, selection_rule='all-SKU feasibility first; infeasible fallback SKU pass share, mean RR, turnover')
    (OUT / 'selection.json').write_text(json.dumps(selection, indent=2))
    p.set_seed(42)
    data, cfg, e, model, fm, fs = setup(raw, 60)
    trace['final_pretrain'] = p.pretrain_forecaster(model, data, cfg, torch.device('cpu'))
    basis = p.compute_residual_factor_basis(model, data, cfg, torch.device('cpu'))
    rows = []
    for name, config, optimize in [('nn_inventory_gap', pto_dc, False), ('nn_projected', dc, True)]:
        m = evaluate(model, data, cfg, e, basis, config, optimize, OUT / name)
        rows.append(dict(model=name, epoch=0, **asdict(config), **m))
        print(name, m, flush=True)
        save_model(model, basis, fm, fs, config, name, 0, e)
    trace['final_dfl'] = []
    if best_epoch == 0:
        m = evaluate(model, data, cfg, e, basis, dc, True, OUT / 'dfl_v2_selected')
        rows.append(dict(model='dfl_v2_selected', epoch=0, **asdict(dc), **m))
        save_model(model, basis, fm, fs, dc, 'dfl_v2_selected', 0, e)
    for epoch in range(1, 4):
        trace['final_dfl'].append(train_epoch(model, data, cfg, e, basis, dc, epoch))
        if epoch == best_epoch or epoch == 3:
            name = 'dfl_v2_selected' if epoch == best_epoch else 'dfl_v2_3epochs'
            m = evaluate(model, data, cfg, e, basis, dc, True, OUT / name)
            rows.append(dict(model=name, epoch=epoch, **asdict(dc), **m))
            print(name, m, flush=True)
            save_model(model, basis, fm, fs, dc, name, epoch, e)
    old = pd.read_csv(OLD / 'comparison.csv')
    old = old[old.model.isin(['historical_baseline', 'ridge', 'random_forest', 'lightgbm', 'nn_robust', 'dfl_robust_3epochs'])]
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(OUT / 'comparison.csv', index=False)
    (OUT / 'training_trace.json').write_text(json.dumps(trace, indent=2))
    sources = ['models/dfl_improved.py', 'models/dfl_improved.py', 'models/nn_scenario.py', 'models/nn_scenario.py', 'common/ready_evaluation.py']
    manifest = dict(seed=42, data_source=str(OLD / 'data.npz'), data_sha256=hashlib.sha256((OLD / 'data.npz').read_bytes()).hexdigest(), source_sha256={s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources}, capacity=float(e.capacity), initial_inventory_total=float(e.initial.sum()), validation_infeasible=True, unchanged_business_rules=True)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)

if __name__=='__main__':
    from common.cli import start
    start('dfl_improved', globals(), default_protocol='ready_immediate')
