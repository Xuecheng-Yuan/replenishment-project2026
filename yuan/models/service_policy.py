"""models.service_policy: actual model/processing implementation, not a legacy runner alias.

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
OUT=ROOT/'results/generated/service_policy'
WORK=OUT
TABLES=OUT/'图表'
OLD=ROOT/'results/generated/ready_immediate'
import numpy as np
import torch
from torch import nn

class ServicePolicy(nn.Module):

    def __init__(self, n_skus: int, n_features: int, hidden: int=32) -> None:
        super().__init__()
        self.sku_embedding = nn.Embedding(n_skus, 4)
        self.network = nn.Sequential(nn.Linear(n_features + 5, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.normal_(self.sku_embedding.weight, std=0.02)
        nn.init.normal_(self.network[-1].weight, std=0.005)
        nn.init.constant_(self.network[-1].bias, np.log(1.15))

    def forward(self, features, inventory, reference, sku_indices):
        state = (inventory / reference).clamp(0, 20).unsqueeze(1)
        inputs = torch.cat([features, state, self.sku_embedding(sku_indices)], dim=1)
        log_multiplier = self.network(inputs).squeeze(1).clamp(-3, 2)
        return reference * log_multiplier.exp()

def rollout(model, features, reference, demand, initial_inventory, sku_indices):
    """Before each future demand, use ONLY past features/current inventory.

    features[t] describes information known before demand[t]. The target at
    this boundary corresponds to the order placed at the preceding period end.
    """
    inventory = initial_inventory
    inventory_sum = torch.zeros((), dtype=demand.dtype)
    shortage_sum = torch.zeros((), dtype=demand.dtype)
    for t in range(demand.shape[0]):
        target = model(features[t], inventory, reference[t], sku_indices)
        available = torch.maximum(inventory, target)
        inventory = torch.relu(available - demand[t])
        shortage = torch.relu(demand[t] - available)
        inventory_sum = inventory_sum + inventory.sum()
        shortage_sum = shortage_sum + shortage.sum()
    return (inventory_sum, shortage_sum, inventory)

def service_loss(inventory_sum, shortage_sum, demand_sum, multiplier, rho, service=0.95):
    """J/C + lambda*g + rho/2*[g]+² with C=mean total train demand."""
    inventory_term = inventory_sum / demand_sum
    gap = shortage_sum / demand_sum - (1.0 - service)
    return (inventory_term + multiplier * gap + 0.5 * rho * torch.relu(gap).square(), gap)

@torch.no_grad()
def numpy_targets(model, features, reference, demand, initial_inventory, *, pre_period_decision: bool, inventory_scale: float=1.0):
    """Produce period-end targets with endogenous inventory feedback.

    T+1 feature blocks: initial pre-period block, then T period-end blocks.
    Testing starts with supplied initial inventory and zero pipeline.
    """
    n_skus, n_periods = demand.shape
    indices = torch.arange(n_skus)
    inventory = torch.tensor(initial_inventory, dtype=torch.float32)
    features = torch.tensor(features, dtype=torch.float32)
    reference = torch.tensor(reference, dtype=torch.float32)
    targets = np.zeros((n_skus, n_periods), dtype=np.float64)
    if pre_period_decision:
        target = model(features[0], inventory, reference[0], indices) * inventory_scale
        inventory = torch.maximum(inventory, torch.ceil(target))
    evaluation_initial = inventory.numpy().astype(np.float64).copy()
    for t in range(n_periods):
        inventory = torch.relu(inventory - torch.tensor(demand[:, t], dtype=torch.float32))
        target = torch.ceil(model(features[t + 1], inventory, reference[t + 1], indices) * inventory_scale)
        targets[:, t] = target.numpy()
        inventory = torch.maximum(inventory, target)
    return (evaluation_initial, targets)
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import pandas as pd
from common.evaluation import InventoryEvaluator

def wide(ids, values):
    frame = pd.DataFrame(values, columns=[f'period_{t + 1}' for t in range(values.shape[1])])
    frame.insert(0, 'item_id', ids)
    return frame

def evaluate(model, data, start, stop, columns, *, test=False, scale=1.0, save_dir=None, name='policy'):
    evaluator = InventoryEvaluator()
    if test:
        inventory = data['test_initial_inventory']
    else:
        indices = torch.arange(len(data['item_ids']))
        with torch.no_grad():
            _, _, inv = rollout(model, torch.tensor(data['features'][:start - 12, :, columns]), torch.tensor(data['reference'][:start - 12]), torch.tensor(data['demand'][:, 12:start].T), torch.tensor(data['warm_inventory']), indices)
            inventory = inv.numpy()
    initial, targets = numpy_targets(model, data['features'][start - 12:stop - 12 + 1, :, columns], data['reference'][start - 12:stop - 12 + 1], data['demand'][:, start:stop], inventory, pre_period_decision=not test, inventory_scale=scale)
    ids = data['item_ids']
    info = pd.DataFrame({'item_id': ids, 'initial_inventory': initial})
    result = evaluator.evaluate_dynamic_base_stock(wide(ids, data['demand'][:, start:stop]), info, wide(ids, targets), output_path=save_dir / 'test_per_sku_metrics.csv' if save_dir else None, trajectory_output_path=save_dir / 'test_trajectory.csv' if save_dir else None, model_name=name)
    if save_dir:
        wide(ids, targets).to_csv(save_dir / 'test_order_up_to.csv', index=False)
    return result

def selection_key(summary):
    fill = summary['total_fill_rate']
    stock = summary['average_total_inventory']
    return (0, stock) if fill >= 0.95 else (1, -fill, stock)

def train(data, specification, epochs, *, stop=42, select=True):
    torch.manual_seed(specification['seed'])
    np.random.seed(specification['seed'])
    names = data['feature_names'].tolist()
    columns = [k for k, name in enumerate(names) if specification['use_x'] or not name.startswith('x')]
    n_skus = len(data['item_ids'])
    model = ServicePolicy(n_skus, len(columns))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    features = torch.tensor(data['features'][:stop - 12, :, columns])
    reference = torch.tensor(data['reference'][:stop - 12])
    demands = torch.tensor(data['demand'][:, 12:stop].T)
    initial = torch.tensor(data['warm_inventory'])
    indices = torch.arange(n_skus)
    demand_sum = demands.sum()
    multiplier = 3.0
    log = []
    best = None
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        inventory, shortage, _ = rollout(model, features, reference, demands, initial, indices)
        loss, gap = service_loss(inventory, shortage, demand_sum, multiplier, specification['rho'])
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite policy loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if specification['adaptive']:
            multiplier = max(0.0, multiplier + 20.0 * float(gap.detach()))
        row = {'epoch': epoch, 'loss': float(loss.detach()), 'multiplier': multiplier, 'training_fill_rate': 1 - float(shortage.detach() / demand_sum), 'training_average_total_inventory': float(inventory.detach()) / (stop - 12), 'gap': float(gap.detach())}
        if select and (epoch % 5 == 0 or epoch == epochs):
            model.eval()
            result = evaluate(model, data, 42, 60, columns)
            row.update({f'validation_{k}': v for k, v in result.summary.items()})
            key = selection_key(result.summary)
            if best is None or key < best[0]:
                best = (key, copy.deepcopy(model.state_dict()), epoch, result.summary, multiplier)
        log.append(row)
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"{specification['name']} epoch={epoch}/{epochs} trainFR={row['training_fill_rate']:.4%} stock={row['training_average_total_inventory']:.0f} lambda={multiplier:.3f} elapsed={time.perf_counter() - start_time:.1f}s", flush=True)
    if select:
        assert best is not None
        model.load_state_dict(best[1])
        selected_epoch, summary = (best[2], best[3])
    else:
        selected_epoch, summary = (epochs, None)
    return (model.eval(), columns, selected_epoch, summary, log)

def train_legacy():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    data = dict(np.load(OUT / 'prepared_data.npz', allow_pickle=False))
    specifications = [{'name': 'primal_dual_rho100_seed42', 'rho': 100.0, 'adaptive': True, 'use_x': True, 'seed': 42}, {'name': 'primal_dual_rho500_seed42', 'rho': 500.0, 'adaptive': True, 'use_x': True, 'seed': 42}, {'name': 'dual_no_square_seed42', 'rho': 0.0, 'adaptive': True, 'use_x': True, 'seed': 42}, {'name': 'fixed_shortage_seed42', 'rho': 0.0, 'adaptive': False, 'use_x': True, 'seed': 42}, {'name': 'history_only_seed42', 'rho': 100.0, 'adaptive': True, 'use_x': False, 'seed': 42}, {'name': 'primal_dual_rho100_seed7', 'rho': 100.0, 'adaptive': True, 'use_x': True, 'seed': 7}, {'name': 'primal_dual_rho100_seed2026', 'rho': 100.0, 'adaptive': True, 'use_x': True, 'seed': 2026}]
    candidates = []
    models = []
    for specification in specifications:
        directory = OUT / specification['name']
        directory.mkdir(parents=True, exist_ok=True)
        model, columns, epoch, validation, log = train(data, specification, args.epochs)
        pd.DataFrame(log).to_csv(directory / 'training_history.csv', index=False)
        torch.save({'state_dict': model.state_dict(), 'columns': columns, 'specification': specification, 'selected_epoch': epoch}, directory / 'validation_checkpoint.pt')
        row = dict(specification, selected_epoch=epoch, **{f'validation_{k}': v for k, v in validation.items()})
        candidates.append(row)
        models.append((model, columns))
        pd.DataFrame(candidates).to_csv(OUT / 'validation_candidates.csv', index=False)
        print(f"SELECT {specification['name']} epoch={epoch} valFR={validation['total_fill_rate']:.4%} valStock={validation['average_total_inventory']:.0f}", flush=True)
    eligible = [k for k, s in enumerate(specifications) if s['adaptive'] and s['rho'] > 0 and s['use_x']]
    chosen = min(eligible, key=lambda k: selection_key({'total_fill_rate': candidates[k]['validation_total_fill_rate'], 'average_total_inventory': candidates[k]['validation_average_total_inventory']}))
    selection = {'selected_specification': specifications[chosen], 'selected_epoch': candidates[chosen]['selected_epoch'], 'criterion': 'Minimum validation inventory among FR >=95%; otherwise highest FR', 'test_used_for_selection': False, 'epochs_cap': args.epochs}
    (OUT / 'selection.json').write_text(json.dumps(selection, indent=2))
    rows = []
    for k, specification in enumerate(specifications):
        model, columns = models[k]
        directory = OUT / specification['name']
        result = evaluate(model, data, 60, 88, columns, test=True, save_dir=directory, name=specification['name'])
        record = dict(candidates[k], **{f'test_{key}': value for key, value in result.summary.items()})
        record['test_constraint_met'] = result.summary['total_fill_rate'] >= 0.95
        record['role'] = 'validation_checkpoint_ablation'
        rows.append(record)
        (directory / 'summary.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    specification = specifications[chosen]
    final_model, columns, _, _, log = train(data, specification, int(selection['selected_epoch']), stop=60, select=False)
    directory = OUT / 'final_full_train'
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(directory / 'training_history.csv', index=False)
    torch.save({'state_dict': final_model.state_dict(), 'columns': columns, 'specification': specification, 'selected_epoch': selection['selected_epoch']}, directory / 'model.pt')
    result = evaluate(final_model, data, 60, 88, columns, test=True, save_dir=directory, name='service_constrained_policy')
    record = dict(candidates[chosen], name='service_constrained_policy', role='full_train_refit', **{f'test_{key}': value for key, value in result.summary.items()})
    record['test_constraint_met'] = result.summary['total_fill_rate'] >= 0.95
    rows.append(record)
    (directory / 'summary.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    pd.DataFrame(rows).to_csv(OUT / 'experiment_results.csv', index=False)
    existing = pd.read_csv(ROOT / 'results/generated/all_model_comparison.csv')
    comparison = existing[existing.model.isin(['baseline', 'lightgbm', 'xgb_spo_hybrid', 'lgbm_spo_hybrid', 'direct_lgbm_spo_rank', 'direct_xgb_spo_rollout'])]
    common = ['model', 'test_fill_rate', 'test_constraint_met', 'test_average_total_inventory', 'test_total_turnover']
    new_rows = [{'model': row['name'], 'test_fill_rate': row['test_total_fill_rate'], 'test_constraint_met': row['test_constraint_met'], 'test_average_total_inventory': row['test_average_total_inventory'], 'test_total_turnover': row['test_total_turnover']} for row in rows]
    pd.concat([comparison[common], pd.DataFrame(new_rows)], ignore_index=True).to_csv(OUT / 'comparison.csv', index=False)
    print('FINAL ' + json.dumps(record, ensure_ascii=False), flush=True)
    import sys
    sources = [Path(__file__), ROOT / 'common/prepare.py', ROOT / 'models/service_policy.py']
    (OUT / 'run_manifest.json').write_text(json.dumps({'python': sys.executable, 'torch': torch.__version__, 'threads': args.threads, 'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, 'selection': selection}, indent=2))

MODEL_DIR=OUT
SPO_DIR=OUT
EXPERIMENT_DIR=OUT

if __name__=='__main__':
    from common.cli import start
    start('service_policy', globals(), default_protocol='legacy')
