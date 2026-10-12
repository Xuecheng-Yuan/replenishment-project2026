"""models.spo_linear: actual model/processing implementation, not a legacy runner alias.

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
OUT=ROOT/'results/generated/spo/spo_plus_linear'
WORK=OUT
TABLES=OUT/'图表'
OLD=ROOT/'results/generated/ready_immediate'
from dataclasses import dataclass, field
import numpy as np

def action_targets(centers: np.ndarray, scales: np.ndarray, action_multipliers: np.ndarray) -> np.ndarray:
    """Return nonnegative integer order-up-to levels for every row/action."""
    center = np.asarray(centers, dtype=float).reshape(-1, 1)
    scale = np.asarray(scales, dtype=float).reshape(-1, 1)
    actions = np.asarray(action_multipliers, dtype=float).reshape(1, -1)
    if center.shape != scale.shape:
        raise ValueError('centers 和 scales 的长度必须相同。')
    if (scale < 0).any() or not np.isfinite(center).all() or (not np.isfinite(scale).all()):
        raise ValueError('centers/scales 必须有限，且 scales 不能为负。')
    return np.ceil(np.clip(center + scale * actions, 0.0, None))

def action_costs(demand: np.ndarray, targets: np.ndarray, shortage_penalty: float, holding_cost: float=1.0) -> np.ndarray:
    """Build the realized linear cost vector for each sample.

    Component ``a`` is the single-period newsvendor proxy
    ``h(B_a-D)^+ + p(D-B_a)^+``.  It is a training loss only; final reporting
    always uses the full dynamic lost-sales simulator.
    """
    actual = np.asarray(demand, dtype=float).reshape(-1, 1)
    levels = np.asarray(targets, dtype=float)
    if levels.ndim != 2 or levels.shape[0] != len(actual):
        raise ValueError('targets 必须是 样本×动作 的二维数组。')
    if shortage_penalty <= 0 or holding_cost < 0:
        raise ValueError('shortage_penalty 必须为正，holding_cost 不能为负。')
    return (holding_cost * np.maximum(levels - actual, 0.0) + shortage_penalty * np.maximum(actual - levels, 0.0)).astype(np.float32)

@dataclass
class SPOPlusLinear:
    """Linear cost-vector predictor trained with the paper's SPO+ subgradient."""
    epochs: int = 12
    batch_size: int = 2048
    learning_rate: float = 0.05
    ridge: float = 0.0001
    random_seed: int = 42
    coefficients_: np.ndarray | None = field(init=False, default=None, repr=False)
    feature_mean_: np.ndarray | None = field(init=False, default=None, repr=False)
    feature_scale_: np.ndarray | None = field(init=False, default=None, repr=False)
    cost_scale_: float | None = field(init=False, default=None)
    history_: list[dict] = field(init=False, default_factory=list)

    def fit(self, features: np.ndarray, costs: np.ndarray) -> 'SPOPlusLinear':
        """Minimize empirical SPO+ loss by averaged mini-batch subgradient descent."""
        X = np.asarray(features, dtype=np.float32)
        C = np.asarray(costs, dtype=np.float32)
        if X.ndim != 2 or C.ndim != 2 or len(X) != len(C):
            raise ValueError('features/costs 必须是行数相同的二维数组。')
        if len(X) == 0 or C.shape[1] < 2 or (not np.isfinite(X).all()) or (not np.isfinite(C).all()):
            raise ValueError('训练数据为空、动作不足或包含非有限值。')
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0 or (self.ridge < 0):
            raise ValueError('训练超参数不合法。')
        self.feature_mean_ = X.mean(axis=0, dtype=np.float64).astype(np.float32)
        raw_scale = X.std(axis=0, dtype=np.float64).astype(np.float32)
        self.feature_scale_ = np.where(raw_scale > 1e-06, raw_scale, 1.0).astype(np.float32)
        Xn = np.clip((X - self.feature_mean_) / self.feature_scale_, -10.0, 10.0)
        Xn = np.column_stack([np.ones(len(Xn), dtype=np.float32), Xn]).astype(np.float32)
        positive = C[C > 0]
        self.cost_scale_ = float(np.median(positive)) if positive.size else 1.0
        if not np.isfinite(self.cost_scale_) or self.cost_scale_ <= 0:
            self.cost_scale_ = 1.0
        Cn = C / self.cost_scale_
        n_samples, n_features = Xn.shape
        n_actions = Cn.shape[1]
        weights = np.zeros((n_features, n_actions), dtype=np.float32)
        averaged = np.zeros_like(weights)
        averaging_weight = 0.0
        rng = np.random.default_rng(self.random_seed)
        step = 0
        self.history_ = []
        for epoch in range(self.epochs):
            order = rng.permutation(n_samples)
            mismatch_count = 0
            for start in range(0, n_samples, self.batch_size):
                indices = order[start:start + self.batch_size]
                xb = Xn[indices]
                cb = Cn[indices]
                predicted = xb @ weights
                true_best = np.argmin(cb, axis=1)
                adversarial_best = np.argmin(2.0 * predicted - cb, axis=1)
                mismatch_count += int(np.count_nonzero(true_best != adversarial_best))
                direction = np.zeros((len(indices), n_actions), dtype=np.float32)
                rows = np.arange(len(indices))
                direction[rows, true_best] += 2.0
                direction[rows, adversarial_best] -= 2.0
                gradient = xb.T @ direction / len(indices)
                gradient[1:] += self.ridge * weights[1:]
                step += 1
                rate = self.learning_rate / np.sqrt(step)
                weights -= rate * gradient
                averaged += rate * weights
                averaging_weight += rate
            current = averaged / averaging_weight
            predictions = Xn @ current
            chosen = np.argmin(predictions, axis=1)
            oracle = np.argmin(Cn, axis=1)
            spo_loss = float(np.mean(Cn[np.arange(n_samples), chosen] - Cn[np.arange(n_samples), oracle]))
            self.history_.append({'epoch': epoch + 1, 'spo_loss_normalized': spo_loss, 'oracle_mismatch_share': mismatch_count / n_samples})
        self.coefficients_ = averaged / averaging_weight
        return self

    def predict_costs(self, features: np.ndarray) -> np.ndarray:
        """Predict an action-cost vector for each row."""
        if self.coefficients_ is None or self.feature_mean_ is None or self.feature_scale_ is None:
            raise ValueError('模型尚未拟合。')
        X = np.asarray(features, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != len(self.feature_mean_):
            raise ValueError('预测特征维度与训练时不一致。')
        Xn = np.clip((X - self.feature_mean_) / self.feature_scale_, -10.0, 10.0)
        Xn = np.column_stack([np.ones(len(Xn), dtype=np.float32), Xn]).astype(np.float32)
        return Xn @ self.coefficients_ * float(self.cost_scale_)

    def predict_actions(self, features: np.ndarray) -> np.ndarray:
        """Return the optimization oracle decision for every row."""
        return np.argmin(self.predict_costs(features), axis=1)

    def state_dict(self) -> dict[str, np.ndarray | float | list[dict]]:
        if self.coefficients_ is None or self.feature_mean_ is None or self.feature_scale_ is None:
            raise ValueError('模型尚未拟合。')
        return {'coefficients': self.coefficients_, 'feature_mean': self.feature_mean_, 'feature_scale': self.feature_scale_, 'cost_scale': float(self.cost_scale_), 'history': self.history_}
import argparse
import json
import time
from dataclasses import asdict, dataclass
import pandas as pd
from common.evaluation import InventoryEvaluator
from common.features import build_supervised_data
from common.features import load_panel
from common.policy_state import CURRENT_INVENTORY_FEATURE
from common.policy_state import append_current_inventory
from common.policy_state import behavior_ending_inventory
from common.policy_state import inventory_for_target_periods
from common.policy_state import rollout_inventory_aware_policy

@dataclass(frozen=True)
class SPOExperimentConfig:
    service_level: float = 0.95
    validation_start: int = 42
    minimum_history: int = 12
    action_min: float = 0.0
    action_max: float = 8.0
    action_step: float = 0.5
    holding_cost: float = 1.0
    shortage_penalties: tuple[float, ...] = (10.0, 20.0, 40.0, 80.0, 160.0)
    learning_rates: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2)
    tuning_epochs: int = 8
    final_epochs: int = 8
    batch_size: int = 2048
    ridge: float = 0.0001
    random_seed: int = 42

def train_legacy() -> None:
    parser = argparse.ArgumentParser(description='运行论文 Smart Predict-then-Optimize 的 SPO+ 模型。')
    parser.add_argument('--item-id', default='1')
    parser.add_argument('--quick', action='store_true', help='仅做快速代码检查，不作为正式结果。')
    args = parser.parse_args()
    config = SPOExperimentConfig()
    if args.quick:
        config = SPOExperimentConfig(shortage_penalties=(20.0, 80.0), learning_rates=(0.05,), tuning_epochs=2, final_epochs=2)
    start = time.perf_counter()
    SPO_DIR.mkdir(parents=True, exist_ok=True)
    print('读取 train/test 面板……', flush=True)
    train = load_panel(DATA_DIR / 'replenishment_train.xlsx')
    test = load_panel(DATA_DIR / 'replenishment_test.xlsx')
    if not np.array_equal(train.item_ids, test.item_ids):
        raise ValueError('train/test 的 item_id 或顺序不一致。')
    print('构造严格滞后的历史特征、当期库存和有限补货动作……', flush=True)
    training_periods = np.arange(config.minimum_history, config.validation_start)
    validation_periods = np.arange(config.validation_start, train.demand.shape[1])
    all_train_periods = np.arange(config.minimum_history, train.demand.shape[1])
    tune_train = build_supervised_data(train.demand, train.x, train.categories, training_periods)
    validation = build_supervised_data(train.demand, train.x, train.categories, validation_periods)
    validation_extra = build_supervised_data(train.demand, train.x, train.categories, np.asarray([train.demand.shape[1]]), include_target=False)
    final_train = build_supervised_data(train.demand, train.x, train.categories, all_train_periods)
    combined_demand = np.concatenate([train.demand, test.demand], axis=1)
    combined_x = np.concatenate([train.x, test.x], axis=1)
    test_features = build_supervised_data(combined_demand, combined_x, train.categories, np.arange(train.demand.shape[1], combined_demand.shape[1] + 1), include_target=False)
    behavior_inventory = behavior_ending_inventory(train.demand, train.initial_inventory)
    tune_features = append_current_inventory(tune_train.features, inventory_for_target_periods(behavior_inventory, tune_train.target_periods))
    final_features = append_current_inventory(final_train.features, inventory_for_target_periods(behavior_inventory, final_train.target_periods))
    actions = np.arange(config.action_min, config.action_max + config.action_step / 2, config.action_step, dtype=np.float32)
    tune_center, tune_scale = _center_scale(tune_train)
    tune_targets = action_targets(tune_center, tune_scale, actions)
    validation_all_features = np.vstack([validation.features, validation_extra.features])
    n_skus = len(train.item_ids)
    evaluator = InventoryEvaluator(lead_time=1)
    validation_demand_df = _wide(train.item_ids, train.demand[:, config.validation_start:])
    candidates: list[dict] = []
    best: tuple[tuple, dict, SPOPlusLinear] | None = None
    combinations = [(penalty, rate) for penalty in config.shortage_penalties for rate in config.learning_rates]
    for candidate_index, (penalty, rate) in enumerate(combinations, start=1):
        candidate_start = time.perf_counter()
        print(f'候选 {candidate_index}/{len(combinations)}: 缺货惩罚={penalty:g}, 学习率={rate:g}', flush=True)
        costs = action_costs(tune_train.target, tune_targets, penalty, holding_cost=config.holding_cost)
        model = SPOPlusLinear(epochs=config.tuning_epochs, batch_size=config.batch_size, learning_rate=rate, ridge=config.ridge, random_seed=config.random_seed).fit(tune_features, costs)
        validation_rollout = _rollout_linear_policy(model, train.demand[:, config.validation_start:], behavior_inventory[:, config.validation_start - 1], validation_all_features, validation.feature_names, actions, include_pre_period_decision=True)
        validation_actions = validation_rollout.actions.T.reshape(-1)
        validation_result = evaluator.evaluate_dynamic_base_stock(validation_demand_df, pd.DataFrame({'item_id': train.item_ids, 'initial_inventory': validation_rollout.initial_inventory}), _wide(train.item_ids, validation_rollout.targets))
        validation_spo_loss = _decision_regret(validation, validation_actions[:len(validation.target)], actions, penalty, config.holding_cost)
        row = {'candidate': candidate_index, 'shortage_penalty': penalty, 'learning_rate': rate, 'ridge': config.ridge, 'epochs': config.tuning_epochs, 'validation_spo_loss': validation_spo_loss, 'validation_fill_rate': validation_result.summary['total_fill_rate'], 'validation_average_total_inventory': validation_result.summary['average_total_inventory'], 'validation_total_turnover': validation_result.summary['total_turnover'], 'validation_constraint_met': validation_result.summary['total_fill_rate'] >= config.service_level, 'runtime_seconds': time.perf_counter() - candidate_start}
        candidates.append(row)
        key = _selection_key(validation_result.summary, validation_spo_loss, config.service_level)
        if best is None or key < best[0]:
            best = (key, row, model)
        print(f"验证 Fill Rate={row['validation_fill_rate']:.4%}, 平均总库存={row['validation_average_total_inventory']:.0f}, SPO regret={validation_spo_loss:.4f}", flush=True)
    assert best is not None
    selected = best[1]
    pd.DataFrame(candidates).to_csv(SPO_DIR / 'validation_candidates.csv', index=False)
    print(f'使用最佳候选在全部训练期重训: {selected}', flush=True)
    final_center, final_scale = _center_scale(final_train)
    final_targets = action_targets(final_center, final_scale, actions)
    final_costs = action_costs(final_train.target, final_targets, selected['shortage_penalty'], holding_cost=config.holding_cost)
    final_model = SPOPlusLinear(epochs=config.final_epochs, batch_size=config.batch_size, learning_rate=selected['learning_rate'], ridge=config.ridge, random_seed=config.random_seed).fit(final_features, final_costs)
    state = final_model.state_dict()
    np.savez_compressed(SPO_DIR / 'model.npz', coefficients=state['coefficients'], feature_mean=state['feature_mean'], feature_scale=state['feature_scale'], cost_scale=np.asarray([state['cost_scale']]), action_multipliers=actions)
    pd.DataFrame(final_model.history_).to_csv(SPO_DIR / 'training_history.csv', index=False)
    test_rollout = _rollout_linear_policy(final_model, test.demand, test.initial_inventory, test_features.features, test_features.feature_names, actions, include_pre_period_decision=False)
    test_actions = test_rollout.actions.T.reshape(-1)
    test_targets = test_rollout.targets
    test_forecasts = _feature_matrix(test_features.features, test_features.feature_names, 'demand_mean_12', len(test_features.target_periods), n_skus)[:, :-1]
    result = evaluator.evaluate_dynamic_base_stock(_wide(train.item_ids, test.demand), pd.DataFrame({'item_id': train.item_ids, 'initial_inventory': test.initial_inventory}), _wide(train.item_ids, test_targets), output_path=SPO_DIR / 'test_per_sku_metrics.csv', trajectory_output_path=SPO_DIR / 'test_trajectory.csv', model_name='spo_plus_linear', forecasts_df=_wide(train.item_ids, test_forecasts))
    _wide(train.item_ids, test_targets).to_csv(SPO_DIR / 'test_order_up_to.csv', index=False)
    _wide(train.item_ids, test_forecasts).to_csv(SPO_DIR / 'test_forecasts.csv', index=False)
    selected_action_path = actions[test_rollout.actions]
    _wide(train.item_ids, selected_action_path).to_csv(SPO_DIR / 'test_selected_actions.csv', index=False)
    test_spo_loss = _test_next_period_regret(test, test_features, test_rollout.actions, actions, selected['shortage_penalty'], config.holding_cost)
    row = {'model': 'spo_plus_linear', 'family': 'decision_focused_learning', 'parameters': json.dumps({'loss': 'SPO+', 'shortage_penalty': selected['shortage_penalty'], 'holding_cost': config.holding_cost, 'action_grid': [config.action_min, config.action_max, config.action_step], 'learning_rate': selected['learning_rate'], 'ridge': config.ridge, 'epochs': config.final_epochs, 'inventory_feature': CURRENT_INVENTORY_FEATURE}, ensure_ascii=False), 'selected_z': np.nan, 'training_rows': len(final_train.target), 'validation_wape': np.nan, 'test_wape': np.nan, 'validation_spo_loss': selected['validation_spo_loss'], 'test_spo_loss': test_spo_loss, 'validation_fill_rate': selected['validation_fill_rate'], 'validation_average_total_inventory': selected['validation_average_total_inventory'], 'test_fill_rate': result.summary['total_fill_rate'], 'test_constraint_met': result.summary['total_fill_rate'] >= config.service_level, 'test_average_total_inventory': result.summary['average_total_inventory'], 'test_total_turnover': result.summary['total_turnover'], 'test_total_lost_sales': result.summary['total_lost_sales'], 'test_sku_95_share': result.summary['sku_service_level_95_share'], 'runtime_seconds': time.perf_counter() - start}
    summary = {'result': row, 'test': result.summary, 'selected_validation_candidate': selected, 'config': asdict(config), 'method_note': 'Finite-action exact specialization of SPO+: predict the linear cost vector of candidate order-up-to levels and choose its minimum-cost component. Current ending inventory is fed back sequentially at every decision.'}
    (SPO_DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    _save_item_trajectory(args.item_id)
    combined = _update_comparison(row)
    _plot_comparison(combined)
    print('\nSPO+ 实验完成', flush=True)
    print(pd.DataFrame([row])[['model', 'test_fill_rate', 'test_average_total_inventory', 'test_total_turnover']].to_string(index=False))
    print(f'总耗时: {time.perf_counter() - start:.2f} 秒', flush=True)

def _center_scale(supervised) -> tuple[np.ndarray, np.ndarray]:
    names = supervised.feature_names
    return (supervised.features[:, names.index('demand_mean_12')].astype(float), supervised.features[:, names.index('demand_std_12')].astype(float))

def _selected_target_path(features: np.ndarray, feature_names: list[str], selected_actions: np.ndarray, actions: np.ndarray, n_periods: int, n_skus: int) -> np.ndarray:
    center = features[:, feature_names.index('demand_mean_12')]
    scale = features[:, feature_names.index('demand_std_12')]
    selected = np.ceil(np.clip(center + scale * actions[selected_actions], 0.0, None))
    return _period_matrix(selected, n_periods, n_skus)

def _rollout_linear_policy(model: SPOPlusLinear, demand: np.ndarray, initial_inventory: np.ndarray, decision_features: np.ndarray, feature_names: list[str], actions: np.ndarray, *, include_pre_period_decision: bool):
    center_index = feature_names.index('demand_mean_12')
    scale_index = feature_names.index('demand_std_12')

    def selector(_block_index: int, augmented: np.ndarray):
        selected = model.predict_actions(augmented)
        center = augmented[:, center_index]
        scale = augmented[:, scale_index]
        target = np.ceil(np.clip(center + scale * actions[selected], 0.0, None))
        return (selected, target)
    return rollout_inventory_aware_policy(demand, initial_inventory, decision_features, selector, include_pre_period_decision=include_pre_period_decision)

def _feature_matrix(features: np.ndarray, feature_names: list[str], name: str, n_periods: int, n_skus: int) -> np.ndarray:
    return _period_matrix(features[:, feature_names.index(name)], n_periods, n_skus)

def _decision_regret(supervised, selected_actions: np.ndarray, actions: np.ndarray, shortage_penalty: float, holding_cost: float) -> float:
    center, scale = _center_scale(supervised)
    costs = action_costs(supervised.target, action_targets(center, scale, actions), shortage_penalty, holding_cost)
    chosen = costs[np.arange(len(costs)), selected_actions]
    return float(np.mean(chosen - costs.min(axis=1)))

def _test_next_period_regret(test, test_features, selected_action_path: np.ndarray, actions: np.ndarray, shortage_penalty: float, holding_cost: float) -> float:
    n_skus = len(test.item_ids)
    n_periods = test.demand.shape[1] - 1
    if n_periods <= 0:
        return float('nan')
    center, scale = _center_scale(test_features)
    center_path = _period_matrix(center, len(test_features.target_periods), n_skus)[:, 1:-1]
    scale_path = _period_matrix(scale, len(test_features.target_periods), n_skus)[:, 1:-1]
    selected = selected_action_path[:, :-1].T.reshape(-1)
    costs = action_costs(test.demand[:, 1:].T.reshape(-1), action_targets(center_path.T.reshape(-1), scale_path.T.reshape(-1), actions), shortage_penalty, holding_cost)
    chosen = costs[np.arange(n_periods * n_skus), selected]
    return float(np.mean(chosen - costs.min(axis=1)))

def _selection_key(summary: dict, spo_loss: float, service_level: float) -> tuple:
    feasible = summary['total_fill_rate'] >= service_level
    return (not feasible, summary['average_total_inventory'] if feasible else -summary['total_fill_rate'], spo_loss)

def _period_matrix(values: np.ndarray, n_periods: int, n_skus: int) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.size != n_periods * n_skus:
        raise ValueError('数值数量与 SKU×周期数不一致。')
    return vector.reshape(n_periods, n_skus).T

def _wide(item_ids: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    matrix = np.asarray(values, dtype=float)
    frame = pd.DataFrame(matrix, columns=[f'period_{index}' for index in range(1, matrix.shape[1] + 1)])
    frame.insert(0, 'item_id', item_ids)
    return frame

def _save_item_trajectory(item_id: str) -> None:
    frame = pd.read_csv(SPO_DIR / 'test_trajectory.csv')
    matched = frame.loc[frame['item_id'].astype(str) == str(item_id)].copy()
    if matched.empty:
        raise ValueError(f'找不到 item_id={item_id}。')
    destination = OUTPUT_DIR / 'item_diagnostics' / 'spo_plus_linear'
    destination.mkdir(parents=True, exist_ok=True)
    matched.to_csv(destination / f'item_{item_id}_test_trajectory.csv', index=False)

def _update_comparison(row: dict) -> pd.DataFrame:
    destination = OUTPUT_DIR / 'all_model_comparison.csv'
    existing = pd.read_csv(destination) if destination.exists() else pd.DataFrame()
    combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True, sort=False)
    combined = combined.drop_duplicates('model', keep='last')
    combined['test_constraint_met'] = combined['test_fill_rate'] >= 0.95
    combined = combined.sort_values(['test_constraint_met', 'test_average_total_inventory'], ascending=[False, True])
    combined.to_csv(destination, index=False)
    focused_names = {'baseline', 'moving_average_12', 'lightgbm', 'xgboost', 'moment_dro_ma12', 'wasserstein_dro_ma12', 'chi_square_dro_ma12', 'spo_plus_linear'}
    combined.loc[combined['model'].isin(focused_names)].to_csv(OUTPUT_DIR / 'spo' / 'spo_vs_existing_comparison.csv', index=False)
    return combined

def _plot_comparison(comparison: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'baseline': '#6B7280', 'time_series': '#4C78A8', 'machine_learning': '#59A14F', 'dro': '#E15759', 'decision_focused_learning': '#7C3AED'}
    figure, axis = plt.subplots(figsize=(12, 7.5))
    for family, group in comparison.groupby('family', dropna=False):
        axis.scatter(group['test_average_total_inventory'], group['test_fill_rate'] * 100, s=92, color=colors.get(str(family), '#9C755F'), edgecolor='white', linewidth=0.8, label=str(family), zorder=3)
        for record in group.itertuples(index=False):
            axis.annotate(record.model, (record.test_average_total_inventory, record.test_fill_rate * 100), xytext=(5, 5), textcoords='offset points', fontsize=7.2)
    axis.axhline(95, color='#B45309', linestyle='--', linewidth=1.4, label='95% constraint')
    axis.set_xscale('log')
    axis.set_xlabel('Average total inventory (log scale; lower is better)')
    axis.set_ylabel('Test total Fill Rate (%)')
    axis.set_title('All replenishment models: inventory versus service', loc='left', fontweight='bold')
    axis.grid(color='#D9DEE3', linewidth=0.7, alpha=0.8)
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / 'all_model_comparison.png', dpi=180, bbox_inches='tight')
    plt.close(figure)

MODEL_DIR=OUT
SPO_DIR=OUT
EXPERIMENT_DIR=OUT

if __name__=='__main__':
    from common.cli import start
    start('spo_linear', globals(), default_protocol='legacy')
