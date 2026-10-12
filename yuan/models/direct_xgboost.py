"""models.direct_xgboost: actual model/processing implementation, not a legacy runner alias.

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
OUT=ROOT/'results/generated/spo/direct_xgb_spo'
WORK=OUT
TABLES=OUT/'图表'
OLD=ROOT/'results/generated/ready_immediate'
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

def hybrid_grad_hess(true_costs: np.ndarray, predicted_costs: np.ndarray, *, alpha: float, temperature: float, hessian_floor: float=1e-06) -> tuple[np.ndarray, np.ndarray]:
    """Return gradient and diagonal Hessian of MSE + smooth SPO+.

    The feasible decisions are one-hot action vectors.  ``alpha=0`` is pure
    cost-vector squared error and ``alpha=1`` is pure smooth SPO+.
    """
    actual = np.asarray(true_costs, dtype=float)
    prediction = np.asarray(predicted_costs, dtype=float)
    if actual.shape != prediction.shape or actual.ndim != 2:
        raise ValueError('true_costs 和 predicted_costs 必须是同形二维数组。')
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('alpha 必须在 [0, 1] 内。')
    if temperature <= 0 or hessian_floor <= 0:
        raise ValueError('temperature 和 hessian_floor 必须为正。')
    n_actions = actual.shape[1]
    logits = (actual - 2.0 * prediction) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    oracle = np.argmin(actual, axis=1)
    oracle_one_hot = np.zeros_like(actual)
    oracle_one_hot[np.arange(len(actual)), oracle] = 1.0
    gradient = (1.0 - alpha) * (prediction - actual) / n_actions + 2.0 * alpha * (oracle_one_hot - probabilities)
    hessian = (1.0 - alpha) / n_actions + 4.0 * alpha * probabilities * (1.0 - probabilities) / temperature + hessian_floor
    return (gradient.astype(np.float32), hessian.astype(np.float32))

@dataclass
class XGBSPOHybrid:
    """Nonlinear action-cost predictor for decision-focused replenishment."""
    alpha: float
    temperature: float = 0.5
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    n_jobs: int = 8
    random_seed: int = 42
    model_: object | None = field(init=False, default=None, repr=False)
    cost_scale_: float | None = field(init=False, default=None)

    def fit(self, features: np.ndarray, costs: np.ndarray) -> 'XGBSPOHybrid':
        from xgboost import XGBRegressor
        X = np.asarray(features, dtype=np.float32)
        C = np.asarray(costs, dtype=np.float32)
        if X.ndim != 2 or C.ndim != 2 or len(X) != len(C) or (C.shape[1] < 2):
            raise ValueError('features/costs 必须是行数相同的二维数组，且至少有两个动作。')
        if not np.isfinite(X).all() or not np.isfinite(C).all():
            raise ValueError('训练数据包含非有限值。')
        positive = C[C > 0]
        self.cost_scale_ = float(np.median(positive)) if positive.size else 1.0
        if not np.isfinite(self.cost_scale_) or self.cost_scale_ <= 0:
            self.cost_scale_ = 1.0
        normalized = C / self.cost_scale_

        def objective(y_true: np.ndarray, y_pred: np.ndarray):
            return hybrid_grad_hess(y_true, y_pred, alpha=self.alpha, temperature=self.temperature)
        self.model_ = XGBRegressor(objective=objective, multi_strategy='one_output_per_tree', n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate, subsample=self.subsample, colsample_bytree=self.colsample_bytree, reg_lambda=self.reg_lambda, tree_method='hist', max_bin=256, n_jobs=self.n_jobs, random_state=self.random_seed)
        self.model_.fit(X, normalized, verbose=False)
        return self

    def predict_costs(self, features: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.cost_scale_ is None:
            raise ValueError('模型尚未拟合。')
        prediction = np.asarray(self.model_.predict(np.asarray(features, dtype=np.float32)), dtype=float)
        return prediction * self.cost_scale_

    def predict_actions(self, features: np.ndarray) -> np.ndarray:
        return np.argmin(self.predict_costs(features), axis=1)

    def save_model(self, path: str | Path) -> None:
        if self.model_ is None:
            raise ValueError('模型尚未拟合。')
        self.model_.save_model(Path(path))
import argparse
import json
import time
from dataclasses import asdict, dataclass
import pandas as pd
from common.evaluation import InventoryEvaluator
from common.features import build_supervised_data
from common.features import load_panel
from common.policy_state import POLICY_STATE_FEATURE_NAMES
from common.policy_state import augment_policy_features
from common.policy_state import rolling_reference
from common.policy_state import rollout_action_costs

@dataclass(frozen=True)
class DirectPolicyConfig:
    service_level: float = 0.95
    validation_service_buffer: float = 0.02
    minimum_history: int = 12
    validation_start: int = 42
    reference_window: int = 12
    action_min: float = 0.0
    action_max: float = 5.0
    action_step: float = 0.5
    holding_cost: float = 1.0
    shortage_penalties: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
    rollout_horizons: tuple[int, ...] = (1, 3, 6)
    alpha: float = 0.75
    temperature: float = 0.5
    tune_estimators: int = 40
    final_estimators: int = 40
    max_depth: int = 6
    learning_rate: float = 0.05
    behavior_action: float = 2.0
    policy_refinement_steps: int = 0
    random_seed: int = 42
    n_jobs: int = 8

def train_legacy() -> None:
    parser = argparse.ArgumentParser(description='运行不预测需求、直接输出补货动作的XGBoost-SPO模型。')
    parser.add_argument('--n-jobs', type=int, default=8)
    parser.add_argument('--item-id', default='1')
    parser.add_argument('--quick', action='store_true', help='快速链路检查，不作为正式结果。')
    parser.add_argument('--reuse-validation', action='store_true', help='复用已有validation_candidates.csv，只重训和评价最终模型。')
    args = parser.parse_args()
    if args.n_jobs < 1:
        raise ValueError('--n-jobs 必须 >= 1。')
    config = DirectPolicyConfig(n_jobs=args.n_jobs)
    if args.quick:
        config = DirectPolicyConfig(shortage_penalties=(40.0,), rollout_horizons=(1, 3), tune_estimators=12, final_estimators=16, policy_refinement_steps=0, n_jobs=args.n_jobs)
    start_time = time.perf_counter()
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    print('读取train/test面板……', flush=True)
    train = load_panel(DATA_DIR / 'replenishment_train.xlsx')
    test = load_panel(DATA_DIR / 'replenishment_test.xlsx')
    if not np.array_equal(train.item_ids, test.item_ids):
        raise ValueError('train/test 的item_id或顺序不一致。')
    actions = np.arange(config.action_min, config.action_max + config.action_step / 2, config.action_step, dtype=np.float32)
    train_reference = rolling_reference(train.demand, config.reference_window)
    all_train_periods = np.arange(config.minimum_history - 1, train.demand.shape[1])
    all_train_features = _decision_features(train.demand, train.x, train.categories, all_train_periods)
    behavior_state = _simulate_states(train.demand, train_reference, train.initial_inventory, all_train_periods, all_train_features, actions, behavior_action=config.behavior_action)
    print('验证集调参：缺货惩罚是95% Fill Rate约束的拉格朗日乘子。', flush=True)
    candidates: list[dict] = []
    best_by_variant: dict[str, tuple[tuple, dict]] = {}
    reuse_validation = args.reuse_validation and (EXPERIMENT_DIR / 'validation_candidates.csv').exists()
    tuning_penalties = () if reuse_validation else config.shortage_penalties
    total_candidates = len(tuning_penalties) * len(config.rollout_horizons)
    candidate_number = 0
    for horizon in config.rollout_horizons:
        variant = 'direct_xgb_spo_one_step' if horizon == 1 else 'direct_xgb_spo_rollout'
        fit_periods = np.arange(config.minimum_history - 1, config.validation_start - horizon)
        fit_base = _select_period_blocks(all_train_features, all_train_periods, fit_periods, len(train.item_ids))
        for penalty in tuning_penalties:
            candidate_number += 1
            candidate_start = time.perf_counter()
            print(f'候选 {candidate_number}/{total_candidates}: horizon={horizon}, 缺货惩罚={penalty:g}', flush=True)
            policy_features, costs = _training_arrays(train.demand, train_reference, fit_periods, fit_base, behavior_state, actions, horizon, penalty, config)
            model = _new_model(config.tune_estimators, config).fit(policy_features, costs)
            validation_targets, validation_initial, _ = _policy_targets(model, train.demand, train_reference, train.initial_inventory, all_train_periods, all_train_features, actions, config, collect_start=config.validation_start)
            validation_result = InventoryEvaluator(lead_time=1).evaluate_dynamic_base_stock(_wide(train.item_ids, train.demand[:, config.validation_start:]), pd.DataFrame({'item_id': train.item_ids, 'initial_inventory': validation_initial}), _wide(train.item_ids, validation_targets))
            row = {'candidate': candidate_number, 'variant': variant, 'rollout_horizon': horizon, 'shortage_penalty': penalty, 'alpha': config.alpha, 'temperature': config.temperature, 'n_estimators': config.tune_estimators, 'training_rows': len(policy_features), 'validation_fill_rate': validation_result.summary['total_fill_rate'], 'validation_average_total_inventory': validation_result.summary['average_total_inventory'], 'validation_total_turnover': validation_result.summary['total_turnover'], 'validation_constraint_met': validation_result.summary['total_fill_rate'] >= config.service_level, 'runtime_seconds': time.perf_counter() - candidate_start}
            candidates.append(row)
            key = _selection_key(validation_result.summary, config.service_level + config.validation_service_buffer)
            if variant not in best_by_variant or key < best_by_variant[variant][0]:
                best_by_variant[variant] = (key, row)
            print(f"  验证Fill Rate={row['validation_fill_rate']:.4%}, 平均总库存={row['validation_average_total_inventory']:.0f}", flush=True)
    if reuse_validation:
        candidate_frame = pd.read_csv(EXPERIMENT_DIR / 'validation_candidates.csv')
        candidates = candidate_frame.to_dict('records')
        for row in candidates:
            summary = {'total_fill_rate': row['validation_fill_rate'], 'average_total_inventory': row['validation_average_total_inventory']}
            key = _selection_key(summary, config.service_level + config.validation_service_buffer)
            variant = str(row['variant'])
            if variant not in best_by_variant or key < best_by_variant[variant][0]:
                best_by_variant[variant] = (key, row)
        print('已复用验证候选表，只重训最终模型。', flush=True)
    else:
        candidate_frame = pd.DataFrame(candidates)
        candidate_frame.to_csv(EXPERIMENT_DIR / 'validation_candidates.csv', index=False)
    result_rows: list[dict] = []
    for variant in ('direct_xgb_spo_one_step', 'direct_xgb_spo_rollout'):
        if variant not in best_by_variant:
            continue
        selected = best_by_variant[variant][1]
        print(f'\n训练最终模型 {variant}: {selected}', flush=True)
        horizon = int(selected['rollout_horizon'])
        penalty = float(selected['shortage_penalty'])
        final_periods = np.arange(config.minimum_history - 1, train.demand.shape[1] - horizon)
        final_base = _select_period_blocks(all_train_features, all_train_periods, final_periods, len(train.item_ids))
        current_state = behavior_state
        model: XGBSPOHybrid | None = None
        for iteration in range(config.policy_refinement_steps + 1):
            policy_features, costs = _training_arrays(train.demand, train_reference, final_periods, final_base, current_state, actions, horizon, penalty, config)
            model = _new_model(config.final_estimators, config).fit(policy_features, costs)
            if iteration < config.policy_refinement_steps:
                print(f'  策略状态再采样 {iteration + 1}/{config.policy_refinement_steps}', flush=True)
                current_state = _simulate_states(train.demand, train_reference, train.initial_inventory, all_train_periods, all_train_features, actions, model=model, behavior_action=config.behavior_action)
        assert model is not None
        model_dir = OUTPUT_DIR / 'spo' / variant
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(model_dir / 'cost_xgb_spo_model.json')
        selected_frame = candidate_frame.loc[candidate_frame['variant'] == variant]
        selected_frame.to_csv(model_dir / 'validation_candidates.csv', index=False)
        combined_demand = np.concatenate([train.demand, test.demand], axis=1)
        combined_x = np.concatenate([train.x, test.x], axis=1)
        test_global_periods = np.arange(train.demand.shape[1], combined_demand.shape[1])
        test_base = _decision_features(combined_demand, combined_x, train.categories, test_global_periods)
        combined_reference = rolling_reference(combined_demand, config.reference_window)
        test_targets, _, test_actions = _policy_targets(model, test.demand, combined_reference[:, train.demand.shape[1]:], test.initial_inventory, np.arange(test.demand.shape[1]), test_base, actions, config, collect_start=0)
        evaluator = InventoryEvaluator(lead_time=1)
        result = evaluator.evaluate_dynamic_base_stock(_wide(train.item_ids, test.demand), pd.DataFrame({'item_id': train.item_ids, 'initial_inventory': test.initial_inventory}), _wide(train.item_ids, test_targets), output_path=model_dir / 'test_per_sku_metrics.csv', trajectory_output_path=model_dir / 'test_trajectory.csv', model_name=variant)
        _wide(train.item_ids, test_targets).to_csv(model_dir / 'test_order_up_to.csv', index=False)
        _wide(train.item_ids, test_actions).to_csv(model_dir / 'test_selected_actions.csv', index=False)
        (model_dir / 'model_metadata.json').write_text(json.dumps({'cost_scale': model.cost_scale_, 'actions': actions.tolist(), 'feature_names': [*build_supervised_data(train.demand, train.x, train.categories, np.asarray([config.minimum_history])).feature_names, *POLICY_STATE_FEATURE_NAMES], 'selected_candidate': selected, 'demand_forecast_used': False}, ensure_ascii=False, indent=2), encoding='utf-8')
        row = {'model': variant, 'family': 'direct_decision_focused_learning', 'parameters': json.dumps({'demand_forecast': None, 'action': 'order_up_to_multiplier', 'rollout_horizon': horizon, 'shortage_penalty': penalty, 'alpha': config.alpha, 'action_grid': [config.action_min, config.action_max, config.action_step], 'policy_state_refinement_steps': config.policy_refinement_steps}, ensure_ascii=False), 'selected_z': np.nan, 'training_rows': len(policy_features), 'validation_wape': np.nan, 'test_wape': np.nan, 'validation_spo_loss': np.nan, 'test_spo_loss': np.nan, 'validation_fill_rate': selected['validation_fill_rate'], 'validation_average_total_inventory': selected['validation_average_total_inventory'], 'test_fill_rate': result.summary['total_fill_rate'], 'test_constraint_met': result.summary['total_fill_rate'] >= config.service_level, 'test_average_total_inventory': result.summary['average_total_inventory'], 'test_total_turnover': result.summary['total_turnover'], 'test_total_lost_sales': result.summary['total_lost_sales'], 'test_sku_95_share': result.summary['sku_service_level_95_share'], 'runtime_seconds': time.perf_counter() - start_time}
        summary = {'result': row, 'test': result.summary, 'selected_validation_candidate': selected, 'config': asdict(config), 'decision_timing': 'After observing period t demand and ending inventory, select an order-up-to action; the order arrives before period t+1 demand.', 'validation_selection_rule': f'Choose the lowest-inventory candidate whose validation Fill Rate is at least {config.service_level + config.validation_service_buffer:.1%}; the final test constraint remains {config.service_level:.1%}.', 'evaluation_caveat': 'The 2 percentage-point validation buffer was introduced during development after diagnosing service degradation on the current test window. Treat this window as a development benchmark for these direct policies; use a new untouched future window for an unbiased final claim.', 'leakage_control': 'Policy features use demand/x through the decision period only. Validation and test demand are never used for earlier decisions.'}
        (model_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        _save_item_trajectory(model_dir, variant, args.item_id)
        result_rows.append(row)
    combined = _update_comparison(result_rows)
    _plot_comparison(combined)
    print('\n直接决策XGBoost-SPO实验完成', flush=True)
    print(pd.DataFrame(result_rows)[['model', 'test_fill_rate', 'test_average_total_inventory', 'test_total_turnover']].to_string(index=False), flush=True)
    print(f'总耗时: {time.perf_counter() - start_time:.2f}秒', flush=True)

def _decision_features(demand: np.ndarray, x: np.ndarray, categories: np.ndarray, decision_periods: np.ndarray) -> np.ndarray:
    """Features at decisions after t are built as features for demand at t+1."""
    supervised = build_supervised_data(demand, x, categories, np.asarray(decision_periods, dtype=int) + 1, include_target=False)
    return supervised.features

def _simulate_states(demand: np.ndarray, reference: np.ndarray, initial_inventory: np.ndarray, decision_periods: np.ndarray, decision_features: np.ndarray, actions: np.ndarray, *, model: object | None=None, behavior_action: float) -> dict[str, np.ndarray]:
    n_skus, n_periods = demand.shape
    feature_lookup = {int(period): decision_features[index * n_skus:(index + 1) * n_skus] for index, period in enumerate(decision_periods)}
    opening = np.zeros_like(demand, dtype=float)
    ending = np.zeros_like(demand, dtype=float)
    lost = np.zeros_like(demand, dtype=float)
    targets = np.zeros_like(demand, dtype=float)
    selected_actions = np.zeros_like(demand, dtype=float)
    on_hand = np.asarray(initial_inventory, dtype=float).copy()
    arrival = np.zeros(n_skus, dtype=float)
    for period in range(n_periods):
        available = on_hand + arrival
        sales = np.minimum(available, demand[:, period])
        current_ending = available - sales
        current_lost = demand[:, period] - sales
        if period in feature_lookup and model is not None:
            policy_x = augment_policy_features(feature_lookup[period], reference=reference[:, period], opening_inventory=available, ending_inventory=current_ending, current_demand=demand[:, period], current_lost_sales=current_lost)
            if getattr(model, 'uses_action_targets', False):
                candidate_targets = np.ceil(reference[:, period, None] * actions[None, :])
                selected = model.predict_actions(policy_x, candidate_targets)
            else:
                selected = model.predict_actions(policy_x)
            multiplier = actions[selected]
        else:
            multiplier = np.full(n_skus, behavior_action, dtype=float)
        target = np.ceil(reference[:, period] * multiplier)
        arrival = np.maximum(0.0, target - current_ending)
        on_hand = current_ending
        opening[:, period] = available
        ending[:, period] = current_ending
        lost[:, period] = current_lost
        targets[:, period] = target
        selected_actions[:, period] = multiplier
    return {'opening': opening, 'ending': ending, 'lost': lost, 'targets': targets, 'actions': selected_actions}

def _policy_targets(model: object, demand: np.ndarray, reference: np.ndarray, initial_inventory: np.ndarray, decision_periods: np.ndarray, decision_features: np.ndarray, actions: np.ndarray, config: DirectPolicyConfig, *, collect_start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = _simulate_states(demand, reference, initial_inventory, decision_periods, decision_features, actions, model=model, behavior_action=config.behavior_action)
    if collect_start == 0:
        initial_available = np.asarray(initial_inventory, dtype=float)
    else:
        previous_ending = states['ending'][:, collect_start - 1]
        previous_target = states['targets'][:, collect_start - 1]
        initial_available = np.maximum(previous_ending, previous_target)
    return (states['targets'][:, collect_start:], initial_available, states['actions'][:, collect_start:])

def _training_arrays(demand: np.ndarray, reference: np.ndarray, decision_periods: np.ndarray, base_features: np.ndarray, states: dict[str, np.ndarray], actions: np.ndarray, horizon: int, penalty: float, config: DirectPolicyConfig) -> tuple[np.ndarray, np.ndarray]:
    opening = _period_vector(states['opening'], decision_periods)
    ending = _period_vector(states['ending'], decision_periods)
    current_demand = _period_vector(demand, decision_periods)
    current_lost = _period_vector(states['lost'], decision_periods)
    current_reference = _period_vector(reference, decision_periods)
    features = augment_policy_features(base_features, reference=current_reference, opening_inventory=opening, ending_inventory=ending, current_demand=current_demand, current_lost_sales=current_lost)
    future_demand = np.vstack([demand[:, period + 1:period + 1 + horizon] for period in decision_periods])
    reference_path = np.vstack([reference[:, period:period + horizon] for period in decision_periods])
    costs = rollout_action_costs(ending, future_demand, reference_path, actions, shortage_penalty=penalty, holding_cost=config.holding_cost)
    return (features, costs)

def _new_model(n_estimators: int, config: DirectPolicyConfig) -> XGBSPOHybrid:
    return XGBSPOHybrid(alpha=config.alpha, temperature=config.temperature, n_estimators=n_estimators, max_depth=config.max_depth, learning_rate=config.learning_rate, n_jobs=config.n_jobs, random_seed=config.random_seed)

def _select_period_blocks(features: np.ndarray, all_periods: np.ndarray, selected_periods: np.ndarray, n_skus: int) -> np.ndarray:
    indices = {int(period): index for index, period in enumerate(all_periods)}
    return np.vstack([features[indices[int(period)] * n_skus:(indices[int(period)] + 1) * n_skus] for period in selected_periods])

def _period_vector(matrix: np.ndarray, periods: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[:, periods], dtype=float).T.reshape(-1)

def _wide(item_ids: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    matrix = np.asarray(values, dtype=float)
    frame = pd.DataFrame(matrix, columns=[f'period_{index}' for index in range(1, matrix.shape[1] + 1)])
    frame.insert(0, 'item_id', item_ids)
    return frame

def _selection_key(summary: dict, service_level: float) -> tuple:
    feasible = summary['total_fill_rate'] >= service_level
    return (not feasible, summary['average_total_inventory'] if feasible else -summary['total_fill_rate'])

def _save_item_trajectory(model_dir: Path, model_name: str, item_id: str) -> None:
    frame = pd.read_csv(model_dir / 'test_trajectory.csv')
    matched = frame.loc[frame['item_id'].astype(str) == str(item_id)].copy()
    if matched.empty:
        raise ValueError(f'找不到item_id={item_id}。')
    destination = OUTPUT_DIR / 'item_diagnostics' / model_name
    destination.mkdir(parents=True, exist_ok=True)
    matched.to_csv(destination / f'item_{item_id}_test_trajectory.csv', index=False)

def _update_comparison(rows: list[dict]) -> pd.DataFrame:
    destination = OUTPUT_DIR / 'all_model_comparison.csv'
    existing = pd.read_csv(destination) if destination.exists() else pd.DataFrame()
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True, sort=False)
    combined = combined.drop_duplicates('model', keep='last')
    combined['test_constraint_met'] = combined['test_fill_rate'] >= 0.95
    combined = combined.sort_values(['test_constraint_met', 'test_average_total_inventory'], ascending=[False, True])
    combined.to_csv(destination, index=False)
    focused = combined.loc[combined['model'].isin({'baseline', 'lightgbm', 'xgb_spo_hybrid', 'lgbm_spo_hybrid', 'direct_xgb_spo_one_step', 'direct_xgb_spo_rollout'})]
    focused.to_csv(EXPERIMENT_DIR / 'comparison.csv', index=False)
    return combined

def _plot_comparison(comparison: pd.DataFrame) -> None:
    from models.xgboost_spo import _plot_comparison as plot
    plot(comparison)

MODEL_DIR=OUT
SPO_DIR=OUT
EXPERIMENT_DIR=OUT

if __name__=='__main__':
    from common.cli import start
    start('direct_xgboost', globals(), default_protocol='legacy')
