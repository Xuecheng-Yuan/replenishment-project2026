"""models.direct_lightgbm: actual model/processing implementation, not a legacy runner alias.

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
OUT=ROOT/'results/generated/spo/direct_lgbm_spo_rank'
WORK=OUT
TABLES=OUT/'图表'
OLD=ROOT/'results/generated/ready_immediate'
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import numpy as np

def direct_spo_rank_grad_hess(true_costs: np.ndarray, predicted_costs: np.ndarray, *, spo_weight: float, rank_weight: float, spo_temperature: float=0.5, rank_temperature: float=1.0, hessian_floor: float=1e-06) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and diagonal Hessian of MSE + smooth SPO+ + pairwise rank.

    The pairwise term compares every action with the realized oracle action.
    Its weight is proportional to the normalized realized regret of selecting
    that action, so confusing two nearly equivalent actions is penalized less.
    """
    actual = np.asarray(true_costs, dtype=float)
    prediction = np.asarray(predicted_costs, dtype=float)
    if actual.shape != prediction.shape or actual.ndim != 2:
        raise ValueError('true_costs 和 predicted_costs 必须是同形二维数组。')
    if spo_weight < 0 or rank_weight < 0 or spo_weight + rank_weight > 1:
        raise ValueError('损失权重必须非负且 spo_weight + rank_weight <= 1。')
    if spo_temperature <= 0 or rank_temperature <= 0 or hessian_floor <= 0:
        raise ValueError('温度和Hessian下界必须为正。')
    n_samples, n_actions = actual.shape
    mse_weight = 1.0 - spo_weight - rank_weight
    gradient = mse_weight * (prediction - actual) / n_actions
    hessian = np.full_like(actual, mse_weight / n_actions)
    oracle = np.argmin(actual, axis=1)
    rows = np.arange(n_samples)
    if spo_weight > 0:
        logits = (actual - 2.0 * prediction) / spo_temperature
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        oracle_one_hot = np.zeros_like(actual)
        oracle_one_hot[rows, oracle] = 1.0
        gradient += 2.0 * spo_weight * (oracle_one_hot - probabilities)
        hessian += 4.0 * spo_weight * probabilities * (1.0 - probabilities) / spo_temperature
    if rank_weight > 0:
        oracle_cost = actual[rows, oracle][:, None]
        regret = np.maximum(actual - oracle_cost, 0.0)
        regret_sum = regret.sum(axis=1, keepdims=True)
        pair_weight = np.divide(regret, regret_sum, out=np.zeros_like(regret), where=regret_sum > 0)
        oracle_prediction = prediction[rows, oracle][:, None]
        pair_logits = np.clip((oracle_prediction - prediction) / rank_temperature, -40.0, 40.0)
        pair_probability = 1.0 / (1.0 + np.exp(-pair_logits))
        pair_gradient = pair_weight * pair_probability / rank_temperature
        pair_curvature = pair_weight * pair_probability * (1.0 - pair_probability) / rank_temperature ** 2
        gradient -= rank_weight * pair_gradient
        hessian += rank_weight * pair_curvature
        np.add.at(gradient, (rows, oracle), rank_weight * pair_gradient.sum(axis=1))
        np.add.at(hessian, (rows, oracle), rank_weight * pair_curvature.sum(axis=1))
    hessian += hessian_floor
    return (gradient.astype(np.float32), hessian.astype(np.float32))

@dataclass
class LGBMDirectSPOPolicy:
    """Score finite replenishment actions directly with LightGBM."""
    spo_weight: float
    rank_weight: float
    spo_temperature: float = 0.5
    rank_temperature: float = 1.0
    n_estimators: int = 40
    num_leaves: int = 63
    learning_rate: float = 0.05
    min_child_samples: int = 100
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    n_jobs: int = 8
    random_seed: int = 42
    model_: object | None = field(init=False, default=None, repr=False)
    cost_scale_: float | None = field(init=False, default=None)
    actions_: np.ndarray | None = field(init=False, default=None, repr=False)
    uses_action_targets: bool = field(init=False, default=True, repr=False)

    def fit(self, features: np.ndarray, targets: np.ndarray, costs: np.ndarray, actions: np.ndarray) -> 'LGBMDirectSPOPolicy':
        from lightgbm import LGBMRegressor
        X = np.asarray(features, dtype=np.float32)
        B = np.asarray(targets, dtype=np.float32)
        C = np.asarray(costs, dtype=np.float32)
        A = np.asarray(actions, dtype=np.float32)
        if X.ndim != 2 or B.ndim != 2 or C.shape != B.shape or (len(X) != len(B)):
            raise ValueError('features、targets和costs的维度不一致。')
        if B.shape[1] != len(A) or len(A) < 2:
            raise ValueError('动作数量与targets列数不一致。')
        if not np.isfinite(X).all() or not np.isfinite(B).all() or (not np.isfinite(C).all()):
            raise ValueError('训练数据包含非有限值。')
        positive = C[C > 0]
        self.cost_scale_ = float(np.median(positive)) if positive.size else 1.0
        if not np.isfinite(self.cost_scale_) or self.cost_scale_ <= 0:
            self.cost_scale_ = 1.0
        self.actions_ = A.copy()
        expanded = self._expand_features(X, B, A)
        labels = (C / self.cost_scale_).reshape(-1)
        n_actions = len(A)

        def objective(y_true: np.ndarray, y_pred: np.ndarray):
            gradient, hessian = direct_spo_rank_grad_hess(np.asarray(y_true).reshape(-1, n_actions), np.asarray(y_pred).reshape(-1, n_actions), spo_weight=self.spo_weight, rank_weight=self.rank_weight, spo_temperature=self.spo_temperature, rank_temperature=self.rank_temperature)
            return (gradient.reshape(-1), hessian.reshape(-1))
        self.model_ = LGBMRegressor(objective=objective, n_estimators=self.n_estimators, num_leaves=self.num_leaves, learning_rate=self.learning_rate, min_child_samples=self.min_child_samples, subsample=self.subsample, colsample_bytree=self.colsample_bytree, reg_lambda=self.reg_lambda, n_jobs=self.n_jobs, random_state=self.random_seed, verbosity=-1)
        self.model_.fit(expanded, labels)
        return self

    def predict_costs(self, features: np.ndarray, targets: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.cost_scale_ is None or self.actions_ is None:
            raise ValueError('模型尚未拟合。')
        X = np.asarray(features, dtype=np.float32)
        B = np.asarray(targets, dtype=np.float32)
        if B.shape != (len(X), len(self.actions_)):
            raise ValueError('预测targets维度与模型动作数不一致。')
        expanded = self._expand_features(X, B, self.actions_)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='X does not have valid feature names')
            prediction = np.asarray(self.model_.predict(expanded), dtype=float)
        return prediction.reshape(len(X), len(self.actions_)) * self.cost_scale_

    def predict_actions(self, features: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return np.argmin(self.predict_costs(features, targets), axis=1)

    def save_model(self, path: str | Path) -> None:
        if self.model_ is None:
            raise ValueError('模型尚未拟合。')
        self.model_.booster_.save_model(str(Path(path)))

    @staticmethod
    def _expand_features(features: np.ndarray, targets: np.ndarray, actions: np.ndarray) -> np.ndarray:
        n_samples = len(features)
        n_actions = len(actions)
        return np.column_stack([np.repeat(features, n_actions, axis=0), np.tile(actions, n_samples), targets.reshape(-1)]).astype(np.float32)
import argparse
import json
import time
from dataclasses import asdict, dataclass
import pandas as pd
from models.direct_xgboost import _decision_features, _period_vector, _plot_comparison, _policy_targets, _save_item_trajectory, _select_period_blocks, _selection_key, _simulate_states, _training_arrays, _wide
from common.evaluation import InventoryEvaluator
from common.features import build_supervised_data
from common.features import load_panel
from common.policy_state import POLICY_STATE_FEATURE_NAMES
from common.policy_state import action_levels
from common.policy_state import rolling_reference
MODEL_NAME = 'direct_lgbm_spo_rank'
LOSS_CANDIDATES = ({'loss_name': 'cost_mse', 'spo_weight': 0.0, 'rank_weight': 0.0}, {'loss_name': 'smooth_spo_plus', 'spo_weight': 0.75, 'rank_weight': 0.0}, {'loss_name': 'regret_weighted_rank', 'spo_weight': 0.0, 'rank_weight': 0.75}, {'loss_name': 'spo_rank_hybrid', 'spo_weight': 0.4, 'rank_weight': 0.35})

@dataclass(frozen=True)
class DirectLGBMConfig:
    service_level: float = 0.95
    validation_service_buffer: float = 0.02
    minimum_history: int = 12
    validation_start: int = 42
    reference_window: int = 12
    action_min: float = 0.0
    action_max: float = 5.0
    action_step: float = 0.5
    holding_cost: float = 1.0
    shortage_penalties: tuple[float, ...] = (1.5, 1.75, 2.0, 2.5, 3.0, 4.0)
    rollout_horizons: tuple[int, ...] = (1, 3, 6)
    ablation_horizon: int = 3
    ablation_penalty: float = 4.0
    spo_temperature: float = 0.5
    rank_temperature: float = 1.0
    n_estimators: int = 30
    num_leaves: int = 63
    learning_rate: float = 0.05
    min_child_samples: int = 100
    behavior_action: float = 2.0
    random_seed: int = 42
    n_jobs: int = 8

def train_legacy() -> None:
    parser = argparse.ArgumentParser(description='运行不预测需求、直接选择补货动作的LightGBM-SPO排序模型。')
    parser.add_argument('--n-jobs', type=int, default=8)
    parser.add_argument('--item-id', default='1')
    parser.add_argument('--quick', action='store_true', help='快速链路检查，不作为正式结果。')
    args = parser.parse_args()
    if args.n_jobs < 1:
        raise ValueError('--n-jobs 必须 >= 1。')
    config = DirectLGBMConfig(n_jobs=args.n_jobs)
    loss_candidates = LOSS_CANDIDATES
    if args.quick:
        config = DirectLGBMConfig(shortage_penalties=(1.25,), rollout_horizons=(1, 6), n_estimators=5, n_jobs=args.n_jobs)
        loss_candidates = (LOSS_CANDIDATES[-1],)
    start_time = time.perf_counter()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print('读取train/test面板……', flush=True)
    train = load_panel(DATA_DIR / 'replenishment_train.xlsx')
    test = load_panel(DATA_DIR / 'replenishment_test.xlsx')
    if not np.array_equal(train.item_ids, test.item_ids):
        raise ValueError('train/test 的item_id或顺序不一致。')
    actions = np.arange(config.action_min, config.action_max + config.action_step / 2, config.action_step, dtype=np.float32)
    n_skus = len(train.item_ids)
    train_reference = rolling_reference(train.demand, config.reference_window)
    all_train_periods = np.arange(config.minimum_history - 1, train.demand.shape[1])
    all_train_features = _decision_features(train.demand, train.x, train.categories, all_train_periods)
    behavior_state = _simulate_states(train.demand, train_reference, train.initial_inventory, all_train_periods, all_train_features, actions, behavior_action=config.behavior_action)
    evaluator = InventoryEvaluator(lead_time=1)
    rows: list[dict] = []
    print('第一阶段：比较成本MSE、平滑SPO+、排序和混合损失。', flush=True)
    loss_best: tuple[tuple, dict] | None = None
    for candidate in loss_candidates:
        row = _fit_and_validate(train, train_reference, all_train_periods, all_train_features, behavior_state, actions, config.ablation_horizon, config.ablation_penalty, candidate, config, evaluator, stage='loss_ablation')
        rows.append(row)
        key = _selection_key({'total_fill_rate': row['validation_fill_rate'], 'average_total_inventory': row['validation_average_total_inventory']}, config.service_level + config.validation_service_buffer)
        if loss_best is None or key < loss_best[0]:
            loss_best = (key, row)
    assert loss_best is not None
    selected_loss = {'loss_name': loss_best[1]['loss_name'], 'spo_weight': loss_best[1]['spo_weight'], 'rank_weight': loss_best[1]['rank_weight']}
    print(f'损失函数消融胜出：{selected_loss}', flush=True)
    print('第二阶段：搜索滚动期数和缺货惩罚。', flush=True)
    operational_best: tuple[tuple, dict] | None = None
    total = len(config.rollout_horizons) * len(config.shortage_penalties)
    index = 0
    for horizon in config.rollout_horizons:
        for penalty in config.shortage_penalties:
            index += 1
            print(f'运营候选 {index}/{total}: H={horizon}, lambda={penalty:g}', flush=True)
            row = _fit_and_validate(train, train_reference, all_train_periods, all_train_features, behavior_state, actions, horizon, penalty, selected_loss, config, evaluator, stage='operational_tuning')
            rows.append(row)
            key = _selection_key({'total_fill_rate': row['validation_fill_rate'], 'average_total_inventory': row['validation_average_total_inventory']}, config.service_level + config.validation_service_buffer)
            if operational_best is None or key < operational_best[0]:
                operational_best = (key, row)
    assert operational_best is not None
    candidates = pd.DataFrame(rows)
    candidates.insert(0, 'candidate', np.arange(1, len(candidates) + 1))
    candidates.to_csv(MODEL_DIR / 'validation_candidates.csv', index=False)
    selected = operational_best[1]
    print(f'最终验证候选：{selected}', flush=True)
    horizon = int(selected['rollout_horizon'])
    penalty = float(selected['shortage_penalty'])
    final_periods = np.arange(config.minimum_history - 1, train.demand.shape[1] - horizon)
    final_base = _select_period_blocks(all_train_features, all_train_periods, final_periods, n_skus)
    final_features, final_costs = _training_arrays(train.demand, train_reference, final_periods, final_base, behavior_state, actions, horizon, penalty, config)
    final_targets = action_levels(_period_vector(train_reference, final_periods), actions)
    final_model = _new_model(selected, config).fit(final_features, final_targets, final_costs, actions)
    final_model.save_model(MODEL_DIR / 'direct_lgbm_spo_model.txt')
    _save_feature_importance(final_model, final_features.shape[1], train, config)
    combined_demand = np.concatenate([train.demand, test.demand], axis=1)
    combined_x = np.concatenate([train.x, test.x], axis=1)
    test_global_periods = np.arange(train.demand.shape[1], combined_demand.shape[1])
    test_base = _decision_features(combined_demand, combined_x, train.categories, test_global_periods)
    test_reference = rolling_reference(combined_demand, config.reference_window)[:, train.demand.shape[1]:]
    test_targets, _, test_actions = _policy_targets(final_model, test.demand, test_reference, test.initial_inventory, np.arange(test.demand.shape[1]), test_base, actions, config, collect_start=0)
    result = evaluator.evaluate_dynamic_base_stock(_wide(train.item_ids, test.demand), pd.DataFrame({'item_id': train.item_ids, 'initial_inventory': test.initial_inventory}), _wide(train.item_ids, test_targets), output_path=MODEL_DIR / 'test_per_sku_metrics.csv', trajectory_output_path=MODEL_DIR / 'test_trajectory.csv', model_name=MODEL_NAME)
    _wide(train.item_ids, test_targets).to_csv(MODEL_DIR / 'test_order_up_to.csv', index=False)
    _wide(train.item_ids, test_actions).to_csv(MODEL_DIR / 'test_selected_actions.csv', index=False)
    feature_names = build_supervised_data(train.demand, train.x, train.categories, np.asarray([config.minimum_history])).feature_names
    metadata = {'actions': actions.tolist(), 'feature_names': [*feature_names, *POLICY_STATE_FEATURE_NAMES, 'action', 'action_target'], 'cost_scale': final_model.cost_scale_, 'selected_candidate': selected, 'demand_forecast_used': False}
    (MODEL_DIR / 'model_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    row = {'model': MODEL_NAME, 'family': 'direct_decision_focused_learning', 'parameters': json.dumps({'demand_forecast': None, 'loss': selected['loss_name'], 'spo_weight': selected['spo_weight'], 'rank_weight': selected['rank_weight'], 'rollout_horizon': horizon, 'shortage_penalty': penalty, 'action_grid': [config.action_min, config.action_max, config.action_step]}, ensure_ascii=False), 'selected_z': np.nan, 'training_rows': len(final_features), 'validation_wape': np.nan, 'test_wape': np.nan, 'validation_spo_loss': np.nan, 'test_spo_loss': np.nan, 'validation_fill_rate': selected['validation_fill_rate'], 'validation_average_total_inventory': selected['validation_average_total_inventory'], 'test_fill_rate': result.summary['total_fill_rate'], 'test_constraint_met': result.summary['total_fill_rate'] >= config.service_level, 'test_average_total_inventory': result.summary['average_total_inventory'], 'test_total_turnover': result.summary['total_turnover'], 'test_total_lost_sales': result.summary['total_lost_sales'], 'test_sku_95_share': result.summary['sku_service_level_95_share'], 'runtime_seconds': time.perf_counter() - start_time}
    summary = {'result': row, 'test': result.summary, 'selected_validation_candidate': selected, 'selected_loss_ablation': loss_best[1], 'config': asdict(config), 'literature_basis': ['Elmachtoub and Grigas (2022), Smart Predict, then Optimize', 'Mandi et al. (2022), Decision-Focused Learning Through the Lens of Learning to Rank', 'Elmachtoub, Liang, and McNellis (2020), SPO Trees'], 'evaluation_caveat': 'The current test workbook is a development benchmark because earlier model iterations have already been inspected. Use a new future window for a final blind claim.'}
    (MODEL_DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    _save_item_trajectory(MODEL_DIR, MODEL_NAME, args.item_id)
    comparison = _update_comparison(row)
    _plot_comparison(comparison)
    print('\nDirect LightGBM-SPO实验完成', flush=True)
    print(pd.DataFrame([row])[['model', 'test_fill_rate', 'test_average_total_inventory', 'test_total_turnover']].to_string(index=False), flush=True)
    print(f'总耗时: {time.perf_counter() - start_time:.2f}秒', flush=True)

def _fit_and_validate(train, reference: np.ndarray, all_periods: np.ndarray, all_features: np.ndarray, behavior_state: dict[str, np.ndarray], actions: np.ndarray, horizon: int, penalty: float, loss: dict, config: DirectLGBMConfig, evaluator: InventoryEvaluator, *, stage: str) -> dict:
    started = time.perf_counter()
    n_skus = len(train.item_ids)
    fit_periods = np.arange(config.minimum_history - 1, config.validation_start - horizon)
    fit_base = _select_period_blocks(all_features, all_periods, fit_periods, n_skus)
    features, costs = _training_arrays(train.demand, reference, fit_periods, fit_base, behavior_state, actions, horizon, penalty, config)
    targets = action_levels(_period_vector(reference, fit_periods), actions)
    model = _new_model(loss, config).fit(features, targets, costs, actions)
    validation_targets, validation_initial, _ = _policy_targets(model, train.demand, reference, train.initial_inventory, all_periods, all_features, actions, config, collect_start=config.validation_start)
    result = evaluator.evaluate_dynamic_base_stock(_wide(train.item_ids, train.demand[:, config.validation_start:]), pd.DataFrame({'item_id': train.item_ids, 'initial_inventory': validation_initial}), _wide(train.item_ids, validation_targets))
    row = {'stage': stage, 'loss_name': loss['loss_name'], 'spo_weight': float(loss['spo_weight']), 'rank_weight': float(loss['rank_weight']), 'rollout_horizon': horizon, 'shortage_penalty': penalty, 'training_rows': len(features), 'validation_fill_rate': result.summary['total_fill_rate'], 'validation_average_total_inventory': result.summary['average_total_inventory'], 'validation_total_turnover': result.summary['total_turnover'], 'validation_constraint_met': result.summary['total_fill_rate'] >= config.service_level, 'validation_selection_buffer_met': result.summary['total_fill_rate'] >= config.service_level + config.validation_service_buffer, 'runtime_seconds': time.perf_counter() - started}
    print(f"  {loss['loss_name']}, H={horizon}, lambda={penalty:g}: Fill={row['validation_fill_rate']:.4%}, 库存={row['validation_average_total_inventory']:.0f}", flush=True)
    return row

def _new_model(loss: dict, config: DirectLGBMConfig) -> LGBMDirectSPOPolicy:
    return LGBMDirectSPOPolicy(spo_weight=float(loss['spo_weight']), rank_weight=float(loss['rank_weight']), spo_temperature=config.spo_temperature, rank_temperature=config.rank_temperature, n_estimators=config.n_estimators, num_leaves=config.num_leaves, learning_rate=config.learning_rate, min_child_samples=config.min_child_samples, n_jobs=config.n_jobs, random_seed=config.random_seed)

def _save_feature_importance(model, n_policy_features: int, train, config) -> None:
    values = np.asarray(model.model_.feature_importances_, dtype=float)
    names = build_supervised_data(train.demand, train.x, train.categories, np.asarray([config.minimum_history])).feature_names
    expanded_names = [*names, *POLICY_STATE_FEATURE_NAMES, 'action', 'action_target']
    if len(values) != n_policy_features + 2 or len(values) != len(expanded_names):
        raise ValueError('LightGBM特征重要性维度不一致。')
    pd.DataFrame({'feature': expanded_names, 'importance': values}).sort_values('importance', ascending=False).to_csv(MODEL_DIR / 'feature_importance.csv', index=False)

def _update_comparison(row: dict) -> pd.DataFrame:
    destination = OUTPUT_DIR / 'all_model_comparison.csv'
    existing = pd.read_csv(destination) if destination.exists() else pd.DataFrame()
    combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True, sort=False)
    combined = combined.drop_duplicates('model', keep='last')
    combined['test_constraint_met'] = combined['test_fill_rate'] >= 0.95
    combined = combined.sort_values(['test_constraint_met', 'test_average_total_inventory'], ascending=[False, True])
    combined.to_csv(destination, index=False)
    focused = combined.loc[combined['model'].isin({'baseline', 'lightgbm', 'lgbm_spo_hybrid', 'direct_xgb_spo_rollout', MODEL_NAME})]
    focused.to_csv(OUTPUT_DIR / 'spo' / 'direct_lgbm_spo_comparison.csv', index=False)
    return combined

MODEL_DIR=OUT
SPO_DIR=OUT
EXPERIMENT_DIR=OUT

if __name__=='__main__':
    from common.cli import start
    start('direct_lightgbm', globals(), default_protocol='legacy')
