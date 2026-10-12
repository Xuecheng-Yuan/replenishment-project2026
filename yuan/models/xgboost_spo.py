"""models.xgboost_spo: actual model/processing implementation, not a legacy runner alias.

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
OUT=ROOT/'results/generated/spo/xgb_spo_hybrid'
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
from common.regressors import build_regressor
from common.policy_state import CURRENT_INVENTORY_FEATURE
from models.spo_linear import action_costs
from models.spo_linear import action_targets
from common.policy_state import behavior_ending_inventory
from common.policy_state import inventory_for_target_periods
from common.policy_state import rollout_inventory_aware_policy
MODEL_NAME = 'xgb_spo_hybrid'

@dataclass(frozen=True)
class HybridExperimentConfig:
    service_level: float = 0.95
    minimum_history: int = 12
    oof_start: int = 24
    validation_start: int = 42
    block_size: int = 6
    action_min: float = 0.0
    action_max: float = 4.0
    action_step: float = 0.5
    holding_cost: float = 1.0
    shortage_penalties: tuple[float, ...] = (20.0, 40.0, 80.0, 160.0)
    alphas: tuple[float, ...] = (0.0, 0.5, 0.75, 0.9, 1.0)
    temperature: float = 0.5
    tune_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.05
    random_seed: int = 42
    n_jobs: int = 8
BASE_XGB_PARAMETERS = {'n_estimators': 240, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.9}

def train_legacy() -> None:
    parser = argparse.ArgumentParser(description='运行XGBoost与SPO+混合损失补货模型。')
    parser.add_argument('--n-jobs', type=int, default=8)
    parser.add_argument('--item-id', default='1')
    parser.add_argument('--quick', action='store_true', help='快速链路检查，不作为正式结果。')
    args = parser.parse_args()
    if args.n_jobs < 1:
        raise ValueError('--n-jobs 必须 >= 1。')
    config = HybridExperimentConfig(n_jobs=args.n_jobs)
    if args.quick:
        config = HybridExperimentConfig(shortage_penalties=(40.0,), alphas=(0.5,), tune_estimators=15, n_jobs=args.n_jobs)
    start_time = time.perf_counter()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print('读取train/test面板……', flush=True)
    train = load_panel(DATA_DIR / 'replenishment_train.xlsx')
    test = load_panel(DATA_DIR / 'replenishment_test.xlsx')
    if not np.array_equal(train.item_ids, test.item_ids):
        raise ValueError('train/test 的item_id或顺序不一致。')
    n_skus = len(train.item_ids)
    print('构造严格滞后的监督学习特征……', flush=True)
    full_train = build_supervised_data(train.demand, train.x, train.categories, np.arange(config.minimum_history, train.demand.shape[1]))
    validation = build_supervised_data(train.demand, train.x, train.categories, np.arange(config.validation_start, train.demand.shape[1]))
    validation_extra = build_supervised_data(train.demand, train.x, train.categories, np.asarray([train.demand.shape[1]]), include_target=False)
    combined_demand = np.concatenate([train.demand, test.demand], axis=1)
    combined_x = np.concatenate([train.x, test.x], axis=1)
    test_features = build_supervised_data(combined_demand, combined_x, train.categories, np.arange(train.demand.shape[1], combined_demand.shape[1] + 1), include_target=False)
    behavior_inventory = behavior_ending_inventory(train.demand, train.initial_inventory)
    print('生成扩展窗口的XGBoost训练期外预测中心……', flush=True)
    oof_periods, oof_centers = _expanding_oof_predictions(full_train, n_skus, config)
    oof_data = _take_periods(full_train, oof_periods, n_skus)
    tune_mask = oof_periods < config.validation_start
    tune_periods = oof_periods[tune_mask]
    tune_rows = np.repeat(tune_mask, n_skus)
    tune_features = oof_data.features[tune_rows]
    tune_target = oof_data.target[tune_rows]
    tune_centers = oof_centers[tune_rows]
    tune_scale = _per_sku_scale(tune_target - tune_centers, len(tune_periods), n_skus)
    final_scale = _per_sku_scale(oof_data.target - oof_centers, len(oof_periods), n_skus)
    print('拟合验证期和测试期的基础XGBoost需求预测……', flush=True)
    base_validation = _fit_base_xgb(full_train.features[_period_mask(full_train.target_periods, full_train.target_periods < config.validation_start, n_skus)], full_train.target[_period_mask(full_train.target_periods, full_train.target_periods < config.validation_start, n_skus)], config)
    validation_all_features = np.vstack([validation.features, validation_extra.features])
    validation_centers = _nonnegative(base_validation.predict(validation_all_features))
    base_test = _fit_base_xgb(full_train.features, full_train.target, config)
    test_centers = _nonnegative(base_test.predict(test_features.features))
    base_test.save_model(MODEL_DIR / 'base_demand_xgboost.json')
    actions = np.arange(config.action_min, config.action_max + config.action_step / 2, config.action_step, dtype=np.float32)
    tune_augmented = _augment_cost_features(tune_features, tune_centers, tune_scale, len(tune_periods), n_skus, inventory_for_target_periods(behavior_inventory, tune_periods))
    tune_targets = action_targets(tune_centers, _repeat_scale(tune_scale, len(tune_periods)), actions)
    validation_demand_df = _wide(train.item_ids, train.demand[:, config.validation_start:])
    evaluator = InventoryEvaluator(lead_time=1)
    candidates: list[dict] = []
    best: tuple[tuple, dict] | None = None
    combinations = [(penalty, alpha) for penalty in config.shortage_penalties for alpha in config.alphas]
    for candidate_index, (penalty, alpha) in enumerate(combinations, start=1):
        candidate_start = time.perf_counter()
        variant = _variant_name(alpha)
        print(f'候选 {candidate_index}/{len(combinations)}: {variant}, 缺货惩罚={penalty:g}', flush=True)
        costs = action_costs(tune_target, tune_targets, penalty, holding_cost=config.holding_cost)
        model = _new_cost_model(alpha, config).fit(tune_augmented, costs)
        validation_rollout = _rollout_hybrid_policy(model, train.demand[:, config.validation_start:], behavior_inventory[:, config.validation_start - 1], validation_all_features, validation_centers, tune_scale, actions, include_pre_period_decision=True)
        validation_actions = validation_rollout.actions.T.reshape(-1)
        validation_result = evaluator.evaluate_dynamic_base_stock(validation_demand_df, pd.DataFrame({'item_id': train.item_ids, 'initial_inventory': validation_rollout.initial_inventory}), _wide(train.item_ids, validation_rollout.targets))
        validation_costs = action_costs(validation.target, action_targets(validation_centers[:len(validation.target)], _repeat_scale(tune_scale, len(validation.target_periods)), actions), penalty, holding_cost=config.holding_cost)
        chosen = validation_costs[np.arange(len(validation_costs)), validation_actions[:len(validation.target)]]
        regret = float(np.mean(chosen - validation_costs.min(axis=1)))
        row = {'candidate': candidate_index, 'variant': variant, 'alpha': alpha, 'temperature': config.temperature, 'shortage_penalty': penalty, 'n_estimators': config.tune_estimators, 'validation_spo_regret': regret, 'validation_fill_rate': validation_result.summary['total_fill_rate'], 'validation_average_total_inventory': validation_result.summary['average_total_inventory'], 'validation_total_turnover': validation_result.summary['total_turnover'], 'validation_constraint_met': validation_result.summary['total_fill_rate'] >= config.service_level, 'runtime_seconds': time.perf_counter() - candidate_start}
        candidates.append(row)
        key = _selection_key(validation_result.summary, regret, config.service_level)
        if best is None or key < best[0]:
            best = (key, row)
        print(f"验证Fill Rate={row['validation_fill_rate']:.4%}, 平均总库存={row['validation_average_total_inventory']:.0f}", flush=True)
    assert best is not None
    selected = best[1]
    pd.DataFrame(candidates).to_csv(MODEL_DIR / 'validation_candidates.csv', index=False)
    print(f'最佳候选：{selected}', flush=True)
    final_augmented = _augment_cost_features(oof_data.features, oof_centers, final_scale, len(oof_periods), n_skus, inventory_for_target_periods(behavior_inventory, oof_periods))
    final_targets = action_targets(oof_centers, _repeat_scale(final_scale, len(oof_periods)), actions)
    final_costs = action_costs(oof_data.target, final_targets, selected['shortage_penalty'], holding_cost=config.holding_cost)
    print('使用最佳损失参数在全部训练期外样本上重训成本模型……', flush=True)
    final_model = _new_cost_model(float(selected['alpha']), config).fit(final_augmented, final_costs)
    final_model.save_model(MODEL_DIR / 'cost_xgb_spo_model.json')
    (MODEL_DIR / 'model_metadata.json').write_text(json.dumps({'cost_scale': final_model.cost_scale_, 'actions': actions.tolist(), 'feature_names': [*full_train.feature_names, 'base_xgb_forecast', 'residual_scale', CURRENT_INVENTORY_FEATURE], 'selected_candidate': selected}, ensure_ascii=False, indent=2), encoding='utf-8')
    test_rollout = _rollout_hybrid_policy(final_model, test.demand, test.initial_inventory, test_features.features, test_centers, final_scale, actions, include_pre_period_decision=False)
    test_targets = test_rollout.targets
    test_forecasts = _period_matrix(test_centers, len(test_features.target_periods), n_skus)[:, :-1]
    result = evaluator.evaluate_dynamic_base_stock(_wide(train.item_ids, test.demand), pd.DataFrame({'item_id': train.item_ids, 'initial_inventory': test.initial_inventory}), _wide(train.item_ids, test_targets), output_path=MODEL_DIR / 'test_per_sku_metrics.csv', trajectory_output_path=MODEL_DIR / 'test_trajectory.csv', model_name=MODEL_NAME, forecasts_df=_wide(train.item_ids, test_forecasts))
    _wide(train.item_ids, test_targets).to_csv(MODEL_DIR / 'test_order_up_to.csv', index=False)
    _wide(train.item_ids, test_forecasts).to_csv(MODEL_DIR / 'test_forecasts.csv', index=False)
    test_action_path = actions[test_rollout.actions]
    _wide(train.item_ids, test_action_path).to_csv(MODEL_DIR / 'test_selected_actions.csv', index=False)
    row = {'model': MODEL_NAME, 'family': 'decision_focused_learning', 'parameters': json.dumps({'base_forecast': 'expanding_oof_xgboost', 'loss': 'cost_mse_plus_smooth_spo_plus', 'alpha': selected['alpha'], 'temperature': config.temperature, 'shortage_penalty': selected['shortage_penalty'], 'action_grid': [config.action_min, config.action_max, config.action_step], 'n_estimators': config.tune_estimators, 'max_depth': config.max_depth, 'learning_rate': config.learning_rate, 'inventory_feature': CURRENT_INVENTORY_FEATURE}, ensure_ascii=False), 'selected_z': np.nan, 'training_rows': len(oof_data.target), 'validation_wape': _wape(validation.target, validation_centers[:len(validation.target)]), 'test_wape': _wape(test.demand.T.reshape(-1), test_centers[:-n_skus]), 'validation_spo_loss': selected['validation_spo_regret'], 'test_spo_loss': np.nan, 'validation_fill_rate': selected['validation_fill_rate'], 'validation_average_total_inventory': selected['validation_average_total_inventory'], 'test_fill_rate': result.summary['total_fill_rate'], 'test_constraint_met': result.summary['total_fill_rate'] >= config.service_level, 'test_average_total_inventory': result.summary['average_total_inventory'], 'test_total_turnover': result.summary['total_turnover'], 'test_total_lost_sales': result.summary['total_lost_sales'], 'test_sku_95_share': result.summary['sku_service_level_95_share'], 'runtime_seconds': time.perf_counter() - start_time}
    summary = {'result': row, 'test': result.summary, 'selected_validation_candidate': selected, 'config': asdict(config), 'base_xgboost_parameters': BASE_XGB_PARAMETERS, 'leakage_control': "Cost-model training centers are expanding-window out-of-fold XGBoost predictions; validation and test centers are generated by models fitted only on earlier periods. Current ending inventory is generated historically for training and fed back from the model's own sequential rollout at validation and test time."}
    (MODEL_DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    _save_item_trajectory(args.item_id)
    combined = _update_comparison(row)
    _plot_comparison(combined)
    print('\nXGB-SPO Hybrid实验完成', flush=True)
    print(pd.DataFrame([row])[['model', 'test_fill_rate', 'test_average_total_inventory', 'test_total_turnover']].to_string(index=False), flush=True)
    print(f'总耗时: {time.perf_counter() - start_time:.2f}秒', flush=True)

def _fit_base_xgb(features: np.ndarray, target: np.ndarray, config: HybridExperimentConfig):
    model = build_regressor('xgboost', BASE_XGB_PARAMETERS, config.n_jobs)
    model.fit(features, target)
    return model

def _expanding_oof_predictions(supervised, n_skus: int, config: HybridExperimentConfig):
    prediction_periods = np.arange(config.oof_start, supervised.target_periods.max() + 1)
    predictions: list[np.ndarray] = []
    actual_periods: list[int] = []
    for start in range(config.oof_start, int(supervised.target_periods.max()) + 1, config.block_size):
        end = min(start + config.block_size, int(supervised.target_periods.max()) + 1)
        fit_period_mask = supervised.target_periods < start
        predict_period_mask = (supervised.target_periods >= start) & (supervised.target_periods < end)
        fit_rows = _period_mask(supervised.target_periods, fit_period_mask, n_skus)
        predict_rows = _period_mask(supervised.target_periods, predict_period_mask, n_skus)
        model = _fit_base_xgb(supervised.features[fit_rows], supervised.target[fit_rows], config)
        predictions.append(_nonnegative(model.predict(supervised.features[predict_rows])))
        actual_periods.extend(supervised.target_periods[predict_period_mask].tolist())
        print(f'  基础预测：训练至Period_{start}之前，预测Period_{start + 1}–{end}', flush=True)
    periods = np.asarray(actual_periods, dtype=int)
    if not np.array_equal(periods, prediction_periods):
        raise ValueError('扩展窗口预测周期不连续。')
    return (periods, np.concatenate(predictions).astype(np.float32))

def _take_periods(supervised, periods: np.ndarray, n_skus: int):
    from common.features import SupervisedData
    selected = np.isin(supervised.target_periods, periods)
    rows = _period_mask(supervised.target_periods, selected, n_skus)
    return SupervisedData(features=supervised.features[rows], target=supervised.target[rows], feature_names=supervised.feature_names, target_periods=supervised.target_periods[selected])

def _period_mask(target_periods: np.ndarray, period_mask: np.ndarray, n_skus: int) -> np.ndarray:
    if len(target_periods) != len(period_mask):
        raise ValueError('周期掩码长度不一致。')
    return np.repeat(np.asarray(period_mask, dtype=bool), n_skus)

def _per_sku_scale(errors: np.ndarray, n_periods: int, n_skus: int) -> np.ndarray:
    matrix = _period_matrix(errors, n_periods, n_skus)
    scale = matrix.std(axis=1, ddof=1)
    fallback = float(np.std(errors, ddof=1))
    return np.maximum(np.nan_to_num(scale, nan=fallback, posinf=fallback, neginf=fallback), 1e-06)

def _repeat_scale(scale: np.ndarray, n_periods: int) -> np.ndarray:
    return np.tile(np.asarray(scale, dtype=np.float32), n_periods)

def _augment_cost_features(features: np.ndarray, centers: np.ndarray, scale: np.ndarray, n_periods: int, n_skus: int, current_inventory: np.ndarray) -> np.ndarray:
    repeated_scale = _repeat_scale(scale, n_periods)
    inventory = np.asarray(current_inventory, dtype=float).reshape(-1)
    if len(features) != len(centers) or len(features) != len(repeated_scale) or len(features) != len(inventory):
        raise ValueError('成本模型增强特征长度不一致。')
    return np.column_stack([features, centers, repeated_scale, inventory]).astype(np.float32)

def _rollout_hybrid_policy(model, demand: np.ndarray, initial_inventory: np.ndarray, decision_features: np.ndarray, centers: np.ndarray, scale: np.ndarray, actions: np.ndarray, *, include_pre_period_decision: bool, predict_with_targets: bool=False):
    n_skus, n_periods = np.asarray(demand).shape
    center_path = _period_matrix(centers, n_periods + 1, n_skus)

    def selector(block_index: int, features_with_inventory: np.ndarray):
        base = features_with_inventory[:, :-1]
        inventory = features_with_inventory[:, -1]
        center = center_path[:, block_index]
        augmented = _augment_cost_features(base, center, scale, 1, n_skus, inventory)
        candidates = action_targets(center, scale, actions)
        if predict_with_targets:
            selected = model.predict_actions(augmented, candidates)
        else:
            selected = model.predict_actions(augmented)
        target = np.ceil(np.clip(candidates[np.arange(n_skus), selected], 0.0, None))
        return (selected, target)
    return rollout_inventory_aware_policy(demand, initial_inventory, decision_features, selector, include_pre_period_decision=include_pre_period_decision)

def _new_cost_model(alpha: float, config: HybridExperimentConfig) -> XGBSPOHybrid:
    return XGBSPOHybrid(alpha=alpha, temperature=config.temperature, n_estimators=config.tune_estimators, max_depth=config.max_depth, learning_rate=config.learning_rate, n_jobs=config.n_jobs, random_seed=config.random_seed)

def _selected_target_path(centers: np.ndarray, scale: np.ndarray, selected_actions: np.ndarray, actions: np.ndarray, n_periods: int, n_skus: int) -> np.ndarray:
    target = np.ceil(np.clip(centers + _repeat_scale(scale, n_periods) * actions[selected_actions], 0.0, None))
    return _period_matrix(target, n_periods, n_skus)

def _period_matrix(values: np.ndarray, n_periods: int, n_skus: int) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.size != n_periods * n_skus:
        raise ValueError('数值数量与SKU×周期数不一致。')
    return vector.reshape(n_periods, n_skus).T

def _wide(item_ids: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    matrix = np.asarray(values, dtype=float)
    frame = pd.DataFrame(matrix, columns=[f'period_{index}' for index in range(1, matrix.shape[1] + 1)])
    frame.insert(0, 'item_id', item_ids)
    return frame

def _nonnegative(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.clip(np.asarray(values, dtype=float), 0.0, None))

def _wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denominator = float(np.asarray(actual).sum())
    return float(np.abs(np.asarray(actual) - np.asarray(forecast)).sum() / denominator) if denominator else 0.0

def _variant_name(alpha: float) -> str:
    if alpha == 0.0:
        return 'xgb_cost_mse'
    if alpha == 1.0:
        return 'xgb_smooth_spo_plus'
    return 'xgb_spo_hybrid'

def _selection_key(summary: dict, regret: float, service_level: float) -> tuple:
    feasible = summary['total_fill_rate'] >= service_level
    return (not feasible, summary['average_total_inventory'] if feasible else -summary['total_fill_rate'], regret)

def _save_item_trajectory(item_id: str) -> None:
    frame = pd.read_csv(MODEL_DIR / 'test_trajectory.csv')
    matched = frame.loc[frame['item_id'].astype(str) == str(item_id)].copy()
    if matched.empty:
        raise ValueError(f'找不到item_id={item_id}。')
    destination = OUTPUT_DIR / 'item_diagnostics' / MODEL_NAME
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
    focused = combined.loc[combined['model'].isin({'baseline', 'moving_average_12', 'xgboost', 'lightgbm', 'spo_plus_linear', MODEL_NAME})]
    focused.to_csv(OUTPUT_DIR / 'spo' / 'hybrid_vs_existing_comparison.csv', index=False)
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
    start('xgboost_spo', globals(), default_protocol='legacy')
