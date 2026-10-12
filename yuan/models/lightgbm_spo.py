"""models.lightgbm_spo: actual model/processing implementation, not a legacy runner alias.

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
OUT=ROOT/'results/generated/spo/lgbm_spo_hybrid'
WORK=OUT
TABLES=OUT/'图表'
OLD=ROOT/'results/generated/ready_immediate'
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from models.xgboost_spo import hybrid_grad_hess

@dataclass
class LGBMSPOHybrid:
    """Learn coupled action costs by expanding each sample across its actions."""
    alpha: float
    temperature: float = 0.5
    n_estimators: int = 100
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

    def fit(self, features: np.ndarray, targets: np.ndarray, costs: np.ndarray, actions: np.ndarray) -> 'LGBMSPOHybrid':
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
            gradient, hessian = hybrid_grad_hess(np.asarray(y_true).reshape(-1, n_actions), np.asarray(y_pred).reshape(-1, n_actions), alpha=self.alpha, temperature=self.temperature)
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
from models.xgboost_spo import _augment_cost_features, _nonnegative, _per_sku_scale, _period_mask, _period_matrix, _plot_comparison, _repeat_scale, _rollout_hybrid_policy, _selection_key, _take_periods, _variant_name, _wape, _wide
from common.evaluation import InventoryEvaluator
from common.features import build_supervised_data
from common.features import load_panel
from common.regressors import build_regressor
from common.policy_state import CURRENT_INVENTORY_FEATURE
from models.spo_linear import action_costs
from models.spo_linear import action_targets
from common.policy_state import behavior_ending_inventory
from common.policy_state import inventory_for_target_periods
MODEL_NAME = 'lgbm_spo_hybrid'

@dataclass(frozen=True)
class LGBMHybridConfig:
    service_level: float = 0.95
    minimum_history: int = 12
    oof_start: int = 24
    validation_start: int = 42
    block_size: int = 6
    action_min: float = 0.0
    action_max: float = 4.0
    action_step: float = 0.5
    holding_cost: float = 1.0
    shortage_penalties: tuple[float, ...] = (20.0, 40.0, 80.0)
    alphas: tuple[float, ...] = (0.0, 0.5, 0.9, 1.0)
    temperature: float = 0.5
    tune_estimators: int = 100
    num_leaves: int = 63
    learning_rate: float = 0.05
    min_child_samples: int = 100
    random_seed: int = 42
    n_jobs: int = 8
BASE_LGBM_PARAMETERS = {'n_estimators': 450, 'num_leaves': 127, 'learning_rate': 0.04, 'min_child_samples': 50}

def train_legacy() -> None:
    parser = argparse.ArgumentParser(description='运行LightGBM与SPO+补货模型。')
    parser.add_argument('--n-jobs', type=int, default=8)
    parser.add_argument('--item-id', default='1')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    if args.n_jobs < 1:
        raise ValueError('--n-jobs必须>=1。')
    config = LGBMHybridConfig(n_jobs=args.n_jobs)
    if args.quick:
        config = LGBMHybridConfig(shortage_penalties=(20.0,), alphas=(1.0,), tune_estimators=15, n_jobs=args.n_jobs)
    start_time = time.perf_counter()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print('读取train/test面板……', flush=True)
    train = load_panel(DATA_DIR / 'replenishment_train.xlsx')
    test = load_panel(DATA_DIR / 'replenishment_test.xlsx')
    if not np.array_equal(train.item_ids, test.item_ids):
        raise ValueError('train/test的item_id或顺序不一致。')
    n_skus = len(train.item_ids)
    print('构造严格滞后的监督学习特征……', flush=True)
    full_train = build_supervised_data(train.demand, train.x, train.categories, np.arange(config.minimum_history, train.demand.shape[1]))
    validation = build_supervised_data(train.demand, train.x, train.categories, np.arange(config.validation_start, train.demand.shape[1]))
    validation_extra = build_supervised_data(train.demand, train.x, train.categories, np.asarray([train.demand.shape[1]]), include_target=False)
    combined_demand = np.concatenate([train.demand, test.demand], axis=1)
    combined_x = np.concatenate([train.x, test.x], axis=1)
    test_features = build_supervised_data(combined_demand, combined_x, train.categories, np.arange(train.demand.shape[1], combined_demand.shape[1] + 1), include_target=False)
    behavior_inventory = behavior_ending_inventory(train.demand, train.initial_inventory)
    print('生成扩展窗口的LightGBM训练期外预测中心……', flush=True)
    oof_periods, oof_centers = _expanding_oof_predictions(full_train, n_skus, config)
    oof_data = _take_periods(full_train, oof_periods, n_skus)
    tune_period_mask = oof_periods < config.validation_start
    tune_periods = oof_periods[tune_period_mask]
    tune_rows = np.repeat(tune_period_mask, n_skus)
    tune_features = oof_data.features[tune_rows]
    tune_target = oof_data.target[tune_rows]
    tune_centers = oof_centers[tune_rows]
    tune_scale = _per_sku_scale(tune_target - tune_centers, len(tune_periods), n_skus)
    final_scale = _per_sku_scale(oof_data.target - oof_centers, len(oof_periods), n_skus)
    print('拟合验证期和测试期的基础LightGBM需求预测……', flush=True)
    fit_periods = full_train.target_periods < config.validation_start
    fit_rows = _period_mask(full_train.target_periods, fit_periods, n_skus)
    base_validation = _fit_base_lgbm(full_train.features[fit_rows], full_train.target[fit_rows], config)
    validation_all_features = np.vstack([validation.features, validation_extra.features])
    validation_centers = _nonnegative(base_validation.predict(validation_all_features))
    base_test = _fit_base_lgbm(full_train.features, full_train.target, config)
    test_centers = _nonnegative(base_test.predict(test_features.features))
    base_test.booster_.save_model(str(MODEL_DIR / 'base_demand_lightgbm.txt'))
    actions = np.arange(config.action_min, config.action_max + config.action_step / 2, config.action_step, dtype=np.float32)
    tune_augmented = _augment_cost_features(tune_features, tune_centers, tune_scale, len(tune_periods), n_skus, inventory_for_target_periods(behavior_inventory, tune_periods))
    tune_targets = action_targets(tune_centers, _repeat_scale(tune_scale, len(tune_periods)), actions)
    validation_targets_all = action_targets(validation_centers, _repeat_scale(tune_scale, len(validation.target_periods) + 1), actions)
    evaluator = InventoryEvaluator(lead_time=1)
    validation_demand_df = _wide(train.item_ids, train.demand[:, config.validation_start:])
    candidates: list[dict] = []
    best: tuple[tuple, dict] | None = None
    combinations = [(penalty, alpha) for penalty in config.shortage_penalties for alpha in config.alphas]
    for candidate_index, (penalty, alpha) in enumerate(combinations, start=1):
        candidate_start = time.perf_counter()
        variant = _variant_name(alpha).replace('xgb', 'lgbm')
        print(f'候选 {candidate_index}/{len(combinations)}: {variant}, 缺货惩罚={penalty:g}', flush=True)
        costs = action_costs(tune_target, tune_targets, penalty, holding_cost=config.holding_cost)
        model = _new_cost_model(alpha, config).fit(tune_augmented, tune_targets, costs, actions)
        validation_rollout = _rollout_hybrid_policy(model, train.demand[:, config.validation_start:], behavior_inventory[:, config.validation_start - 1], validation_all_features, validation_centers, tune_scale, actions, include_pre_period_decision=True, predict_with_targets=True)
        validation_actions = validation_rollout.actions.T.reshape(-1)
        validation_result = evaluator.evaluate_dynamic_base_stock(validation_demand_df, pd.DataFrame({'item_id': train.item_ids, 'initial_inventory': validation_rollout.initial_inventory}), _wide(train.item_ids, validation_rollout.targets))
        validation_costs = action_costs(validation.target, validation_targets_all[:len(validation.target)], penalty, holding_cost=config.holding_cost)
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
    print('使用最佳参数在全部训练期外样本上重训成本模型……', flush=True)
    final_model = _new_cost_model(float(selected['alpha']), config).fit(final_augmented, final_targets, final_costs, actions)
    final_model.save_model(MODEL_DIR / 'cost_lgbm_spo_model.txt')
    (MODEL_DIR / 'model_metadata.json').write_text(json.dumps({'cost_scale': final_model.cost_scale_, 'actions': actions.tolist(), 'feature_names': [*full_train.feature_names, 'base_lgbm_forecast', 'residual_scale', CURRENT_INVENTORY_FEATURE, 'action_multiplier', 'candidate_target'], 'selected_candidate': selected}, ensure_ascii=False, indent=2), encoding='utf-8')
    test_rollout = _rollout_hybrid_policy(final_model, test.demand, test.initial_inventory, test_features.features, test_centers, final_scale, actions, include_pre_period_decision=False, predict_with_targets=True)
    test_targets = test_rollout.targets
    test_forecasts = _period_matrix(test_centers, len(test_features.target_periods), n_skus)[:, :-1]
    result = evaluator.evaluate_dynamic_base_stock(_wide(train.item_ids, test.demand), pd.DataFrame({'item_id': train.item_ids, 'initial_inventory': test.initial_inventory}), _wide(train.item_ids, test_targets), output_path=MODEL_DIR / 'test_per_sku_metrics.csv', trajectory_output_path=MODEL_DIR / 'test_trajectory.csv', model_name=MODEL_NAME, forecasts_df=_wide(train.item_ids, test_forecasts))
    _wide(train.item_ids, test_targets).to_csv(MODEL_DIR / 'test_order_up_to.csv', index=False)
    _wide(train.item_ids, test_forecasts).to_csv(MODEL_DIR / 'test_forecasts.csv', index=False)
    test_action_path = actions[test_rollout.actions]
    _wide(train.item_ids, test_action_path).to_csv(MODEL_DIR / 'test_selected_actions.csv', index=False)
    row = {'model': MODEL_NAME, 'family': 'decision_focused_learning', 'parameters': json.dumps({'base_forecast': 'expanding_oof_lightgbm', 'loss': 'cost_mse_plus_smooth_spo_plus', 'alpha': selected['alpha'], 'temperature': config.temperature, 'shortage_penalty': selected['shortage_penalty'], 'action_grid': [config.action_min, config.action_max, config.action_step], 'n_estimators': config.tune_estimators, 'num_leaves': config.num_leaves, 'learning_rate': config.learning_rate, 'inventory_feature': CURRENT_INVENTORY_FEATURE}, ensure_ascii=False), 'selected_z': np.nan, 'training_rows': len(oof_data.target), 'validation_wape': _wape(validation.target, validation_centers[:len(validation.target)]), 'test_wape': _wape(test.demand.T.reshape(-1), test_centers[:-n_skus]), 'validation_spo_loss': selected['validation_spo_regret'], 'test_spo_loss': np.nan, 'validation_fill_rate': selected['validation_fill_rate'], 'validation_average_total_inventory': selected['validation_average_total_inventory'], 'test_fill_rate': result.summary['total_fill_rate'], 'test_constraint_met': result.summary['total_fill_rate'] >= config.service_level, 'test_average_total_inventory': result.summary['average_total_inventory'], 'test_total_turnover': result.summary['total_turnover'], 'test_total_lost_sales': result.summary['total_lost_sales'], 'test_sku_95_share': result.summary['sku_service_level_95_share'], 'runtime_seconds': time.perf_counter() - start_time}
    summary = {'result': row, 'test': result.summary, 'selected_validation_candidate': selected, 'config': asdict(config), 'base_lightgbm_parameters': BASE_LGBM_PARAMETERS, 'implementation': 'Each sample-action pair is one LightGBM row. The custom objective reshapes rows by action group to compute coupled smooth SPO+ gradients and Hessians. Current ending inventory is generated historically for training and fed back sequentially at validation and test time.'}
    (MODEL_DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    _save_item_trajectory(args.item_id)
    combined = _update_comparison(row)
    _plot_comparison(combined)
    print('\nLightGBM–SPO+实验完成', flush=True)
    print(pd.DataFrame([row])[['model', 'test_fill_rate', 'test_average_total_inventory', 'test_total_turnover']].to_string(index=False), flush=True)
    print(f'总耗时: {time.perf_counter() - start_time:.2f}秒', flush=True)

def _fit_base_lgbm(features: np.ndarray, target: np.ndarray, config: LGBMHybridConfig):
    model = build_regressor('lightgbm', BASE_LGBM_PARAMETERS, config.n_jobs)
    model.fit(features, target)
    return model

def _expanding_oof_predictions(supervised, n_skus: int, config: LGBMHybridConfig):
    prediction_periods = np.arange(config.oof_start, supervised.target_periods.max() + 1)
    predictions: list[np.ndarray] = []
    actual_periods: list[int] = []
    for start in range(config.oof_start, int(supervised.target_periods.max()) + 1, config.block_size):
        end = min(start + config.block_size, int(supervised.target_periods.max()) + 1)
        fit_period_mask = supervised.target_periods < start
        predict_period_mask = (supervised.target_periods >= start) & (supervised.target_periods < end)
        fit_rows = _period_mask(supervised.target_periods, fit_period_mask, n_skus)
        predict_rows = _period_mask(supervised.target_periods, predict_period_mask, n_skus)
        model = _fit_base_lgbm(supervised.features[fit_rows], supervised.target[fit_rows], config)
        predictions.append(_nonnegative(model.predict(supervised.features[predict_rows])))
        actual_periods.extend(supervised.target_periods[predict_period_mask].tolist())
        print(f'  基础预测：训练至Period_{start}之前，预测Period_{start + 1}–{end}', flush=True)
    periods = np.asarray(actual_periods, dtype=int)
    if not np.array_equal(periods, prediction_periods):
        raise ValueError('扩展窗口预测周期不连续。')
    return (periods, np.concatenate(predictions).astype(np.float32))

def _new_cost_model(alpha: float, config: LGBMHybridConfig) -> LGBMSPOHybrid:
    return LGBMSPOHybrid(alpha=alpha, temperature=config.temperature, n_estimators=config.tune_estimators, num_leaves=config.num_leaves, learning_rate=config.learning_rate, min_child_samples=config.min_child_samples, n_jobs=config.n_jobs, random_seed=config.random_seed)

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
    focused = combined.loc[combined['model'].isin({'baseline', 'lightgbm', 'xgboost', 'spo_plus_linear', 'xgb_spo_hybrid', MODEL_NAME})]
    focused.to_csv(OUTPUT_DIR / 'spo' / 'tree_spo_comparison.csv', index=False)
    return combined

MODEL_DIR=OUT
SPO_DIR=OUT
EXPERIMENT_DIR=OUT

if __name__=='__main__':
    from common.cli import start
    start('lightgbm_spo', globals(), default_protocol='legacy')
