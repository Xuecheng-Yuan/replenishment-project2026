"""common.ml_training: actual model/processing implementation, not a legacy runner alias.

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
OUT=ROOT/'results/generated/machine_learning'
WORK=OUT
TABLES=OUT/'图表'
OLD=ROOT/'results/generated/ready_immediate'
import argparse
import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from common.evaluation import InventoryEvaluator
from common.trajectories import plot_model_comparison
from common.features import build_supervised_data
from common.features import load_panel
from common.regressors import model_specs
from common.regressors import ModelSpec, build_regressor

@dataclass(frozen=True)
class MLExperimentConfig:
    service_level: float = 0.95
    validation_start: int = 42
    minimum_history: int = 12
    z_min: float = 0.0
    z_max: float = 8.0
    z_step: float = 0.05
    random_seed: int = 42
    n_jobs: int = 8

def train() -> None:
    parser = argparse.ArgumentParser(description='运行独立机器学习补货模型。')
    parser.add_argument('--n-jobs', type=int, default=8)
    parser.add_argument('--item-id', default='1')
    parser.add_argument('--models', default='all', help='逗号分隔模型名；默认all。可用于失败后单独重跑某个模型。')
    args = parser.parse_args()
    if args.n_jobs < 1:
        raise ValueError('--n-jobs 必须 >= 1。')
    config = MLExperimentConfig(n_jobs=args.n_jobs)
    start = time.perf_counter()
    ML_DIR.mkdir(parents=True, exist_ok=True)
    print('读取完整 train/test 面板……', flush=True)
    train = load_panel(DATA_DIR / 'replenishment_train.xlsx')
    test = load_panel(DATA_DIR / 'replenishment_test.xlsx')
    _validate_panels(train, test, config)
    n_skus = len(train.item_ids)
    print('构造严格滞后的监督学习特征……', flush=True)
    training_periods = np.arange(config.minimum_history, config.validation_start)
    validation_periods = np.arange(config.validation_start, train.demand.shape[1])
    all_train_periods = np.arange(config.minimum_history, train.demand.shape[1])
    tune_train = build_supervised_data(train.demand, train.x, train.categories, training_periods)
    validation = build_supervised_data(train.demand, train.x, train.categories, validation_periods)
    validation_extra = build_supervised_data(train.demand, train.x, train.categories, np.asarray([train.demand.shape[1]]), include_target=False)
    final_train = build_supervised_data(train.demand, train.x, train.categories, all_train_periods)
    combined_demand = np.concatenate([train.demand, test.demand], axis=1)
    combined_x = np.concatenate([train.x, test.x], axis=1)
    test_periods_with_extra = np.arange(train.demand.shape[1], combined_demand.shape[1] + 1)
    test_features = build_supervised_data(combined_demand, combined_x, train.categories, test_periods_with_extra, include_target=False)
    feature_metadata = {'feature_count': len(final_train.feature_names), 'feature_names': final_train.feature_names, 'training_target_periods': [int(value + 1) for value in training_periods], 'validation_target_periods': [int(value + 1) for value in validation_periods], 'x_timing': 'target period t only uses x from t-1 or earlier', 'pca': 'SVR only, retaining 95% variance', 'kmeans': 'not used because a fixed full-history cluster would leak future information'}
    (ML_DIR / 'feature_metadata.json').write_text(json.dumps(feature_metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    requested = _requested_models(args.models)
    specifications = [spec for spec in model_specs() if spec.name in requested]
    unknown = requested - {spec.name for spec in model_specs()}
    if unknown:
        raise ValueError(f'未知模型: {sorted(unknown)}')
    evaluator = InventoryEvaluator(lead_time=1)
    comparison_rows: list[dict] = []
    item_trajectories = _existing_item_trajectories(args.item_id)
    for specification in specifications:
        print(f'\n===== {specification.name} =====', flush=True)
        row, trajectory = _run_model(specification, tune_train, validation, validation_extra, final_train, test_features, train, test, config, evaluator, args.item_id)
        comparison_rows.append(row)
        item_trajectories[specification.name] = trajectory
        _write_comparison(comparison_rows)
    _write_comparison(comparison_rows)
    if item_trajectories:
        plot_model_comparison(item_trajectories, args.item_id, OUTPUT_DIR / 'item_comparisons' / f'item_{args.item_id}_all_models_comparison.png')
    print(f'\n机器学习实验完成，总耗时: {time.perf_counter() - start:.2f} 秒', flush=True)
    if comparison_rows:
        print(pd.DataFrame(comparison_rows)[['model', 'selected_z', 'test_fill_rate', 'test_average_total_inventory', 'test_total_turnover']].to_string(index=False), flush=True)

def _run_model(specification: ModelSpec, tune_train, validation, validation_extra, final_train, test_features, train, test, config: MLExperimentConfig, evaluator: InventoryEvaluator, item_id: str) -> tuple[dict, pd.DataFrame]:
    model_start = time.perf_counter()
    n_skus = len(train.item_ids)
    tune_period_rows = np.repeat(tune_train.target_periods, n_skus)
    tune_indices = _stratified_sample_indices(tune_train.target, tune_period_rows, specification.tune_sample_limit, config.random_seed)
    X_fit = tune_train.features[tune_indices]
    y_fit = tune_train.target[tune_indices]
    candidate_rows: list[dict] = []
    best: tuple[tuple, dict, np.ndarray, np.ndarray, float, dict] | None = None
    for index, parameters in enumerate(specification.candidates, start=1):
        candidate_start = time.perf_counter()
        print(f'候选 {index}/{len(specification.candidates)}: {parameters}', flush=True)
        model = build_regressor(specification.name, parameters, config.n_jobs)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X_fit, y_fit)
        train_prediction = _nonnegative(model.predict(tune_train.features))
        validation_prediction = _nonnegative(model.predict(validation.features))
        extra_prediction = _nonnegative(model.predict(validation_extra.features))
        training_scale = _per_sku_scale(tune_train.target - train_prediction, len(tune_train.target_periods), n_skus)
        validation_path = np.column_stack([_period_matrix(validation_prediction, len(validation.target_periods), n_skus), extra_prediction])
        selected_z, validation_summary = _calibrate_z(evaluator, train.item_ids, train.demand[:, config.validation_start:], validation_path, training_scale, config)
        validation_wape = _wape(validation.target, validation_prediction)
        elapsed = time.perf_counter() - candidate_start
        candidate_row = {'candidate': index, 'parameters': json.dumps(parameters, ensure_ascii=False), 'selected_z': selected_z, 'validation_wape': validation_wape, 'validation_fill_rate': validation_summary['total_fill_rate'], 'validation_average_total_inventory': validation_summary['average_total_inventory'], 'fit_and_validation_seconds': elapsed}
        candidate_rows.append(candidate_row)
        key = _selection_key(validation_summary, validation_wape)
        if best is None or key < best[0]:
            best = (key, parameters, validation_prediction, extra_prediction, selected_z, validation_summary)
    assert best is not None
    _, best_parameters, validation_prediction, _, selected_z, validation_summary = best
    model_output = ML_DIR / specification.name
    model_output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidate_rows).to_csv(model_output / 'validation_candidates.csv', index=False)
    final_period_rows = np.repeat(final_train.target_periods, n_skus)
    final_indices = _stratified_sample_indices(final_train.target, final_period_rows, specification.final_sample_limit, config.random_seed)
    print(f'使用最佳参数重训: {best_parameters}', flush=True)
    final_model = build_regressor(specification.name, best_parameters, config.n_jobs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        final_model.fit(final_train.features[final_indices], final_train.target[final_indices])
    joblib.dump(final_model, model_output / 'model.joblib', compress=3)
    _save_feature_importance(final_model, final_train.feature_names, model_output)
    test_path_flat = _nonnegative(final_model.predict(test_features.features))
    test_path = _period_matrix(test_path_flat, len(test_features.target_periods), n_skus)
    validation_scale = _per_sku_scale(validation.target - validation_prediction, len(validation.target_periods), n_skus)
    test_targets = _targets(test_path[:, 1:], validation_scale, selected_z)
    test_forecasts = test_path[:, :-1]
    test_demand_df = _wide(train.item_ids, test.demand)
    test_info = pd.DataFrame({'item_id': train.item_ids, 'initial_inventory': test.initial_inventory})
    result = evaluator.evaluate_dynamic_base_stock(test_demand_df, test_info, _wide(train.item_ids, test_targets), output_path=model_output / 'test_per_sku_metrics.csv', trajectory_output_path=model_output / 'test_trajectory.csv', model_name=specification.name, forecasts_df=_wide(train.item_ids, test_forecasts))
    _wide(train.item_ids, test_forecasts).to_csv(model_output / 'test_forecasts.csv', index=False)
    _wide(train.item_ids, test_targets).to_csv(model_output / 'test_order_up_to.csv', index=False)
    test_wape = _wape(test.demand.T.reshape(-1), test_path_flat[:-n_skus])
    row = {'model': specification.name, 'family': 'machine_learning', 'parameters': json.dumps(best_parameters, ensure_ascii=False), 'selected_z': selected_z, 'training_rows': int(len(final_indices)), 'validation_wape': _wape(validation.target, validation_prediction), 'test_wape': test_wape, 'validation_fill_rate': validation_summary['total_fill_rate'], 'validation_average_total_inventory': validation_summary['average_total_inventory'], 'test_fill_rate': result.summary['total_fill_rate'], 'test_constraint_met': result.summary['total_fill_rate'] >= config.service_level, 'test_average_total_inventory': result.summary['average_total_inventory'], 'test_total_turnover': result.summary['total_turnover'], 'test_total_lost_sales': result.summary['total_lost_sales'], 'test_sku_95_share': result.summary['sku_service_level_95_share'], 'runtime_seconds': time.perf_counter() - model_start}
    summary = {'result': row, 'validation': validation_summary, 'test': result.summary, 'config': asdict(config), 'sampling': {'tuning_rows': int(len(tune_indices)), 'final_rows': int(len(final_indices)), 'total_tuning_rows': int(len(tune_train.target)), 'total_final_rows': int(len(final_train.target))}}
    (model_output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    trajectory = _save_item_trajectory(specification.name, item_id)
    print(f"测试 Fill Rate={result.summary['total_fill_rate']:.4%}, 平均总库存={result.summary['average_total_inventory']:.0f}", flush=True)
    return (row, trajectory)

def _calibrate_z(evaluator: InventoryEvaluator, item_ids: np.ndarray, validation_demand: np.ndarray, forecast_path: np.ndarray, scale: np.ndarray, config: MLExperimentConfig) -> tuple[float, dict]:
    demand_df = _wide(item_ids, validation_demand)
    candidates = np.arange(config.z_min, config.z_max + config.z_step / 2, config.z_step)
    cache: dict[int, dict] = {}

    def evaluate(index: int) -> dict:
        if index not in cache:
            z = float(candidates[index])
            initial = pd.DataFrame({'item_id': item_ids, 'initial_inventory': _targets(forecast_path[:, [0]], scale, z)[:, 0]})
            targets = _wide(item_ids, _targets(forecast_path[:, 1:], scale, z))
            cache[index] = evaluator.evaluate_dynamic_base_stock(demand_df, initial, targets).summary
        return cache[index]
    low, high = (0, len(candidates) - 1)
    if evaluate(high)['total_fill_rate'] < config.service_level:
        return (float(candidates[high]), evaluate(high))
    while low < high:
        middle = (low + high) // 2
        if evaluate(middle)['total_fill_rate'] >= config.service_level:
            high = middle
        else:
            low = middle + 1
    return (float(candidates[low]), evaluate(low))

def _stratified_sample_indices(target: np.ndarray, periods: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or len(target) <= limit:
        return np.arange(len(target))
    log_target = np.log1p(target)
    boundaries = np.unique(np.quantile(log_target, np.linspace(0, 1, 11)))
    demand_bin = np.digitize(log_target, boundaries[1:-1], right=True)
    period_values = np.unique(periods)
    period_map = {value: index for index, value in enumerate(period_values)}
    period_bin = np.asarray([period_map[value] // 3 for value in periods])
    labels = period_bin * 10 + demand_bin
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=limit, random_state=seed)
    sampled, _ = next(splitter.split(np.zeros(len(target)), labels))
    return np.sort(sampled)

def _per_sku_scale(errors: np.ndarray, n_periods: int, n_skus: int) -> np.ndarray:
    matrix = _period_matrix(errors, n_periods, n_skus)
    scale = matrix.std(axis=1, ddof=1)
    fallback = float(np.std(errors, ddof=1))
    return np.nan_to_num(scale, nan=fallback, posinf=fallback, neginf=fallback)

def _period_matrix(values: np.ndarray, n_periods: int, n_skus: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size != n_periods * n_skus:
        raise ValueError('预测数量与SKU×周期数不一致。')
    return values.reshape(n_periods, n_skus).T

def _targets(forecast: np.ndarray, scale: np.ndarray, z: float) -> np.ndarray:
    return np.ceil(np.clip(forecast + z * scale[:, None], 0, None))

def _wide(item_ids: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    frame = pd.DataFrame(values, columns=[f'period_{index}' for index in range(1, values.shape[1] + 1)])
    frame.insert(0, 'item_id', item_ids)
    return frame

def _nonnegative(prediction: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.clip(np.asarray(prediction, dtype=float), 0, None))

def _wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denominator = float(np.sum(actual))
    return float(np.abs(actual - forecast).sum() / denominator) if denominator else 0.0

def _selection_key(summary: dict, validation_wape: float) -> tuple:
    feasible = summary['total_fill_rate'] >= MLExperimentConfig.service_level
    return (not feasible, summary['average_total_inventory'] if feasible else -summary['total_fill_rate'], validation_wape)

def _requested_models(raw: str) -> set[str]:
    names = {spec.name for spec in model_specs()}
    return names if raw.strip().lower() == 'all' else {part.strip() for part in raw.split(',') if part.strip()}

def _validate_panels(train, test, config: MLExperimentConfig) -> None:
    if not np.array_equal(train.item_ids, test.item_ids):
        raise ValueError('train/test 的 item_id 或顺序不一致。')
    if not np.array_equal(train.categories.astype(str), test.categories.astype(str)):
        raise ValueError('train/test 的品类不一致。')
    if train.demand.shape[1] <= config.validation_start:
        raise ValueError('训练期不足以建立验证窗口。')

def _save_item_trajectory(model_name: str, item_id: str) -> pd.DataFrame:
    frame = pd.read_csv(ML_DIR / model_name / 'test_trajectory.csv')
    matched = frame.loc[frame['item_id'].astype(str) == str(item_id)].copy()
    if matched.empty:
        raise ValueError(f'找不到 item_id={item_id}。')
    destination = OUTPUT_DIR / 'item_diagnostics' / model_name
    destination.mkdir(parents=True, exist_ok=True)
    matched.to_csv(destination / f'item_{item_id}_test_trajectory.csv', index=False)
    return matched.drop(columns=['model', 'item_id'], errors='ignore')

def _existing_item_trajectories(item_id: str) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    root = OUTPUT_DIR / 'item_diagnostics'
    for path in sorted(root.glob(f'*/item_{item_id}_test_trajectory.csv')):
        model_name = path.parent.name
        if path.exists():
            result[model_name] = pd.read_csv(path).drop(columns=['model', 'item_id'], errors='ignore')
    return result

def _write_comparison(current_rows: list[dict]) -> None:
    rows_by_model = {row['model']: row for row in current_rows}
    for summary_path in ML_DIR.glob('*/summary.json'):
        payload = json.loads(summary_path.read_text(encoding='utf-8'))
        if 'result' in payload:
            rows_by_model.setdefault(payload['result']['model'], payload['result'])
    current = pd.DataFrame(rows_by_model.values()).sort_values('model')
    current.to_csv(ML_DIR / 'model_comparison.csv', index=False)
    existing_path = OUTPUT_DIR / 'model_comparison.csv'
    frames = [pd.read_csv(existing_path)] if existing_path.exists() else []
    all_models_path = OUTPUT_DIR / 'all_model_comparison.csv'
    if all_models_path.exists():
        previous_all = pd.read_csv(all_models_path)
        if 'family' in previous_all:
            dro_rows = previous_all.loc[previous_all['family'] == 'dro']
            if not dro_rows.empty:
                frames.append(dro_rows)
    frames.append(current)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates('model', keep='last')
    time_series_names = {'naive', 'moving_average_3', 'moving_average_6', 'moving_average_12', 'exponential_smoothing', 'arima'}
    combined.loc[combined['model'] == 'baseline', 'family'] = 'baseline'
    combined.loc[combined['model'].isin(time_series_names), 'family'] = 'time_series'
    combined['test_constraint_met'] = combined['test_fill_rate'] >= MLExperimentConfig.service_level
    combined = combined.sort_values(['test_constraint_met', 'test_average_total_inventory'], ascending=[False, True])
    combined.to_csv(OUTPUT_DIR / 'all_model_comparison.csv', index=False)

def _save_feature_importance(model, feature_names: list[str], output_dir: Path) -> None:
    estimator = model
    if hasattr(estimator, 'regressor_'):
        estimator = estimator.regressor_
    if hasattr(estimator, 'steps'):
        estimator = estimator.steps[-1][1]
    values = None
    if hasattr(estimator, 'feature_importances_'):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, 'coef_'):
        values = np.abs(np.ravel(estimator.coef_)).astype(float)
    if values is None or len(values) != len(feature_names):
        return
    pd.DataFrame({'feature': feature_names, 'importance': values}).sort_values('importance', ascending=False).to_csv(output_dir / 'feature_importance.csv', index=False)

ML_DIR=OUT
