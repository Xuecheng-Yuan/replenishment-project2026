"""models.dro_chi_square: actual model/processing implementation, not a legacy runner alias.

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
OUT=ROOT/'results/generated/dro'
WORK=OUT
TABLES=OUT/'图表'
OLD=ROOT/'results/generated/ready_immediate'
from dataclasses import dataclass
import numpy as np
SUPPORTED_METHODS = {'moment', 'wasserstein', 'chi_square'}

@dataclass(frozen=True)
class RobustBufferModel:
    """Fitted SKU-level safety buffers for one ambiguity-set specification."""
    method: str
    ambiguity_parameter: float
    service_level: float
    moving_average_window: int
    buffer_multiplier: float
    mean_error: np.ndarray
    residual_scale: np.ndarray
    safety_buffer: np.ndarray
    robust_constraint_gap: float
    n_residual_periods: int

    def targets(self, forecasts: np.ndarray) -> np.ndarray:
        """Convert one-step forecasts into nonnegative integer base-stock targets."""
        values = np.asarray(forecasts, dtype=float)
        if values.ndim != 2 or values.shape[0] != len(self.safety_buffer):
            raise ValueError('forecasts 必须是 SKU×周期二维数组。')
        return np.ceil(np.clip(values + self.safety_buffer[:, None], 0.0, None))

def rolling_moving_average_residuals(demand: np.ndarray, window: int=12) -> tuple[np.ndarray, np.ndarray]:
    """Return strictly one-step moving-average forecasts and residuals."""
    values = np.asarray(demand, dtype=float)
    if values.ndim != 2 or values.shape[1] <= window:
        raise ValueError('demand 必须是二维数组，且历史期数必须大于窗口。')
    if window < 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('window 或 demand 不合法。')
    forecasts = np.column_stack([values[:, period - window:period].mean(axis=1) for period in range(window, values.shape[1])])
    return (forecasts, values[:, window:] - forecasts)

def fit_robust_buffer(demand: np.ndarray, method: str, ambiguity_parameter: float, *, service_level: float=0.95, window: int=12, max_multiplier: float=25.0, tolerance: float=1e-08) -> RobustBufferModel:
    """Fit the smallest common scale multiplier satisfying a robust Fill-Rate bound.

    The robust constraint is written as

        sup_P E[(D-B)^+ - (1-service_level) D] <= 0.

    It is imposed for the historical forecast period with the smallest total
    forecast, which consequently protects every fitted historical period.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f'未知 DRO 方法: {method}')
    if ambiguity_parameter < 0:
        raise ValueError('ambiguity_parameter 不能为负数。')
    if method == 'moment' and ambiguity_parameter < 1:
        raise ValueError('moment DRO 的方差放大系数必须 >= 1。')
    if not 0 < service_level < 1:
        raise ValueError('service_level 必须在 (0, 1) 内。')
    fitted_forecasts, errors = rolling_moving_average_residuals(demand, window)
    mean_error = errors.mean(axis=1)
    residual_scale = errors.std(axis=1, ddof=1)
    fallback = float(errors.std(ddof=1))
    residual_scale = np.nan_to_num(residual_scale, nan=fallback, posinf=fallback, neginf=fallback)
    shortage_fraction = 1.0 - service_level
    minimum_total_forecast = float(fitted_forecasts.sum(axis=0).min())

    def gap(multiplier: float) -> float:
        return _robust_gap(method, ambiguity_parameter, multiplier, errors, mean_error, residual_scale, minimum_total_forecast, shortage_fraction)
    if gap(max_multiplier) > tolerance:
        raise ValueError(f'{method} 在 multiplier<={max_multiplier} 内无法满足鲁棒服务约束。')
    low, high = (0.0, float(max_multiplier))
    if gap(low) <= tolerance:
        high = low
    else:
        for _ in range(60):
            middle = (low + high) / 2.0
            if gap(middle) <= 0:
                high = middle
            else:
                low = middle
    if method == 'moment':
        safety_buffer = mean_error + high * residual_scale
    else:
        safety_buffer = high * residual_scale
    return RobustBufferModel(method=method, ambiguity_parameter=float(ambiguity_parameter), service_level=float(service_level), moving_average_window=int(window), buffer_multiplier=float(high), mean_error=mean_error, residual_scale=residual_scale, safety_buffer=safety_buffer, robust_constraint_gap=float(gap(high)), n_residual_periods=int(errors.shape[1]))

def _robust_gap(method: str, ambiguity_parameter: float, multiplier: float, errors: np.ndarray, mean_error: np.ndarray, residual_scale: np.ndarray, minimum_total_forecast: float, shortage_fraction: float) -> float:
    """Worst-case aggregate shortage-minus-allowance upper bound."""
    if method == 'moment':
        distance = multiplier * residual_scale
        variance = ambiguity_parameter * np.square(residual_scale)
        worst_shortage = 0.5 * (np.sqrt(np.square(distance) + variance) - distance)
        expected_demand = minimum_total_forecast + float(mean_error.sum())
        return float(worst_shortage.sum() - shortage_fraction * expected_demand)
    distance = multiplier * residual_scale[:, None]
    scenario_loss = np.maximum(errors - distance, 0.0) - shortage_fraction * errors
    empirical_mean = scenario_loss.mean(axis=1)
    if method == 'wasserstein':
        lipschitz = max(shortage_fraction, 1.0 - shortage_fraction)
        robust_value = empirical_mean + lipschitz * ambiguity_parameter * residual_scale
    else:
        robust_value = empirical_mean + np.sqrt(ambiguity_parameter * scenario_loss.var(axis=1, ddof=0))
    return float(robust_value.sum() - shortage_fraction * minimum_total_forecast)
import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import pandas as pd
from common.evaluation import InventoryEvaluator
from common.trajectories import plot_model_comparison
from common.features import load_panel
from models.moving_average import moving_average_path

@dataclass(frozen=True)
class DROExperimentConfig:
    service_level: float = 0.95
    validation_start: int = 42
    moving_average_window: int = 12
    lead_time: int = 1
METHODS = {'moment_dro_ma12': {'method': 'moment', 'parameter_name': 'variance_multiplier', 'candidates': (1.0, 1.25, 1.5, 2.0)}, 'wasserstein_dro_ma12': {'method': 'wasserstein', 'parameter_name': 'radius_ratio', 'candidates': (0.0025, 0.005, 0.01, 0.02, 0.03, 0.05)}, 'chi_square_dro_ma12': {'method': 'chi_square', 'parameter_name': 'radius', 'candidates': (0.01, 0.025, 0.05, 0.1, 0.2, 0.4)}}

def train_legacy() -> None:
    parser = argparse.ArgumentParser(description='运行多个 DRO 补货模型。')
    parser.add_argument('--item-id', default='1')
    parser.add_argument('--models', default='all', help='逗号分隔 DRO 模型名；默认 all。')
    args = parser.parse_args()
    requested = set(METHODS) if args.models.strip().lower() == 'all' else {value.strip() for value in args.models.split(',') if value.strip()}
    unknown = requested - set(METHODS)
    if unknown:
        raise ValueError(f'未知 DRO 模型: {sorted(unknown)}')
    start = time.perf_counter()
    config = DROExperimentConfig()
    DRO_DIR.mkdir(parents=True, exist_ok=True)
    print('读取 train/test 面板……', flush=True)
    train = load_panel(DATA_DIR / 'replenishment_train.xlsx')
    test = load_panel(DATA_DIR / 'replenishment_test.xlsx')
    if not np.array_equal(train.item_ids, test.item_ids):
        raise ValueError('train/test 的 item_id 或顺序不一致。')
    fit_history = train.demand[:, :config.validation_start]
    validation_actual = train.demand[:, config.validation_start:]
    validation_path = moving_average_path(fit_history, validation_actual, config.moving_average_window)
    test_path = moving_average_path(train.demand, test.demand, config.moving_average_window)
    evaluator = InventoryEvaluator(lead_time=config.lead_time)
    validation_demand = _wide(train.item_ids, validation_actual)
    comparison_rows: list[dict] = []
    item_trajectories = _existing_item_trajectories(args.item_id)
    for model_name, specification in METHODS.items():
        if model_name not in requested:
            continue
        print(f'\n===== {model_name} =====', flush=True)
        model_start = time.perf_counter()
        candidate_rows: list[dict] = []
        best: tuple[tuple, dict] | None = None
        for parameter in specification['candidates']:
            robust_model = fit_robust_buffer(fit_history, specification['method'], parameter, service_level=config.service_level, window=config.moving_average_window)
            validation_targets = robust_model.targets(validation_path.forecasts)
            initial = pd.DataFrame({'item_id': train.item_ids, 'initial_inventory': validation_targets[:, 0]})
            validation_result = evaluator.evaluate_dynamic_base_stock(validation_demand, initial, _wide(train.item_ids, validation_targets[:, 1:]))
            candidate = {specification['parameter_name']: parameter, 'buffer_multiplier': robust_model.buffer_multiplier, 'robust_constraint_gap': robust_model.robust_constraint_gap, 'validation_fill_rate': validation_result.summary['total_fill_rate'], 'validation_average_total_inventory': validation_result.summary['average_total_inventory'], 'validation_total_turnover': validation_result.summary['total_turnover'], 'validation_constraint_met': validation_result.summary['total_fill_rate'] >= config.service_level}
            candidate_rows.append(candidate)
            key = _selection_key(validation_result.summary, parameter, config.service_level)
            if best is None or key < best[0]:
                best = (key, candidate)
            print(f"{specification['parameter_name']}={parameter:g}, k={robust_model.buffer_multiplier:.4f}, 验证Fill Rate={validation_result.summary['total_fill_rate']:.4%}", flush=True)
        assert best is not None
        selected = best[1]
        selected_parameter = float(selected[specification['parameter_name']])
        final_model = fit_robust_buffer(train.demand, specification['method'], selected_parameter, service_level=config.service_level, window=config.moving_average_window)
        test_targets_all = final_model.targets(test_path.forecasts)
        test_targets = test_targets_all[:, 1:]
        model_dir = DRO_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        test_result = evaluator.evaluate_dynamic_base_stock(_wide(train.item_ids, test.demand), pd.DataFrame({'item_id': train.item_ids, 'initial_inventory': test.initial_inventory}), _wide(train.item_ids, test_targets), output_path=model_dir / 'test_per_sku_metrics.csv', trajectory_output_path=model_dir / 'test_trajectory.csv', model_name=model_name, forecasts_df=_wide(train.item_ids, test_path.forecasts[:, :-1]))
        pd.DataFrame(candidate_rows).to_csv(model_dir / 'validation_candidates.csv', index=False)
        _wide(train.item_ids, test_path.forecasts[:, :-1]).to_csv(model_dir / 'test_forecasts.csv', index=False)
        _wide(train.item_ids, test_targets).to_csv(model_dir / 'test_order_up_to.csv', index=False)
        pd.DataFrame({'item_id': train.item_ids, 'mean_error': final_model.mean_error, 'residual_scale': final_model.residual_scale, 'safety_buffer': final_model.safety_buffer}).to_csv(model_dir / 'robust_buffer.csv', index=False)
        runtime = time.perf_counter() - model_start
        row = {'model': model_name, 'family': 'dro', 'parameters': json.dumps({'forecast': 'moving_average_12', 'ambiguity_set': specification['method'], specification['parameter_name']: selected_parameter, 'buffer_multiplier': final_model.buffer_multiplier}, ensure_ascii=False), 'selected_z': np.nan, 'robust_parameter': selected_parameter, 'buffer_multiplier': final_model.buffer_multiplier, 'training_rows': int(train.demand.shape[0] * final_model.n_residual_periods), 'validation_wape': _wape(validation_actual, validation_path.forecasts[:, :-1]), 'test_wape': _wape(test.demand, test_path.forecasts[:, :-1]), 'validation_fill_rate': selected['validation_fill_rate'], 'validation_average_total_inventory': selected['validation_average_total_inventory'], 'test_fill_rate': test_result.summary['total_fill_rate'], 'test_constraint_met': test_result.summary['total_fill_rate'] >= config.service_level, 'test_average_total_inventory': test_result.summary['average_total_inventory'], 'test_total_turnover': test_result.summary['total_turnover'], 'test_total_lost_sales': test_result.summary['total_lost_sales'], 'test_sku_95_share': test_result.summary['sku_service_level_95_share'], 'runtime_seconds': runtime}
        comparison_rows.append(row)
        summary = {'result': row, 'config': asdict(config), 'ambiguity_method': specification, 'selected_validation_candidate': selected, 'final_robust_model': {'buffer_multiplier': final_model.buffer_multiplier, 'robust_constraint_gap': final_model.robust_constraint_gap, 'n_residual_periods': final_model.n_residual_periods}, 'test': test_result.summary}
        (model_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        trajectory = _save_item_trajectory(model_name, args.item_id)
        item_trajectories[model_name] = trajectory
        print(f"测试Fill Rate={test_result.summary['total_fill_rate']:.4%}, 平均总库存={test_result.summary['average_total_inventory']:.0f}", flush=True)
    rows_by_model = {row['model']: row for row in comparison_rows}
    for summary_path in DRO_DIR.glob('*/summary.json'):
        payload = json.loads(summary_path.read_text(encoding='utf-8'))
        if 'result' in payload:
            previous = payload['result']
            rows_by_model.setdefault(previous['model'], previous)
    comparison = pd.DataFrame(rows_by_model.values()).sort_values(['test_constraint_met', 'test_average_total_inventory'], ascending=[False, True])
    comparison.to_csv(DRO_DIR / 'model_comparison.csv', index=False)
    combined = _update_all_model_comparison(comparison)
    _plot_summary(combined, OUTPUT_DIR / 'all_model_comparison.png')
    _write_focused_comparison(combined)
    if item_trajectories:
        plot_model_comparison(item_trajectories, args.item_id, OUTPUT_DIR / 'item_comparisons' / f'item_{args.item_id}_all_models_with_dro_comparison.png')
    print('\nDRO 实验完成', flush=True)
    print(comparison[['model', 'robust_parameter', 'buffer_multiplier', 'test_fill_rate', 'test_average_total_inventory', 'test_total_turnover']].to_string(index=False), flush=True)
    print(f'总耗时: {time.perf_counter() - start:.2f} 秒', flush=True)

def _selection_key(summary: dict, parameter: float, service_level: float) -> tuple:
    feasible = summary['total_fill_rate'] >= service_level
    return (not feasible, summary['average_total_inventory'] if feasible else -summary['total_fill_rate'], parameter)

def _wide(item_ids: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    matrix = np.asarray(values, dtype=float)
    frame = pd.DataFrame(matrix, columns=[f'period_{index}' for index in range(1, matrix.shape[1] + 1)])
    frame.insert(0, 'item_id', item_ids)
    return frame

def _wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denominator = float(np.asarray(actual).sum())
    return float(np.abs(np.asarray(actual) - np.asarray(forecast)).sum() / denominator) if denominator else 0.0

def _save_item_trajectory(model_name: str, item_id: str) -> pd.DataFrame:
    source = DRO_DIR / model_name / 'test_trajectory.csv'
    frame = pd.read_csv(source)
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
    for source in sorted(root.glob(f'*/item_{item_id}_test_trajectory.csv')):
        result[source.parent.name] = pd.read_csv(source).drop(columns=['model', 'item_id'], errors='ignore')
    return result

def _update_all_model_comparison(dro_comparison: pd.DataFrame) -> pd.DataFrame:
    destination = OUTPUT_DIR / 'all_model_comparison.csv'
    existing = pd.read_csv(destination) if destination.exists() else pd.DataFrame()
    if not existing.empty:
        existing = existing.loc[existing.get('family', '') != 'dro'].copy()
    combined = pd.concat([existing, dro_comparison], ignore_index=True, sort=False)
    combined['test_constraint_met'] = combined['test_fill_rate'] >= 0.95
    combined = combined.drop_duplicates('model', keep='last').sort_values(['test_constraint_met', 'test_average_total_inventory'], ascending=[False, True])
    combined.to_csv(destination, index=False)
    return combined

def _write_focused_comparison(combined: pd.DataFrame) -> None:
    existing_names = {'baseline', 'moving_average_12', 'lightgbm', 'xgboost', 'catboost'}
    focused = combined.loc[combined['model'].isin(existing_names) | (combined['family'] == 'dro')].copy()
    focused.to_csv(DRO_DIR / 'dro_vs_existing_comparison.csv', index=False)
    _plot_summary(focused, DRO_DIR / 'dro_vs_existing_comparison.png')

def _plot_summary(comparison: pd.DataFrame, destination: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'baseline': '#6B7280', 'time_series': '#4C78A8', 'machine_learning': '#59A14F', 'dro': '#E15759'}
    figure, axis = plt.subplots(figsize=(12, 7.5))
    for family, group in comparison.groupby('family', dropna=False):
        axis.scatter(group['test_average_total_inventory'], group['test_fill_rate'] * 100, s=92, color=colors.get(str(family), '#9C755F'), edgecolor='white', linewidth=0.8, label=str(family), zorder=3)
        for row in group.itertuples(index=False):
            axis.annotate(row.model, (row.test_average_total_inventory, row.test_fill_rate * 100), xytext=(5, 5), textcoords='offset points', fontsize=7.5)
    axis.axhline(95, color='#B45309', linestyle='--', linewidth=1.4, label='95% constraint')
    axis.set_xscale('log')
    axis.set_xlabel('Average total inventory (log scale; lower is better)')
    axis.set_ylabel('Test total Fill Rate (%)')
    axis.set_title('All replenishment models: inventory versus service', loc='left', fontweight='bold')
    axis.grid(color='#D9DEE3', linewidth=0.7, alpha=0.8)
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(figure)

DRO_DIR=OUT

if __name__=='__main__':
    from common.cli import start
    start('dro_chi_square', globals(), default_protocol='legacy')
