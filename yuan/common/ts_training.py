"""common.ts_training: actual model/processing implementation, not a legacy runner alias.

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
OUT=ROOT/'results/generated/time_series'
WORK=OUT
TABLES=OUT/'图表'
OLD=ROOT/'results/generated/ready_immediate'
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from common.evaluation import InventoryEvaluator
from models.baseline import MeanVarianceBaseline
from common.ts_support import ForecastPath

@dataclass(frozen=True)
class ExperimentConfig:
    service_level: float = 0.95
    validation_fraction: float = 0.3
    z_min: float = 0.0
    z_max: float = 8.0
    z_step: float = 0.05
    lead_time: int = 1

def _run_forecast_model(model_name: str, model_parameters: dict, validation_path: ForecastPath, test_path: ForecastPath, train_demand: pd.DataFrame, test_demand: pd.DataFrame, test_info: pd.DataFrame, item_ids: np.ndarray, split: int, config: ExperimentConfig, evaluator: InventoryEvaluator, item_id: str) -> tuple[dict, pd.DataFrame]:
    selected_z, validation_summary = _calibrate_z(evaluator, train_demand, item_ids, split, validation_path, config)
    test_targets = _targets(test_path.forecasts[:, 1:], test_path.residual_scale, selected_z)
    test_result = _evaluate_test(model_name, evaluator, test_demand, test_info, item_ids, test_targets, test_path.forecasts[:, :-1])
    validation_actual = train_demand[_period_columns(train_demand)[split:]].to_numpy(dtype=float)
    test_actual = test_demand[_period_columns(test_demand)].to_numpy(dtype=float)
    validation_wape = _wape(validation_actual, validation_path.forecasts[:, :-1])
    test_wape = _wape(test_actual, test_path.forecasts[:, :-1])
    model_output = MODEL_DIR / model_name
    summary = {'model': model_name, 'model_parameters': model_parameters, 'selected_z': selected_z, 'validation_wape': validation_wape, 'test_wape': test_wape, 'validation': validation_summary, 'test': test_result.summary, 'experiment_config': asdict(config)}
    (model_output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    _wide(item_ids, test_path.forecasts[:, :-1]).to_csv(model_output / 'test_forecasts.csv', index=False)
    _wide(item_ids, test_targets).to_csv(model_output / 'test_order_up_to.csv', index=False)
    item_frame = _save_item_trajectory(model_name, item_id)
    return (_comparison_row(model_name, json.dumps(model_parameters, ensure_ascii=False), selected_z, validation_summary, test_result.summary, validation_wape, test_wape), item_frame)

def _calibrate_z(evaluator: InventoryEvaluator, train_demand: pd.DataFrame, item_ids: np.ndarray, split: int, path: ForecastPath, config: ExperimentConfig) -> tuple[float, dict]:
    validation_cols = _period_columns(train_demand)[split:]
    validation_demand = train_demand[['item_id', *validation_cols]].copy()
    validation_demand.columns = ['item_id', *[f'period_{i}' for i in range(1, len(validation_cols) + 1)]]
    candidates = np.arange(config.z_min, config.z_max + config.z_step / 2, config.z_step)
    cache: dict[int, dict] = {}

    def evaluate_index(index: int) -> dict:
        if index not in cache:
            z = float(candidates[index])
            initial = pd.DataFrame({'item_id': item_ids, 'initial_inventory': _targets(path.forecasts[:, [0]], path.residual_scale, z)[:, 0]})
            decision_targets = _wide(item_ids, _targets(path.forecasts[:, 1:], path.residual_scale, z))
            cache[index] = evaluator.evaluate_dynamic_base_stock(validation_demand, initial, decision_targets).summary
        return cache[index]
    low, high = (0, len(candidates) - 1)
    if evaluate_index(high)['total_fill_rate'] < config.service_level:
        return (float(candidates[high]), evaluate_index(high))
    while low < high:
        middle = (low + high) // 2
        if evaluate_index(middle)['total_fill_rate'] >= config.service_level:
            high = middle
        else:
            low = middle + 1
    return (float(candidates[low]), evaluate_index(low))

def _evaluate_test(model_name: str, evaluator: InventoryEvaluator, test_demand: pd.DataFrame, test_info: pd.DataFrame, item_ids: np.ndarray, targets: np.ndarray, forecasts: np.ndarray | None):
    model_output = MODEL_DIR / model_name
    model_output.mkdir(parents=True, exist_ok=True)
    return evaluator.evaluate_dynamic_base_stock(test_demand, test_info, _wide(item_ids, targets), output_path=model_output / 'test_per_sku_metrics.csv', trajectory_output_path=model_output / 'test_trajectory.csv', model_name=model_name, forecasts_df=_wide(item_ids, forecasts) if forecasts is not None else None)

def _save_item_trajectory(model_name: str, item_id: str) -> pd.DataFrame:
    source = MODEL_DIR / model_name / 'test_trajectory.csv'
    frame = pd.read_csv(source)
    matched = frame.loc[frame['item_id'].astype(str) == str(item_id)].copy()
    if matched.empty:
        raise ValueError(f'找不到 item_id={item_id}。')
    output_dir = OUTPUT_DIR / 'item_diagnostics' / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    matched.to_csv(output_dir / f'item_{item_id}_test_trajectory.csv', index=False)
    return matched.drop(columns=['model', 'item_id'], errors='ignore')

def _save_baseline_compatibility(baseline: MeanVarianceBaseline, parameters: pd.DataFrame, result) -> None:
    parameters.to_csv(OUTPUT_DIR / 'baseline_parameters.csv', index=False)
    baseline.get_statistics().to_csv(OUTPUT_DIR / 'baseline_demand_statistics.csv', index=False)
    result.per_sku.to_csv(OUTPUT_DIR / 'baseline_test_per_sku_metrics.csv', index=False)
    report = {'model': baseline.get_summary(), 'test': result.summary}
    (OUTPUT_DIR / 'baseline_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

def _wide(item_ids: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    frame = pd.DataFrame(values, columns=[f'period_{i}' for i in range(1, values.shape[1] + 1)])
    frame.insert(0, 'item_id', item_ids)
    return frame

def _targets(forecasts: np.ndarray, scale: np.ndarray, z: float) -> np.ndarray:
    return np.ceil(np.clip(forecasts + z * scale[:, None], 0, None))

def _period_columns(frame: pd.DataFrame) -> list[str]:
    return sorted([column for column in frame if str(column).startswith('period_')], key=lambda column: int(str(column).split('_')[-1]))

def _wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denominator = float(actual.sum())
    return float(np.abs(actual - forecast).sum() / denominator) if denominator else 0.0

def _selection_key(summary: dict, z: float) -> tuple[bool, float, float]:
    feasible = summary['total_fill_rate'] >= ExperimentConfig.service_level
    return (not feasible, summary['average_total_inventory'] if feasible else -summary['total_fill_rate'], z)

def _comparison_row(model: str, parameters: str, selected_z: float, validation: dict, test: dict, validation_wape: float, test_wape: float) -> dict:
    return {'model': model, 'parameters': parameters, 'selected_z': selected_z, 'validation_wape': validation_wape, 'test_wape': test_wape, 'validation_fill_rate': validation.get('total_fill_rate'), 'validation_average_total_inventory': validation.get('average_total_inventory'), 'test_fill_rate': test['total_fill_rate'], 'test_constraint_met': test['total_fill_rate'] >= ExperimentConfig.service_level, 'test_average_total_inventory': test['average_total_inventory'], 'test_total_turnover': test['total_turnover'], 'test_total_lost_sales': test['total_lost_sales'], 'test_sku_95_share': test['sku_service_level_95_share']}

def _plot_model_summary(comparison: pd.DataFrame, output_path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(10.5, 6.5))
    feasible = comparison['test_fill_rate'] >= ExperimentConfig.service_level
    colors = np.where(feasible, '#2F6B9A', '#B9C1C8')
    axis.scatter(comparison['test_average_total_inventory'], comparison['test_fill_rate'] * 100, s=90, c=colors, edgecolor='white', linewidth=0.8, zorder=3)
    for row in comparison.itertuples(index=False):
        axis.annotate(row.model, (row.test_average_total_inventory, row.test_fill_rate * 100), xytext=(6, 5), textcoords='offset points', fontsize=8.5)
    axis.axhline(95, color='#C76B3C', linestyle='--', linewidth=1.4, label='95% constraint')
    axis.set_xscale('log')
    axis.set_xlabel('Average total inventory (log scale; lower is better)')
    axis.set_ylabel('Test total Fill Rate (%)')
    axis.set_title('Inventory and service comparison', loc='left', fontsize=15, fontweight='bold')
    axis.grid(color='#D9DEE3', linewidth=0.7, alpha=0.8)
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(figure)

MODEL_DIR=OUT
