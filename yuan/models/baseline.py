"""models.baseline: actual model/processing implementation, not a legacy runner alias.

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
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from common.evaluation import InventoryEvaluator

def _period_columns(df: pd.DataFrame) -> list[str]:
    columns = [column for column in df.columns if str(column).startswith('period_')]
    try:
        return sorted(columns, key=lambda column: int(str(column).split('_')[-1]))
    except ValueError as exc:
        raise ValueError('周期列必须采用 period_1、period_2 ... 的格式。') from exc

@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for the mean-variance baseline."""
    service_level: float = 0.95
    lead_time: int = 1
    validation_fraction: float = 0.3
    z_min: float = 0.0
    z_max: float = 4.0
    z_step: float = 0.05
    integer_quantity: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.service_level <= 1:
            raise ValueError('service_level 必须在 (0, 1] 内。')
        if self.lead_time < 1:
            raise ValueError('lead_time 必须 >= 1。')
        if not 0 < self.validation_fraction < 1:
            raise ValueError('validation_fraction 必须在 (0, 1) 内。')
        if self.z_min < 0 or self.z_max < self.z_min or self.z_step <= 0:
            raise ValueError('z 的搜索范围不合法。')

class MeanVarianceBaseline:
    """Fit per-SKU base-stock levels from demand mean and variance."""

    def __init__(self, config: BaselineConfig | None=None) -> None:
        self.config = config or BaselineConfig()
        self.selected_z_: float | None = None
        self.statistics_: pd.DataFrame | None = None
        self.parameters_: pd.DataFrame | None = None
        self.validation_summary_: dict | None = None

    def fit(self, demand_df: pd.DataFrame) -> 'MeanVarianceBaseline':
        self._validate_demand(demand_df)
        period_cols = _period_columns(demand_df)
        split = int(np.floor(len(period_cols) * (1 - self.config.validation_fraction)))
        split = min(max(split, 1), len(period_cols) - 1)
        fit_cols, validation_cols = (period_cols[:split], period_cols[split:])
        calibration_stats = self._statistics(demand_df, fit_cols)
        validation_demand = demand_df[['item_id', *validation_cols]].copy()
        evaluator = InventoryEvaluator(lead_time=self.config.lead_time)
        selected_z = self.config.z_max
        selected_summary: dict | None = None
        for z in self._z_candidates():
            parameters = self._make_parameters(calibration_stats, z)
            initial_inventory = parameters[['item_id', 'S']].rename(columns={'S': 'initial_inventory'})
            result = evaluator.evaluate(validation_demand, initial_inventory, parameters)
            selected_z, selected_summary = (float(z), result.summary)
            if result.summary['total_fill_rate'] >= self.config.service_level:
                break
        self.selected_z_ = selected_z
        self.validation_summary_ = selected_summary
        self.statistics_ = self._statistics(demand_df, period_cols)
        self.parameters_ = self._make_parameters(self.statistics_, selected_z)
        return self

    def get_parameters(self) -> pd.DataFrame:
        self._check_fitted()
        return self.parameters_.copy()

    def get_statistics(self) -> pd.DataFrame:
        self._check_fitted()
        return self.statistics_.copy()

    def get_summary(self) -> dict:
        self._check_fitted()
        return {'selected_z': self.selected_z_, 'validation': dict(self.validation_summary_ or {}), 'config': asdict(self.config)}

    def _make_parameters(self, statistics: pd.DataFrame, z: float) -> pd.DataFrame:
        level = (statistics['mean_demand'] * self.config.lead_time + z * statistics['std_demand'] * np.sqrt(self.config.lead_time)).clip(lower=0)
        if self.config.integer_quantity:
            level = np.ceil(level)
        return pd.DataFrame({'item_id': statistics['item_id'], 's': level.astype(float), 'S': level.astype(float), 'z': float(z)})

    def _z_candidates(self) -> np.ndarray:
        count = int(np.floor((self.config.z_max - self.config.z_min) / self.config.z_step))
        candidates = self.config.z_min + np.arange(count + 1) * self.config.z_step
        if candidates[-1] < self.config.z_max:
            candidates = np.append(candidates, self.config.z_max)
        return candidates

    @staticmethod
    def _statistics(demand_df: pd.DataFrame, period_cols: list[str]) -> pd.DataFrame:
        values = demand_df[period_cols].to_numpy(dtype=float)
        return pd.DataFrame({'item_id': demand_df['item_id'].to_numpy(), 'mean_demand': values.mean(axis=1), 'variance_demand': values.var(axis=1, ddof=1), 'std_demand': values.std(axis=1, ddof=1)})

    @staticmethod
    def _validate_demand(demand_df: pd.DataFrame) -> None:
        if 'item_id' not in demand_df:
            raise ValueError('demand_df 缺少 item_id。')
        if demand_df['item_id'].isna().any() or demand_df['item_id'].duplicated().any():
            raise ValueError('item_id 不能为空或重复。')
        period_cols = _period_columns(demand_df)
        if len(period_cols) < 2:
            raise ValueError('baseline 至少需要两个历史周期。')
        values = demand_df[period_cols].apply(pd.to_numeric, errors='coerce')
        if values.isna().any().any() or (values < 0).any().any():
            raise ValueError('需求必须是非负数且不能缺失。')

    def _check_fitted(self) -> None:
        if self.parameters_ is None:
            raise RuntimeError('模型尚未 fit。')
import json
import time
from common.data import load_historical_demand

def train_legacy() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('读取 train/test 数据……')
    train_demand, _ = load_historical_demand(DATA_DIR / 'replenishment_train.xlsx')
    test_demand, test_info = load_historical_demand(DATA_DIR / 'replenishment_test.xlsx')
    config = BaselineConfig(service_level=0.95, lead_time=1, validation_fraction=0.3)
    model = MeanVarianceBaseline(config).fit(train_demand)
    parameters = model.get_parameters()
    parameters.to_csv(OUTPUT_DIR / 'baseline_parameters.csv', index=False)
    model.get_statistics().to_csv(OUTPUT_DIR / 'baseline_demand_statistics.csv', index=False)
    evaluator = InventoryEvaluator(lead_time=config.lead_time)
    result = evaluator.evaluate(test_demand, test_info, parameters, output_path=OUTPUT_DIR / 'baseline_test_per_sku_metrics.csv')
    report = {'model': model.get_summary(), 'test': result.summary}
    with (OUTPUT_DIR / 'baseline_summary.json').open('w', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print('\nBaseline 完成')
    print(f'选择的 z              : {model.selected_z_:.2f}')
    print(f"验证集总体 Fill Rate  : {model.validation_summary_['total_fill_rate']:.4%}")
    print(f"测试集总体 Fill Rate  : {result.summary['total_fill_rate']:.4%}")
    print(f"测试集总体周转率      : {result.summary['total_turnover']:.4f}")
    print(f"测试集平均总库存      : {result.summary['average_total_inventory']:.2f}")
    print(f"达到 95% 的 SKU 比例  : {result.summary['sku_service_level_95_share']:.4%}")
    print(f'结果目录              : {OUTPUT_DIR}')
    print(f'总耗时                : {time.perf_counter() - start:.2f} 秒')

if __name__=='__main__':
    from common.cli import start
    start('baseline', globals(), default_protocol='legacy')
