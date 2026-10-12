"""common.evaluation: actual model/processing implementation, not a legacy runner alias.

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
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass
class EvaluationResult:
    per_sku: pd.DataFrame
    summary: dict

class InventoryEvaluator:
    """Simulate an (s, S) lost-sales policy and calculate Fill Rate metrics."""

    def __init__(self, lead_time: int=1) -> None:
        if lead_time < 1:
            raise ValueError('lead_time 必须 >= 1。')
        self.lead_time = lead_time

    def evaluate(self, demand_df: pd.DataFrame, item_info: pd.DataFrame, parameters: pd.DataFrame, output_path: str | Path | None=None) -> EvaluationResult:
        data = self._align_inputs(demand_df, item_info, parameters)
        period_cols = sorted([c for c in demand_df if str(c).startswith('period_')], key=lambda c: int(str(c).split('_')[-1]))
        demands = data[period_cols].to_numpy(dtype=float)
        on_hand = data['initial_inventory'].to_numpy(dtype=float, copy=True)
        reorder_point = data['s'].to_numpy(dtype=float)
        order_up_to = data['S'].to_numpy(dtype=float)
        pipeline = np.zeros((self.lead_time, len(data)), dtype=float)
        total_demand = demands.sum(axis=1)
        total_sales = np.zeros(len(data), dtype=float)
        inventory_sum = np.zeros(len(data), dtype=float)
        order_count = np.zeros(len(data), dtype=int)
        total_order_quantity = np.zeros(len(data), dtype=float)
        total_inventory_by_period: list[float] = []
        for period in range(demands.shape[1]):
            on_hand += pipeline[0]
            pipeline[:-1] = pipeline[1:]
            pipeline[-1] = 0.0
            sales = np.minimum(on_hand, demands[:, period])
            on_hand -= sales
            inventory_position = on_hand + pipeline.sum(axis=0)
            order_quantity = np.where(inventory_position <= reorder_point, np.maximum(0.0, order_up_to - inventory_position), 0.0)
            pipeline[-1] += order_quantity
            total_sales += sales
            inventory_sum += on_hand
            order_count += order_quantity > 0
            total_order_quantity += order_quantity
            total_inventory_by_period.append(float(on_hand.sum()))
        average_inventory = inventory_sum / demands.shape[1]
        fill_rate = np.divide(total_sales, total_demand, out=np.ones_like(total_sales), where=total_demand > 0)
        turnover = np.divide(total_sales, average_inventory, out=np.where(total_sales > 0, np.inf, 0.0), where=average_inventory > 0)
        per_sku = pd.DataFrame({'item_id': data['item_id'], 'fill_rate': fill_rate, 'average_inventory': average_inventory, 'turnover': turnover, 'total_demand': total_demand, 'total_sales': total_sales, 'lost_sales': total_demand - total_sales, 'order_count': order_count, 'total_order_quantity': total_order_quantity})
        total_demand_all = float(total_demand.sum())
        total_sales_all = float(total_sales.sum())
        average_total_inventory = float(np.mean(total_inventory_by_period))
        summary = {'n_skus': len(data), 'n_periods': demands.shape[1], 'total_fill_rate': total_sales_all / total_demand_all if total_demand_all else 1.0, 'total_turnover': total_sales_all / average_total_inventory if average_total_inventory > 0 else float('inf') if total_sales_all > 0 else 0.0, 'average_total_inventory': average_total_inventory, 'total_demand': total_demand_all, 'total_sales': total_sales_all, 'total_lost_sales': total_demand_all - total_sales_all, 'sku_service_level_95_share': float((fill_rate >= 0.95).mean())}
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            per_sku.to_csv(path, index=False)
        return EvaluationResult(per_sku=per_sku, summary=summary)

    def evaluate_dynamic_base_stock(self, demand_df: pd.DataFrame, item_info: pd.DataFrame, order_up_to_df: pd.DataFrame, output_path: str | Path | None=None, trajectory_output_path: str | Path | None=None, model_name: str='model', forecasts_df: pd.DataFrame | None=None) -> EvaluationResult:
        """Evaluate time-varying ``s_t = S_t`` targets.

        ``period_t`` in ``order_up_to_df`` is the target used for the order
        placed after period t. With lead time one, that order arrives before
        period t+1. The first test-period inventory therefore always comes
        from ``item_info`` and is never back-filled from a forecast.
        """
        if self.lead_time != 1:
            raise ValueError('动态 time-series 评价目前只支持 lead_time=1。')
        data, period_cols = self._align_dynamic_inputs(demand_df, item_info, order_up_to_df, forecasts_df)
        demands = data[period_cols].to_numpy(dtype=float)
        targets = data[[f'target_{column}' for column in period_cols]].to_numpy(dtype=float)
        forecasts = data[[f'forecast_{column}' for column in period_cols]].to_numpy(dtype=float) if forecasts_df is not None else None
        on_hand = data['initial_inventory'].to_numpy(dtype=float, copy=True)
        arrivals = np.zeros(len(data), dtype=float)
        total_demand = demands.sum(axis=1)
        total_sales = np.zeros(len(data), dtype=float)
        inventory_sum = np.zeros(len(data), dtype=float)
        order_count = np.zeros(len(data), dtype=int)
        total_order_quantity = np.zeros(len(data), dtype=float)
        total_inventory_by_period: list[float] = []
        trajectory_parts: list[pd.DataFrame] = []
        for period_index, period_col in enumerate(period_cols):
            opening_inventory = on_hand.copy()
            arrival = arrivals.copy()
            on_hand += arrival
            sales = np.minimum(on_hand, demands[:, period_index])
            on_hand -= sales
            lost_sales = demands[:, period_index] - sales
            target = targets[:, period_index]
            order_quantity = np.maximum(0.0, target - on_hand)
            arrivals = order_quantity
            total_sales += sales
            inventory_sum += on_hand
            order_count += order_quantity > 0
            total_order_quantity += order_quantity
            total_inventory_by_period.append(float(on_hand.sum()))
            if trajectory_output_path is not None:
                part = pd.DataFrame({'model': model_name, 'item_id': data['item_id'].to_numpy(), 'period': period_index + 1, 'opening_inventory': opening_inventory, 'arrival': arrival, 'demand': demands[:, period_index], 'sales': sales, 'lost_sales': lost_sales, 'ending_inventory': on_hand, 'order_quantity': order_quantity, 'order_up_to': target})
                if forecasts is not None:
                    part['forecast'] = forecasts[:, period_index]
                trajectory_parts.append(part)
        result = self._build_result(data['item_id'], total_demand, total_sales, inventory_sum, order_count, total_order_quantity, total_inventory_by_period, demands.shape[1])
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            result.per_sku.to_csv(path, index=False)
        if trajectory_output_path is not None:
            path = Path(trajectory_output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(trajectory_parts, ignore_index=True).to_csv(path, index=False)
        return result

    @staticmethod
    def _build_result(item_ids: pd.Series, total_demand: np.ndarray, total_sales: np.ndarray, inventory_sum: np.ndarray, order_count: np.ndarray, total_order_quantity: np.ndarray, total_inventory_by_period: list[float], n_periods: int) -> EvaluationResult:
        average_inventory = inventory_sum / n_periods
        fill_rate = np.divide(total_sales, total_demand, out=np.ones_like(total_sales), where=total_demand > 0)
        turnover = np.divide(total_sales, average_inventory, out=np.where(total_sales > 0, np.inf, 0.0), where=average_inventory > 0)
        per_sku = pd.DataFrame({'item_id': item_ids.to_numpy(), 'fill_rate': fill_rate, 'average_inventory': average_inventory, 'turnover': turnover, 'total_demand': total_demand, 'total_sales': total_sales, 'lost_sales': total_demand - total_sales, 'order_count': order_count, 'total_order_quantity': total_order_quantity})
        total_demand_all = float(total_demand.sum())
        total_sales_all = float(total_sales.sum())
        average_total_inventory = float(np.mean(total_inventory_by_period))
        summary = {'n_skus': len(item_ids), 'n_periods': n_periods, 'total_fill_rate': total_sales_all / total_demand_all if total_demand_all else 1.0, 'total_turnover': total_sales_all / average_total_inventory if average_total_inventory > 0 else float('inf') if total_sales_all > 0 else 0.0, 'average_total_inventory': average_total_inventory, 'total_demand': total_demand_all, 'total_sales': total_sales_all, 'total_lost_sales': total_demand_all - total_sales_all, 'sku_service_level_95_share': float((fill_rate >= 0.95).mean())}
        return EvaluationResult(per_sku=per_sku, summary=summary)

    @staticmethod
    def _align_inputs(demand_df: pd.DataFrame, item_info: pd.DataFrame, parameters: pd.DataFrame) -> pd.DataFrame:
        required = {'demand_df': (demand_df, {'item_id'}), 'item_info': (item_info, {'item_id', 'initial_inventory'}), 'parameters': (parameters, {'item_id', 's', 'S'})}
        for name, (frame, columns) in required.items():
            if not columns.issubset(frame.columns):
                raise ValueError(f'{name} 缺少字段: {sorted(columns - set(frame.columns))}')
            if frame['item_id'].isna().any() or frame['item_id'].duplicated().any():
                raise ValueError(f'{name} 的 item_id 不能为空或重复。')
        period_cols = [c for c in demand_df if str(c).startswith('period_')]
        data = demand_df.merge(item_info[['item_id', 'initial_inventory']], on='item_id', how='left', validate='one_to_one').merge(parameters[['item_id', 's', 'S']], on='item_id', how='left', validate='one_to_one')
        numeric_cols = [*period_cols, 'initial_inventory', 's', 'S']
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
        if data[numeric_cols].isna().any().any():
            raise ValueError('需求、初始库存或策略参数存在缺失/非数值。')
        if (data[[*period_cols, 'initial_inventory']] < 0).any().any():
            raise ValueError('需求和初始库存不能为负数。')
        if (data['S'] < data['s']).any():
            raise ValueError('必须满足 S >= s。')
        return data

    @staticmethod
    def _align_dynamic_inputs(demand_df: pd.DataFrame, item_info: pd.DataFrame, order_up_to_df: pd.DataFrame, forecasts_df: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
        period_cols = sorted([column for column in demand_df if str(column).startswith('period_')], key=lambda column: int(str(column).split('_')[-1]))
        required = {'demand_df': (demand_df, {'item_id', *period_cols}), 'item_info': (item_info, {'item_id', 'initial_inventory'}), 'order_up_to_df': (order_up_to_df, {'item_id', *period_cols})}
        if forecasts_df is not None:
            required['forecasts_df'] = (forecasts_df, {'item_id', *period_cols})
        for name, (frame, columns) in required.items():
            if not columns.issubset(frame.columns):
                raise ValueError(f'{name} 缺少字段: {sorted(columns - set(frame.columns))}')
            if frame['item_id'].isna().any() or frame['item_id'].duplicated().any():
                raise ValueError(f'{name} 的 item_id 不能为空或重复。')
        targets = order_up_to_df[['item_id', *period_cols]].rename(columns={column: f'target_{column}' for column in period_cols})
        data = demand_df[['item_id', *period_cols]].merge(item_info[['item_id', 'initial_inventory']], on='item_id', how='left', validate='one_to_one').merge(targets, on='item_id', how='left', validate='one_to_one')
        if forecasts_df is not None:
            forecasts = forecasts_df[['item_id', *period_cols]].rename(columns={column: f'forecast_{column}' for column in period_cols})
            data = data.merge(forecasts, on='item_id', how='left', validate='one_to_one')
        numeric_cols = [column for column in data if column != 'item_id']
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
        if data[numeric_cols].isna().any().any() or (data[numeric_cols] < 0).any().any():
            raise ValueError('需求、库存、预测或动态目标必须非负且不能缺失。')
        return (data, period_cols)
