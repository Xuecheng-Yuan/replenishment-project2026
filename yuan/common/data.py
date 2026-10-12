"""common.data: actual model/processing implementation, not a legacy runner alias.

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
import re
from pathlib import Path
import pandas as pd

def load_historical_demand(file_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return wide demand data and the initial inventory table."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f'找不到数据文件: {path.resolve()}')
    period_frames: dict[int, pd.DataFrame] = {}
    item_info: pd.DataFrame | None = None
    with pd.ExcelFile(path) as workbook:
        for sheet_name in workbook.sheet_names:
            frame = pd.read_excel(workbook, sheet_name=sheet_name)
            frame.columns = [str(column).strip() for column in frame.columns]
            demand_cols = [c for c in frame if re.fullmatch('demand_\\d+', c)]
            if 'item_id' not in frame or len(demand_cols) != 1:
                raise ValueError(f"sheet '{sheet_name}' 的 item_id 或 demand_x 列不合法。")
            if frame['item_id'].isna().any() or frame['item_id'].duplicated().any():
                raise ValueError(f"sheet '{sheet_name}' 的 item_id 为空或重复。")
            demand_col = demand_cols[0]
            period = int(demand_col.split('_')[-1])
            if period in period_frames:
                raise ValueError(f'period_{period} 重复。')
            period_frames[period] = frame[['item_id', demand_col]].rename(columns={demand_col: f'period_{period}'})
            if period == 1:
                if not {'item_id', 'initial_inventory'}.issubset(frame.columns):
                    raise ValueError('Period_1 缺少 initial_inventory。')
                info_cols = [c for c in ['item_id', 'category', 'initial_inventory'] if c in frame]
                item_info = frame[info_cols].copy()
    if not period_frames or item_info is None:
        raise ValueError('工作簿中没有完整的周期数据。')
    expected = set(range(1, max(period_frames) + 1))
    if set(period_frames) != expected:
        raise ValueError(f'缺少周期: {sorted(expected - set(period_frames))}')
    demand = period_frames[1]
    for period in range(2, max(period_frames) + 1):
        demand = demand.merge(period_frames[period], on='item_id', how='outer', validate='one_to_one')
    demand_cols = [f'period_{period}' for period in range(1, max(period_frames) + 1)]
    demand[demand_cols] = demand[demand_cols].apply(pd.to_numeric, errors='coerce')
    if demand[demand_cols].isna().any().any() or (demand[demand_cols] < 0).any().any():
        raise ValueError('需求数据包含缺失值、非数值或负数。')
    item_info['initial_inventory'] = pd.to_numeric(item_info['initial_inventory'], errors='coerce')
    if item_info['initial_inventory'].isna().any() or (item_info['initial_inventory'] < 0).any():
        raise ValueError('initial_inventory 必须是非负数且不能缺失。')
    return (demand.sort_values('item_id').reset_index(drop=True), item_info.sort_values('item_id').reset_index(drop=True))
