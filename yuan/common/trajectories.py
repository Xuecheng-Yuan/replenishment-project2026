"""common.trajectories: actual model/processing implementation, not a legacy runner alias.

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
from pathlib import Path
from collections.abc import Mapping
import numpy as np
import pandas as pd

def simulate_item_trajectory(demands: np.ndarray, initial_inventory: float, s: float, S: float, lead_time: int=1) -> tuple[pd.DataFrame, dict]:
    """Return period-level inventory and replenishment records for one SKU."""
    demands = np.asarray(demands, dtype=float)
    if demands.size == 0 or np.isnan(demands).any() or (demands < 0).any():
        raise ValueError('demands 必须是非空、非负且无缺失的一维数组。')
    if initial_inventory < 0 or lead_time < 1 or S < s:
        raise ValueError('必须满足初始库存非负、lead_time >= 1 且 S >= s。')
    on_hand = float(initial_inventory)
    pipeline = [0.0] * lead_time
    records: list[dict] = []
    cumulative_demand = 0.0
    cumulative_sales = 0.0
    for period, demand in enumerate(demands, start=1):
        opening_inventory = on_hand
        arrival = pipeline.pop(0)
        available_inventory = opening_inventory + arrival
        sales = min(available_inventory, float(demand))
        lost_sales = float(demand) - sales
        ending_inventory = available_inventory - sales
        pipeline_before_order = float(sum(pipeline))
        inventory_position = ending_inventory + pipeline_before_order
        order_quantity = max(0.0, S - inventory_position) if inventory_position <= s else 0.0
        pipeline.append(order_quantity)
        on_hand = ending_inventory
        cumulative_demand += float(demand)
        cumulative_sales += sales
        records.append({'period': period, 'opening_inventory': opening_inventory, 'arrival': arrival, 'available_inventory': available_inventory, 'demand': float(demand), 'sales': sales, 'lost_sales': lost_sales, 'ending_inventory': ending_inventory, 'pipeline_before_order': pipeline_before_order, 'inventory_position_before_order': inventory_position, 'order_quantity': order_quantity, 'pipeline_after_order': float(sum(pipeline)), 'cumulative_fill_rate': cumulative_sales / cumulative_demand if cumulative_demand > 0 else 1.0})
    trajectory = pd.DataFrame(records)
    total_demand = float(trajectory['demand'].sum())
    total_sales = float(trajectory['sales'].sum())
    average_inventory = float(trajectory['ending_inventory'].mean())
    summary = {'fill_rate': total_sales / total_demand if total_demand > 0 else 1.0, 'average_ending_inventory': average_inventory, 'total_demand': total_demand, 'total_sales': total_sales, 'total_lost_sales': float(trajectory['lost_sales'].sum()), 'total_order_quantity': float(trajectory['order_quantity'].sum()), 'order_count': int((trajectory['order_quantity'] > 0).sum())}
    return (trajectory, summary)

def plot_item_trajectory(trajectory: pd.DataFrame, item_id: object, s: float, S: float, output_path: str | Path) -> Path:
    """Save one readable timeline chart using a single quantity axis."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = trajectory['period'].to_numpy()
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(x - width / 2, trajectory['demand'], width, color='#D8E5EF', label='Demand')
    ax.bar(x + width / 2, trajectory['order_quantity'], width, color='#D59A2F', label='Order quantity')
    ax.step(x, trajectory['available_inventory'], where='mid', color='#7896AA', linewidth=1.4, label='Available inventory')
    ax.step(x, trajectory['ending_inventory'], where='mid', color='#2F6B9A', linewidth=2.2, label='Ending inventory')
    if s == S:
        ax.axhline(S, color='#303942', linestyle='--', linewidth=1.2, label=f's = S = {S:g}')
    else:
        ax.axhline(s, color='#303942', linestyle='--', linewidth=1.2, label=f's = {s:g}')
        ax.axhline(S, color='#303942', linestyle=':', linewidth=1.2, label=f'S = {S:g}')
    shortage = trajectory['lost_sales'] > 0
    if shortage.any():
        ax.scatter(x[shortage], np.zeros(shortage.sum()), marker='x', s=55, color='#C76B3C', label='Stockout period', zorder=5)
    fig.suptitle(f'Test inventory and replenishment trajectory — item {item_id}', x=0.1, y=0.98, ha='left', fontsize=15, fontweight='bold')
    ax.set_title('Bars show demand/orders; lines show inventory states. All values use the same quantity unit.', loc='left', fontsize=9, color='#5F6B75', pad=12)
    ax.set_xlabel('Test period')
    ax.set_ylabel('Quantity')
    ax.set_xlim(0.3, len(trajectory) + 0.7)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', color='#D9DEE3', linewidth=0.7, alpha=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    fig.subplots_adjust(top=0.86, bottom=0.19)
    fig.savefig(path, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path

def combine_model_trajectories(trajectories: Mapping[str, pd.DataFrame], item_id: object) -> pd.DataFrame:
    """Validate and combine model trajectories into one long-format table."""
    if not trajectories:
        raise ValueError('至少需要一个模型轨迹。')
    required = {'period', 'demand', 'sales', 'lost_sales', 'ending_inventory', 'order_quantity'}
    combined: list[pd.DataFrame] = []
    reference_period: np.ndarray | None = None
    reference_demand: np.ndarray | None = None
    for model_name, frame in trajectories.items():
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f'模型 {model_name} 的轨迹缺少字段: {sorted(missing)}')
        current = frame.copy().sort_values('period').reset_index(drop=True)
        period = current['period'].to_numpy()
        demand = current['demand'].to_numpy(dtype=float)
        if reference_period is None:
            reference_period, reference_demand = (period, demand)
        elif not np.array_equal(period, reference_period) or not np.allclose(demand, reference_demand):
            raise ValueError(f'模型 {model_name} 的周期或需求与其他模型不一致。')
        current.insert(0, 'item_id', item_id)
        current.insert(0, 'model', str(model_name))
        combined.append(current)
    return pd.concat(combined, ignore_index=True)

def plot_model_comparison(trajectories: Mapping[str, pd.DataFrame], item_id: object, output_path: str | Path) -> Path:
    """Plot inventory and replenishment for several models in one figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    combined = combine_model_trajectories(trajectories, item_id)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = ['#2F6B9A', '#D59A2F', '#C76B3C', '#708C55', '#A65A78', '#6E63A8', '#2F8F83', '#8A6A45', '#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']
    figure, (inventory_ax, order_ax) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.08})
    first = next(iter(trajectories.values())).sort_values('period')
    inventory_ax.bar(first['period'], first['demand'], color='#E1E6EA', width=0.78, label='Demand (shared)', zorder=1)
    for index, (model_name, frame) in enumerate(trajectories.items()):
        data = frame.sort_values('period')
        color = colors[index % len(colors)]
        inventory_ax.plot(data['period'], data['ending_inventory'], color=color, linewidth=2.1, marker='o', markersize=3.2, label=f'{model_name} inventory', zorder=3)
        order_ax.plot(data['period'], data['order_quantity'], color=color, linewidth=1.8, linestyle='--', marker='o', markersize=3.0, label=f'{model_name} order')
    figure.suptitle(f'Test inventory and replenishment by model — item {item_id}', x=0.09, y=0.98, ha='left', fontsize=15, fontweight='bold')
    inventory_ax.set_title('Color identifies the model; demand is identical for every model.', loc='left', fontsize=9, color='#5F6B75', pad=10)
    inventory_ax.set_ylabel('Ending inventory')
    order_ax.set_ylabel('Order quantity')
    order_ax.set_xlabel('Test period')
    for axis in (inventory_ax, order_ax):
        axis.set_ylim(bottom=0)
        axis.grid(axis='y', color='#D9DEE3', linewidth=0.7, alpha=0.8)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(frameon=False, ncol=3, loc='upper left')
    figure.subplots_adjust(top=0.88)
    figure.savefig(path, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(figure)
    return path
