"""common.policy_state: actual model/processing implementation, not a legacy runner alias.

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
import numpy as np

def rolling_reference(demand: np.ndarray, window: int=12) -> np.ndarray:
    """Return the trailing demand mean available after every period."""
    values = np.asarray(demand, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError('demand 必须是非空二维数组。')
    if window < 1:
        raise ValueError('window 必须 >= 1。')
    cumulative = np.cumsum(values, axis=1)
    result = np.empty_like(values, dtype=float)
    for period in range(values.shape[1]):
        start = max(0, period + 1 - window)
        total = cumulative[:, period]
        if start:
            total = total - cumulative[:, start - 1]
        result[:, period] = total / (period + 1 - start)
    return np.maximum(result, 1e-06)

def action_levels(reference: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Map dimensionless actions to nonnegative integer inventory targets."""
    scale = np.asarray(reference, dtype=float).reshape(-1)
    multipliers = np.asarray(actions, dtype=float).reshape(-1)
    if scale.size == 0 or multipliers.size < 2:
        raise ValueError('reference 不能为空，actions 至少需要两个候选。')
    if (scale < 0).any() or (multipliers < 0).any():
        raise ValueError('reference 和 actions 不能为负。')
    return np.ceil(scale[:, None] * multipliers[None, :])

def rollout_action_costs(ending_inventory: np.ndarray, future_demand: np.ndarray, reference_path: np.ndarray, actions: np.ndarray, *, shortage_penalty: float, holding_cost: float=1.0, discount: float=1.0) -> np.ndarray:
    """Calculate realized finite-horizon cost for every candidate action.

    A candidate multiplier is held fixed within the short rollout, while its
    physical target is rescaled by the trailing-demand reference at each step.
    The first replenishment arrives before the first future demand.
    """
    inventory = np.asarray(ending_inventory, dtype=float).reshape(-1)
    demand = np.asarray(future_demand, dtype=float)
    reference = np.asarray(reference_path, dtype=float)
    multipliers = np.asarray(actions, dtype=float).reshape(-1)
    if demand.ndim != 2 or reference.shape != demand.shape:
        raise ValueError('future_demand 和 reference_path 必须是同形二维数组。')
    if len(inventory) != len(demand):
        raise ValueError('ending_inventory 与未来需求行数不一致。')
    if shortage_penalty < 0 or holding_cost < 0 or (not 0 < discount <= 1):
        raise ValueError('成本系数必须非负，discount 必须在 (0, 1] 内。')
    if (inventory < 0).any() or (demand < 0).any() or (reference < 0).any():
        raise ValueError('库存、需求和reference不能为负。')
    state = np.repeat(inventory[:, None], len(multipliers), axis=1)
    costs = np.zeros_like(state)
    for step in range(demand.shape[1]):
        target = np.ceil(reference[:, step, None] * multipliers[None, :])
        available = np.maximum(state, target)
        sales = np.minimum(available, demand[:, step, None])
        state = available - sales
        lost = demand[:, step, None] - sales
        weight = discount ** step
        costs += weight * (holding_cost * state + shortage_penalty * lost)
    return costs.astype(np.float32)

def augment_policy_features(history_features: np.ndarray, *, reference: np.ndarray, opening_inventory: np.ndarray, ending_inventory: np.ndarray, current_demand: np.ndarray, current_lost_sales: np.ndarray) -> np.ndarray:
    """Append the decision state observed when the replenishment order is set."""
    history = np.asarray(history_features, dtype=np.float32)
    arrays = [np.asarray(value, dtype=float).reshape(-1) for value in (reference, opening_inventory, ending_inventory, current_demand, current_lost_sales)]
    if history.ndim != 2 or any((len(value) != len(history) for value in arrays)):
        raise ValueError('历史特征与库存状态的行数必须一致。')
    scale = np.maximum(arrays[0], 1e-06)
    state = np.column_stack([*arrays, arrays[1] / scale, arrays[2] / scale, arrays[3] / scale])
    result = np.column_stack([history, state]).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('策略特征包含非有限值。')
    return result
POLICY_STATE_FEATURE_NAMES = ['reference_demand', 'opening_inventory', 'ending_inventory', 'current_demand', 'current_lost_sales', 'opening_to_reference', 'ending_to_reference', 'demand_to_reference']
from dataclasses import dataclass
from typing import Callable
CURRENT_INVENTORY_FEATURE = 'current_ending_inventory'

@dataclass(frozen=True)
class InventoryAwareRollout:
    """Targets and actions produced by a sequential inventory-aware policy."""
    initial_inventory: np.ndarray
    targets: np.ndarray
    actions: np.ndarray
    ending_inventory: np.ndarray

def behavior_ending_inventory(demand: np.ndarray, initial_inventory: np.ndarray, *, reference_window: int=12, behavior_multiplier: float=2.0) -> np.ndarray:
    """Generate leakage-free training states with a fixed historical policy.

    The state in column ``t`` is inventory after demand in period ``t`` and
    before the replenishment decision made at the end of that period.
    """
    values = np.asarray(demand, dtype=float)
    initial = np.asarray(initial_inventory, dtype=float).reshape(-1)
    if values.ndim != 2 or len(initial) != values.shape[0]:
        raise ValueError('demand 与 initial_inventory 的 SKU 数量不一致。')
    if behavior_multiplier < 0:
        raise ValueError('behavior_multiplier 不能为负。')
    reference = rolling_reference(values, reference_window)
    ending = np.zeros_like(values, dtype=float)
    on_hand = initial.copy()
    arrival = np.zeros(values.shape[0], dtype=float)
    for period in range(values.shape[1]):
        available = on_hand + arrival
        sales = np.minimum(available, values[:, period])
        on_hand = available - sales
        ending[:, period] = on_hand
        target = np.ceil(reference[:, period] * behavior_multiplier)
        arrival = np.maximum(0.0, target - on_hand)
    return ending

def inventory_for_target_periods(ending_inventory: np.ndarray, target_periods: np.ndarray) -> np.ndarray:
    """Return period-major inventory observed just before each target period."""
    ending = np.asarray(ending_inventory, dtype=float)
    periods = np.asarray(target_periods, dtype=int)
    if ending.ndim != 2 or (periods <= 0).any() or (periods > ending.shape[1]).any():
        raise ValueError('target_periods 必须能映射到上一期的期末库存。')
    return ending[:, periods - 1].T.reshape(-1).astype(np.float32)

def append_current_inventory(features: np.ndarray, current_inventory: np.ndarray) -> np.ndarray:
    """Append current ending inventory as one observable decision feature."""
    matrix = np.asarray(features, dtype=np.float32)
    inventory = np.asarray(current_inventory, dtype=float).reshape(-1)
    if matrix.ndim != 2 or len(matrix) != len(inventory):
        raise ValueError('特征与当期库存的行数不一致。')
    result = np.column_stack([matrix, inventory]).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('加入库存后的特征包含非有限值。')
    return result

def rollout_inventory_aware_policy(demand: np.ndarray, initial_inventory: np.ndarray, decision_features: np.ndarray, selector: Callable[[int, np.ndarray], tuple[np.ndarray, np.ndarray]], *, include_pre_period_decision: bool) -> InventoryAwareRollout:
    """Roll a policy forward while feeding its own current inventory back in.

    ``decision_features`` contains one block per target period in period-major
    order.  It must contain ``T + 1`` blocks: the block before the first demand
    period plus one block for every end-of-period decision.  Test evaluation
    sets ``include_pre_period_decision=False`` because its supplied initial
    inventory is authoritative; validation uses ``True`` to bridge from the
    preceding training period.
    """
    values = np.asarray(demand, dtype=float)
    initial = np.asarray(initial_inventory, dtype=float).reshape(-1)
    feature_matrix = np.asarray(decision_features, dtype=np.float32)
    if values.ndim != 2 or len(initial) != values.shape[0]:
        raise ValueError('demand 与 initial_inventory 的 SKU 数量不一致。')
    n_skus, n_periods = values.shape
    if len(feature_matrix) != (n_periods + 1) * n_skus:
        raise ValueError('滚动特征必须包含 T+1 个周期块。')

    def block(index: int, inventory: np.ndarray) -> np.ndarray:
        base = feature_matrix[index * n_skus:(index + 1) * n_skus]
        return append_current_inventory(base, inventory)
    on_hand = initial.copy()
    action_columns: list[np.ndarray] = []
    if include_pre_period_decision:
        selected, target = selector(0, block(0, on_hand))
        selected = np.asarray(selected, dtype=int).reshape(-1)
        target = np.asarray(target, dtype=float).reshape(-1)
        if len(selected) != n_skus or len(target) != n_skus:
            raise ValueError('selector 返回的动作或目标长度不正确。')
        action_columns.append(selected)
        on_hand = np.maximum(on_hand, target)
    evaluation_initial = on_hand.copy()
    targets = np.zeros((n_skus, n_periods), dtype=float)
    ending = np.zeros_like(targets)
    for period in range(n_periods):
        sales = np.minimum(on_hand, values[:, period])
        current_ending = on_hand - sales
        ending[:, period] = current_ending
        selected, target = selector(period + 1, block(period + 1, current_ending))
        selected = np.asarray(selected, dtype=int).reshape(-1)
        target = np.asarray(target, dtype=float).reshape(-1)
        if len(selected) != n_skus or len(target) != n_skus:
            raise ValueError('selector 返回的动作或目标长度不正确。')
        action_columns.append(selected)
        targets[:, period] = target
        on_hand = np.maximum(current_ending, target)
    return InventoryAwareRollout(initial_inventory=evaluation_initial, targets=targets, actions=np.column_stack(action_columns), ending_inventory=ending)
