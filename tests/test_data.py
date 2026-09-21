"""数据边界与 train/test 结构检查测试。"""

from pathlib import Path

import pandas as pd
import pytest

from replenishment.data import ReplenishmentWorkbook, validate_train_test_boundary


def _workbook(item_ids: list[int]) -> ReplenishmentWorkbook:
    demand = pd.DataFrame({"item_id": item_ids, "period_1": [1] * len(item_ids)})
    item_info = pd.DataFrame(
        {
            "item_id": item_ids,
            "category": ["长尾品"] * len(item_ids),
            "initial_inventory": [1] * len(item_ids),
        }
    )
    return ReplenishmentWorkbook(
        demand=demand,
        item_info=item_info,
        period_columns=("period_1",),
        source_path=Path("dummy.xlsx"),
        sha256="dummy",
    )


def test_train_test_must_have_same_item_set() -> None:
    """防止评价阶段因 SKU 缺失而静默丢行。"""

    with pytest.raises(ValueError, match="SKU 集合不一致"):
        validate_train_test_boundary(_workbook([1, 2]), _workbook([1, 3]))


def test_train_test_category_must_match() -> None:
    """同一 SKU 的分类变化必须报错，避免分组结果不可比。"""

    train = _workbook([1, 2])
    test = _workbook([1, 2])
    test.item_info.loc[test.item_info["item_id"] == 2, "category"] = "爆品"

    with pytest.raises(ValueError, match="category 不一致"):
        validate_train_test_boundary(train, test)
