"""补货数据的读取、结构校验与文件指纹计算。

本模块只负责把 Excel 工作簿转换成明确的数据结构，不做模型训练。
这样可以把“数据读取”和“模型拟合”分开，便于审计训练集与测试集的使用边界。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


_DEMAND_COLUMN_PATTERN = re.compile(r"demand_(\d+)")


@dataclass(frozen=True)
class ReplenishmentWorkbook:
    """一个补货工作簿解析后的标准形式。

    Attributes:
        demand: 宽表需求数据，每行一个 SKU，每列一个时期。
        item_info: SKU 静态信息，至少包含 item_id 和 initial_inventory。
        period_columns: 已按时间顺序排列的需求列名。
        source_path: 数据文件的绝对路径，仅用于审计和报错。
        sha256: 原始文件指纹，用于确认实验使用了哪一份数据。
    """

    demand: pd.DataFrame
    item_info: pd.DataFrame
    period_columns: tuple[str, ...]
    source_path: Path
    sha256: str

    @property
    def n_items(self) -> int:
        """返回 SKU 数量。"""

        return len(self.demand)

    @property
    def n_periods(self) -> int:
        """返回时期数量。"""

        return len(self.period_columns)


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """分块计算文件 SHA-256，避免一次把大文件全部读入内存。"""

    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_replenishment_workbook(
    path: str | Path,
    *,
    expected_periods: int | None = None,
) -> ReplenishmentWorkbook:
    """读取多工作表补货数据，并执行高风险数据质量检查。

    每个工作表必须表示一个时期，并且恰好包含一个 ``demand_t`` 字段。
    Period 1 还必须包含初始库存。函数只提取 baseline 所需的需求和库存；
    ``x1`` 至 ``x20`` 等当期特征不会进入 baseline，防止无意使用未来信息。
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"找不到数据文件：{source}")

    period_frames: dict[int, pd.DataFrame] = {}
    item_info: pd.DataFrame | None = None

    # 按工作表逐期读取。工作表物理顺序不作为时间顺序，时间由 demand_t 决定。
    with pd.ExcelFile(source) as workbook:
        for sheet_name in workbook.sheet_names:
            frame = pd.read_excel(workbook, sheet_name=sheet_name)
            frame.columns = [str(column).strip() for column in frame.columns]

            if "item_id" not in frame.columns:
                raise ValueError(f"工作表 {sheet_name!r} 缺少 item_id。")

            demand_columns = [
                column
                for column in frame.columns
                if _DEMAND_COLUMN_PATTERN.fullmatch(str(column))
            ]
            if len(demand_columns) != 1:
                raise ValueError(
                    f"工作表 {sheet_name!r} 必须且只能有一个 demand_t 字段，"
                    f"实际找到 {demand_columns}。"
                )

            if frame["item_id"].isna().any() or frame["item_id"].duplicated().any():
                raise ValueError(f"工作表 {sheet_name!r} 的 item_id 存在空值或重复。")

            demand_column = demand_columns[0]
            period = int(_DEMAND_COLUMN_PATTERN.fullmatch(demand_column).group(1))  # type: ignore[union-attr]
            if period in period_frames:
                raise ValueError(f"时期 {period} 在工作簿中重复出现。")

            period_frames[period] = frame[["item_id", demand_column]].rename(
                columns={demand_column: f"period_{period}"}
            )

            # 初始库存只从第一期读取，避免不同工作表的重复字段产生歧义。
            if period == 1:
                required = {"item_id", "category", "initial_inventory"}
                if not required.issubset(frame.columns):
                    raise ValueError("第一期工作表缺少 category 或 initial_inventory。")
                info_columns = [
                    column
                    for column in ("item_id", "category", "initial_inventory")
                    if column in frame.columns
                ]
                item_info = frame[info_columns].copy()

    if not period_frames or item_info is None:
        raise ValueError("工作簿没有形成完整的时期数据或缺少第一期库存信息。")

    last_period = max(period_frames)
    expected_sequence = set(range(1, last_period + 1))
    missing_periods = sorted(expected_sequence - set(period_frames))
    if missing_periods:
        raise ValueError(f"时期编号不连续，缺少：{missing_periods}")
    if expected_periods is not None and last_period != expected_periods:
        raise ValueError(f"预期 {expected_periods} 期，实际读取到 {last_period} 期。")

    # 外连接能把不同期缺失的 SKU 暴露为缺失值，而不是静默丢掉样本。
    demand = period_frames[1]
    for period in range(2, last_period + 1):
        demand = demand.merge(
            period_frames[period],
            on="item_id",
            how="outer",
            validate="one_to_one",
        )

    period_columns = tuple(f"period_{period}" for period in range(1, last_period + 1))
    demand[list(period_columns)] = demand[list(period_columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    if demand["item_id"].isna().any() or demand["item_id"].duplicated().any():
        raise ValueError("合并后的需求数据存在空 item_id 或重复 item_id。")
    if demand[list(period_columns)].isna().any().any():
        raise ValueError("需求数据存在缺失值、非数值，或某些 SKU 在部分时期缺失。")
    if (demand[list(period_columns)] < 0).any().any():
        raise ValueError("需求不能为负数。")

    item_info["initial_inventory"] = pd.to_numeric(
        item_info["initial_inventory"], errors="coerce"
    )
    if item_info["initial_inventory"].isna().any():
        raise ValueError("initial_inventory 存在缺失值或非数值。")
    if (item_info["initial_inventory"] < 0).any():
        raise ValueError("initial_inventory 不能为负数。")
    if item_info["category"].isna().any():
        raise ValueError("category 不能为空。")
    item_info["category"] = item_info["category"].astype(str).str.strip()
    if (item_info["category"] == "").any():
        raise ValueError("category 不能为空字符串。")

    demand = demand.sort_values("item_id").reset_index(drop=True)
    item_info = item_info.sort_values("item_id").reset_index(drop=True)
    if set(demand["item_id"]) != set(item_info["item_id"]):
        raise ValueError("需求表与初始库存表的 SKU 集合不一致。")

    return ReplenishmentWorkbook(
        demand=demand,
        item_info=item_info,
        period_columns=period_columns,
        source_path=source,
        sha256=file_sha256(source),
    )


def validate_train_test_boundary(
    train: ReplenishmentWorkbook,
    test: ReplenishmentWorkbook,
) -> None:
    """检查 train/test 的结构兼容性，不把 test 内容传入模型。"""

    train_items = set(train.demand["item_id"])
    test_items = set(test.demand["item_id"])
    if train_items != test_items:
        missing_in_test = len(train_items - test_items)
        unseen_in_test = len(test_items - train_items)
        raise ValueError(
            "train/test 的 SKU 集合不一致："
            f"test 缺少 {missing_in_test} 个，test 新增 {unseen_in_test} 个。"
        )

    # 同一个 SKU 的类别必须在 train/test 中保持不变，否则分组指标不可比较。
    category_comparison = train.item_info[["item_id", "category"]].merge(
        test.item_info[["item_id", "category"]],
        on="item_id",
        how="inner",
        suffixes=("_train", "_test"),
        validate="one_to_one",
    )
    mismatches = category_comparison[
        category_comparison["category_train"] != category_comparison["category_test"]
    ]
    if not mismatches.empty:
        raise ValueError(f"train/test 有 {len(mismatches)} 个 SKU 的 category 不一致。")
