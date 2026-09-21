"""无测试泄漏的 baseline 训练与最终评估流水线。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .data import load_replenishment_workbook, validate_train_test_boundary
from .evaluation import EvaluationResult, InventoryEvaluator
from .models import BaselineConfig, MeanVarianceBaseline


@dataclass(frozen=True)
class BaselineExperimentResult:
    """baseline 实验完成后返回给调用者的核心结果。"""

    model_summary: dict[str, object]
    test_result: EvaluationResult
    output_files: dict[str, Path]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置，并保证根节点是映射。"""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("配置文件根节点必须是键值映射。")
    return config


def run_baseline_experiment(
    *,
    train_path: str | Path,
    test_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> BaselineExperimentResult:
    """先用 train 完成全部训练，再读取 test 做一次最终评估。

    关键防泄漏设计：

    1. 先读取 train，并完成 z 校准和最终参数重估；
    2. 获取冻结的模型参数后，才读取 test；
    3. test 只传入评价器，不会传回 ``fit`` 或任何参数选择函数。
    """

    config = load_yaml_config(config_path)
    data_config = _required_mapping(config, "data")
    baseline_config = _required_mapping(config, "baseline")
    expected_train_periods = int(data_config.get("expected_train_periods", 60))
    expected_test_periods = int(data_config.get("expected_test_periods", 28))

    # 阶段一：只读取训练集。此时测试文件尚未打开。
    train = load_replenishment_workbook(
        train_path,
        expected_periods=expected_train_periods,
    )

    # 阶段二：只用 train 的前 42 期估计、后 18 期校准，再用完整 60 期重估。
    model = MeanVarianceBaseline(
        BaselineConfig(
            target_service_level=float(baseline_config["target_service_level"]),
            lead_time=int(baseline_config["lead_time"]),
            validation_periods=int(baseline_config["validation_periods"]),
            z_min=float(baseline_config["z_min"]),
            z_max=float(baseline_config["z_max"]),
            z_step=float(baseline_config["z_step"]),
            integer_quantity=bool(baseline_config.get("integer_quantity", True)),
        )
    ).fit(train.demand)
    artifacts = model.get_artifacts()

    # 阶段三：模型已经冻结后才读取测试集。下面没有任何重新拟合或调参调用。
    test = load_replenishment_workbook(
        test_path,
        expected_periods=expected_test_periods,
    )
    validate_train_test_boundary(train, test)

    evaluator = InventoryEvaluator(lead_time=model.config.lead_time)
    test_result = evaluator.evaluate(
        test.demand,
        test.item_info,
        artifacts.parameters,
        record_trajectory=False,
    )

    # 阶段四：保存可审计结果。输出中记录 train/test 指纹和各自用途。
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    output_files = {
        "parameters": destination / "baseline_parameters.csv",
        "train_statistics": destination / "baseline_train_statistics.csv",
        "test_per_item_metrics": destination / "baseline_test_per_item_metrics.csv",
        "summary": destination / "baseline_summary.json",
    }
    artifacts.parameters.to_csv(output_files["parameters"], index=False)
    artifacts.train_statistics.to_csv(output_files["train_statistics"], index=False)
    test_result.per_item.to_csv(output_files["test_per_item_metrics"], index=False)

    model_summary = model.get_summary()
    report = {
        "data_usage": {
            "train": {
                "path": str(train.source_path),
                "sha256": train.sha256,
                "n_items": train.n_items,
                "n_periods": train.n_periods,
                "usage": "模型估计、train 内部验证校准、最终参数重估",
            },
            "test": {
                "path": str(test.source_path),
                "sha256": test.sha256,
                "n_items": test.n_items,
                "n_periods": test.n_periods,
                "usage": "模型冻结后的最终服务水平与周转率评估",
            },
            "leakage_guard": (
                "test 在 baseline.fit 完成并取得冻结参数后才读取；"
                "test 数据未进入均值、标准差、安全系数或其他参数估计。"
            ),
        },
        "model": model_summary,
        "test": test_result.summary,
    }
    with output_files["summary"].open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)

    return BaselineExperimentResult(
        model_summary=model_summary,
        test_result=test_result,
        output_files=output_files,
    )


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    """读取必需配置分组并给出明确错误。"""

    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"配置缺少 {key!r} 分组或该分组不是键值映射。")
    return value

