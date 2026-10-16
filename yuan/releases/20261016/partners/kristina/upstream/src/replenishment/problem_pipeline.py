"""problem_based_methods 的无泄漏训练与最终评价流水线。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .data import load_replenishment_workbook, validate_train_test_boundary
from .problem_based_methods import ProblemBasedConfig, ProblemBasedMethods
from .problem_evaluation import EvaluationResult, InventoryEvaluator


@dataclass(frozen=True)
class ProblemBasedExperimentResult:
    """正式实验的模型摘要、测试结果和输出路径。"""

    model_summary: dict[str, object]
    test_result: EvaluationResult
    output_files: dict[str, Path]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """读取YAML配置。"""

    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("配置文件根节点必须是键值映射。")
    return config


def run_problem_based_experiment(
    *,
    train_path: str | Path,
    test_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> ProblemBasedExperimentResult:
    """先在train中完成校准，冻结后才读取test并评价一次。"""

    config = load_yaml_config(config_path)
    data_config = _required_mapping(config, "data")
    method_config = _required_mapping(config, "problem_based_methods")

    # 第一阶段只读取train，test不会进入参数估计或z选择。
    train = load_replenishment_workbook(
        train_path,
        expected_periods=int(data_config.get("expected_train_periods", 60)),
    )
    model = ProblemBasedMethods(
        ProblemBasedConfig(
            target_service_level=float(method_config["target_service_level"]),
            lead_time=int(method_config["lead_time"]),
            review_period=int(method_config["review_period"]),
            planning_horizon=int(method_config["planning_horizon"]),
            validation_periods=int(method_config["validation_periods"]),
            z_min=float(method_config["z_min"]),
            z_max=float(method_config["z_max"]),
            z_step=float(method_config["z_step"]),
            integer_quantity=bool(method_config.get("integer_quantity", True)),
        )
    ).fit(train.demand)
    artifacts = model.get_artifacts()

    # 第二阶段参数已经冻结，test仅用于4期和28期的最终评价。
    test = load_replenishment_workbook(
        test_path,
        expected_periods=int(data_config.get("expected_test_periods", 28)),
    )
    validate_train_test_boundary(train, test)
    evaluator = InventoryEvaluator(
        lead_time=model.config.lead_time,
        planning_horizon=model.config.planning_horizon,
    )
    test_result = evaluator.evaluate(test.demand, test.item_info, artifacts.parameters)

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    output_files = {
        "per_item_metrics": destination / "problem_based_methods_per_item_metrics.csv",
        "summary": destination / "problem_based_methods_summary.json",
    }
    test_result.per_item.to_csv(output_files["per_item_metrics"], index=False)

    model_summary = model.get_summary()
    report = {
        "data_usage": {
            "train": {
                "file": train.source_path.name,
                "sha256": train.sha256,
                "n_items": train.n_items,
                "n_periods": train.n_periods,
                "usage": "需求分布估计、内部验证和安全系数选择",
            },
            "test": {
                "file": test.source_path.name,
                "sha256": test.sha256,
                "n_items": test.n_items,
                "n_periods": test.n_periods,
                "usage": "冻结策略后的前4期与完整28期评价",
            },
            "leakage_guard": (
                "fit接口只接收train；模型与z冻结后才读取test；"
                "4期计划只使用train估计的需求分布，不读取test未来需求。"
            ),
        },
        "model": model_summary,
        "test": test_result.summary,
    }
    with output_files["summary"].open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)

    return ProblemBasedExperimentResult(model_summary, test_result, output_files)


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    """取得必需配置分组。"""

    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"配置缺少 {key!r} 分组或该分组不是键值映射。")
    return value
