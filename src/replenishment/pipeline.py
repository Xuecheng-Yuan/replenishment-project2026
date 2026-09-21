"""无测试泄漏的 baseline 训练与最终评估流水线。"""

from __future__ import annotations

import json
from itertools import product
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .data import load_replenishment_workbook, validate_train_test_boundary
from .evaluation import EvaluationResult, InventoryEvaluator
from .models import BaselineConfig, MeanVarianceBaseline
from .time_series import (
    build_targets,
    cold_start_forecast_path,
    simulate_dynamic_policy,
)


@dataclass(frozen=True)
class BaselineExperimentResult:
    """baseline 实验完成后返回给调用者的核心结果。"""

    model_summary: dict[str, object]
    test_result: EvaluationResult
    output_files: dict[str, Path]


@dataclass(frozen=True)
class TimeSeriesExperimentResult:
    """时间序列实验返回的最终模型和测试结果。"""

    model_summary: dict[str, object]
    test_result: EvaluationResult
    comparison: pd.DataFrame
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
                "file": train.source_path.name,
                "sha256": train.sha256,
                "n_items": train.n_items,
                "n_periods": train.n_periods,
                "usage": "模型估计、train 内部验证校准、最终参数重估",
            },
            "test": {
                "file": test.source_path.name,
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


def run_time_series_experiment(
    *,
    train_path: str | Path,
    test_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> TimeSeriesExperimentResult:
    """只用 train 选择时间序列策略，冻结后再读取 test。"""

    config = load_yaml_config(config_path)
    data_config = _required_mapping(config, "data")
    ts_config = _required_mapping(config, "time_series")
    train = load_replenishment_workbook(
        train_path,
        expected_periods=int(data_config.get("expected_train_periods", 60)),
    )
    categories = _aligned_categories(train.demand, train.item_info)
    train_values = train.demand[list(train.period_columns)].to_numpy(dtype=float)

    # 所有模型和安全系数都只在三个 train 滚动窗口中比较。
    comparison_rows: list[dict[str, object]] = []
    model_results: list[tuple[dict[str, object], dict[str, object]]] = []
    for model in ts_config["models"]:
        result = _validate_time_series_model(train_values, categories, model, ts_config)
        model_results.append((model, result))
        comparison_rows.append(
            {
                "model": model["id"],
                "parameters": json.dumps(model, ensure_ascii=False, sort_keys=True),
                "feasible_on_every_fold": result["feasible"],
                "validation_service_level": result["service_level"],
                "validation_worst_fold_service": result["worst_fold_service"],
                "validation_turnover": result["turnover"],
                "safety_factors": json.dumps(result["safety_factors"], ensure_ascii=False),
            }
        )

    feasible = [(model, result) for model, result in model_results if result["feasible"]]
    if not feasible:
        raise RuntimeError("没有时间序列候选在每个 train 验证折达到95%服务水平。")
    selected_model, selected = max(feasible, key=lambda item: float(item[1]["turnover"]))
    # 只有全部模型和安全系数已经冻结后，才读取 test。
    test = load_replenishment_workbook(
        test_path,
        expected_periods=int(data_config.get("expected_test_periods", 28)),
    )
    validate_train_test_boundary(train, test)
    test_values = test.demand[list(test.period_columns)].to_numpy(dtype=float)
    test_categories = _aligned_categories(test.demand, test.item_info)
    evaluator = InventoryEvaluator(lead_time=int(ts_config["lead_time"]))
    selected_test_result: EvaluationResult | None = None
    test_summaries: dict[str, dict[str, object]] = {}
    for model, validation in model_results:
        path = cold_start_forecast_path(
            train_values,
            test_values,
            test_categories,
            model,
            error_window=int(ts_config["error_window"]),
        )
        target_values = build_targets(path, test_categories, validation["safety_factors"])
        targets = pd.DataFrame(target_values, columns=test.period_columns)
        targets.insert(0, "item_id", test.demand["item_id"].to_numpy())
        result = evaluator.evaluate_dynamic(test.demand, test.item_info, targets)
        test_summaries[str(model["id"])] = result.summary
        if model["id"] == selected_model["id"]:
            selected_test_result = result
    assert selected_test_result is not None
    test_result = selected_test_result

    # test 列只用于透明报告，不参与上面的模型与安全系数选择。
    for row in comparison_rows:
        summary = test_summaries[str(row["model"])]
        row.update(
            {
                "test_service_level": summary["overall_service_level"],
                "test_constraint_met": float(summary["overall_service_level"]) >= 0.95,
                "test_turnover": summary["overall_turnover"],
                "test_average_inventory": summary["average_total_ending_inventory"],
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["feasible_on_every_fold", "validation_turnover"], ascending=[False, False]
    )

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    output_files = {
        "comparison": destination / "time_series_comparison.csv",
        "test_per_item_metrics": destination / "time_series_test_per_item_metrics.csv",
        "summary": destination / "time_series_summary.json",
    }
    comparison.to_csv(output_files["comparison"], index=False)
    test_result.per_item.to_csv(output_files["test_per_item_metrics"], index=False)
    model_summary = {
        "selected_model": selected_model,
        "selected_safety_factors": selected["safety_factors"],
        "validation": selected,
        "selection_rule": "每个验证折服务水平>=95%，可行方案中综合周转率最高",
    }
    report = {
        "data_usage": {
            "train": {
                "file": train.source_path.name,
                "sha256": train.sha256,
                "n_items": train.n_items,
                "n_periods": train.n_periods,
                "usage": "学习增长先验和误差比例、滚动验证、模型与安全系数选择",
            },
            "test": {
                "file": test.source_path.name,
                "sha256": test.sha256,
                "n_items": test.n_items,
                "n_periods": test.n_periods,
                "usage": "冻结策略后的在线滚动评价",
            },
            "leakage_guard": (
                "模型和安全系数在读取test前已冻结；train绝对水平和期末状态不传入test；"
                "每个test真实需求只更新下一期预测。"
            ),
        },
        "model": model_summary,
        "test": {
            **test_result.summary,
            "constraint_met": (
                float(test_result.summary["overall_service_level"])
                >= float(ts_config["target_service_level"])
            ),
        },
    }
    with output_files["summary"].open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
    return TimeSeriesExperimentResult(model_summary, test_result, comparison, output_files)


def _validate_time_series_model(
    train: np.ndarray,
    categories: np.ndarray,
    model: dict[str, object],
    config: dict[str, Any],
) -> dict[str, object]:
    """用相同的滚动窗口和库存目标比较一个预测模型。"""

    folds = []
    for fold in config["validation_folds"]:
        start = int(fold["history_periods"])
        stop = start + int(fold["validation_periods"])
        path = cold_start_forecast_path(
            train[:, :start],
            train[:, start:stop],
            categories,
            model,
            error_window=int(config["error_window"]),
        )
        folds.append((train[:, start:stop], path))

    values = np.arange(
        float(config["safety_min"]),
        float(config["safety_max"]) + float(config["safety_step"]) / 2,
        float(config["safety_step"]),
    )
    category_names = sorted(set(np.asarray(categories).astype(str)))
    best: dict[str, object] | None = None
    for combination in product(values, repeat=len(category_names)):
        factors = dict(zip(category_names, map(float, combination), strict=True))
        summaries = []
        for episode, path in folds:
            targets = build_targets(path, categories, factors)
            # 验证块第1期只用于冷启动；从第2期起评价可控的补货决策。
            summaries.append(simulate_dynamic_policy(episode[:, 1:], targets[:, 0], targets[:, 1:]))

        sales = sum(summary["total_sales"] for summary in summaries)
        demand_total = sum(summary["total_demand"] for summary in summaries)
        inventory_sum = sum(summary["inventory_sum"] for summary in summaries)
        n_periods = sum(episode.shape[1] - 1 for episode, _ in folds)
        candidate = {
            "feasible": all(
                summary["service_level"] >= float(config["target_service_level"])
                for summary in summaries
            ),
            "service_level": sales / demand_total,
            "worst_fold_service": min(summary["service_level"] for summary in summaries),
            "turnover": sales / (inventory_sum / n_periods),
            "safety_factors": factors,
            "folds": summaries,
        }
        if best is None or _validation_key(candidate) > _validation_key(best):
            best = candidate
    assert best is not None
    return best


def _validation_key(result: dict[str, object]) -> tuple[bool, float, float]:
    """优先可行性；可行时最大化周转率，不可行时最大化最差服务水平。"""

    feasible = bool(result["feasible"])
    primary = float(result["turnover"] if feasible else result["worst_fold_service"])
    return feasible, primary, float(result["service_level"])


def _aligned_categories(demand: pd.DataFrame, item_info: pd.DataFrame) -> np.ndarray:
    """按 demand 行顺序返回类别，避免依赖工作簿物理排序。"""

    aligned = demand[["item_id"]].merge(
        item_info[["item_id", "category"]], on="item_id", validate="one_to_one"
    )
    return aligned["category"].astype(str).to_numpy()


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    """读取必需配置分组并给出明确错误。"""

    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"配置缺少 {key!r} 分组或该分组不是键值映射。")
    return value
