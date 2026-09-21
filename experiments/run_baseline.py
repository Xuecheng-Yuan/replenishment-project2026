"""运行均值–方差 baseline，并在模型冻结后评估测试集。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from replenishment.pipeline import run_baseline_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """解析命令行路径，默认使用项目标准目录。"""

    parser = argparse.ArgumentParser(description="训练并评估补货 baseline。")
    parser.add_argument(
        "--train",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "replenishment_train.xlsx",
        help="60 期训练工作簿。",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "replenishment_test.xlsx",
        help="仅用于最终评价的测试工作簿。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline.yaml",
        help="baseline 配置文件。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tables",
        help="CSV 与 JSON 结果目录。",
    )
    return parser.parse_args()


def main() -> None:
    """按固定顺序执行训练、冻结和测试评估。"""

    args = parse_args()
    start = time.perf_counter()
    print("第一步：读取 60 期 train，并在 train 内部完成 baseline 校准。", flush=True)
    print("第二步：冻结模型后读取 test，只计算服务水平和周转率。", flush=True)

    result = run_baseline_experiment(
        train_path=args.train,
        test_path=args.test,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    model = result.model_summary
    test = result.test_result.summary

    print("\nBaseline 运行完成")
    print(f"train 拟合期数          : {model['fit_periods']}")
    print(f"train 验证期数          : {model['validation_periods']}")
    print(f"选择的安全系数 z        : {float(model['selected_z']):.2f}")
    print(
        "train 验证总体服务水平 : "
        f"{float(model['validation']['overall_service_level']):.4%}"
    )
    print(f"test 总体服务水平       : {float(test['overall_service_level']):.4%}")
    print(f"test 总体周转率         : {float(test['overall_turnover']):.4f}")
    print(
        "test 平均总期末库存    : "
        f"{float(test['average_total_ending_inventory']):,.2f}"
    )
    print(
        "test SKU 95% 达标比例  : "
        f"{float(test['item_service_level_95_share']):.4%}"
    )
    category_metrics = test.get("category_metrics", {})
    for category in ("爆品", "长尾品"):
        if category in category_metrics:
            metrics = category_metrics[category]
            print(f"test {category}服务水平      : {float(metrics['service_level']):.4%}")
            print(f"test {category}周转率        : {float(metrics['turnover']):.4f}")
    print(f"输出目录                : {args.output_dir.resolve()}")
    print(f"总耗时                  : {time.perf_counter() - start:.2f} 秒")


if __name__ == "__main__":
    main()
