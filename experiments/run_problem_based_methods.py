"""运行 problem_based_methods，并报告test前4期和完整28期结果。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from replenishment.problem_pipeline import run_problem_based_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="运行problem_based_methods补货基准。")
    parser.add_argument(
        "--train",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "replenishment_train.xlsx",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "replenishment_test.xlsx",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "problem_based_methods.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tables",
    )
    return parser.parse_args()


def main() -> None:
    """执行实验并打印最重要的指标。"""

    args = parse_args()
    start = time.perf_counter()
    print("第一步：仅用train估计需求分布并选择安全系数。", flush=True)
    print("第二步：冻结策略后，评价test前4期和完整28期。", flush=True)
    result = run_problem_based_experiment(
        train_path=args.train,
        test_path=args.test,
        config_path=args.config,
        output_dir=args.output_dir,
    )

    print("\nproblem_based_methods运行完成")
    print(f"选择的安全系数z：{float(result.model_summary['selected_z']):.2f}")
    for window, metrics in result.test_result.summary["windows"].items():
        print(f"\n评价窗口：{window}")
        print(f"库存/需求比：{float(metrics['inventory_demand_ratio']):.6f}")
        print(f"严格正库存服务率：{float(metrics['positive_inventory_service_rate']):.4%}")
        print(f"无缺货服务率：{float(metrics['no_stockout_service_rate']):.4%}")
        print(f"满足95%约束：{bool(metrics['constraint_met'])}")
    print(f"输出目录：{args.output_dir.resolve()}")
    print(f"总耗时：{time.perf_counter() - start:.2f}秒")


if __name__ == "__main__":
    main()
