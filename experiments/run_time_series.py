"""运行 train-only 时间序列选择，并在策略冻结后评价 test。"""

from pathlib import Path

from replenishment.pipeline import run_time_series_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result = run_time_series_experiment(
        train_path=PROJECT_ROOT / "data/raw/replenishment_train.xlsx",
        test_path=PROJECT_ROOT / "data/raw/replenishment_test.xlsx",
        config_path=PROJECT_ROOT / "configs/time_series.yaml",
        output_dir=PROJECT_ROOT / "outputs/tables",
    )
    summary = result.test_result.summary
    print("时间序列实验完成")
    print(f"最终模型：{result.model_summary['selected_model']['id']}")
    print(f"总体服务水平：{float(summary['overall_service_level']):.4%}")
    print(f"满足95%约束：{'是' if float(summary['overall_service_level']) >= 0.95 else '否'}")
    print(f"总体周转率：{float(summary['overall_turnover']):.4f}")
    print(f"平均总期末库存：{float(summary['average_total_ending_inventory']):,.2f}")
    print(f"汇总文件：{result.output_files['summary']}")


if __name__ == "__main__":
    main()
