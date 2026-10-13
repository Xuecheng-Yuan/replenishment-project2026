# 项目目录与文件职责

## 当前滚动DRO交付

当前代码及最终结果在`experiments/rolling_dro_v3/`；中文交付报告和聚类解释在`docs/rolling_dro_v3/`。该实验使用“期末正库存服务水平95%–95.5%”口径，与下方既有problem-based基准分别运行和保存。

```text
experiments/rolling_dro_v3/
├── README.md                     # 运行、查看和核验说明
├── run_v3.py                     # 当前五策略实验入口
├── model_v2.py                   # 聚类、预测、DRO与库存模拟
├── build_visualizations.py       # 普通Python生成图表和HTML
├── publish_results.py           # gzip全量结果与哈希清单
├── verify_published.py          # 无需重跑即可核验发布结果
├── tests/test_v2.py             # 10项模型数学、正库存边界及因果性测试
├── visualization/               # HTML、PNG、SVG展示
└── outputs_v4/                  # 报告、Excel、CSV、完整CSV.gz轨迹
docs/rolling_dro_v3/
├── DELIVERY_REPORT.md
└── CLUSTERING.md
```

## 既有目录

```text
replenishment_project2026/
├── README.md
├── PROJECT_STRUCTURE.md
├── pyproject.toml
├── configs/
│   ├── base.yaml
│   ├── baseline.yaml
│   └── problem_based_methods.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── docs/
│   ├── problem_based_methods.md
│   ├── baseline_method.md
│   └── research_plan.md
├── src/replenishment/
│   ├── __init__.py
│   ├── data.py
│   ├── evaluation.py
│   ├── features.py
│   ├── models.py
│   ├── optimization.py
│   ├── problem_based_methods.py
│   ├── problem_evaluation.py
│   ├── problem_pipeline.py
│   └── pipeline.py
├── experiments/
│   ├── run_problem_based_methods.py
│   └── run_baseline.py
├── scripts/
│   └── run_pipeline.py
├── tests/
│   ├── test_data.py
│   ├── test_evaluation.py
│   ├── test_models.py
│   ├── test_problem_based_methods.py
│   ├── test_problem_evaluation.py
│   └── test_smoke.py
└── outputs/
    ├── figures/
    ├── tables/
    └── models/
```

## 核心文件

- `src/replenishment/data.py`：读取Excel，校验时期、SKU、缺失值、负值和train/test边界。
- `src/replenishment/problem_based_methods.py`：估计需求分布、构造 $s=S$、在train内部选择安全系数。
- `src/replenishment/problem_evaluation.py`：执行1期提前期和4期滚动计划，计算库存需求比及两种服务率。
- `src/replenishment/problem_pipeline.py`：先train拟合并冻结，再读取test评价，保存两份结果。
- `src/replenishment/pipeline.py`：兼容入口，转发到新的problem-based流水线。
- `src/replenishment/evaluation.py`、`models.py`、`optimization.py`：旧导入路径兼容层，内部统一调用当前方法。
- `experiments/run_problem_based_methods.py`：正式实验入口。
- `experiments/run_baseline.py`：兼容入口，与正式实验运行同一方法。
- `docs/problem_based_methods.md`：完整模型、公式、时序、防泄漏规则和结果解释。
- `docs/baseline_method.md`：旧文件名兼容说明，不再保存旧方法。

## 输出规则

`outputs/tables/`默认忽略自动生成结果，只保留目录占位文件。正式运行仅生成：

- `problem_based_methods_summary.json`；
- `problem_based_methods_per_item_metrics.csv`。

原始Excel只读。共享代码和文档不得包含本地绝对路径、用户名、访问令牌或其他凭据。
