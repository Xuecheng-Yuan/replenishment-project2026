# 项目目录与文件职责

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
