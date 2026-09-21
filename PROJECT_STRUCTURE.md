# 项目目录与文件职责

```text
replenishment_project2026/
├── README.md
├── PROJECT_STRUCTURE.md
├── pyproject.toml
├── configs/
│   ├── base.yaml
│   └── baseline.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── docs/
│   ├── baseline_method.md
│   └── research_plan.md
├── src/replenishment/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── optimization.py
│   ├── evaluation.py
│   └── pipeline.py
├── experiments/
│   └── run_baseline.py
├── scripts/
│   └── run_pipeline.py
├── tests/
│   ├── test_data.py
│   ├── test_evaluation.py
│   ├── test_models.py
│   └── test_smoke.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── models/
├── notebooks/
├── paper/
└── references/
```

## 根目录

- `README.md`：项目目标、baseline 方法、安装、运行和防泄漏规则。
- `PROJECT_STRUCTURE.md`：当前文件，说明目录和文件职责。
- `pyproject.toml`：Python 依赖、包发现、pytest 和 Ruff 配置。
- `.gitignore`：忽略环境、缓存、默认数据和自动生成结果。
- `.gitattributes`：如使用 Git LFS，则记录 Excel 等大文件的跟踪规则。

## 配置

- `configs/base.yaml`：项目通用目录和随机种子配置。
- `configs/baseline.yaml`：baseline 的时期数、服务水平、提前期和 `z` 搜索网格。

## 数据

- `data/raw/`：只读原始 Excel。代码不覆盖源文件。
- `data/processed/`：后续模型产生的可再生成中间数据。
- `data/external/`：公开或第三方数据。

## 核心代码

- `src/replenishment/data.py`：读取多工作表 Excel，校验时期、SKU、缺失、负值和 train/test 边界，并计算文件指纹。
- `src/replenishment/features.py`：后续预测模型的特征接口；baseline 明确不使用 `x1`–`x20`。
- `src/replenishment/models.py`：只接收 train 的均值–方差 baseline、时间切分和安全系数校准。
- `src/replenishment/optimization.py`：根据均值、标准差和安全系数生成 `s=S` 策略参数。
- `src/replenishment/evaluation.py`：统一执行库存流转，计算总体及爆品/长尾品的服务水平、库存和周转率。
- `src/replenishment/pipeline.py`：先 train 拟合并冻结，再读取 test 评价，保存可审计输出。
- `src/replenishment/__init__.py`：公开稳定接口和项目版本。

## 入口与测试

- `experiments/run_baseline.py`：正式 baseline 实验入口，打印关键结果。
- `scripts/run_pipeline.py`：统一入口，当前调用 baseline。
- `tests/test_data.py`：检查 train/test SKU 边界。
- `tests/test_evaluation.py`：检查到货—需求—下单顺序、服务水平和周转率。
- `tests/test_models.py`：检查 42/18 时间切分、`s=S` 和 fit 接口无 test 参数。
- `tests/test_smoke.py`：检查包能正常导入。

## 文档与输出

- `docs/baseline_method.md`：数学公式、数据使用边界、指标定义和已知局限。
- `docs/research_plan.md`：当前进度和下一阶段研究路线。
- `outputs/tables/`：参数、训练统计、含 category 的测试 SKU 指标和含分组指标的 JSON 摘要，默认不提交 Git；分组分析不另建文件。
- `outputs/figures/`：后续自动生成图表。
- `outputs/models/`：后续需要序列化的模型文件。
- `paper/`：论文或研究报告草稿。
- `references/`：论文、数据集和参考实现记录。
