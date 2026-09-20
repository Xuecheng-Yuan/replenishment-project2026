# 项目目录与文件职责

本文档说明当前项目中每个初始目录和文件准备承担的职责。后续新增文件时，也应在这里补充说明。

```text
replenishment_project2026/
├── README.md
├── PROJECT_STRUCTURE.md
├── pyproject.toml
├── .gitignore
├── configs/
│   └── base.yaml
├── data/
│   ├── raw/.gitkeep
│   ├── processed/.gitkeep
│   └── external/.gitkeep
├── docs/
│   └── research_plan.md
├── notebooks/
│   └── README.md
├── src/replenishment/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── optimization.py
│   ├── evaluation.py
│   └── pipeline.py
├── experiments/
│   └── README.md
├── scripts/
│   └── run_pipeline.py
├── tests/
│   └── test_smoke.py
├── outputs/
│   ├── figures/.gitkeep
│   ├── tables/.gitkeep
│   └── models/.gitkeep
├── paper/
│   └── README.md
└── references/
    └── README.md
```

## 根目录文件

- `README.md`：项目入口，说明研究目标、环境安装、运行顺序和 Git 原则。
- `PROJECT_STRUCTURE.md`：当前文件，即目录地图和文件职责清单。
- `pyproject.toml`：Python 包、依赖、测试和代码格式工具的统一配置。
- `.gitignore`：排除数据、模型结果、虚拟环境、缓存和系统临时文件。

## 配置与数据

- `configs/base.yaml`：公共实验参数，包括随机种子、时间列、SKU 列、需求列和成本参数。具体实验可新增独立配置文件，不要把参数散落在代码里。
- `data/raw/.gitkeep`：保留原始数据目录。原始文件只读保存，不纳入 Git。
- `data/processed/.gitkeep`：保留清洗、聚合或特征化后的中间数据目录，不纳入 Git。
- `data/external/.gitkeep`：保留来自外部公开数据源或第三方的数据目录，不纳入 Git。

## 研究文档与探索

- `docs/research_plan.md`：记录研究问题、假设、评价指标、实验路线和阶段任务。
- `notebooks/README.md`：规定 Notebook 的命名和用途；Notebook 只用于探索与展示，稳定逻辑应迁移到 `src/`。
- `references/README.md`：记录论文、数据集、代码仓库及其与本项目的关系。

## 核心代码

- `src/replenishment/__init__.py`：把目录声明为 Python 包，并保存版本号。
- `src/replenishment/data.py`：数据读取、字段检查、类型转换、缺失值处理和数据切分。
- `src/replenishment/features.py`：滞后量、滚动统计、日历变量、库存状态等特征工程。
- `src/replenishment/models.py`：需求预测模型的训练、预测、保存和加载接口。
- `src/replenishment/optimization.py`：补货决策、库存转移、成本函数和优化模型。
- `src/replenishment/evaluation.py`：预测指标与库存指标的统一计算和结果汇总。
- `src/replenishment/pipeline.py`：串联数据、预测、优化和评估，形成端到端流程。

## 实验、脚本与测试

- `experiments/README.md`：规定实验脚本的命名、输入、输出和结果记录方式；每个正式实验建议有独立脚本。
- `scripts/run_pipeline.py`：命令行运行入口，后续读取配置并调用端到端流程。
- `tests/test_smoke.py`：最小冒烟测试，确认项目包可以被导入；后续为各模块增加对应单元测试。

## 结果与论文

- `outputs/figures/.gitkeep`：保留自动生成图表的目录，图表本身默认不纳入 Git。
- `outputs/tables/.gitkeep`：保留指标表、对比表等自动生成文件的目录。
- `outputs/models/.gitkeep`：保留训练模型、参数文件和检查点的目录。
- `paper/README.md`：说明论文草稿、附录和投稿材料应如何组织；正式写作时可增加 `draft.md` 或 LaTeX 文件。

## 新增文件的约定

新增代码前先判断它属于可复用逻辑、单次实验还是自动生成结果。可复用逻辑放 `src/`，实验入口放 `experiments/`，一次性执行工具放 `scripts/`，生成物放 `outputs/`。不要在根目录堆放临时文件。

