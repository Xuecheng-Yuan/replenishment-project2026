# Replenishment Project 2026

这是一个重新搭建的补货优化研究项目，用于从数据清洗、需求预测、补货决策到策略评估，形成可复现的完整实验流程。

## 快速开始

```bash
cd replenishment_project2026
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

项目目前只提供骨架，不包含旧项目的数据、模型或实验结果。开始开发前，请先阅读 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) 和 [docs/research_plan.md](docs/research_plan.md)。

## 推荐工作顺序

1. 将原始数据放进 `data/raw/`，不要直接修改原文件。
2. 在 `src/replenishment/data.py` 中实现读取、校验和清洗逻辑。
3. 在 `src/replenishment/models.py` 中实现需求预测模型。
4. 在 `src/replenishment/optimization.py` 中实现补货决策模型。
5. 在 `src/replenishment/evaluation.py` 中统一计算成本、缺货率和服务水平。
6. 用 `experiments/` 记录实验，用 `tests/` 保证核心逻辑正确。

## Git 提交原则

- 提交：源代码、测试、配置、研究文档和论文草稿。
- 不提交：原始数据、模型文件、自动生成的图表和临时缓存。
- 每次提交只解决一个清晰问题，并写可读的提交说明。

