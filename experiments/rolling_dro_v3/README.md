# 四期滚动分布鲁棒补货：当前交付版

上传分支为`codex/rolling-dro-delivery-20261011`，交付入口为[Pull Request #1](https://github.com/Xuecheng-Yuan/replenishment-project2026/pull/1)，合并前请在该分支查阅本目录。

本实验采用需求满足率≥95%、提前期1、无额外盘点等待、缺货丢失；测试Period_1重新起算。每次预测未来4期并生成4期计划，只执行最近1期订单，观察本期需求后滚动更新。

完整测试集6000个SKU×28期上，爆品/长尾DRO需求满足率96.56%，库存需求比0.168340，低于传统(s,S)的0.172097，降低2.18%。库存需求比＝所有SKU、所有评价时期的期末库存之和÷总需求，越低越好。已按此指标重新选参并完整重跑。

## 先看结果

- [可视化图表与离线交互页面](visualization/README.md)：三张图在GitHub可直接查看；下载后双击`index.html`，可切换策略查看28期数据。
- [Excel决策对照表](outputs_v4/replenishment_comparison.xlsx)：策略总表、逐期比较、6000个SKU汇总、完整样例和最近一期四期预测及五策略决策。
- [结果汇总](outputs_v4/summary.csv)、[逐期结果](outputs_v4/results_by_period.csv)、[冻结参数](outputs_v4/frozen_parameters.json)。
- [聚类依据](../../docs/rolling_dro_v3/CLUSTERING.md)与[交付报告](../../docs/rolling_dro_v3/DELIVERY_REPORT.md)。

## 本地运行

需要Python 3.10+。在仓库根目录执行：

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r experiments/rolling_dro_v3/requirements_v3.txt
python experiments/rolling_dro_v3/run_v3.py
python experiments/rolling_dro_v3/build_visualizations.py
```

默认读取仓库`data/raw/`中的两份原始Excel。这两份文件由既有仓库用Git LFS管理，克隆时应安装Git LFS并执行`git lfs pull`。也可指定你已有的原始文件：

```bash
python experiments/rolling_dro_v3/run_v3.py --data-dir "你的原始Excel目录"
```

首次运行会逐表读取88个工作表，后续复用通过源文件与缓存SHA256核验的`.cache/period_data.npz`。模型训练、测试、CSV结果和HTML/PNG/SVG图表都由普通Python环境生成；数值求解不需要商业求解器或安装RSOME。

## 验证

无需重跑模型即可核验仓库发布的完整结果：

```bash
python -m unittest discover -s experiments/rolling_dro_v3/tests -p test_v2.py -v
python experiments/rolling_dro_v3/verify_published.py
python experiments/rolling_dro_v3/verify_workbook_v3.py
```

完整重跑后还可以执行`python experiments/rolling_dro_v3/verify_results_v3.py`，将决策与原始测试数组核对。`verify_published.py`检查发布文件哈希、库存守恒、1期提前期、丢失销售、整数订单、四期预测、实际执行订单及汇总指标。

## 全量结果与刷新

每种策略的全量轨迹都有168,000行，Git中以`outputs_v4/detail_*.csv.gz`保存；五种策略并列对照为`decision_comparison_full.csv.gz`。Pandas可以直接读取gzip：

```python
import pandas as pd
results = pd.read_csv("experiments/rolling_dro_v3/outputs_v4/decision_comparison_full.csv.gz")
```

重新运行模型产生未压缩CSV。运行`python experiments/rolling_dro_v3/publish_results.py`可刷新gzip与SHA256清单。读取工具在未压缩文件存在时优先读取它。

交付Excel已核对本版数值。`build_visualizations.py`会刷新图表和HTML；重新导出Excel使用`build_workbook_v3.mjs`，需要Node.js与Codex提供的`@oai/artifact-tool`组件。普通Python运行模型不依赖该组件，重新训练也不会自动刷新Excel。

## 文件职责

| 文件 | 职责 |
|---|---|
| `run_v3.py` | 三种DRO、传统(s,S)、零半径经验模型的训练选参及测试回放主入口 |
| `model_v2.py` | 特征聚类、滚动预测、W∞鲁棒满足率约束与库存模拟 |
| `run_v2.py` | 为当前主入口提供(s,S)分位收缩与特征诊断函数；历史入口运行时转到run_v3.py |
| `data_io.py`、`config.py` | 原始工作簿解析与缓存路径配置；当前策略参数由`run_v3.py`确定 |
| `report_v3.py`、`regenerate_report_v3.py` | 生成与重建数学模型及实验报告 |
| `build_visualizations.py` | 从结果CSV生成PNG、SVG和独立HTML展示 |
| `publish_results.py` | 压缩完整轨迹并记录发布文件SHA256 |
| `verify_*.py`、`tests/test_v2.py` | 结果核对、Excel核对和7项模型测试 |
| `verify_dashboard.mjs` | 可选Node.js检查：HTML策略切换、140条嵌入数据和离线文件链接 |

当前聚类数6与主成分权重0.5为预设值，未在本流水线中自动选择最优聚类数。最终点预测采用SKU自身趋势，聚类主要作用于预测误差分布与鲁棒补货水位。测试集曾参与早期版本的偏差诊断，所以当前结果是修订后的重复评估；详见实验报告。
