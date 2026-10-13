# 滚动分布鲁棒补货项目交付报告

交付日期：2026年10月11日。目标仓库：[Xuecheng-Yuan/replenishment-project2026](https://github.com/Xuecheng-Yuan/replenishment-project2026)。

上传状态：代码及结果已发布到独立分支`codex/rolling-dro-delivery-20261011`，通过[Pull Request #1](https://github.com/Xuecheng-Yuan/replenishment-project2026/pull/1)交付，尚未合并到`main`。核心代码及结果提交为`0196364`，后续补充交付状态与跨平台核验说明。可以直接在PR的Files changed页或该分支中查阅所有文件。

## 结论与查阅入口

完整测试集6000个SKU×28期上，爆品/长尾DRO需求满足率96.56%，库存需求比0.168340，低于传统(s,S)的0.172097，降低2.18%。库存需求比＝所有SKU、所有评价时期的期末库存之和÷总需求，越低越好。已按此指标重新选参并完整重跑。

五种方法均达到完整28期总体需求满足率95%的目标。完整排名为爆品/长尾DRO、特征聚类DRO、传统(s,S)、不分组DRO、零半径经验模型。

| 方法 | 需求满足率 | 库存需求比（越低越好） | 相对(s,S)变化 |
|---|---:|---:|---:|
| 特征聚类DRO | 96.55% | 0.169025 | -1.78% |
| 不分组DRO | 96.78% | 0.177925 | +3.39% |
| 爆品/长尾DRO | 96.56% | 0.168340 | -2.18% |
| 传统(s,S) | 96.57% | 0.172097 | +0.00% |
| 零半径经验模型 | 96.94% | 0.186779 | +8.53% |

数学定义：$J_H=\sum_{t=1}^H\sum_i I^{end}_{it}/\sum_{t=1}^H\sum_i D_{it}$。分母是完整需求，包含损失需求；分子累计每期末库存，不是仅使用最后一期库存。该比值没有“次”的单位。爆品/长尾DRO对应15,228,944÷90,465,437＝0.168340。服务水平仍为总销量÷总需求≥95%。

推荐按以下顺序查阅：

1. [在线可见的三张图表](../visualization/README.md)。下载后打开同目录`index.html`，可离线切换策略查看逐期数值。
2. [Excel决策比较](../outputs_v4/replenishment_comparison.xlsx)：查看总表、逐期、SKU、最近一期决策和完整样例。
3. [数学模型及实验报告](../outputs_v4/REPORT.md)：查看约束、库存时序、四期滚动计划、传统(s,S)基线和结果解释。
4. [聚类说明](CLUSTERING.md)：查看5项需求形态、3项主成分、K-means参数和分组作用。
5. [运行说明](../README.md)：复现、验证及文件职责。

## 文件组织

| 位置 | 内容 |
|---|---|
| `experiments/rolling_dro_v3/run_v3.py` | 当前模型的主运行入口 |
| 同目录`model_v2.py` | 聚类、预测、W∞鲁棒约束和库存模拟 |
| 同目录`visualization/` | 独立HTML交互展示及PNG、SVG图表 |
| 同目录`outputs_v4/` | 汇总、逐期、四期预测、订单、全量轨迹、冻结参数、Excel和校验记录 |
| 同目录`tests/`及`verify_*.py` | 模型测试和结果核验 |
| `docs/rolling_dro_v3/` | 交付报告与聚类解释 |
| `.github/workflows/rolling-dro-v3.yml` | Python模型测试及发布结果核验 |

每种策略的全量轨迹为6000×28=168,000行。五策略并列对照存为`decision_comparison_full.csv.gz`，每策略详细轨迹存为`detail_*.csv.gz`。gzip压缩不改变CSV数值，Pandas可直接读取；小型CSV保持原格式。`artifact_manifest.json`记录发布结果的字节数和SHA256。

仓库原有代码与本实验的口径不同：既有problem-based基准使用额外复查周期1和无缺货SKU－周期比例；本实验使用无额外盘点等待及销量/需求的需求满足率。因此保留独立目录与结果，两版库存需求比现已统一，但服务水平定义和补货时序仍有差异，结果比较需保持相同实验条件。

## 本地复现

需要Python 3.10+。从仓库根目录执行：

```bash
python -m pip install -r experiments/rolling_dro_v3/requirements_v3.txt
python experiments/rolling_dro_v3/run_v3.py
python experiments/rolling_dro_v3/build_visualizations.py
```

默认读取`data/raw/replenishment_train.xlsx`和`data/raw/replenishment_test.xlsx`。它们由既有仓库的Git LFS管理；下载后需要确认文件是实际Excel而非LFS指针。亦可用`--data-dir`指定现有原始文件目录。数值求解与图表生成无需商业求解器、RSOME或前端依赖。

随仓库附带的Excel已经核对当前数值。模型重跑生成CSV，图表脚本生成HTML/PNG/SVG；Excel重新导出使用保留的`build_workbook_v3.mjs`，依赖Codex的Node.js `@oai/artifact-tool`组件，不属于普通Python依赖。详情见运行说明。

## 验证证据

本次完成以下检查：

- 五策略按原项目库存需求比重新选参并完整重跑，核验28期及第2–28期共10行汇总。
- 7项模型测试通过：库存需求比独立算例、按原指标选择可行参数、鲁棒端点解与独立LP一致、鲁棒半径性质、1期提前期与库存守恒、未来需求不改变当前订单、无跨期耦合下的四期分解。
- 每策略168,000行轨迹核验通过：库存守恒、丢失销售守恒、订单非负整数、到货提前期、最近1期执行、四期预测和汇总指标。
- Excel的5种策略总表、140行逐期数据、6000行SKU汇总、6000行最近一期决策、336行样例与CSV对齐。
- 完整轨迹gzip、发布哈希清单及离线HTML图表生成完成。
- 离线HTML核验通过：140条逐期数据、5个策略选择、每策略28行显示及本地文件链接。

重复验证命令见运行说明。持续集成只运行模型测试和已发布结果核对，不重新执行训练选参或下载LFS原始数据。[GitHub自动检查运行记录](https://github.com/Xuecheng-Yuan/replenishment-project2026/actions/workflows/rolling-dro-v3.yml)可查阅各提交的执行状态。

## 方法与结果的解释边界

K=6、三个主成分和PCA权重0.5为预设值，没有在当前流水线里选出最优聚类数。最终预测权重为1，点预测采用SKU自身趋势；聚类主要影响相对预测误差的共享和鲁棒水位。

测试集此前参与过早期版本偏差诊断，当前结果为修订后的重复评估。完整28期满足率95%不代表每个SKU或每期都达标。第1期没有模型订单到货，第28期订单在第29期到货，不计入28期在手库存。

本次修订将训练选参、测试评价、排名、图表及Excel全部统一为原项目库存需求比最小化，保留已确认的95%需求满足率和提前期1。训练验证窗口的需求总量对所有候选参数相同，因此最小化累计期末库存与最小化库存需求比等价。上传后可用`git log -1 -- experiments/rolling_dro_v3 docs/rolling_dro_v3`查阅本次交付提交。
