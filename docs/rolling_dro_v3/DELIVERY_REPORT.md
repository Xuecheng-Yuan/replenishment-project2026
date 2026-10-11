# 滚动分布鲁棒补货项目交付报告

交付日期：2026年10月11日。目标仓库：[Xuecheng-Yuan/replenishment-project2026](https://github.com/Xuecheng-Yuan/replenishment-project2026)。

## 结论与查阅入口

当前可运行模型、可视化、Excel对照表及全量最终结果集中在`experiments/rolling_dro_v3/`，方法解读和交付说明集中在`docs/rolling_dro_v3/`。本次迁移完整重跑后，全部10行汇总结果与此前交付结果一致。

五种方法均达到完整28期总体需求满足率95%的目标。爆品/长尾DRO的28期库存周转率最高，为41.522次，较传统(s,S)提高0.57%；平均库存降低约0.58%。六类特征聚类DRO为41.477次。

| 方法 | 需求满足率 | 28期库存周转率（次） | 平均库存/品种 |
|---|---:|---:|---:|
| 特征聚类DRO | 96.55% | 41.477 | 350.96 |
| 不分组DRO | 96.78% | 40.945 | 356.38 |
| 爆品/长尾DRO | 96.56% | 41.522 | 350.63 |
| 传统(s,S) | 96.57% | 41.285 | 352.68 |
| 零半径经验模型 | 96.94% | 40.423 | 361.59 |

库存周转率使用数量口径：测试28期实际销量总量÷同期全仓平均在手库存，单位为次，未年化。缺货按丢失销售处理，未满足的需求不计入销量。

推荐按以下顺序查阅：

1. [在线可见的三张图表](../../experiments/rolling_dro_v3/visualization/README.md)。下载后打开同目录`index.html`，可离线切换策略查看逐期数值。
2. [Excel决策比较](../../experiments/rolling_dro_v3/outputs_v4/replenishment_comparison.xlsx)：查看总表、逐期、SKU、最近一期决策和完整样例。
3. [数学模型及实验报告](../../experiments/rolling_dro_v3/outputs_v4/REPORT.md)：查看约束、库存时序、四期滚动计划、传统(s,S)基线和结果解释。
4. [聚类说明](CLUSTERING.md)：查看5项需求形态、3项主成分、K-means参数和分组作用。
5. [运行说明](../../experiments/rolling_dro_v3/README.md)：复现、验证及文件职责。

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

仓库原有代码与本实验的口径不同：既有problem-based基准使用额外复查周期1和无缺货SKU－周期比例；本实验使用无额外盘点等待及销量/需求的需求满足率。因此保留独立目录与结果，避免把两套服务率或周转指标直接混在同一表中。

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

- 五策略完整重跑，28期及第2–28期共10行汇总均与此前交付结果一致。
- 5项模型测试通过：鲁棒端点解与独立LP一致、鲁棒半径性质、1期提前期与库存守恒、未来需求不改变当前订单、无跨期耦合下的四期分解。
- 每策略168,000行轨迹核验通过：库存守恒、丢失销售守恒、订单非负整数、到货提前期、最近1期执行、四期预测和汇总指标。
- Excel的5种策略总表、140行逐期数据、6000行SKU汇总、6000行最近一期决策、336行样例与CSV对齐。
- 完整轨迹gzip、发布哈希清单及离线HTML图表生成完成。
- 离线HTML核验通过：140条逐期数据、5个策略选择、每策略28行显示及本地文件链接。

重复验证命令见运行说明。持续集成只运行模型测试和已发布结果核对，不重新执行训练选参或下载LFS原始数据。首次GitHub运行结果需在仓库Actions页查看。

## 方法与结果的解释边界

K=6、三个主成分和PCA权重0.5为预设值，没有在当前流水线里选出最优聚类数。最终预测权重为1，点预测采用SKU自身趋势；聚类主要影响相对预测误差的共享和鲁棒水位。

测试集此前参与过早期版本偏差诊断，当前结果为修订后的重复评估。完整28期满足率95%不代表每个SKU或每期都达标。第1期没有模型订单到货，第28期订单在第29期到货，不计入28期在手库存。

本版迁移只调整仓库相对路径、压缩读取、可视化与交付组织；库存数学模型和数值结果保持此次核对确认的一致性。上传后可用`git log -1 -- experiments/rolling_dro_v3 docs/rolling_dro_v3`查阅本次交付提交。
