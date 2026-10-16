# Replenishment Project 2026

## 新旧两版定稿实验结果报告（2026-10-16）

[旧版：需求满足率最终报告](experiments/final_results_reports/reports/旧版_需求满足率_最终实验结果报告.pdf) · [新版：正库存比例最终报告](experiments/final_results_reports/reports/新版_正库存比例_最终实验结果报告.pdf) · [可读文字、核验记录与复现代码](experiments/final_results_reports/README.md)

两版各7页，分别包含数学模型与解读、全60期训练、六方法正式结果、爆品与长尾、整体及逐期图、四期预测误差和复现路径。重新回放201.6万条正式记录，结果未改：旧版分类DRO最低J=0.168340；新版(s,S)最低J=0.345958、P=95.2810%，相对分类DRO的稳定优势尚未确认。与下面的约束变化原因探索报告分开交付。

## 约束变化原因重做实验（2026-10-16）

[机制探索报告PDF](experiments/constraint_mechanism/reports/机制探索报告.pdf) · [完整实验报告PDF](experiments/constraint_mechanism/reports/完整实验报告.pdf) · [代码、原始结果与复现说明](experiments/constraint_mechanism/README.md)

重做1620训练候选、108实验单元，逐行核验806.4万条测试轨迹。单独改约束会改变可行域及部分排序；焦点两方法的数值翻转还涉及反馈与风险包络，最终预测差异已排除。新版(s,S)相对分类DRO的差异区间跨0，稳定优势尚未确认。该诊断实验与下方正式95.28%左右的交付结果独立保存。

## 当前：服务水平95%–95.5%的完整60期版本

[本次交付报告与数学模型](experiments/service95_revision/REPORT.md) · [当前六方法Excel与三版对照](experiments/service95_revision/outputs_v4/replenishment_comparison.xlsx) · [全部商品、爆品、长尾动画](汇报成果/2026-10-15/动画/index.html)

六种方法独立测试28期的正库存比例为95.2798%–95.2893%，全部满足95%–95.5%。传统(s,S)库存需求比最低0.345958；三种DRO中爆品/长尾DRO最低0.346907。训练文件全部60期最终拟合，独立测试文件从Period_1重新开始，提前期1、失销、原始库存、零初始在途。

当前增加因果服务预算反馈，逐期仅使用已经发生的需求、正库存计数与预测误差调整后续水位。反馈参数在训练内部28期回溯重启留出中选择；三轮向前/重启12期前缀另外诊断。完整训练、时间关系、108个反馈候选和42个已选情景均在报告和Excel说明。先前99%以上高库存实验单独归档，不作为当前结果。

复现：`python experiments/ro_extension/service_band_revision.py`。核验：`python experiments/ro_extension/verify_service95.py`。下方各历史版本仅用于追溯，当前交付以本节链接为准。


## 历史：PR7三轮滚动验证与完整60期训练版

[修正与交付报告](experiments/full60_revision/CHANGE_REPORT.md) · [当前结果及Excel](experiments/full60_revision/README.md) · [服务约束单因素报告](experiments/full60_revision/REPORT.md)

新版和单因素实验均在训练文件内按1–24→25–36、1–36→37–48、1–48→49–60三轮滚动验证选参，合并36期原始总量后计算指标，再用全部60期最终拟合。独立测试使用`replenishment_test.xlsx`的28期，Period_1重新起算；测试实际达标状态见上方报告。下文原32期新版结果作为历史追溯。

## 最新六策略场景动画（2026-10-15）

- [动画、启动方式与指标说明](汇报成果/2026-10-15/README.md)
- [六策略货架与小人场景动画（下载后打开）](汇报成果/2026-10-15/动画/index.html)
- [分布鲁棒公式互动（下载后打开）](汇报成果/2026-10-15/动画/公式互动.html)

动画已同步当前三轮选参、全60期最终拟合的真实测试轨迹，并保留历史数量满足率版切换。支持全部6000种商品、全部480种爆品、全部5520种长尾品和固定代表商品，六个独立窗口同步播放。离线HTML无需安装依赖；P按逐商品期末库存严格大于0计数，群体数据来自全量轨迹，小人和货架图标为示意。

## 历史：新旧两版新增传统RO（2026-10-15）

历史两版均已补充“传统区间鲁棒优化RO”，各有六种方法。该历史交付位于各版的 `with_ro` 子文件夹，原五方法结果保留；当前新版为service95_revision。

- [旧版：六方法报告、Excel与可视化](experiments/demand_fill_v3/with_ro/README.md)
- [新版：六方法报告、Excel与可视化](experiments/rolling_dro_v3/with_ro/README.md)
- [两版六方法比较](experiments/version_comparison/COMPARISON.md)
- [RO代码、数学模型说明与本地运行](experiments/ro_extension/README.md)

旧版RO需求满足率96.7100%，库存需求比0.174962；新版RO正库存天数比例95.0077%，库存需求比0.589326。原有五种方法逐商品逐期轨迹未变，两版最优方法仍分别为爆品/长尾DRO和传统(s,S)。

## 联合汇报与交互网站归档（2026-10-14）

- [最新汇报PPT](汇报成果/2026-10-14/补货项目汇报_20261014.pptx)
- [交互网站离线版](汇报成果/2026-10-14/补货方案对比网站_离线版.html)
- [打开方式与版本说明](汇报成果/2026-10-14/README.md)

本轮展示区分yuan xuecheng的预测与决策学习方向、kristina的鲁棒优化方向。既有代码与历史汇报保留不变。

## 历史交付：被替代的32期拟合版本

旧的需求满足率约束版已完整独立保存在[experiments/demand_fill_v3](experiments/demand_fill_v3/README.md)，包含原报告、Excel、代码和结果。[两版对比报告](experiments/version_comparison/COMPARISON.md)同时列出服务定义、建模差异和新增RO后的六种策略结果。

可运行代码、可视化和最终结果位于[experiments/rolling_dro_v3](experiments/rolling_dro_v3/README.md)，采用**正库存服务水平95%–95.5%、提前期1、无额外盘点等待、缺货丢失**。包含特征聚类DRO、不分组DRO、爆品/长尾DRO、传统(s,S)和零半径经验模型，预测4期、只执行最近1期。

本版已将服务约束改为所有商品期末库存大于0的商品－天占比，范围95%–95.5%；优化目标仍为累计期末库存÷累计需求最小。五种策略在完整28天测试中均落在服务范围内，传统(s,S)库存需求比最低，为0.515894；三种DRO中，特征聚类DRO最低，为0.550971，比传统(s,S)高6.80%，本次没有获得优于传统基线的库存结果。

- [交付报告与文件导航](docs/rolling_dro_v3/DELIVERY_REPORT.md)
- [图表与离线交互展示](experiments/rolling_dro_v3/visualization/README.md)
- [最终数学模型与实验报告](experiments/rolling_dro_v3/outputs_v4/REPORT.md)
- [Excel决策对照表](experiments/rolling_dro_v3/outputs_v4/replenishment_comparison.xlsx)
- [特征聚类依据](docs/rolling_dro_v3/CLUSTERING.md)

从仓库根目录运行：

```bash
python -m pip install -r experiments/rolling_dro_v3/requirements_v3.txt
python experiments/rolling_dro_v3/run_v3.py
python experiments/rolling_dro_v3/build_visualizations.py
```

默认使用`data/raw/`中的Git LFS原始Excel，也可通过`--data-dir`指定现有数据目录。详细复现和校验命令见实验目录README。

## 既有problem-based基准（另一套评价口径）

本项目研究单级、多SKU、周期复查库存系统中的补货决策。当前正式基准为 `problem_based_methods`：仅使用60期train估计需求分布和选择安全系数，冻结后在28期test上评价前4期与完整28期表现。

## 数据

- `data/raw/replenishment_train.xlsx`：6,000个SKU、60期，只用于需求分布估计、train内部验证和最终参数重估。
- `data/raw/replenishment_test.xlsx`：6,000个SKU、28期，只用于冻结策略后的最终评价。
- 当前基准不使用 `x1` 至 `x20`，不会读取test未来需求。

代码不会修改原始Excel。共享输出只保存文件名和SHA-256，不保存本地绝对路径或用户名。

## Problem-based methods

系统固定使用1期提前期和1期复查周期：

$$
L=1,\qquad r=1.
$$

根据train历史需求估计每个SKU的均值 $\mu_i$ 和方差 $\sigma_i^2$。在没有固定订货成本的情况下，$(s,S)$ 退化为 $s_i=S_i$：

$$
s_i=S_i=
\left\lceil
(L+r)\mu_i+z\sqrt{(L+r)\sigma_i^2}
\right\rceil.
$$

代入 $L=r=1$：

$$
s_i=S_i=
\left\lceil
2\mu_i+z\sqrt{2\sigma_i^2}
\right\rceil.
$$

每一期都生成未来4期订单计划，但只执行第一期订单；下一期获得新的库存状态后重新规划。

完整符号、假设、公式和库存时序见 [docs/problem_based_methods.md](docs/problem_based_methods.md)。

## 目标与约束

Boss指定的周转指标为所有SKU、所有评价时期的总期末库存除以总需求：

$$
J_H=
\frac{\sum_{t=1}^{H}\sum_i I_{it}}
{\sum_{t=1}^{H}\sum_i D_{it}}.
$$

由于评价窗口内总需求固定，正式目标为：

$$
\min J_H.
$$

正式服务约束为无缺货SKU-时期占比：

$$
SL_H^{NS}=
\frac{1}{|\mathcal I|H}
\sum_{t=1}^{H}\sum_i\mathbf 1\{U_{it}=0\}
\ge 0.95.
$$

同时报告严格正库存服务率：

$$
SL_H^{+}=
\frac{1}{|\mathcal I|H}
\sum_{t=1}^{H}\sum_i\mathbf 1\{I_{it}>0\}.
$$

## 库存事件顺序

每期严格按照以下顺序运行：

1. 期初收到上期期末订单；
2. 本期需求发生，未满足部分作为lost sales；
3. 需求发生后生成未来4期补货计划；
4. 只执行第一期订单，该订单在下一期期初到达。

test第1期没有模型订单到货，只能使用工作簿给定的 `initial_inventory`。

## 参数选择与防泄漏

60期train按时间顺序切分：

- 第1–42期估计候选需求分布；
- 第43–60期选择安全系数 $z$；
- 候选必须在验证前4期和完整18期都满足95%无缺货服务约束；
- 在可行候选中最小化完整验证窗口的库存需求比；
- $z$冻结后，再用完整60期train重估最终库存参数；
- 完成上述步骤后才读取test。

## 评价窗口

同一条test库存轨迹报告：

- `first_4`：test第1–4期；
- `full_28`：test第1–28期。

## 当前结果

train内部验证选择 $z=2.10$。

| test窗口 | 库存需求比 | 严格正库存服务率 | 无缺货服务率 | 95%约束 |
|---|---:|---:|---:|---|
| 前4期 | 11.494377 | 97.3208% | 97.3583% | 满足 |
| 完整28期 | 5.212871 | 99.5018% | 99.5083% | 满足 |

爆品前4期无缺货服务率为75%。480个爆品在第1期全部缺货，但第2–4期全部不缺货。第1期爆品初始库存合计59,613，而需求为821,414，仅覆盖7.2574%；由于第一期没有到货，这一结果主要反映外生初始库存不足。

## 安装与运行

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
python -m experiments.run_problem_based_methods
```

默认只生成两份结果：

- `outputs/tables/problem_based_methods_summary.json`；
- `outputs/tables/problem_based_methods_per_item_metrics.csv`。

## Git规则

提交前运行测试和隐私扫描。结果文件默认不加入Git，避免把本地路径、临时输出或大规模实验明细提交到共享仓库。
