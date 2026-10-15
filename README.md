# Replenishment Project 2026

## 最新六策略场景动画（2026-10-15）

- [动画、启动方式与指标说明](汇报成果/2026-10-15/README.md)
- [六策略货架与小人场景动画（下载后打开）](汇报成果/2026-10-15/动画/index.html)
- [分布鲁棒公式互动（下载后打开）](汇报成果/2026-10-15/动画/公式互动.html)

新增传统区间 RO，支持当前正库存服务版和历史数量满足率版切换，六个独立窗口同步播放。离线 HTML 无需安装依赖；动画数值来自冻结真实轨迹，小人和货架图标为示意。

## 新旧两版新增传统RO（2026-10-15）

两版均已补充“传统区间鲁棒优化RO”，现在各有六种方法。最新交付位于各版的 `with_ro` 子文件夹，原五方法结果保留。

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

## 当前交付：四期滚动分布鲁棒补货

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
