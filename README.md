# Replenishment Project 2026

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
