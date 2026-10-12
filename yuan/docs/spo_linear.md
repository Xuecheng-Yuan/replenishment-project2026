# spo_linear：模型实现与说明

[总览](../README.md) · [实际代码](../models/spo_linear.py) · [全部变体记录](../results/variants.json)

## 1. 文件如何对应

本模型所有定义、关键决策/损失函数或估计器构造在`models/spo_linear.py`，不是转发到旧experiments目录的脚本。数据/评价/公共训练流程位于common/。参数、窗口、种子和消融不另建模型文件。历史125项仅为结果记录，完整索引见results/variants.json。

## 2. 信息时序与模型原理

### A组

直接预测17个候选目标库存的成本向量，选预测最低成本动作；输入59维特征加动态库存。真实成本包含持有与缺货，次梯度训练SPO+。有限动作代理不是原动态库存问题的精确等价。

SKU是商品编号；x1–x20是外部属性/特征，不是需求本身。历史滞后表示预测目标日前已观察值。库存状态是今天卖完后的库存，不是明天已实现的库存。预测层误差小不自动意味着决策服务达标。

### 核心公式

$$
\ell_{\mathrm{SPO+}}(\widehat c,c)=\max_{w\in\mathcal W}(c-2\widehat c)^\top w+2\widehat c^\top w^*(c)-c^\top w^*(c)
$$

d为需求，I为期末库存，s为实际销量，q为下单量；预测目标用S表示，C为公共容量，z为风险余量系数；a_i为训练期商品尺度，theta为学习参数。上式是核心计算关系/目标概要，不把代理损失写成业务硬约束。完整推导和实现细节见后面的完整推导；共享符号、Fill Rate与Ready Rate区别见[评价标准](evaluation.md)。


## 3. 运行命令

从本仓库根目录运行；默认只查看历史结果，不训练：

```bash
python -m models.spo_linear
python -m models.spo_linear --help
python -m models.spo_linear --train
```

普通ML与baseline可用`--protocol legacy`、`--protocol ready_immediate`、`--protocol lead_one`切换已存在实验口径。lead_one为旧冻结预测的安全库存移除/固定z4对照，不是重新在该口径调参。移动平均用`--window 3/6/12`；MPC用`--round all/1/2/3/4`。高级模型默认沿用原批次的验证/消融，但实现全在同一文件，不会调用旧脚本树。

原始Excel放data/。运行只写results/generated/，不覆盖results/variants.json中的历史快照。改变输入请新目录重跑，不允许旧缓存自动作为新数据证据。

## 4. 历史结果（所有配置集中列在这里）

|原标签|组|角色|Fill Rate|Ready Rate达标比例|平均总库存|原始周转|加权周转|
|---|---|---|---:|---:|---:|---:|---:|
|legacy_spo_plus_linear|A|主方案|0.950235|未记录|3.68461e+06|23.3304|未记录|

小数比例0.95就是95%。参数与来源SHA256逐项保存在results/variants.json。历史业务失败没有改成通过；没有因为此次重构而重训全部真实数据。

## 5. 完整数学与训练细节

以下为本模型相关历史说明的完整正文，记录旧研究假设；当时“最优”只限于自己的实验口径。运行请用上面的新命令，以下旧文件路径仅作为出处，不表示新目录仍有旧结构。

### 原始说明：spo_plus_method_report.md

### Smart Predict-then-Optimize（SPO+）补货方法报告

## 1. 一句话结论

本项目已经按照 Elmachtoub 与 Grigas 的 **Smart “Predict, then Optimize”** 思路，新增了一个可复现的 `spo_plus_linear` 模型。正式测试结果为：

| 模型 | 测试总体 Fill Rate | 平均总库存 | 总周转率 | 满足 95% |
|---|---:|---:|---:|---:|
| SPO+ 线性决策模型（含当期库存） | **95.0235%** | **3,684,607** | **23.3304** | 是 |

它比原始均值—方差 baseline 少用 **69.16%** 的平均库存，并仍满足总体 Fill Rate 约束。因此，这篇论文的方法在本问题上是可行的。不过它没有超过 LightGBM、XGBoost、CatBoost，也没有超过简单的 12 期移动平均；当前不能把 SPO+ 称为最优方案。

## 2. 论文到底改变了什么

普通的“先预测、再决策”分成两步：

1. 用均方误差等指标训练需求预测模型；
2. 把预测需求放入库存公式，得到补货量。

这种做法默认“预测越准，决策就越好”，但两者并不完全相同。例如，把需求 100 预测成 90 和预测成 110，绝对误差都是 10；在库存中，前者可能造成缺货，后者只会多留一点库存，两种错误的业务后果不同。

SPO 的想法是：不再只看预测数值错了多少，而是看预测导致的决策比真正最优决策多付了多少成本。SPO+ 则是一个可计算、凸的替代损失，可用次梯度方法训练。

## 3. 为什么不能把论文公式原封不动套到动态库存模型

论文的标准问题是

$$
z^*(c)=\min_{w\in S} c^\top w,
$$

其中未知量 $c$ 只在线性目标函数中，约束集合 $S$ 不随 $c$ 改变。

本项目原来的动态库存模型不完全满足这个形式：真实需求不仅影响成本，还进入库存状态、销售量和缺货量的约束。论文结论也明确把“未知参数出现在约束中”留作进一步研究。因此，如果直接说原动态问题完全等同于论文模型，是不严谨的。

当前采用一个透明的有限动作适配：每个时期先产生一组候选补货目标，把每个候选在真实需求下的成本组成一个向量。模型预测的不是单个需求值，而是整个候选动作成本向量，再选择预测成本最低的动作。这个有限动作选择问题严格符合论文的线性目标形式；最终效果仍由原来的动态库存仿真器评价。

## 4. 补货问题如何变成 SPO 问题

### 4.1 候选动作

对于样本 $j$，只用目标期以前的信息计算 12 期滚动均值 $\mu_j$ 和标准差 $\sigma_j$。候选动作集合为

$$
\mathcal A=\{0,0.5,1.0,\ldots,8.0\}.
$$

动作 $a$ 对应的 order-up-to 目标是

$$
B_{ja}=\left\lceil\max(0,\mu_j+a\sigma_j)\right\rceil.
$$

这里的 $a$ 看起来像安全系数，但它不是一个全局固定的 $z$。SPO+ 可以对不同 SKU、不同周期选择不同动作。

### 4.2 真实动作成本向量

知道样本的真实下一期需求 $D_j$ 后，可计算动作 $a$ 的单期代理成本：

$$
c_{ja}
=h(B_{ja}-D_j)^+
+p(D_j-B_{ja})^+,
$$

其中：

- $(u)^+=\max(u,0)$；
- $h=1$ 是持有成本权重；
- $p$ 是缺货惩罚权重；
- 第一项惩罚多余库存，第二项惩罚缺货。

把全部 17 个动作的成本放在一起，就得到真实成本向量

$$
c_j=(c_{j1},\ldots,c_{j17})^\top.
$$

### 4.3 线性优化形式

令 $e_a$ 表示“选择动作 $a$”的 one-hot 向量，决策集合为

$$
S=\{e_1,e_2,\ldots,e_{17}\}.
$$

已知成本向量时，最优动作是

$$
w^*(c_j)\in\arg\min_{w\in S}c_j^\top w.
$$

因为 $w$ 是 one-hot 向量，这个优化问题就是从 17 个成本中找最小值，不需要 Gurobi、CPLEX 等求解器。

## 5. SPO 与 SPO+ 的数学模型

### 5.1 成本预测器

每个样本有 59 个严格滞后的特征 $x_j$，包括历史需求、滚动统计量、`x1-x20` 的滞后值和变化量等。线性模型预测动作成本向量：

$$
\hat c_j=W x_j.
$$

程序会先标准化特征并加入截距。预测第 $t$ 期时只允许使用 $t-1$ 期及更早的信息。

### 5.2 SPO 损失

模型根据预测成本选择 $w^*(\hat c_j)$。真实成本下的决策后悔值为

$$
\ell_{\mathrm{SPO}}(\hat c_j,c_j)
=c_j^\top w^*(\hat c_j)-z^*(c_j),
$$

其中

$$
z^*(c_j)=\min_{w\in S}c_j^\top w.
$$

它回答的是：“模型选的动作，比事后知道真实需求时的最佳候选动作多花了多少成本？”

### 5.3 SPO+ 损失

直接优化 SPO 损失通常不连续、非凸。论文给出的凸上界是

$$
\ell_{\mathrm{SPO+}}(\hat c,c)
=\max_{w\in S}\{c^\top w-2\hat c^\top w\}
+2\hat c^\top w^*(c)-z^*(c).
$$

其关于预测成本 $\hat c$ 的一个次梯度为

$$
2\left[w^*(c)-w^*(2\hat c-c)\right].
$$

对线性模型 $\hat c_j=W x_j$，矩阵 $W$ 的样本次梯度为

$$
2\left[w^*(c_j)-w^*(2W x_j-c_j)\right]x_j^\top.
$$

当前实现用小批量随机次梯度下降、递减学习率和加权平均迭代点训练，并增加很小的 Ridge 正则项：

$$
\min_W\frac1n\sum_{j=1}^n
\ell_{\mathrm{SPO+}}(Wx_j,c_j)
+\frac{\lambda}{2}\lVert W\rVert_F^2.
$$

这正是论文线性预测器与 mini-batch 次梯度算法在有限动作库存问题上的特例。

## 6. 训练、验证与测试协议

| 阶段 | 目标期 | 用途 |
|---|---|---|
| 调参训练 | 训练集第 13–42 期 | 拟合候选 SPO+ 模型 |
| 验证 | 训练集第 43–60 期 | 选择缺货惩罚和学习率 |
| 最终训练 | 训练集第 13–60 期 | 用选定参数重新训练 |
| 正式测试 | `replenishment_test.xlsx` 的 28 期 | 只做最终评价 |

搜索范围：

- 缺货惩罚 $p$：`10, 20, 40, 80, 160`；
- 学习率：`0.02, 0.05, 0.10, 0.20`；
- Ridge：`0.0001`；
- 动作：`0` 到 `8`，步长 `0.5`；
- 调参和最终训练：均为 8 个 epoch；
- batch size：`2048`。

所有候选先跑完整验证期库存仿真。若多个候选达到 95% 总体 Fill Rate，就选择平均总库存最低者；否则选择 Fill Rate 最高者。最终选中：

```text
shortage_penalty = 80
learning_rate = 0.20
ridge = 0.0001
epochs = 8
```

测试集没有参与参数选择。`test_wape` 留空是有意的，因为该模型预测动作成本而不是需求，拿 WAPE 评价它没有明确含义。

## 7. 最终动态库存评价

SPO+ 的训练成本是单期代理成本，但正式评价没有使用代理指标。选出的每期 order-up-to 水平被送入与所有其他模型相同的动态库存评价器：

1. 期初接收上一期下单的货物；
2. 本期需求发生；
3. 销售量为现货与需求的较小值，未满足需求按 lost sales 处理；
4. 期末下单至 SPO+ 选择的下一期目标；
5. 提前期为 1。

总体 Fill Rate 仍为

$$
\text{Fill Rate}
=\frac{\sum_{i,t}\text{Sales}_{it}}
{\sum_{i,t}\text{Demand}_{it}}.
$$

因此 SPO+ 与 baseline、时间序列、机器学习和 DRO 的结果可以直接比较。

## 8. 与已有方案的比较

| 排序仅限合格方案 | 模型 | Fill Rate | 平均总库存 | 总周转率 |
|---:|---|---:|---:|---:|
| 1 | LightGBM | 96.9577% | 585,283 | 149.8646 |
| 2 | XGBoost | 97.8206% | 838,215 | 105.5741 |
| 3 | CatBoost | 97.5982% | 995,719 | 88.6722 |
| 4 | Moving Average 12 | 95.8390% | 2,937,675 | 29.5135 |
| 5 | Wasserstein DRO + MA12 | 96.4349% | 3,133,250 | 27.8434 |
| 6 | 卡方 DRO + MA12 | 96.7745% | 3,268,233 | 26.7874 |
| 11 | **SPO+ Linear** | **95.0235%** | **3,684,607** | **23.3304** |
| 8 | 矩信息 DRO + MA12 | 98.2566% | 4,309,164 | 20.6277 |
| 9 | Mean–Variance Baseline | 98.7838% | 11,949,377 | 7.4787 |
| 10 | SVR | 99.1511% | 54,791,768 | 1.6371 |

这里先按是否达到 95% 筛选，再按平均库存从低到高排序。SPO+ 的具体比较是：

- 相对 baseline：平均库存减少 69.16%，Fill Rate 下降 3.760 个百分点，但仍合格；
- 相对 MA12：库存增加 30.97%，Fill Rate 低 0.530 个百分点；
- 相对 Wasserstein DRO：库存增加 22.80%，Fill Rate 低 1.126 个百分点；
- 相对 LightGBM：库存约高 557.38%，Fill Rate 低 1.649 个百分点。

## 9. 为什么论文方法没有成为第一名

这不表示论文方法错误，主要有五个原因：

1. **训练问题是原动态库存问题的代理。** 为满足论文“未知参数在线性目标中”的条件，训练时使用单期候选动作成本；动态提前期和跨期库存只在最终仿真中体现。
2. **当前成本预测器是线性的。** 论文实验也指出，当真实关系存在非线性或模型错设时，SPO+ 可能优于普通回归，但这不意味着线性 SPO+ 一定超过强大的非线性提升树。
3. **动作被离散化。** 模型只能从 17 个 $\mu+a\sigma$ 目标中选择，表达能力低于直接输出任意连续目标。
4. **持有与缺货成本来自代理权重。** 数据没有给出真实业务成本，只能在验证集搜索 $p$。若有真实毛利、持有费和缺货损失，SPO 目标会更有业务含义。
5. **仍只有一个验证窗口。** 训练后期和测试期分布有变化，单窗口选择不够稳定。

论文方法最重要的贡献在这里已经体现：模型的学习目标从“尽量预测准需求”变成“尽量选对库存动作”。结果合格，但当前工程证据支持把它保留为研究候选，而不是替换 LightGBM。

## 10. 工程文件

| 文件 | 作用 |
|---|---|
| `src/models/spo/spo_plus.py` | 候选动作、真实动作成本、SPO+ 线性模型和次梯度训练 |
| `experiments/run_spo_plus.py` | 数据切分、调参、最终重训、测试和统一比较入口 |
| `tests/test_spo_plus.py` | 动作成本计算和 SPO+ 可学习性的单元测试 |
| `outputs/spo/spo_plus_linear/model.npz` | 最终系数、标准化参数和动作网格 |
| `outputs/spo/spo_plus_linear/validation_candidates.csv` | 20 个候选的验证结果 |
| `outputs/spo/spo_plus_linear/training_history.csv` | 每个 epoch 的训练诊断 |
| `outputs/spo/spo_plus_linear/test_selected_actions.csv` | 每个 SKU、每期选择的安全倍数动作 |
| `outputs/spo/spo_plus_linear/test_order_up_to.csv` | 每个 SKU、每期目标库存 |
| `outputs/spo/spo_plus_linear/test_per_sku_metrics.csv` | 每个 SKU 的 Fill Rate、库存和周转率 |
| `outputs/spo/spo_plus_linear/test_trajectory.csv` | 逐 SKU、逐期库存和补货轨迹 |
| `outputs/spo/spo_plus_linear/summary.json` | 参数与正式指标摘要 |
| `outputs/spo/spo_vs_existing_comparison.csv` | SPO+ 与代表性已有模型的比较 |

## 11. 完整复现命令

请直接在终端粘贴：

```bash
cd "<PROJECT_ROOT>"

python -m pip install -r requirements.txt

mkdir -p /private/tmp/replenishment_font_cache

XDG_CACHE_HOME=/private/tmp/replenishment_font_cache \
PYTHONPYCACHEPREFIX=/private/tmp/replenishment_spo_pycache \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/private/tmp/matplotlib \
python \
  -u -m experiments.run_spo_plus --item-id 1
```

仅检查代码链路时可以加 `--quick`，但 quick 模式只训练 2 个 epoch，结果不能写入正式比较。

## 12. 下一步怎样改进才有研究价值

优先级从高到低建议如下：

1. 用多个 rolling-origin 验证窗口选择 $p$、学习率和 epoch，降低单窗口偶然性；
2. 把线性成本预测器升级为非线性 SPO+ 模型，但仍单独评价，不立即做 stacking；
3. 设计能更准确反映提前期和跨期库存的动作成本，而不只使用单期 newsvendor 代理；
4. 按品类或需求规模设置动作网格，避免所有 SKU 共用同一组安全倍数；
5. 如果获得真实持有成本、缺货成本和毛利，用真实业务成本替代人工惩罚权重；
6. 在新的、从未查看过的最终留出集上重新比较，避免反复查看当前测试集造成研究选择偏差。

当前最合理的定位是：`spo_plus_linear` 是一个数学上对应论文、工程上可运行、指标上合格，但仍有明显改进空间的决策导向学习基线。


## 6. 限制

不跨A/B/C组混排；总量周转和加权周转分开；软服务罚项不等于95%硬约束。多轮开发使用过同一个测试集，不是独立确认性结果。有限搜索、首阶近似、容量不足等限制按原报告保留。此次代码重构以语法、导入、小型逻辑与数值一致性校验为主，真实全量重训不是此次任务。
