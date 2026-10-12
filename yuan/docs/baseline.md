# baseline：模型实现与说明

[总览](../README.md) · [实际代码](../models/baseline.py) · [全部变体记录](../results/variants.json)

## 1. 文件如何对应

本模型所有定义、关键决策/损失函数或估计器构造在`models/baseline.py`，不是转发到旧experiments目录的脚本。数据/评价/公共训练流程位于common/。参数、窗口、种子和消融不另建模型文件。历史125项仅为结果记录，完整索引见results/variants.json。

## 2. 信息时序与模型原理

### A组

每SKU计算历史均值和样本标准差，验证选择公共安全系数z；s=S退化为目标库存策略。不使用x、不训练ML。

SKU是商品编号；x1–x20是外部属性/特征，不是需求本身。历史滞后表示预测目标日前已观察值。库存状态是今天卖完后的库存，不是明天已实现的库存。预测层误差小不自动意味着决策服务达标。

### 核心公式

$$
S_i=s_i=\lceil\max(0,\mu_iL+z\sigma_i\sqrt L)\rceil,\qquad L=1
$$

d为需求，I为期末库存，s为实际销量，q为下单量；预测目标用S表示，C为公共容量，z为风险余量系数；a_i为训练期商品尺度，theta为学习参数。上式是核心计算关系/目标概要，不把代理损失写成业务硬约束。完整推导和实现细节见后面的完整推导；共享符号、Fill Rate与Ready Rate区别见[评价标准](evaluation.md)。


### B组

历史统计或38维全局预测器：17个严格滞后需求统计、品类、20个当期x。数量特征按训练SKU均值缩放；ML目标为log(1+d)减SKU均值偏置。验证选择参数和安全系数z∈{0,0.5,1,2,3,4}，共享容量超出时比例缩小订单。当前x已知、补货立即到货：不是最新业务标准。

SKU是商品编号；x1–x20是外部属性/特征，不是需求本身。历史滞后表示预测目标日前已观察值。库存状态是今天卖完后的库存，不是明天已实现的库存。预测层误差小不自动意味着决策服务达标。

### 核心公式

$$
S_i=\widehat d_i+z\sigma_i,\quad q_i=(S_i-I_i)^+\min\{1,C/\max(\sum_j(S_j-I_j)^+,10^{-8})\}
$$

d为需求，I为期末库存，s为实际销量，q为下单量；预测目标用S表示，C为公共容量，z为风险余量系数；a_i为训练期商品尺度，theta为学习参数。上式是核心计算关系/目标概要，不把代理损失写成业务硬约束。完整推导和实现细节见后面的完整推导；共享符号、Fill Rate与Ready Rate区别见[评价标准](evaluation.md)。


### C组

复用旧版严格滞后ML冻结预测，不重新选择模型。今天需求揭晓后根据明天预测下单，明天到货；无安全库存目标等于预测，参考目标加4倍训练需求标准差。零首日在途、最后一期不下单、共享容量同比缩小。该入口批量复算全部20项；z4是固定参考，不是此标准重新调优的最优策略。

SKU是商品编号；x1–x20是外部属性/特征，不是需求本身。历史滞后表示预测目标日前已观察值。库存状态是今天卖完后的库存，不是明天已实现的库存。预测层误差小不自动意味着决策服务达标。

### 核心公式

$$
S_{i,t+1}=\widehat d_{i,t+1}+z\sigma_i,\quad q_{it}=(S_{i,t+1}-I_{it})^+\min(1,C/\max(\sum_j(S_{j,t+1}-I_{jt})^+,10^{-8}))
$$

d为需求，I为期末库存，s为实际销量，q为下单量；预测目标用S表示，C为公共容量，z为风险余量系数；a_i为训练期商品尺度，theta为学习参数。上式是核心计算关系/目标概要，不把代理损失写成业务硬约束。完整推导和实现细节见后面的完整推导；共享符号、Fill Rate与Ready Rate区别见[评价标准](evaluation.md)。


## 3. 运行命令

从本仓库根目录运行；默认只查看历史结果，不训练：

```bash
python -m models.baseline
python -m models.baseline --help
python -m models.baseline --train
```

普通ML与baseline可用`--protocol legacy`、`--protocol ready_immediate`、`--protocol lead_one`切换已存在实验口径。lead_one为旧冻结预测的安全库存移除/固定z4对照，不是重新在该口径调参。移动平均用`--window 3/6/12`；MPC用`--round all/1/2/3/4`。高级模型默认沿用原批次的验证/消融，但实现全在同一文件，不会调用旧脚本树。

原始Excel放data/。运行只写results/generated/，不覆盖results/variants.json中的历史快照。改变输入请新目录重跑，不允许旧缓存自动作为新数据证据。

## 4. 历史结果（所有配置集中列在这里）

|原标签|组|角色|Fill Rate|Ready Rate达标比例|平均总库存|原始周转|加权周转|
|---|---|---|---:|---:|---:|---:|---:|
|legacy_baseline|A|主方案|0.987838|未记录|1.19494e+07|7.47865|未记录|
|immediate_historical_baseline|B|主方案|0.998037|0.9835|1.47446e+07|6.12346|7.27086|
|lead1_historical_baseline_no_safety|C|无安全库存/固定安全参考|0.912783|0.512333|2.91793e+06|28.2993|34.5798|
|lead1_historical_baseline_fixed_z4_reference|C|无安全库存/固定安全参考|0.998037|0.9835|1.42847e+07|6.32061|7.51282|

小数比例0.95就是95%。参数与来源SHA256逐项保存在results/variants.json。历史业务失败没有改成通过；没有因为此次重构而重训全部真实数据。

## 5. 完整数学与训练细节

以下为本模型相关历史说明的完整正文，记录旧研究假设；当时“最优”只限于自己的实验口径。运行请用上面的新命令，以下旧文件路径仅作为出处，不表示新目录仍有旧结构。

### 原始说明：replenishment_solution_report.md

### 补货问题方案与实验结果总报告

> 项目：Replenishment Problem  
> 报告口径：基于当前项目中已经实现并完成测试的 25 个方案  
> 数据文件：`data/replenishment_train.xlsx` 与 `data/replenishment_test.xlsx`  
> 核心目标：在测试期总体 Fill Rate 不低于 95% 的前提下，使平均总库存尽可能低

## 0. 一页结论

本项目研究的是一个包含 6,000 个 SKU 的单级库存补货问题。每个周期开始时，上期下达的订单到货；随后发生需求；未满足的需求直接损失，不延期交付；周期结束时，系统决定补多少货，订单在下一周期开始时到达。因此当前提前期为 1 个周期。

截至本报告，已经实现并实际运行了五类共 25 个方案：

- 1 个均值—方差 base-stock baseline；
- 6 个只使用历史需求的时间序列方案；
- 9 个“先预测、再决策”的机器学习方案；
- 3 个使用不同模糊集合的分布鲁棒优化（DRO）方案。
- 6 个决策导向SPO+方案：线性SPO+、XGBoost–SPO+、LightGBM–SPO+，不预测需求的直接XGBoost–SPO一步版和滚动版，以及直接LightGBM–SPO+–Rank。

按照项目预先确定的选择规则——**先筛除测试总体 Fill Rate 低于 95% 的方案，再在合格方案中选择平均总库存最低者**——当前最优方案是 **LightGBM**：

- 测试总体 Fill Rate：**96.9577%**；
- 平均总库存：**585,283**；
- 总周转率：**149.8646**；
- 6,000 个 SKU 中，单 SKU Fill Rate 达到 95% 的比例：**91.5167%**；
- 相比 baseline，平均总库存下降 **95.10%**。

baseline 的 Fill Rate 为 **98.7838%**，看起来非常好，但平均总库存达到 **11,949,377**。它不是算错了，而是用很高的库存换取了高服务率。训练集每个“SKU × 周期”的平均需求约为 965.8，测试集只有 538.5，下降约 44.2%；固定库存上限的 baseline 没有随测试期需求下降而及时降低目标库存，因此产生大量冗余库存。高 Fill Rate 本身不能证明补货方案优秀，必须和库存同时看。

在当前总量约束下，十五个合格方案按平均库存从低到高排列如下：

| 排名 | 方案 | 测试总体 Fill Rate | 平均总库存 | 相比 baseline 降库存 | 结论 |
|---:|---|---:|---:|---:|---|
| 1 | LightGBM | 96.9577% | 585,283 | 95.10% | 当前主推荐方案 |
| 2 | XGBoost–SPO+ | 97.3647% | 825,619 | 93.09% | 决策导向学习的最佳方案 |
| 3 | XGBoost | 97.8206% | 838,215 | 92.99% | 服务更高，库存略多 |
| 4 | LightGBM–SPO+ | 97.6059% | 947,906 | 92.07% | 可行，但未超过XGBoost–SPO+ |
| 5 | CatBoost | 97.5982% | 995,719 | 91.67% | SKU 服务覆盖率很高 |
| 6 | Direct XGB–SPO Rollout | 95.0384% | 2,128,309 | 82.19% | 不预测需求，直接决策的最佳版本 |
| 7 | Direct LightGBM–SPO+–Rank | 95.9206% | 2,198,542 | 81.60% | 不预测需求；混合决策损失 |
| 8 | 12 期移动平均 | 95.8390% | 2,937,675 | 75.42% | 简单、透明的强基准 |
| 9 | Wasserstein DRO + MA12 | 96.4349% | 3,133,250 | 73.78% | MA12 的温和稳健增强 |
| 10 | 卡方散度 DRO + MA12 | 96.7745% | 3,268,233 | 72.65% | 强调历史坏场景的概率上升 |
| 11 | SPO+ Linear | 95.0235% | 3,684,607 | 69.16% | 决策导向学习，当前未超过MA12 |
| 12 | 矩信息 DRO + MA12 | 98.2566% | 4,309,164 | 63.94% | 最保守的 DRO |
| 13 | Direct XGB–SPO One-step | 96.4354% | 4,871,152 | 59.24% | 一步成本偏短视，库存高于滚动版 |
| 14 | 均值—方差 baseline | 98.7838% | 11,949,377 | 0.00% | 稳健但明显过量备货 |
| 15 | SVR | 99.1511% | 54,791,768 | -358.53% | 虽合格，但库存不可接受 |

这里的“当前最优”只针对“总体 Fill Rate ≥ 95%，然后最小化平均库存”这一目标。如果业务要求**每个 SKU 都尽量达到 95%**，结论会发生变化：XGBoost 和 CatBoost 的单 SKU 达标比例分别为 97.7667% 和 98.1167%，明显高于 LightGBM 的 91.5167%。因此总量服务和 SKU 服务公平性是两个不同目标。

---

## 1. 这个补货问题到底在解决什么

零售或供应链系统每天都要回答一个问题：每个 SKU 今天应该补多少货？

补少了会缺货，损失销售；补多了会占用资金和仓储空间。这个项目把目标写成一个带约束的优化问题：

```text
最小化：测试期平均总库存
约束：测试期总体 Fill Rate ≥ 95%
```

“平均总库存”中的“总”表示先把 6,000 个 SKU 在同一周期的期末库存相加，再对 28 个测试周期取平均。它不是单个 SKU 的平均库存。

### 1.1 一个周期内发生事件的顺序

所有方案都由同一个评价器模拟，事件顺序完全一致：

```text
周期开始
   ↓
上周期订单到货
   ↓
本周期需求发生
   ↓
现有库存尽量满足需求；满足不了的部分记为 lost sales
   ↓
记录期末库存
   ↓
根据目标库存下达补货订单
   ↓
订单在下一周期开始时到货
```

提前期 `lead_time = 1`。第一测试期的库存来自测试文件中的 `initial_inventory`，任何模型都无法在第一测试期开始前用自己的预测“补写”一笔到货。这保证了比较的一致性。

### 1.2 为什么当前写成 `s = S`

经典 `(s,S)` 策略的含义是：库存位置降到或低于 `s` 时，下单补到 `S`。当前问题没有固定订货成本、最小起订量或整箱约束，因此没有必要人为制造一个 `s` 与 `S` 的间隔。我们采用退化后的 base-stock 策略：

```text
s = S = 目标库存水平 B
```

换句话说，每个周期都把下一期需要的库存补到一个目标水平。这是本项目最简单、也最容易公平比较的决策层。

---

## 2. 数据、规模和可用信息

### 2.1 数据规模

| 数据 | SKU 数 | 周期数 | 匿名特征数 | SKU × 周期样本 | 需求总量 |
|---|---:|---:|---:|---:|---:|
| 训练集 | 6,000 | 60 | 20 | 360,000 | 347,692,252 |
| 测试集 | 6,000 | 28 | 20 | 168,000 | 90,465,437 |

此外，数据中还有 `item_id`、`category` 和 `initial_inventory`。当前发现 2 个品类。

训练集每个 SKU-周期的平均需求为 965.81，测试集为 538.48，测试/训练比值约为 55.75%。这说明训练期和测试期存在明显的需求规模变化，也是固定目标库存 baseline 在测试期库存偏高的重要原因。

### 2.2 信息可用时间：禁止使用当期 `x`

业务设定是：预测下一期需求时，下一期的 `x1`–`x20` **不能提前知道**。因此：

- baseline 和时间序列模型完全不使用 `x1`–`x20`；
- 当前三个 DRO 以 MA12 为中心预测，也不使用 `x1`–`x20`；
- 机器学习预测第 `t` 期需求时，只能使用第 `t-1` 期及更早的需求与 `x`；
- 机器学习会使用 `x(t-1)` 以及 `x(t-1)-x(t-2)`，绝不使用 `x(t)`；
- 测试采用滚动一步预测：观察完本期真实需求和特征后，才允许构造下一期预测。

这条规则很重要。如果把测试期当期 `x` 直接放入模型，离线结果可能更漂亮，但在真实决策时拿不到这些变量，结果属于数据泄漏，无法部署。

---

## 3. 统一数学模型与评价指标

这一节先把所有方案共同面对的库存系统写成数学形式。后面的 baseline、时间序列和机器学习只是用不同方法生成目标库存，库存流转和评价约束完全相同。

### 3.1 符号表

| 符号 | 类型 | 含义 |
|---|---|---|
| $\mathcal I=\{1,\ldots,N\}$ | 集合 | SKU 集合，当前 $N=6000$ |
| $\mathcal T=\{1,\ldots,T\}$ | 集合 | 评价周期集合，测试集 $T=28$ |
| $D_{it}$ | 外生参数/随机量 | SKU $i$ 在周期 $t$ 的真实需求 |
| $H_{i0}$ | 外生参数 | 测试文件给出的初始现货库存 |
| $A_{it}$ | 状态量 | 周期 $t$ 期初到货量 |
| $Y_{it}$ | 结果量 | 周期 $t$ 实际销售量 |
| $U_{it}$ | 结果量 | 周期 $t$ 损失销售量 |
| $H_{it}$ | 状态量 | 满足周期 $t$ 需求后的期末现货库存 |
| $Q_{it}$ | 决策变量 | 周期 $t$ 期末下达的补货量，周期 $t+1$ 到货 |
| $B_{i,t+1}$ | 策略输出 | 周期 $t$ 决策时为下一期设定的目标库存 |
| $\widehat D_{i,t+1\mid t}$ | 预测值 | 利用截至周期 $t$ 的信息预测周期 $t+1$ 需求 |
| $\widehat\sigma_i$ | 估计参数 | SKU $i$ 的历史需求或预测误差标准差 |
| $z$ | 策略参数 | 全部 SKU 共用的安全系数 |
| $\mathcal F_t$ | 信息集合 | 周期 $t$ 做决策时已经观察到的信息 |

大写 $D,H,Q$ 等表示需求、状态或决策；小写参数 $z$ 表示需要校准的安全系数。下标 $i$ 表示 SKU，下标 $t$ 表示周期。

### 3.2 建模假设及其作用

**假设 1：单级库存系统。** 每个 SKU 只考虑一个库存节点，不描述供应商—仓库—门店的多级网络。该假设使补货和库存流转可以逐 SKU 模拟；放松后需要加入节点和运输弧下标。

**假设 2：提前期固定为一个周期。** 周期 $t$ 期末下单，恰好在 $t+1$ 期初到货。它与当前评价器一致；若不同 SKU 的提前期不同，目标库存必须覆盖各自提前期内的累计需求。

**假设 3：缺货为 lost sales。** 未满足需求立即损失，不形成延期订单。因此 $U_{it}$ 不进入下一周期状态；若改为 backorder，需要增加欠单状态及欠单成本。

**假设 4：补货无容量、预算和起订量限制。** $Q_{it}$ 只需非负，可取任意整数数量。当前数据没有给出固定订货成本或整箱规则，因此采用 $s=S$；放松后才有必要建立真正分离的 $(s,S)$ 或混合整数模型。

**假设 5：周期内事件顺序固定。** 先到货，再发生需求，最后下单。更改顺序会改变同一组订单产生的 Fill Rate，因此所有模型必须共享这一假设。

**假设 6：决策不能看到未来。** $B_{i,t+1}$ 必须是信息集 $\mathcal F_t$ 的函数。对机器学习而言，$\mathcal F_t$ 最多包含 $D_{i1:t}$ 与 $x_{i1:t}$；预测 $D_{it}$ 时最多使用 $t-1$ 及更早信息。

### 3.3 库存流转约束

第一测试期没有由模型在测试前补写的到货：

$$
A_{i1}=0,\qquad A_{it}=Q_{i,t-1},\quad t=2,\ldots,T. \tag{I1}
$$

可售库存为 $H_{i,t-1}+A_{it}$。实际销售不能超过需求，也不能超过可售库存：

$$
Y_{it}=\min\{D_{it},\ H_{i,t-1}+A_{it}\}. \tag{I2}
$$

损失销售与期末库存分别为：

$$
U_{it}=D_{it}-Y_{it}, \tag{I3}
$$

$$
H_{it}=H_{i,t-1}+A_{it}-Y_{it}. \tag{I4}
$$

给定下一期目标库存 $B_{i,t+1}$，期末补货量为：

$$
Q_{it}=\max\{0,\ B_{i,t+1}-H_{it}\}. \tag{I5}
$$

并满足 $Y_{it},U_{it},H_{it},Q_{it}\ge 0$。公式（I1）—（I5）就是代码中库存评价器逐期执行的数学形式。对于当前一周期提前期，期末已经没有其他在途订单，所以公式（I5）等价于把库存位置补到目标水平。

### 3.4 统一优化目标与信息约束

若把模型或策略记为 $\pi$，它根据当时可见信息产生目标库存：

$$
B_{i,t+1}=g_{\pi}(\mathcal F_t), \qquad
Q_{it}=\max\{0,g_{\pi}(\mathcal F_t)-H_{it}\}. \tag{I6}
$$

本项目希望在候选策略集合 $\Pi$ 中选择：

$$
\begin{aligned}
\min_{\pi\in\Pi}\quad
& \overline H(\pi)=\frac{1}{T}\sum_{t=1}^{T}\sum_{i\in\mathcal I}H_{it}(\pi) \\
\text{s.t.}\quad
& \operatorname{FR}(\pi)=
\frac{\sum_{i\in\mathcal I}\sum_{t=1}^{T}Y_{it}(\pi)}
{\sum_{i\in\mathcal I}\sum_{t=1}^{T}D_{it}}
\ge 0.95, \\
& \text{约束（1）—（6）成立。}
\end{aligned} \tag{I7}
$$

公式（I7）是**策略选择目标**，不是一个可以读取测试期未来需求的“先知优化模型”。实际决策必须满足 $\mathcal F_t$ 的信息约束。代码用训练集和验证集确定策略及参数，再在测试集逐期仿真；测试期真实需求只有到达该周期后才会加入下一次决策的信息集。

当前算法并没有在所有可能策略中求理论全局最优，而是在已经实现的 25 个候选策略中进行工程化比较。严格地说，它们是带数据校准的补货策略或启发式求解程序。

### 3.5 总体 Fill Rate：本项目的硬约束

$$
\operatorname{FR}_{\mathrm{total}}
=\frac{\sum_{i\in\mathcal I}\sum_{t\in\mathcal T}Y_{it}}
{\sum_{i\in\mathcal I}\sum_{t\in\mathcal T}D_{it}}. \tag{M1}
$$

它是按需求量加权的。高销量 SKU 对总体 Fill Rate 的影响更大。当前约束要求总体 Fill Rate 至少为 95%。

### 3.6 单 SKU Fill Rate 与 SKU 达标比例

$$
\operatorname{FR}_i=
\frac{\sum_{t\in\mathcal T}Y_{it}}
{\sum_{t\in\mathcal T}D_{it}}, \tag{M2}
$$

$$
\operatorname{Share}_{95}
=\frac{1}{N}\sum_{i\in\mathcal I}
\mathbb I\{\operatorname{FR}_i\ge0.95\}, \tag{M3}
$$

其中 $\mathbb I\{\cdot\}$ 是指示函数：条件成立取 1，否则取 0。

零需求 SKU 的 Fill Rate 记为 100%。注意：项目当前优化约束是**总体** Fill Rate ≥ 95%，不是“每个 SKU 均达到 95%”。所以总体合格的 LightGBM 仍有一部分低销量 SKU 未达到 95%。

### 3.7 平均总库存：主要优化目标

$$
\overline H_{\mathrm{total}}
=\frac{1}{T}\sum_{t\in\mathcal T}\sum_{i\in\mathcal I}H_{it}. \tag{M4}
$$

数值越低越好，但只能在 Fill Rate 已经合格的模型之间比较。

### 3.8 总周转率

$$
\operatorname{Turnover}_{\mathrm{total}}
=\frac{\sum_{i\in\mathcal I}\sum_{t\in\mathcal T}Y_{it}}
{\overline H_{\mathrm{total}}}. \tag{M5}
$$

该指标表示整个 28 期测试窗口的周转次数，未年化。周转率越高通常越好，但服务率不合格时，高周转率往往只是因为库存太少、缺货太多，不能被视为优秀结果。

### 3.9 WAPE：预测指标，不是最终补货目标

$$
\operatorname{WAPE}
=\frac{\sum_{i\in\mathcal I}\sum_{t\in\mathcal T}
|D_{it}-\widehat D_{it\mid t-1}|}
{\sum_{i\in\mathcal I}\sum_{t\in\mathcal T}D_{it}}. \tag{M6}
$$

WAPE 越低，表示需求预测的总体绝对误差越小。不过 WAPE 对高估和低估对称，而补货业务不对称：低估会导致缺货，高估会导致库存。因此，预测 WAPE 最低的模型不一定是库存决策最优的模型。最终选择必须回到 Fill Rate 与库存仿真。

---

## 4. 实验如何避免“用测试答案调模型”

本项目遵守按时间切分的原则，不随机打乱未来和过去。

### 4.1 Baseline 与时间序列模型

- 训练文件前 42 期（70%）用于估计或初始化模型；
- 后 18 期（30%）作为验证集，用来选择模型参数和安全系数 `z`；
- 参数选定后，再利用完整 60 期历史生成测试阶段所需的模型状态；
- 28 期测试数据只用于最终滚动评价。

### 4.2 机器学习模型

由于最大需求滞后为 12 期，最早可预测目标是第 13 期：

- 目标期 13–42：180,000 行，用于候选模型拟合；
- 目标期 43–60：108,000 行，用于参数与 `z` 选择；
- 参数选定后，用目标期 13–60 的 288,000 行重训最终模型；
- 测试文件的 28 期需求只用于滚动测试。

随机森林调参时最多抽样 120,000 行，但最终用全部 288,000 行重训；SVR 计算量很大，调参抽样 15,000 行、最终训练抽样 20,000 行；抽样同时按时期和需求规模分层，随机种子固定为 42。其他机器学习模型使用全量样本。

### 4.3 DRO 模型

- 前 42 期用于生成 MA12 严格一步预测残差并拟合模糊集合；
- 后 18 期用于选择方差放大系数或分布距离半径；
- 选中参数后，用完整 60 期重新估计残差分布和鲁棒安全倍数；
- 测试期只进行滚动预测与最终评价，不参与模糊集合参数选择。

### 4.4 时间序列与机器学习的安全库存校准

时间序列和机器学习模型都把预测转换成动态 base-stock：

```text
下一期目标库存 = ceil(下一期需求预测 + z × 该 SKU 的历史预测误差标准差)
s_t = S_t = 下一期目标库存
```

其中 `ceil` 表示向上取整，目标库存不会小于 0。程序在验证集上从小到大搜索 `z`，步长为 0.05，选择达到 95% 总体 Fill Rate 所需的最小安全系数。时间序列和机器学习的搜索区间均为 `[0, 8]`；baseline 的区间为 `[0, 4]`。

---

## 5. 方案一：均值—方差 baseline

### 5.1 Baseline 的完整数学模型

Baseline 不预测每一个未来周期，而是从历史需求估计每个 SKU 的固定 base-stock 水平。设校准统计量所使用的历史周期集合为 $\mathcal C$，其长度为 $n_C$。SKU $i$ 的样本均值和样本标准差为：

$$
\overline d_i^{\mathcal C}
=\frac{1}{n_C}\sum_{t\in\mathcal C}D_{it}, \tag{B1}
$$

$$
\widehat\sigma_i^{\mathcal C}
=\sqrt{\frac{1}{n_C-1}
\sum_{t\in\mathcal C}
\left(D_{it}-\overline d_i^{\mathcal C}\right)^2}. \tag{B2}
$$

给定提前期 $L$ 和安全系数 $z$，固定目标库存为：

$$
B_i(z)=
\left\lceil
L\overline d_i^{\mathcal C}
+z\sqrt{L}\widehat\sigma_i^{\mathcal C}
\right\rceil, \tag{B3}
$$

并令：

$$
s_i=S_i=B_i(z). \tag{B4}
$$

一般 $(s,S)$ 策略先定义订货时点的库存位置 $IP_{it}$（现货库存加在途库存，若有欠单还要减欠单）：

$$
Q_{it}=
\bigl(S_i-IP_{it}\bigr)
\mathbb I\{IP_{it}\le s_i\}.
$$

当 $s_i=S_i=B_i$ 时，如果库存位置低于或等于 $B_i$，就补到 $B_i$；如果高于 $B_i$，则不订货。因此可统一写成 $Q_{it}=\max\{0,B_i-IP_{it}\}$。在当前 $L=1$ 且每期下单前没有其他未到在途订单的时序下，$IP_{it}=H_{it}$，于是得到后面的公式（B6）。这说明代码里的 base-stock 确实是 $(s,S)$ 在 $s=S$ 时的特例。

公式（B3）的第一项 $L\overline d_i$ 是提前期内的平均需求，第二项 $z\sqrt L\widehat\sigma_i$ 是安全库存。当前 $L=1$，因此它简化为：

$$
B_i(z)=\left\lceil
\overline d_i^{\mathcal C}+z\widehat\sigma_i^{\mathcal C}
\right\rceil. \tag{B5}
$$

在每个周期需求发生后，baseline 的订货规则为：

$$
Q_{it}=\max\{0,B_i(z)-H_{it}\}, \tag{B6}
$$

该订单在下一周期开始时到货。由于 $B_i$ 不随 $t$ 变化，baseline 是**静态目标库存策略**。

需要特别说明：公式（B3）采用了正态安全库存公式的“均值 + 标准差”结构，但代码没有假定需求一定严格服从正态分布，也没有直接把 $z$ 设为标准正态分布的 95% 分位数 1.645。这里的 $z$ 通过真实历史验证仿真校准，因此是经验安全系数。

### 5.2 为什么要使用后 30% 训练期

`s` 和 `S` 虽然相等，但它们的共同数值仍然未知。后 30% 训练期不是用来“寻找 S 的另一半”，而是用来选择共同安全系数 `z`。

令前 42 期为统计估计集 $\mathcal C$，后 18 期为验证集 $\mathcal V$，候选集合为：

$$
\mathcal Z_B=\{0,0.05,0.10,\ldots,4.00\}. \tag{B7}
$$

为了从一个稳定状态开始校准，baseline 验证仿真把初始库存设为对应候选的目标库存：

$$
H_{i0}^{\mathcal V}(z)=B_i^{\mathcal C}(z). \tag{B7a}
$$

这只用于训练文件内部的验证。正式测试时，所有模型统一采用 `replenishment_test.xlsx` 给出的 $H_{i0}^{\mathrm{test}}$，不会把 baseline 的目标库存当成测试初始库存。

对每一个 $z\in\mathcal Z_B$，先由公式（B1）—（B6）仿真验证期，再选择第一个达到服务约束的候选：

$$
z_B^*=\min\left\{
z\in\mathcal Z_B:\operatorname{FR}_{\mathcal V}(z)\ge0.95
\right\}. \tag{B8}
$$

本次得到 $z_B^*=3.20$。确定安全系数后，再用完整 60 期训练集合 $\mathcal R$ 重新估计 $\overline d_i^{\mathcal R}$ 和 $\widehat\sigma_i^{\mathcal R}$，得到测试阶段最终参数：

$$
B_i^{\text{test}}=
\left\lceil
L\overline d_i^{\mathcal R}
+z_B^*\sqrt L\widehat\sigma_i^{\mathcal R}
\right\rceil,qquad
s_i^{\text{test}}=S_i^{\text{test}}=B_i^{\text{test}}. \tag{B9}
$$

如果候选区间内没有任何 $z$ 达到 95%，代码会采用上界 $z=4$ 并如实报告未达标；本次不存在这个问题。

最终 6,000 个 SKU 的 `S` 合计为 15,674,780，单 SKU 的中位数为 535.5，均值为 2,612.5。由于需求分布高度不均匀，95% 分位数已经达到 20,686.7。

### 5.3 Baseline 结果

| 阶段 | 总体 Fill Rate | 平均总库存 | 总周转率 | SKU 95% 达标比例 |
|---|---:|---:|---:|---:|
| 验证集 | 95.1619% | 2,755,268 | 56.7817 | 74.1167% |
| 测试集 | 98.7838% | 11,949,377 | 7.4787 | 98.7500% |

测试期总需求为 90,465,437，总销售为 89,365,213，损失销售为 1,100,224。

### 5.4 如何正确理解它

baseline 的优势是简单、透明、运行快、服务率高；缺点是目标库存固定，无法适应需求水平随时间下降。它是一个有效的“保守下限比较对象”：任何复杂模型都应该证明自己能在满足服务率的同时显著减少库存。

---

## 6. 方案二：时间序列预测后补货

时间序列模型只看每个 SKU 自己的历史需求，不使用 `x1`–`x20`。每个测试周期都进行滚动一步预测，然后通过共同的安全库存层转成下一期补货目标。

### 6.1 时间序列方案的统一数学结构

设 $\widehat D_{i,t+1\mid t}^{(m)}$ 为模型 $m$ 在周期 $t$ 结束后对下一期需求的预测。模型只能利用 $D_{i1},\ldots,D_{it}$。用模型在训练历史上的一步预测残差：

$$
e_{it}^{(m)}=D_{it}-\widehat D_{it\mid t-1}^{(m)} \tag{T1}
$$

估计每个 SKU 的残差标准差：

$$
\widehat\sigma_{i,m}^{e}
=\operatorname{sd}\left(\{e_{it}^{(m)}:t\in\mathcal C\}\right). \tag{T2}
$$

预测被转换成下一期动态目标库存：

$$
B_{i,t+1}^{(m)}(z)=
\left\lceil
\max\left\{0,
\widehat D_{i,t+1\mid t}^{(m)}
+z\widehat\sigma_{i,m}^{e}
\right\}
\right\rceil. \tag{T3}
$$

再通过 $Q_{it}=\max\{0,B_{i,t+1}^{(m)}-H_{it}\}$ 下单。对固定模型 $m$，验证期安全系数为：

$$
z_m^*=\min\left\{
z\in\{0,0.05,\ldots,8.00\}:
\operatorname{FR}_{\mathcal V}(m,z)\ge0.95
\right\}. \tag{T4}
$$

如果某个模型的所有 $z$ 都不合格，则取 $z=8$ 并报告其实际服务率。对指数平滑还要同时比较平滑参数；满足服务约束的组合中，选择验证平均库存最低者。

### 6.2 六个预测模型的数学表达

**Naive。** 下一期等于最近一期真实需求：

$$
\widehat D_{i,t+1\mid t}=D_{it}. \tag{T5}
$$

**$k$ 期移动平均。** 当前分别令 $k\in\{3,6,12\}$：

$$
\widehat D_{i,t+1\mid t}
=\frac{1}{k}\sum_{r=0}^{k-1}D_{i,t-r}. \tag{T6}
$$

窗口越短，模型对近期变化反应越快但更易受噪声影响；窗口越长，预测更平滑但反应较慢。

**Holt 指数平滑。** 令 $\ell_{it}$ 为水平，$b_{it}$ 为趋势，$\alpha$ 为水平平滑系数，$\beta$ 为趋势平滑系数，$\phi$ 为阻尼系数：

$$
\widehat D_{it\mid t-1}=\ell_{i,t-1}+\phi b_{i,t-1}, \tag{T7}
$$

$$
\ell_{it}=\alpha D_{it}
+(1-\alpha)(\ell_{i,t-1}+\phi b_{i,t-1}), \tag{T8}
$$

$$
b_{it}=\beta(\ell_{it}-\ell_{i,t-1})
+(1-\beta)\phi b_{i,t-1}. \tag{T9}
$$

当 $\beta=0$ 时，代码把趋势初始化为 0，模型退化成简单指数平滑。本次选中 $\alpha=0.8,\beta=0,\phi=1$。

**ARIMA$(p,d,q)$。** 用滞后算子 $\mathcal B D_{it}=D_{i,t-1}$ 表示为：

$$
\Phi_p(\mathcal B)(1-\mathcal B)^dD_{it}
=c_{it}+\Theta_q(\mathcal B)\varepsilon_{it}, \tag{T10}
$$

其中 $\Phi_p$ 是 $p$ 阶自回归多项式，$\Theta_q$ 是 $q$ 阶移动平均误差多项式，$\varepsilon_{it}$ 是创新项。代码对每个 SKU、每个候选阶数进行拟合，并最小化：

$$
\operatorname{AIC}=2k-2\log\widehat L, \tag{T11}
$$

其中 $k$ 为估计参数个数，$\widehat L$ 为最大化后的似然值。选中阶数后采用一步预测，并在观察到当期真实需求后更新状态，但不重新拟合参数。

### 6.3 已实现的 6 个模型

| 模型 | 通俗解释 | 主要参数 |
|---|---|---|
| Naive | 明天的需求等于今天 | 无 |
| Moving Average 3 | 用最近 3 期需求均值预测下一期 | 窗口 3 |
| Moving Average 6 | 用最近 6 期需求均值预测下一期 | 窗口 6 |
| Moving Average 12 | 用最近 12 期需求均值预测下一期 | 窗口 12 |
| Exponential Smoothing | 越近的数据权重越大，并可考虑趋势与阻尼 | 验证集选择 `alpha=0.8, beta=0, damping=1` |
| ARIMA | 用差分、自回归和移动平均误差描述单 SKU 的动态 | 每个 SKU 按 AIC 从 6 组阶数中选择 |

ARIMA 候选阶数为 `(0,1,0)`、`(1,1,0)`、`(0,1,1)`、`(1,1,1)`、`(2,1,0)`、`(0,1,2)`。6,000 个 SKU 中，5,996 个选择 `(0,1,2)`，3 个选择 `(1,1,1)`，1 个选择 `(2,1,0)`；没有发生拟合失败后回退。

### 6.4 时间序列结果

| 模型 | `z` | 验证 WAPE | 测试 WAPE | 测试 Fill Rate | 平均总库存 | 总周转率 | SKU 95% 达标比例 | 合格 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Naive | 0.00 | 7.2657% | 22.5082% | 90.8287% | 156,537 | 524.9131 | 0.0167% | 否 |
| Moving Average 3 | 0.25 | 8.9403% | 34.3458% | 91.5574% | 862,595 | 96.0217 | 6.7000% | 否 |
| Moving Average 6 | 0.80 | 11.5987% | 50.8637% | 93.8454% | 1,507,434 | 56.3193 | 55.7333% | 否 |
| Moving Average 12 | 1.60 | 16.2681% | 76.4146% | 95.8390% | 2,937,675 | 29.5135 | 81.2833% | **是** |
| Exponential Smoothing | 0.00 | 7.3854% | 24.2813% | 90.6290% | 224,689 | 364.8952 | 0.0167% | 否 |
| ARIMA | 0.00 | 8.1975% | 28.6796% | 93.1621% | 442,374 | 190.5166 | 19.2833% | 否 |

时间序列中只有 12 期移动平均在测试集达到 95%。它将 baseline 的平均总库存降低 75.42%，因此是一个非常有价值的简单强基准。其 WAPE 虽然高达 76.41%，却仍满足库存服务约束，再次说明“预测误差最低”和“补货决策最好”不是同一个问题。

Naive 和指数平滑拥有很高的表面周转率，但它们分别损失 8,296,896 和 8,477,485 单位销售，总体 Fill Rate 只有约 91%，所以不能判定为好方案。

---

## 7. 方案三：机器学习先预测、再决策

### 7.1 为什么训练一个“全局模型”

为每个 SKU 单独训练一个机器学习模型，会产生 6,000 个小模型，而且每个 SKU 只有 60 期训练数据，样本太少。当前方案把不同 SKU、不同目标周期拼成一个全局监督学习数据集，让一个模型学习跨 SKU 的共同规律，同时保留每个 SKU 的滞后需求和误差尺度。

模型输出的是下一期需求预测，不直接输出订货量。随后统一使用动态 base-stock 决策层，把预测和安全库存转成订货目标。这样可以把“预测能力”和“库存决策”分开检查，也方便后续替换模型。

### 7.2 当前 59 个特征

| 特征组 | 数量 | 具体内容 |
|---|---:|---|
| 需求滞后 | 5 | 第 1、2、3、6、12 期滞后需求 |
| 滚动均值 | 3 | 最近 3、6、12 期均值 |
| 滚动标准差 | 3 | 最近 3、6、12 期标准差 |
| 变化与趋势 | 4 | 1 期差、3 期差、最近值/6期均值、近期斜率 |
| 对数需求 | 2 | `log(1+最近需求)`、`log(1+3期均值)` |
| 匿名特征滞后 | 20 | `x1`–`x20` 的上一期值 |
| 匿名特征变化 | 20 | `x1`–`x20` 的上一期值减上上期值 |
| 静态/时间 | 2 | 品类编码、目标周期索引 |
| **合计** | **59** | 全部在预测时可获得 |

当前没有对线性模型和树模型做 PCA，因为 59 维并不高，PCA 会牺牲可解释性且未必能帮助树模型。RBF-SVR 对尺度和维度敏感，因此单独使用标准化与保留 95% 方差的 PCA。

当前没有使用 KMeans。若先用完整 60 期历史给早期训练样本分群，会把未来信息泄漏给过去；严格按时间动态重做聚类又会显著增加工程复杂度。在还没有证据证明聚类有增益之前，保持直接特征更稳妥。

### 7.3 监督学习问题的数学定义

把 SKU $i$ 在目标周期 $t$ 的 59 个可用特征记为向量 $\mathbf v_{it}\in\mathbb R^{59}$，标签为 $y_{it}=D_{it}$。模型参数为 $\boldsymbol\theta$，原始预测为 $f_{\boldsymbol\theta}(\mathbf v_{it})$。代码把负需求预测截断为 0：

$$
\widehat D_{it\mid t-1}
=\max\{0,f_{\boldsymbol\theta}(\mathbf v_{it})\}. \tag{ML1}
$$

设训练样本集合为 $\mathcal N$、样本数为 $n$。各模型并不是直接最小化库存，而是先拟合需求；候选参数拟合后，再用验证集库存仿真选择模型和安全系数。

#### 7.3.1 线性模型

**普通线性回归**假设预测是特征的线性组合：

$$
f_{\boldsymbol\theta}(\mathbf v_j)
=\beta_0+\mathbf v_j^\top\boldsymbol\beta,qquad
(\widehat\beta_0,\widehat{\boldsymbol\beta})
=\arg\min_{\beta_0,\boldsymbol\beta}
\frac{1}{n}\sum_{j\in\mathcal N}
\left(y_j-\beta_0-\mathbf v_j^\top\boldsymbol\beta\right)^2. \tag{ML2}
$$

**Ridge**在平方误差上增加 L2 正则项：

$$
\min_{\beta_0,\boldsymbol\beta}
\sum_{j\in\mathcal N}
\left(y_j-\beta_0-\mathbf v_j^\top\boldsymbol\beta\right)^2
+\alpha\lVert\boldsymbol\beta\rVert_2^2. \tag{ML3}
$$

它会缩小所有系数，减轻特征相关性造成的不稳定，但通常不会把系数压成严格的 0。

**Lasso**使用 L1 正则项：

$$
\min_{\beta_0,\boldsymbol\beta}
\frac{1}{2n}\sum_{j\in\mathcal N}
\left(y_j-\beta_0-\mathbf v_j^\top\boldsymbol\beta\right)^2
+\alpha\lVert\boldsymbol\beta\rVert_1. \tag{ML4}
$$

L1 惩罚可以把不重要特征的系数压到 0，相当于同时进行线性预测和稀疏特征选择。

**Elastic Net**同时使用 L1 和 L2：

$$
\min_{\beta_0,\boldsymbol\beta}
\frac{1}{2n}\sum_{j\in\mathcal N}
\left(y_j-\beta_0-\mathbf v_j^\top\boldsymbol\beta\right)^2
+\alpha\rho\lVert\boldsymbol\beta\rVert_1
+\frac{\alpha(1-\rho)}{2}\lVert\boldsymbol\beta\rVert_2^2, \tag{ML5}
$$

其中 $\rho$ 对应代码里的 `l1_ratio`。四种线性模型在拟合前都对特征进行标准化，避免量纲较大的特征仅因为数值范围大而主导正则项。

#### 7.3.2 RBF-SVR

SVR 在核映射 $\varphi(\mathbf v)$ 后寻找一个尽量平坦的函数，并允许误差落在宽度为 $\epsilon$ 的不敏感区间内：

$$
\begin{aligned}
\min_{\mathbf w,b,\xi_j,\xi_j^*}\quad
&\frac{1}{2}\lVert\mathbf w\rVert_2^2
+C\sum_{j\in\mathcal N}(\xi_j+\xi_j^*)\\
\text{s.t.}\quad
&y_j-(\mathbf w^\top\varphi(\mathbf v_j)+b)\le\epsilon+\xi_j,\\
&(\mathbf w^\top\varphi(\mathbf v_j)+b)-y_j\le\epsilon+\xi_j^*,\\
&\xi_j,\xi_j^*\ge0.
\end{aligned} \tag{ML6}
$$

RBF 核为 $K(\mathbf v,\mathbf v')=\exp(-\gamma\lVert\mathbf v-\mathbf v'\rVert_2^2)$。当前 SVR 先标准化特征、再用 PCA 保留 95% 方差；目标变量也会标准化。$C$ 控制违反误差带的惩罚强度，$\epsilon$ 控制不惩罚误差带的宽度。

#### 7.3.3 随机森林

随机森林在 bootstrap/随机特征扰动下训练 $M$ 棵回归树 $T_m$，最终预测取平均：

$$
\widehat D(\mathbf v)=
\frac{1}{M}\sum_{m=1}^{M}T_m(\mathbf v). \tag{ML7}
$$

单棵树可以表达非线性和特征交互，多树平均用于降低方差。当前最优配置使用 180 棵树、每次分裂最多考虑 80% 特征、叶节点至少 3 个样本。

#### 7.3.4 XGBoost、LightGBM 与 CatBoost

三者都属于梯度提升树。第 $m$ 轮在已有模型 $F_{m-1}$ 上增加一棵新树 $h_m$：

$$
F_m(\mathbf v)=F_{m-1}(\mathbf v)+\eta h_m(\mathbf v), \tag{ML8}
$$

并近似最小化带正则的经验风险：

$$
\mathcal L_m=
\sum_{j\in\mathcal N}
\ell\!\left(y_j,F_{m-1}(\mathbf v_j)+h_m(\mathbf v_j)\right)
+\Omega(h_m). \tag{ML9}
$$

$\eta$ 是学习率，$\Omega$ 控制树的复杂度。当前实现中：XGBoost 使用平方误差；LightGBM 使用绝对误差目标 `regression_l1`；CatBoost 使用 RMSE。三者在树的生长、分桶和正则化方式上不同，但都能表达线性模型难以捕捉的阈值效应和特征交互。

#### 7.3.5 从 ML 预测到库存决策

对任意机器学习模型 $m$，用验证期预测残差计算 SKU 级误差尺度 $\widehat\sigma_{i,m}^{e,\mathcal V}$，再设置：

$$
B_{i,t+1}^{(m)}(z)=
\left\lceil\max\left\{0,
\widehat D_{i,t+1\mid t}^{(m)}
+z\widehat\sigma_{i,m}^{e,\mathcal V}
\right\}\right\rceil. \tag{ML10}
$$

对候选超参数 $\lambda\in\Lambda_m$，先分别校准满足验证服务率的最小 $z$，再选择：

$$
(\lambda_m^*,z_m^*)
\in\arg\min_{\lambda\in\Lambda_m,\,z\in\mathcal Z}
\overline H_{\mathcal V}(m,\lambda,z)
\quad\text{s.t.}\quad
\operatorname{FR}_{\mathcal V}(m,\lambda,z)\ge0.95. \tag{ML11}
$$

若所有候选都无法达到约束，程序选择验证 Fill Rate 最高者，并用 WAPE 打破其余平局。选定 $\lambda_m^*$ 后，用全部可用训练样本重训，但 $z_m^*$ 不再用测试集调整。

### 7.4 已实现的 9 个模型

| 模型 | 核心思想 | 当前选中参数 |
|---|---|---|
| Linear Regression | 所有特征与需求之间是线性加总关系 | 默认线性回归 |
| Ridge | 在线性回归上加 L2 惩罚，降低系数不稳定 | `alpha=100` |
| Lasso | 加 L1 惩罚，可把部分系数压到 0 | `alpha=0.1` |
| Elastic Net | 同时使用 L1 和 L2 惩罚 | `alpha=0.01, l1_ratio=0.5` |
| RBF-SVR | 在核空间拟合非线性关系 | `C=10, epsilon=0.05` |
| Random Forest | 多棵随机决策树平均 | 180 棵树，无深度上限，叶节点最少 3 样本 |
| XGBoost | 顺序构造提升树，逐步修正误差 | 240 棵树，深度 6，学习率 0.05，行采样 0.9 |
| LightGBM | 高效的梯度提升树 | 450 棵树，127 叶，学习率 0.04，叶节点最少 50 样本 |
| CatBoost | 带正则和有序提升机制的梯度提升树 | 300 轮，深度 6，学习率 0.05，L2=3 |

### 7.5 机器学习结果

| 模型 | `z` | 验证 WAPE | 测试 WAPE | 测试 Fill Rate | 平均总库存 | 总周转率 | SKU 95% 达标比例 | 合格 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Linear Regression | 0.00 | 7.1733% | 33.4642% | 88.2180% | 405,547 | 196.7879 | 4.9000% | 否 |
| Ridge | 0.00 | 7.1639% | 23.1445% | 93.4795% | 267,653 | 315.9561 | 4.8333% | 否 |
| Lasso | 0.00 | 7.1690% | 22.9336% | 93.4348% | 248,928 | 339.5603 | 6.5500% | 否 |
| Elastic Net | 0.00 | 7.2092% | 26.8753% | 94.1613% | 398,818 | 213.5899 | 6.5333% | 否 |
| SVR | 4.05 | 41.1523% | 192.8902% | 99.1511% | 54,791,768 | 1.6371 | 100.0000% | 是，但不可取 |
| Random Forest | 0.00 | 7.4686% | 23.2410% | 94.2314% | 287,415 | 296.5986 | 16.7000% | 否 |
| XGBoost | 0.60 | 10.3603% | 25.9741% | 97.8206% | 838,215 | 105.5741 | 97.7667% | **是** |
| LightGBM | 0.40 | 9.8760% | 24.5419% | 96.9577% | 585,283 | 149.8646 | 91.5167% | **是** |
| CatBoost | 0.50 | 11.0684% | 32.4880% | 97.5982% | 995,719 | 88.6722 | 98.1167% | **是** |

线性回归、Ridge、Lasso、Elastic Net 和随机森林在验证集均达到 95%，但在测试集失守。这说明单一验证窗口没有完全覆盖测试阶段的分布变化。它们不能因为库存低、周转率高就进入最终候选集。

SVR 是另一个方向的失败：它依靠 `z=4.05` 堆积大量安全库存，把 Fill Rate 推到 99.15%，但平均库存达到 baseline 的 4.59 倍。它满足数学约束，却违背“在约束下最小化库存”的目标。

三个梯度提升树模型都通过测试约束。LightGBM 的库存最低；XGBoost 和 CatBoost 使用更多库存，换来了更高的总服务率和更高的 SKU 达标覆盖率。

---

## 8. 方案四：分布鲁棒优化（DRO）

DRO 不假设历史预测误差分布就是未来的真实分布，而是在历史分布周围构造一组可能分布，并针对其中较不利的分布设置安全库存。当前实现统一使用 12 期移动平均作为中心预测，再分别采用三种模糊集合：

- 矩信息 DRO：只固定预测误差的均值和方差；
- Wasserstein DRO：允许经验误差分布在 Wasserstein 距离内移动；
- 卡方散度 DRO：允许历史误差场景的概率被重新加权。

三种方案全部满足正式测试 95% 总体 Fill Rate：

| 模型 | 测试 Fill Rate | 平均总库存 | 总周转率 | SKU 95% 达标比例 |
|---|---:|---:|---:|---:|
| Wasserstein DRO + MA12 | 96.4349% | 3,133,250 | 27.8434 | 85.4667% |
| 卡方散度 DRO + MA12 | 96.7745% | 3,268,233 | 26.7874 | 87.6500% |
| 矩信息 DRO + MA12 | 98.2566% | 4,309,164 | 20.6277 | 96.4000% |

与普通 MA12 相比，Wasserstein DRO 多使用 6.66% 库存，将 Fill Rate 提高 0.596 个百分点；矩信息 DRO 最保守，库存增加 46.69%，Fill Rate 提高 2.418 个百分点。按照“达到 95% 后只最小化库存”的严格目标，普通 MA12 仍优于三个 DRO；如果业务希望在 95% 以上保留额外安全余量，Wasserstein DRO 是当前代价最低的鲁棒选择。

完整的直觉解释、Fill Rate 等价推导、三种模糊集合公式、参数结果和局限见 `docs/dro_methods_report.md`（历史出处：`dro_methods_report.md`）。

---

## 8.5 方案五：Smart Predict-then-Optimize（SPO+）

SPO+ 不再以需求预测误差为训练目标，而是预测17个候选补货目标的成本，并直接惩罚“选错库存动作”造成的额外成本。加入当期库存后，线性SPO+模型在测试集取得95.0235%的总体Fill Rate、3,684,607的平均总库存和23.3304的总周转率。它满足约束并比原baseline少69.16%的库存，但没有超过提升树、MA12和两种较好的DRO。

论文公式、有限动作适配、完整推导、参数、结果和局限见 `docs/spo_plus_method_report.md`（历史出处：`spo_plus_method_report.md`）。

---

## 8.6 方案六：XGBoost–SPO+

新模型使用训练期外XGBoost需求预测生成9个候选目标，再用多输出XGBoost预测动作成本。实验比较了纯成本MSE、多个MSE/SPO+混合权重和纯平滑SPO+。加入当期库存后，仍由纯平滑SPO+（`alpha=1`）胜出，测试Fill Rate为97.3647%，平均总库存825,619，总周转率106.6853。它成为当前第二名，并比普通XGBoost少1.50%的库存。

完整损失函数、梯度、Hessian、无泄漏扩展窗口流程和20组消融实验见 `docs/xgb_spo_hybrid_report.md`（历史出处：`xgb_spo_hybrid_report.md`）。

---

## 8.7 方案七：LightGBM–SPO+

该模型使用训练期外LightGBM需求预测构造9个候选库存动作。由于LightGBM不直接支持这里的多输出联合成本，程序将每个样本展开成9个“样本—动作”行，再在自定义目标中按组计算平滑SPO+梯度。

12组消融实验最终仍由纯平滑SPO+（`alpha=1`，缺货惩罚20）胜出，说明在当前任务中动作排序比成本数值的精确拟合更重要。加入当期库存后，正式测试Fill Rate为97.6059%，平均总库存947,906，总周转率93.1523，在当前合格方案中排第四。它比CatBoost库存低，但未超过普通LightGBM或XGBoost–SPO+。

完整的数学形式、工程适配、消融结果和局限见 `docs/lgbm_spo_hybrid_report.md`（历史出处：`lgbm_spo_hybrid_report.md`）。

---

## 8.8 方案八：不预测需求的直接XGBoost–SPO

直接决策模型不再输出需求预测，而是根据历史信息与当前库存状态，直接比较11个库存倍率动作的成本。一步版只计算下一期成本；滚动版计算最多六期的持有与缺货成本。当前工程版本的验证选择要求Fill Rate达到97%，为95%业务约束保留两个百分点的安全裕度。该裕度是在诊断早期版本测试失守后加入的，所以当前测试窗口对这两个新模型应视为开发基准；正式论文结论仍需新的盲测时间窗口。

一步版测试Fill Rate为96.4354%，平均总库存4,871,152；六步滚动版测试Fill Rate为95.0384%，平均总库存2,128,309。两者均合格，滚动版比baseline降低82.19%的库存，但仍没有超过普通LightGBM。完整公式、工程实现、命令和限制见 `docs/direct_xgb_spo_policy_report.md`（历史出处：`direct_xgb_spo_policy_report.md`）。

---

## 8.9 方案九：不预测需求的直接LightGBM–SPO+–Rank

这个模型同样不输出需求预测，而是把每个决策状态展开成11个“状态—动作”样本，
让LightGBM预测每个候选补货动作的运营成本并直接选择最小者。为了让损失真正关注
决策质量，训练目标由三部分构成：成本MSE提供稳定数值尺度，平滑SPO+惩罚会导致
高后悔的错误动作，后悔加权排序损失要求真实最优动作排在其他动作之前。

损失消融显示，单独成本MSE的验证Fill Rate只有94.8566%，无法满足约束；最终混合
损失在合格候选中库存最低。运营参数选择三期滚动成本和缺货惩罚4，测试Fill Rate为
95.9206%，平均总库存2,198,542，总周转率39.4693，较baseline降低81.60%的库存。
它证明LightGBM可以直接学习补货决策，但仍没有超过普通LightGBM“先预测、再决策”。
完整推导、文献依据、梯度、消融与限制见
`docs/direct_lgbm_spo_rank_report.md`（历史出处：`direct_lgbm_spo_rank_report.md`）。

---

## 9. 全部 25 个方案的统一比较

下表按“先合格、再按平均库存从低到高”的规则排列。粗体表示满足测试总体 Fill Rate 约束。

| 排名 | 家族 | 模型 | 测试 Fill Rate | 平均总库存 | 总周转率 | 损失销售 | SKU 95% 达标比例 | 约束 |
|---:|---|---|---:|---:|---:|---:|---:|:---:|
| 1 | 机器学习 | **LightGBM** | 96.9577% | 585,283 | 149.8646 | 2,752,206 | 91.5167% | 是 |
| 2 | 决策导向学习 | **XGBoost–SPO+** | 97.3647% | 825,619 | 106.6853 | 2,384,035 | 91.7667% | 是 |
| 3 | 机器学习 | **XGBoost** | 97.8206% | 838,215 | 105.5741 | 1,971,611 | 97.7667% | 是 |
| 4 | 决策导向学习 | **LightGBM–SPO+** | 97.6059% | 947,906 | 93.1523 | 2,165,850 | 95.5167% | 是 |
| 5 | 机器学习 | **CatBoost** | 97.5982% | 995,719 | 88.6722 | 2,172,827 | 98.1167% | 是 |
| 6 | 直接决策学习 | **Direct XGB–SPO Rollout** | 95.0384% | 2,128,309 | 40.3968 | 4,488,565 | 65.5333% | 是 |
| 7 | 直接决策学习 | **Direct LightGBM–SPO+–Rank** | 95.9206% | 2,198,542 | 39.4693 | 3,690,488 | 77.0167% | 是 |
| 8 | 时间序列 | **Moving Average 12** | 95.8390% | 2,937,675 | 29.5135 | 3,764,291 | 81.2833% | 是 |
| 9 | DRO | **Wasserstein DRO + MA12** | 96.4349% | 3,133,250 | 27.8434 | 3,225,198 | 85.4667% | 是 |
| 10 | DRO | **卡方散度 DRO + MA12** | 96.7745% | 3,268,233 | 26.7874 | 2,917,969 | 87.6500% | 是 |
| 11 | 决策导向学习 | **SPO+ Linear** | 95.0235% | 3,684,607 | 23.3304 | 4,501,978 | 49.0167% | 是 |
| 12 | DRO | **矩信息 DRO + MA12** | 98.2566% | 4,309,164 | 20.6277 | 1,577,188 | 96.4000% | 是 |
| 13 | 直接决策学习 | **Direct XGB–SPO One-step** | 96.4354% | 4,871,152 | 17.9097 | 3,224,774 | 77.0167% | 是 |
| 14 | Baseline | **Mean–variance baseline** | 98.7838% | 11,949,377 | 7.4787 | 1,100,224 | 98.7500% | 是 |
| 15 | 机器学习 | **SVR** | 99.1511% | 54,791,768 | 1.6371 | 767,952 | 100.0000% | 是 |
| — | 机器学习 | Random Forest | 94.2314% | 287,415 | 296.5986 | 5,218,559 | 16.7000% | 否 |
| — | 机器学习 | Elastic Net | 94.1613% | 398,818 | 213.5899 | 5,282,019 | 6.5333% | 否 |
| — | 时间序列 | Moving Average 6 | 93.8454% | 1,507,434 | 56.3193 | 5,567,800 | 55.7333% | 否 |
| — | 机器学习 | Ridge | 93.4795% | 267,653 | 315.9561 | 5,898,796 | 4.8333% | 否 |
| — | 机器学习 | Lasso | 93.4348% | 248,928 | 339.5603 | 5,939,275 | 6.5500% | 否 |
| — | 时间序列 | ARIMA | 93.1621% | 442,374 | 190.5166 | 6,185,918 | 19.2833% | 否 |
| — | 时间序列 | Moving Average 3 | 91.5574% | 862,595 | 96.0217 | 7,637,647 | 6.7000% | 否 |
| — | 时间序列 | Naive | 90.8287% | 156,537 | 524.9131 | 8,296,896 | 0.0167% | 否 |
| — | 时间序列 | Exponential Smoothing | 90.6290% | 224,689 | 364.8952 | 8,477,485 | 0.0167% | 否 |
| — | 机器学习 | Linear Regression | 88.2180% | 405,547 | 196.7879 | 10,658,626 | 4.9000% | 否 |

统一的库存—服务率图位于：`outputs/all_model_comparison.png`。

全部模型的库存与服务率比较（历史产物未包含/需运行生成：`../outputs/all_model_comparison.png`）

### 9.1 如何选择最终模型

如果严格遵循当前数学目标，选择 LightGBM 即可。若业务目标更细，可以采用以下解释：

| 业务偏好 | 推荐方案 | 理由 |
|---|---|---|
| 总体 Fill Rate 合格后，库存最低 | LightGBM | 当前平均库存最低 |
| 愿意多持有约 252,932 库存，换更高服务覆盖 | XGBoost | 总 Fill Rate 97.82%，SKU 达标 97.77% |
| 更看重 SKU 达标比例 | CatBoost | SKU 达标 98.12%，但库存高于 XGBoost |
| 需要极简、无需匿名特征、容易解释 | Moving Average 12 | 唯一合格的时间序列方案 |
| 担心需求分布漂移，希望为 MA12 增加安全余量 | Wasserstein DRO | 三个 DRO 中库存最低 |
| 需要保守且非常透明的参照 | Baseline | Fill Rate 高，但库存代价大 |

不推荐 SVR。也不应选择任何测试 Fill Rate 低于 95% 的方案，即使它的库存很低或周转率很高。

---

## 10. 为什么 baseline 看起来“效果很好”

这不是计算错误，也不完全是错觉。它确实实现了很高的 Fill Rate，但有三点容易造成误判：

1. **只看服务率，没有看库存。** baseline 的 98.78% Fill Rate 是用 11,949,377 的平均库存换来的，而 LightGBM 用 585,283 的库存也达到 96.96%。
2. **测试期需求显著低于训练期。** baseline 根据完整训练历史设置固定目标，测试需求下降后没有同步降库存，服务率自然很高。
3. **95% 是总体约束，不要求越高越好。** 当 96% 已经满足约束时，把服务率继续推到 99% 可能需要大量额外库存。当前目标不是最大化 Fill Rate，而是在达到门槛后最小化库存。

因此 baseline 是一个有用且合格的参照，但不是当前最优解。

---

## 11. 工程结构与每个文件怎么看

### 11.1 代码入口

| 文件 | 作用 | 初学者什么时候看 |
|---|---|---|
| `experiments/run_baseline.py` | 读取数据、训练 baseline、调用评价器、保存结果 | 想了解最短的完整实验流程时先看 |
| `experiments/run_time_series.py` | 调优并运行 6 个时间序列模型 | 想看滚动预测和动态补货如何连接 |
| `experiments/run_machine_learning.py` | 构造 ML 训练/验证/测试流程并运行 9 个模型 | 想看“先预测再决策”的总流程 |
| `experiments/run_dro.py` | 调优并运行 3 个分布鲁棒补货模型 | 想看模糊集合如何转成安全库存 |
| `experiments/run_spo_plus.py` | 调优并运行论文的SPO+模型 | 想看模型如何直接学习库存动作 |
| `experiments/run_xgb_spo_hybrid.py` | 运行XGBoost与平滑SPO+模型和消融实验 | 想看非线性决策导向训练 |
| `experiments/run_lgbm_spo_hybrid.py` | 运行LightGBM与平滑SPO+模型和消融实验 | 想看LightGBM的样本—动作展开实现 |
| `experiments/run_direct_xgb_spo.py` | 运行不预测需求的XGBoost–SPO一步与滚动策略 | 想看模型如何从库存状态直接选择动作 |
| `experiments/run_direct_lgbm_spo.py` | 运行直接LightGBM–SPO+–Rank及损失消融 | 想看LightGBM如何绕过需求预测直接选动作 |
| `src/models/baseline/base_stock.py` | baseline 公式与 `z` 校准 | 想理解 baseline 数学逻辑 |
| `src/models/time_series/forecasters.py` | 各时间序列预测器 | 想理解预测值怎么生成 |
| `src/models/machine_learning/features.py` | 读取面板数据并构造 59 个无泄漏特征 | 想理解 ML 输入 |
| `src/models/machine_learning/regressors.py` | 9 个模型和候选参数 | 想修改模型或参数范围 |
| `src/models/dro/robust_base_stock.py` | 三种 DRO 约束上界和安全缓冲求解 | 想理解 DRO 数学模型与代码的对应关系 |
| `src/models/spo/spo_plus.py` | 有限动作成本、SPO+损失和次梯度训练 | 想核对论文公式与代码 |
| `src/models/spo/xgb_spo_hybrid.py` | 混合损失梯度、Hessian和多输出XGBoost | 想理解XGBoost如何使用SPO+ |
| `src/models/spo/lgbm_spo_hybrid.py` | 样本—动作展开和LightGBM自定义目标 | 想理解LightGBM如何使用SPO+ |
| `src/models/spo/direct_policy.py` | 直接决策的参考尺度、状态特征和滚动成本 | 想核对直接动作成本如何构造 |
| `src/models/spo/lgbm_direct_policy.py` | 直接LightGBM的MSE、SPO+与排序混合目标 | 想核对新损失的梯度、Hessian和动作展开 |
| `src/evaluation/inventory_evaluator.py` | 统一库存仿真、Fill Rate、库存和周转率 | 想核对指标和补货时序 |
| `src/evaluation/item_trajectory.py` | 单 SKU 轨迹图 | 想查看某个 SKU 每期发生了什么 |
| `tests/` | 自动测试 | 修改代码后确认核心逻辑没坏 |

建议阅读顺序是：`README.md` → 本报告 → `run_baseline.py` → `inventory_evaluator.py` → 需要研究的具体模型文件。无需从最复杂的机器学习入口开始。

### 11.2 结果文件

| 文件或目录 | 内容 |
|---|---|
| `outputs/all_model_comparison.csv` | 当前全部 25 个模型的统一原始结果，最重要的汇总表 |
| `outputs/all_model_comparison.png` | 25 个模型的库存—服务率图 |
| `outputs/baseline_summary.json` | baseline 配置、验证和测试总体指标 |
| `outputs/baseline_parameters.csv` | 6,000 个 SKU 的固定 `s`、`S`、`z` |
| `outputs/baseline_test_per_sku_metrics.csv` | baseline 每个 SKU 的 Fill Rate、库存、周转率 |
| `outputs/model_comparison.csv` | baseline 与时间序列模型比较 |
| `outputs/time_series/<模型>/` | 时间序列模型的预测、目标库存、逐期轨迹和指标 |
| `outputs/machine_learning/model_comparison.csv` | 9 个机器学习模型比较 |
| `outputs/machine_learning/feature_metadata.json` | 59 个特征及时间口径 |
| `outputs/machine_learning/<模型>/` | 模型文件、候选参数、预测、目标库存、轨迹和指标 |
| `outputs/dro/model_comparison.csv` | 三个 DRO 模型的统一比较 |
| `outputs/dro/dro_vs_existing_comparison.csv` | DRO 与 baseline、MA12 和提升树模型比较 |
| `outputs/dro/<模型>/` | DRO 候选参数、安全缓冲、预测、目标库存、轨迹和指标 |
| `outputs/spo/spo_plus_linear/` | SPO+模型、候选验证、动作、目标库存、轨迹和指标 |
| `outputs/spo/xgb_spo_hybrid/` | XGBoost–SPO+模型、20组消融、轨迹和指标 |
| `outputs/spo/lgbm_spo_hybrid/` | LightGBM–SPO+模型、12组消融、轨迹和指标 |
| `outputs/spo/direct_xgb_spo_one_step/` | 不预测需求的一步直接决策模型、动作、轨迹和指标 |
| `outputs/spo/direct_xgb_spo_rollout/` | 不预测需求的六步滚动直接决策模型、动作、轨迹和指标 |
| `outputs/spo/direct_lgbm_spo_rank/` | 直接LightGBM模型、损失消融、动作、轨迹和指标 |
| `outputs/item_diagnostics/<模型>/` | 指定 SKU 在某个模型下的测试轨迹 |
| `outputs/item_comparisons/` | 同一 SKU 的多模型比较图与比较表 |

每个模型的 `test_trajectory.csv` 有 168,000 行，即 `6,000 SKU × 28 周期`。主要字段含义如下：

| 字段 | 含义 |
|---|---|
| `opening_inventory` | 本周期到货前的现货库存 |
| `arrival` | 本周期开始时到达的上期订单 |
| `demand` | 本周期需求 |
| `sales` | 实际满足的需求 |
| `lost_sales` | 未满足并直接损失的需求 |
| `ending_inventory` | 满足需求后的期末库存 |
| `order_quantity` | 期末下达、下期到货的数量 |
| `order_up_to` | 本期期末采用的动态目标库存 |
| `forecast` | 模型对相关下一期需求的预测（如该模型有预测） |

每个模型的 `test_per_sku_metrics.csv` 只有 6,000 行，适合看每个 SKU 的整个测试期汇总；终端只打印总体结果，避免被 6,000 行明细淹没。

---

## 12. 如何从终端完整复现

以下命令可以直接复制到 macOS 终端。所有命令都从项目根目录运行。

### 12.1 安装依赖并运行测试

```bash
cd "<PROJECT_ROOT>"

python -m pip install -r requirements.txt

PYTHONPYCACHEPREFIX=/private/tmp/replenishment_pycache \
MPLCONFIGDIR=/private/tmp/matplotlib \
python \
  -m unittest discover -s tests -v
```

### 12.2 运行 baseline

```bash
cd "<PROJECT_ROOT>"

PYTHONPYCACHEPREFIX=/private/tmp/replenishment_pycache \
MPLCONFIGDIR=/private/tmp/matplotlib \
python \
  -u -m experiments.run_baseline
```

### 12.3 运行全部时间序列模型

```bash
cd "<PROJECT_ROOT>"

PYTHONPYCACHEPREFIX=/private/tmp/replenishment_pycache \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/private/tmp/matplotlib \
python \
  -u -m experiments.run_time_series --n-jobs 8 --item-id 1
```

`--n-jobs 8` 是 ARIMA 的并行进程数；内存紧张时改成 4。`--item-id 1` 表示额外输出 SKU 1 的轨迹和比较图。

### 12.4 运行全部机器学习模型

```bash
cd "<PROJECT_ROOT>"

PYTHONPYCACHEPREFIX=/private/tmp/replenishment_pycache \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/private/tmp/matplotlib \
python \
  -u -m experiments.run_machine_learning --models all --n-jobs 8 --item-id 1
```

全部机器学习模型的模型内部计时合计约 992.9 秒，其中随机森林约 595.4 秒、SVR 约 269.3 秒，是主要耗时来源。若只想复现当前推荐方案：

```bash
cd "<PROJECT_ROOT>"

PYTHONPYCACHEPREFIX=/private/tmp/replenishment_pycache \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/private/tmp/matplotlib \
python \
  -u -m experiments.run_machine_learning \
  --models lightgbm,xgboost,catboost --n-jobs 8 --item-id 1
```

### 12.5 运行全部 DRO 模型

```bash
cd "<PROJECT_ROOT>"

PYTHONPYCACHEPREFIX=/private/tmp/replenishment_dro_pycache \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/private/tmp/matplotlib \
python \
  -u -m experiments.run_dro --models all --item-id 1
```

该命令运行矩信息、Wasserstein 和卡方散度三个 DRO；本次完整运行约 31 秒。

### 12.6 运行 SPO+ 决策导向模型

```bash
cd "<PROJECT_ROOT>"

mkdir -p /private/tmp/replenishment_font_cache

XDG_CACHE_HOME=/private/tmp/replenishment_font_cache \
PYTHONPYCACHEPREFIX=/private/tmp/replenishment_spo_pycache \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/private/tmp/matplotlib \
python \
  -u -m experiments.run_spo_plus --item-id 1
```

### 12.7 运行XGBoost–SPO+模型

```bash
cd "<PROJECT_ROOT>"

mkdir -p /private/tmp/replenishment_font_cache

XDG_CACHE_HOME=/private/tmp/replenishment_font_cache \
PYTHONPYCACHEPREFIX=/private/tmp/replenishment_hybrid_pycache \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/private/tmp/matplotlib \
python \
  -u -m experiments.run_xgb_spo_hybrid --n-jobs 8 --item-id 1
```

### 12.8 运行LightGBM–SPO+模型

```bash
cd "<PROJECT_ROOT>"

mkdir -p /private/tmp/replenishment_font_cache

XDG_CACHE_HOME=/private/tmp/replenishment_font_cache \
PYTHONPYCACHEPREFIX=/private/tmp/replenishment_lgbm_spo_pycache \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/private/tmp/matplotlib \
LOKY_MAX_CPU_COUNT=8 \
python \
  -u -m experiments.run_lgbm_spo_hybrid --n-jobs 8 --item-id 1
```

### 12.9 运行不预测需求的直接XGBoost–SPO模型

```bash
cd "<PROJECT_ROOT>"

PYTHONPYCACHEPREFIX=/private/tmp/replenishment_direct_pycache \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/private/tmp/matplotlib \
python \
  -u -m experiments.run_direct_xgb_spo --n-jobs 8 --item-id 1
```

本次正式运行约154秒，产生12组消融候选和完整测试轨迹。

### 12.10 运行不预测需求的直接LightGBM–SPO+–Rank模型

```bash
cd "<PROJECT_ROOT>"

PYTHONPYCACHEPREFIX=/private/tmp/replenishment_lgbm_direct_pycache \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/private/tmp/matplotlib \
python \
  -u -m experiments.run_direct_lgbm_spo --n-jobs 8 --item-id 1
```

本次正式运行约118秒，包含4种损失消融、18组运营参数候选和完整测试轨迹。

### 12.11 查看某一个 SKU

仅查看 baseline 的某个 SKU，例如 SKU 1：

```bash
cd "<PROJECT_ROOT>"

PYTHONPYCACHEPREFIX=/private/tmp/replenishment_pycache \
MPLCONFIGDIR=/private/tmp/matplotlib \
python \
  -m experiments.inspect_test_item --item-id 1
```

把已经存在的不同模型轨迹画到同一张图：

```bash
cd "<PROJECT_ROOT>"

PYTHONPYCACHEPREFIX=/private/tmp/replenishment_pycache \
MPLCONFIGDIR=/private/tmp/matplotlib \
python \
  -m experiments.compare_test_item_models --item-id 1
```

SKU 1 的现有全模型比较图位于：`outputs/item_comparisons/item_1_all_models_with_dro_comparison.png`。

---

## 13. 当前结论的边界与不足

这套工程已经能完整跑通并复现结果，但还不能把当前排名理解成永远成立。

1. **只有一个时间验证窗口。** 多个模型在验证期合格、测试期失守，说明单窗口校准对分布变化不够稳健。下一步应采用多个滚动验证窗口。
2. **训练和测试需求分布发生变化。** 测试平均需求明显下降。当前测试是一个确定的数据集，尚未证明模型在需求突然上升时同样稳健。
3. **约束是总体 Fill Rate。** 它可能掩盖低销量 SKU 的较差服务。若业务要求 SKU 级保障，应增加“SKU 达标比例”约束或按 ABC/品类分层设服务目标。
4. **匿名特征的业务含义未知。** 当前只能严格按“不能提前知道”使用滞后值，无法判断哪些特征在现实中可由计划、价格或促销系统提前获得。
5. **库存模型仍然简化。** 当前没有固定订货成本、持有成本、缺货成本、最小起订量、整箱、容量、预算、供应不确定性和多级网络约束。
6. **提前期固定为 1。** 动态评价器当前仅支持 1 期提前期；现实中不同 SKU 可能有不同且随机的提前期。
7. **缺货按 lost sales。** 未满足需求不会延迟到下一期；若实际业务是 backorder，策略与评价都需要改变。
8. **测试集已用于多个方案的最终比较。** 参数没有使用测试集拟合，但随着方案反复开发，测试结果会逐渐影响研究方向。正式论文或生产验收最好保留一个从未查看过的最终留出集。
9. **当前未做融合。** 这是有意为之：先把每个单模型独立跑通，建立清晰、可解释的比较，再决定是否做 stacking 或模型组合。
10. **DRO 保证依赖模糊集合。** 三种 DRO 在当前测试集均合格，但如果真实分布落在设定模糊集合之外，并不能保证未来仍达到 95%。
11. **SPO+使用单期代理成本。** 这是为了把问题写成论文要求的线性目标形式；最终评价是完整动态仿真，但训练目标没有完全包含跨期库存状态。

---

## 14. 建议的下一阶段

建议按以下顺序继续，而不是立即增加更多复杂模型：

1. 将模型选择改为多窗口 rolling-origin 验证，考察 `z` 在不同需求阶段是否稳定；
2. 明确最终约束究竟是总体 Fill Rate、SKU 达标比例，还是两者同时约束；
3. 以 LightGBM、XGBoost、CatBoost、Moving Average 12 和 Wasserstein DRO 作为后续核心候选，暂时停止投入 SVR；
4. 检查不同品类、需求规模和波动等级下的 SKU 服务率，确认低销量 SKU 是否被总体指标掩盖；
5. 尝试 LightGBM 预测加 Wasserstein/卡方 DRO 决策，检验更准确的中心预测能否减少鲁棒安全库存；
6. 在单模型结果稳定后，再考虑简单加权融合或 stacking，并要求它在同一验证协议下证明增益；
7. 若进入业务应用，加入订货成本、持有成本、最小起订量和真实提前期。

---

## 附录 A：参数搜索范围

### A.1 安全系数

| 模型家族 | 搜索范围 | 步长 |
|---|---:|---:|
| Baseline | 0–4 | 0.05 |
| 时间序列 | 0–8 | 0.05 |
| 机器学习 | 0–8 | 0.05 |

### A.2 DRO 模糊集合参数

| DRO 方法 | 候选参数 |
|---|---|
| 矩信息 DRO | 方差放大系数 `{1.0, 1.25, 1.5, 2.0}` |
| Wasserstein DRO | 半径比例 `{0.0025, 0.005, 0.01, 0.02, 0.03, 0.05}` |
| 卡方散度 DRO | 半径 `{0.01, 0.025, 0.05, 0.1, 0.2, 0.4}` |

### A.3 指数平滑

- 无趋势候选：`alpha ∈ {0.2, 0.4, 0.6, 0.8, 1.0}`，`beta=0`，`damping=1`；
- 趋势候选：`alpha ∈ {0.4, 0.6, 0.8, 1.0}`，`beta ∈ {0.05, 0.2, 0.5}`，`damping ∈ {0.9, 1.0}`；
- 最终选择：`alpha=0.8, beta=0, damping=1`。

### A.4 机器学习候选

- Ridge：`alpha ∈ {0.1, 1, 10, 100}`；
- Lasso：`alpha ∈ {0.01, 0.1, 1, 10}`；
- Elastic Net：`alpha ∈ {0.01, 0.1, 1}`，`l1_ratio ∈ {0.2, 0.5, 0.8}`；
- SVR：4 组 `(C, epsilon)`，即 `(1,0.05)`、`(10,0.05)`、`(10,0.10)`、`(50,0.10)`；
- Random Forest：4 组树数量、深度与最小叶节点样本组合；
- XGBoost、LightGBM、CatBoost：各 4 组紧凑候选，完整定义见 `src/models/machine_learning/regressors.py`。

所有候选先在验证期运行库存仿真。若有候选达到 95%，选择合格候选中平均库存最低者；若都不合格，选择验证 Fill Rate 最高者。测试集不参与候选参数选择。

## 附录 B：结果可信度检查

当前项目已对关键逻辑建立自动测试，并对生成结果做过结构检查：

- 评价器满足 `demand = sales + lost_sales`；
- 库存、销售、缺货和订货量非负；
- 每个模型输出 6,000 行 SKU 汇总；
- 每个动态模型输出 168,000 行测试轨迹；
- 每个 SKU、每个周期只出现一次；
- 保存的机器学习模型可以重新加载；
- 汇总 Fill Rate 能由明细重新计算得到；
- DRO 的鲁棒约束、安全倍数单调性和无未来信息特征已单独测试；
- SPO+新增动作成本、直接策略状态、滚动成本，以及两种混合损失的有限差分测试；当前自动测试共21项，均已通过。

真正需要特别留意的不是代码是否算出了数字，而是验证集与测试集分布变化、总体服务率与 SKU 公平性的区别，以及“预测准确”不等于“库存决策优秀”。

### 原始说明：ready_rate_baseline_ml_dfl_report.md

### 新规则实验：历史 Baseline、机器学习与 DFL 鲁棒场景

## 一、实验结论

这次单独重建了历史需求Baseline，重新训练九类机器学习模型，并运行了提供的DFL鲁棒场景原型的真实数据适配版。原始数据和旧模型结果没有覆盖。

在本次6000个SKU、28期测试中：

- 历史Baseline：平均Ready Rate为99.6464%，98.35%的SKU达标，加权周转7.2709，平均库存约1474万。
- Ridge：全部SKU达标，加权周转8.0118，平均库存约1321万。它是本轮**全SKU可行方案中加权周转最高**的方案，较Baseline库存减少约10.4%。
- 随机森林：加权周转8.3057，99.95%的SKU达标，但仍有3个SKU不达标，不能视为全SKU可行。
- DFL原型：服务率100%，但是平均库存约5905万，加权周转约1.5，没有优于简单方案。
- 验证选择了不做DFL微调的检查点；固定三轮微调在测试上将加权周转从1.5262提高到1.5931，约4.4%，但仍明显落后。这个额外测试现象不能替代验证支持。

这只是一次有限参数搜索和时间划分的工程试验，不是全局最优、统计显著或长期稳定的证明。尤其验证集容量不可行、测试供给宽松，见第三节。

## 二、我们现在解决的是什么问题

新规则下，当期x1～x20在下单前已知，当期真实需求尚未知；下单后补货立即到达，再满足需求。缺货是lost sales，不结转欠货。6000个SKU共享每期总补货容量。

目标是提高库存周转，同时要求**每个SKU的Ready Rate至少95%**。Ready Rate不是Fill Rate：前者计算有正期末库存的周期比例，后者计算满足的需求数量比例。一个方法可能满足99%的需求量，却在很多期把库存耗尽。

所有方法沿用原型的参数假设：

- 初始库存：每个SKU训练均值的1.25倍。
- 每期总补货容量：训练期平均总需求的1.05倍。
- 允许连续数量补货，没有最小包装、固定订货费用和货币成本。
- 使用期末库存平均值作为周转分母。

这些是实验假设，不代表已核实的真实业务参数。本轮不使用Excel的initial_inventory列。需求在期末被完整观察，包括缺货未满足的需求；若真实业务只能看到销量，需要处理需求截尾。

旧实验采用Fill Rate、提前一期到货和未来x不可知，解决的是另一个问题；旧表不能与本轮直接混排。

## 三、数据、时间划分与一个关键限制

训练文件是`data/replenishment_train.xlsx`，60期；测试文件是`data/replenishment_test.xlsx`，28期，SKU身份和顺序已校验一致，均为6000个。数据冻结为新目录中的`data.npz`；源文件和源码SHA-256记录在`manifest.json`。

训练内部采用前42期拟合、第43～60期验证。选参数后用全60期重训，最后只在28期测试上评价。ML和NN需要12期历史，因此监督拟合行是第13～42期，最终重训第13～60期；统计均值、标准差、容量和初始库存仍使用完整42期或60期。

测试时允许读取已经结束的测试期需求，更新下一期历史特征；不能读取当期尚未实现的需求或未来需求，不在测试集上拟合模型或选择参数。删除绝对period_index，因为测试工作簿的周期编号重新从1开始，不能直接当连续趋势标签外推。

### 验证集为什么无法全SKU达标

前42期估计容量约458万/期，第43～60期平均需求约913万/期。18期验证总需求约1.644亿，但初始库存加18次最大补货约8793万，明显不够。

18期中17期有货只等于94.44%；因此，每SKU达到95%实际上要求18期全有正期末库存，即所有需求完全满足。总供给小于需求，故**验证集全SKU服务要求在当前容量下数学上不可行**，不是某个模型不收敛造成的。

验证选择规则为：如果有全SKU可行方案，优先选择加权周转最高的方案；否则依次比较达标SKU比例、平均Ready Rate、加权周转。后者是明确的失败回退规则，不等价于原始约束优化目标。所有候选表现保存在`validation_grid.csv`，而不是隐藏不达标情况。

全60期估计测试容量约608万/期，测试平均需求仅323万/期。验证与测试供需压力相差很大，所以本次验证选择的参数不能视为充分泛化校准。后续应先确认实际容量，再采用多个滚动时间窗口验证；本轮没有为获得漂亮结果而调整容量。

## 四、只用历史需求的Baseline

对每个SKU计算训练需求均值和样本标准差，不看x，不训练机器学习预测器。目标库存为：

$$
S_i=\mu_i+z\sigma_i,\qquad s_i=S_i.
$$

没有固定订货费用，因此(s,S)退化为目标库存策略。每期期初，想补的量为：

$$
\widetilde q_{i,t}=\max(S_i-I_{i,t-1},0).
$$

如果大家想补的量超过总容量C，则按比例缩小：

$$
q_{i,t}=\widetilde q_{i,t}\min\left(1,\frac{C}{\max(\sum_j\widetilde q_{j,t},10^{-8})}\right).
$$

安全系数z用验证在0、0.5、1、2、3、4中选择；本轮选中4。它不是“95%服务率直接对应正态95%分位数”，因为共享容量、非平稳需求和周期服务指标改变了关系。

## 五、机器学习模型怎么做

单独训练线性回归、Ridge、Lasso、弹性网络、SVR、随机森林、XGBoost、LightGBM、CatBoost，不做融合。每个家族搜索1～2套预测参数及上述6个z，完整参数和结果保存在CSV中。没有进行无限预算搜索。

所有模型共用38个特征：

|特征|数量|是否使用未来需求|
|---|---:|---|
|过去1、2、3、6、12期需求|5|否|
|过去3、6、12期均值|3|否|
|过去3、6、12期标准差|3|否|
|需求差分、比率、斜率|4|否|
|历史需求及历史均值对数|2|否|
|商品category编码|1|否|
|当期已知x1～x20|20|不含需求|

数量历史特征除以固定训练SKU均值，降低不同SKU数量级的差异。预测目标为`log(1+需求)−log(1+训练SKU均值)`，预测后还原到需求单位。类别编码只是简单数值编码，并非最优类别处理。

线性类模型标准化；SVR另做95%解释方差PCA。SVR因成本限制，候选拟合抽8000行，最终拟合抽10000行；其他模型使用全部监督训练样本。因此不能将差异解释为SVR模型理论上较差。

得到需求预测p后，采用`S=p+z×历史标准差`，再用与Baseline相同的目标库存和共享容量缩放决策。预测层本身不含动态库存，**决策层使用当期库存**。

WAPE是“全部绝对预测误差之和/全部真实需求之和”，越低越好。预测准确不等于补货决策好；当前数据测试首期存在历史需求衔接变化，不同方法可能受到不同影响。

## 六、DFL鲁棒场景方案怎么做

原型复制到独立模型目录，原Downloads文件不改动。保留：概率NN、SKU embedding、log需求高斯NLL预训练、残差SVD低秩因子、8个随机需求场景、10步梯度展开优化、7期滚动任务损失和鲁棒软最差场景混合。

必要适配包括：使用与ML相同的历史+当期x；增加训练SKU对数均值偏置；特征仅在拟合区间标准化；容量、初始库存、评价权重与其他方法一致；原型log需求上限8会把所有高于约2980的需求截断，改为数值保护上限20。它是适配版，不是逐字不变的复现。

比较NN的确定性、随机场景、鲁棒随机场景、不微调与DFL微调。鲁棒半径先在0、0.2、0.4、0.8中验证选择，本轮选0.2，再比较0～3个微调检查点。本轮选择0；固定3轮另外保留，但没有在测试上选它。

DFL任务损失由负对数周转、逐SKU软服务不足平方罚和0.05倍预测NLL组成。训练服务使用`sigmoid(10×(期末库存−0.02))`，真实评价使用`期末库存>1e−8`。两者不是同一个指标：库存0时sigmoid约0.45，库存数量较大时容易饱和。

有限罚项不能保证服务硬约束；归一化softmin不是精确最坏场景值；10步展开不保证最优决策。该方法是**DRO启发的有限场景压力模型**，不是精确Wasserstein-DRO，也不是SPO+定理的直接实现。

训练窗口每次重置初始库存，未采用真实历史库存状态；残差因子使用预训练的样本内残差、微调后冻结。这些是保留原型的局限，后续研究需要再完善。验证微调与最终重训都按单轮调用、重置Adam的同一协议执行，随机种子42。

### 为什么当前DFL库存很多

原型warm start按预测需求比例分配几乎全部容量，未扣当前库存；unused capacity的初始logit很小。6000维上10次梯度更新不足以明显减少总订货。测试需求约323万/期，而原型仍订约608万/期，所以库存不断积累。

这是当前实现机制的解释，不能推导出“DFL在这个问题上一定无效”。相比相同NN、同一决策层，不微调与固定三轮微调的测试加权周转分别为1.5262和1.5931；总量周转几乎不变。它主要改善SKU间分配，而没有减少总体积压，且验证没有选中微调模型。训练NLL与任务损失轨迹见`dfl_training_trace.json`。

## 七、完整测试比较

完整15行结果见结果表（历史产物未包含/需运行生成：`../outputs/ready_rate_dfl_dro_20261005/result_table.md`）和比较CSV（历史出处：`../outputs/ready_rate_dfl_dro_20261005/comparison.csv`）。其中10行是Baseline+九个ML，另外5行是NN/DFL变体；DFL验证选择0轮时与不微调版本重复，这是如实保留的选择结果，不是独立新算法。

|代表方案|需求WAPE|平均Ready Rate|达标SKU比例|加权周转|总量周转|平均库存|
|---|---:|---:|---:|---:|---:|---:|
|历史Baseline|96.80%|99.6464%|98.35%|7.2709|6.1235|14,744,567|
|线性回归|49.70%|100%|100%|7.9876|6.8266|13,251,816|
|Ridge|48.86%|100%|100%|8.0118|6.8483|13,209,973|
|随机森林|29.33%|99.9940%|99.95%|8.3057|7.1356|12,677,918|
|XGBoost|28.35%|99.9851%|99.90%|8.2897|7.1443|12,662,267|
|LightGBM|29.12%|99.9815%|99.8833%|8.2492|7.1098|12,723,520|
|NN鲁棒/验证所选DFL|27.20%|100%|100%|1.5262|1.5319|59,054,650|

**平均Ready Rate≥95%不代表每个SKU≥95%。**28期中每SKU最多允许1期无货，27/28=96.43%。Lasso和弹性网络周转虽高，却明显未达服务要求，不能作为胜者。

DFL与ML补货层不同，这里是完整流水线比较。判断DFL训练收益，应比较相同NN和同一决策层的微调与不微调；不能把全部差异归因于损失函数。

## 八、评价公式与工程文件

每期可用库存=上期期末库存+当期补货；销量=min(可用库存,需求)；期末库存=可用库存−销量。

Ready Rate_i=有正期末库存的期数/H；Fill Rate=总销量/总需求。平均库存_i=各期期末库存之和/H。

总量周转=整个测试窗口总销量/所有SKU平均库存之和。加权周转使用相同公式，但销量和库存均乘SKU权重；权重与`1/max(训练SKU均值,0.5)`成正比。它避免大SKU完全支配评价，但不是简单平均各SKU周转。

这是**数量口径的周转代理**，不是金额口径COGS/平均库存价值，未年化。原型内部未除H的累计比例等于本报告窗口周转/H；内部保留原式，避免悄悄改变罚项相对权重，外部统一使用平均库存口径。

独立评价类是`src/evaluation/ready_rate_evaluator.py`。每个方法目录保存：

- `sku_metrics.csv`：每SKU Ready Rate、Fill Rate、平均库存、周转；不打印6000行到终端。
- `trajectory.npz`：逐SKU逐期补货、需求、销量、期末库存和初始库存。
- ML的`model.joblib`、`forecast.npz`；NN的预训练权重和两种DFL检查点权重。

共同目录`outputs/ready_rate_dfl_dro_20261005/`含`manifest.json`、验证候选表、DFL训练轨迹、比较表、`audit.json`。模型加载仍需同一特征函数、训练SKU尺度和参数记录，不是单独拿权重文件就能跨业务口径调用。

## 九、直接复制到终端的完整命令

```bash
cd '<PROJECT_ROOT>'
python experiments/prepare_ready_rate_data.py
python experiments/run_ready_rate_ml.py
python experiments/run_ready_rate_dfl.py
python experiments/summarize_ready_rate_results.py
python -m unittest tests.test_ready_rate_evaluator
```

以上命令重跑只更新本次新实验目录。ds和torch是当前机器现有环境：前者包含Excel读取与九种ML依赖，后者包含PyTorch；装齐依赖后也可以用同一环境。实际依赖版本见`requirements-ready-rate.txt`，没有要求安装新的包。

独立复核重新计算库存守恒、销量、服务率、周转、非负性和共享容量，并确认所有模型使用完全相同的测试需求。`audit.json`须全部通过后才接受比较表。随机森林本轮约204秒，SVR约105秒，Boosting模型约5～9秒，线性类约1秒以内；NN训练采用CPU两线程。

### 原始说明：no_safety_lead_one_report.md

### 无安全库存实验：提前期1个周期

## 1. 结论

已对历史Baseline及九类现有ML预测运行无安全库存评价。**本次无安全库存方案没有一个满足全部SKU的95% Ready Rate要求。**

XGBoost平均库存约95万、LightGBM约91万，明显减少了库存；但平均Ready Rate分别只有64.03%和61.18%。
XGBoost总体Fill Rate为95.40%，却只有2/6000个SKU的Ready Rate达标。因此，Fill Rate超过95%不能替代Ready Rate要求。

SVR平均Ready Rate为96.60%，但仅5189/6000个SKU达标，仍非全SKU可行。
这不说明取消安全库存一定不可行，只说明**当前已训练预测器加上这套零缓冲补货规则**没有满足要求。

## 2. 先说清楚本次事件顺序

本轮按项目原有提前期1期的期末订货方式执行：

1. 每期期初收到上一期的订单。
2. 当期需求发生，从可用库存销售；未满足部分丢失。
3. 期末观察当期需求、当期x，预测下一期需求并下单。
4. 此订单下一期期初到达，不能用于满足刚刚发生的当期需求。

这里的“提前期1期”是上述完整顺序，不是当期立即到货。如果业务要求在当期需求发生之前下单，需要重新明确保护期和库存位置；不能直接把这张表当作那个事件顺序的结果。

初始库存仍保留上轮假设：每SKU60期训练平均需求的1.25倍；测试开始时在途订单为0。
因此“无安全库存”指**订货策略不额外添加缓冲**，不是清空初始库存。
共享每期订货容量为全训练平均总需求的1.05倍，即6084614.41。
初始库存和容量对无安全库存、参考组完全相同，原始Excel和旧实验文件不变。

测试最后一期不再下单，因为它不能在28期评价窗口内到达；无终值或窗口外销量奖励。
保存的平均每期订货量包括这一期的零订单，同时另存27个有效订货期的平均值。
这是一个带初始条件的28期实验，不是稳态库存系统分析。

## 3. 无安全库存怎样决策

设期末库存为I，下一期需求预测为p：

```text
目标库存 = 下一期需求预测
想订的数量 = max(目标库存 − 期末库存, 0)
若所有SKU想订数量超过容量，则按比例缩减。
```

没有`z×标准差`，没有额外最小库存、正库存余量或向上取整缓冲。
当需求恰好等于可用库存时，该期Fill Rate可以是100%，但是期末库存为0，Ready Rate不达标。

Baseline直接使用完整训练历史均值作为预测，不看x。

## 4. 使用的是哪批ML模型

本次复用`outputs/machine_learning/<模型>/test_forecasts.csv`里的已训练模型预测，未重新训练。
对应原训练代码为`experiments/run_machine_learning.py`：目标期t只使用t−1及更早的需求、x，以及类别与周期特征。
这些预测适用于期末t−1对t下单，不会读取下一期未知x。

**没有使用10月5日立即到货实验中“读取当期x预测当期需求”的预测结果冒充下一期预测。**
它们不是同一批预测器，因此不能将本次结果与10月5日表格相减，声称差异全部由安全库存造成。

已冻结预测器的参数来自此前训练内部选择；本次不在测试集上重训、选模型、选安全系数，也不声称重新优化了Ready Rate。
本次无聚类，未改变现有预测特征。源预测文件、旧模型summary和冻结需求的SHA-256在`manifest.json`中。

## 5. 无安全库存结果

|方案|平均Ready Rate|达到95% Ready Rate的SKU|总Fill Rate|平均库存|加权周转|
|---|---:|---:|---:|---:|---:|
|历史Baseline|83.49%|3074/6000|91.28%|2,917,932|34.580|
|线性回归|62.11%|1/6000|89.54%|961,814|85.141|
|Ridge|57.29%|0/6000|94.75%|899,950|98.167|
|Lasso|58.03%|0/6000|94.76%|903,649|101.921|
|弹性网络|53.68%|0/6000|95.00%左右|960,269|102.856|
|SVR|96.60%|5189/6000|95.59%|5,724,933|4.777|
|随机森林|61.72%|0/6000|95.22%|910,479|99.455|
|XGBoost|64.03%|2/6000|95.40%|948,377|96.692|
|LightGBM|61.18%|0/6000|94.96%|906,242|101.768|
|CatBoost|75.78%|173/6000|95.44%|1,126,813|52.861|

注意弹性网络精确Fill Rate为94.9959%，按两位小数会显示95.00%，不代表确实≥95%。实际约束仍是逐SKU Ready Rate。
28期中每SKU至少27期有正库存才能达到95%，26/28只有92.86%。周转率高却服务不达标，不能算业务目标胜者。

## 6. 同预测器的固定缓冲参考组

为了隔离是否增加安全库存，另设`固定z=4×训练需求标准差`参考组。
它与零缓冲组使用完全相同的预测、初始库存、提前期、容量和评价代码。
z=4是预先固定的参照，不是在这次测试上搜索出的最优值，也不证明需要这么大的库存。

|方案|零缓冲平均库存|固定z4平均库存|零缓冲达标SKU|固定z4达标SKU|
|---|---:|---:|---:|---:|
|XGBoost|948,377|12,227,814|2|6000|
|LightGBM|906,242|12,158,528|0|6000|
|Ridge|899,950|12,104,856|0|6000|
|随机森林|910,479|12,147,169|0|6000|

这只显示两个端点的库存—服务权衡：不加缓冲太容易耗尽库存，固定大缓冲又占用库存。
更合理的后续方向是在训练内部根据样本外预测误差校准较小、SKU/分组适配的库存余量，不能为了让本次测试通过而直接调参。

## 7. 工程文件与复核

输出目录：`outputs/no_safety_lead_one_20261007/`。

- `comparison.csv`：10种方法各两组，共20条同口径结果。
- `result_table.md`：完整对比。
- `no_safety/<模型>/sku_metrics.csv`：每SKU的Ready Rate、Fill Rate、库存、周转，不占用终端。
- `trajectory.npz`：逐SKU逐期预测、目标、实际下单量、到货量、需求、销量和期末库存。
- `fixed_z4_reference/`：固定缓冲参考组。
- `manifest.json`与`audit.json`：事件顺序、来源哈希和20条轨迹的复核结果。

新评价类为`src/evaluation/lead_one_ready_rate.py`。KPI沿用共同评价类，独立复核确认：
订单只能下一期到达、首期在途为0、末期订单为0、库存守恒、销量正确、订单非负、容量不超限、真实服务与两种周转率正确。
三项新增单元测试包括“新订单不能满足当期需求”和“完美预测可Fill Rate100%但Ready Rate0%”。

```bash
cd '<PROJECT_ROOT>'
python experiments/run_no_safety_lead_one.py
python -m unittest tests.test_lead_one_ready_rate tests.test_ready_rate_evaluator
```

该命令重新评价冻结预测，不会重新训练或覆盖旧模型。


## 6. 限制

不跨A/B/C组混排；总量周转和加权周转分开；软服务罚项不等于95%硬约束。多轮开发使用过同一个测试集，不是独立确认性结果。有限搜索、首阶近似、容量不足等限制按原报告保留。此次代码重构以语法、导入、小型逻辑与数值一致性校验为主，真实全量重训不是此次任务。
