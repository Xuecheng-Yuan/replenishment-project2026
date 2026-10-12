# direct_xgboost：模型实现与说明

[总览](../README.md) · [实际代码](../models/direct_xgboost.py) · [全部变体记录](../results/variants.json)

## 1. 文件如何对应

本模型所有定义、关键决策/损失函数或估计器构造在`models/direct_xgboost.py`，不是转发到旧experiments目录的脚本。数据/评价/公共训练流程位于common/。参数、窗口、种子和消融不另建模型文件。历史125项仅为结果记录，完整索引见results/variants.json。

## 2. 信息时序与模型原理

### A组

不输出需求，直接评价11个历史尺度候选目标库存动作；使用一步或六步真实历史展开成本构造监督，再用成本MSE和平滑SPO型联合损失训练树。六步标签仅用于训练；执行不看未来需求。原入口同时训练两个版本。

SKU是商品编号；x1–x20是外部属性/特征，不是需求本身。历史滞后表示预测目标日前已观察值。库存状态是今天卖完后的库存，不是明天已实现的库存。预测层误差小不自动意味着决策服务达标。

### 核心公式

$$
a^*(x,I)=\arg\min_a\widehat c_\theta(x,I,a)
$$

d为需求，I为期末库存，s为实际销量，q为下单量；预测目标用S表示，C为公共容量，z为风险余量系数；a_i为训练期商品尺度，theta为学习参数。上式是核心计算关系/目标概要，不把代理损失写成业务硬约束。完整推导和实现细节见后面的完整推导；共享符号、Fill Rate与Ready Rate区别见[评价标准](evaluation.md)。


## 3. 运行命令

从本仓库根目录运行；默认只查看历史结果，不训练：

```bash
python -m models.direct_xgboost
python -m models.direct_xgboost --help
python -m models.direct_xgboost --train
```

普通ML与baseline可用`--protocol legacy`、`--protocol ready_immediate`、`--protocol lead_one`切换已存在实验口径。lead_one为旧冻结预测的安全库存移除/固定z4对照，不是重新在该口径调参。移动平均用`--window 3/6/12`；MPC用`--round all/1/2/3/4`。高级模型默认沿用原批次的验证/消融，但实现全在同一文件，不会调用旧脚本树。

原始Excel放data/。运行只写results/generated/，不覆盖results/variants.json中的历史快照。改变输入请新目录重跑，不允许旧缓存自动作为新数据证据。

## 4. 历史结果（所有配置集中列在这里）

|原标签|组|角色|Fill Rate|Ready Rate达标比例|平均总库存|原始周转|加权周转|
|---|---|---|---:|---:|---:|---:|---:|
|legacy_direct_xgb_spo_rollout|A|主方案|0.950384|未记录|2.12831e+06|40.3968|未记录|
|legacy_direct_xgb_spo_one_step|A|主方案|0.964354|未记录|4.87115e+06|17.9097|未记录|

小数比例0.95就是95%。参数与来源SHA256逐项保存在results/variants.json。历史业务失败没有改成通过；没有因为此次重构而重训全部真实数据。

## 5. 完整数学与训练细节

以下为本模型相关历史说明的完整正文，记录旧研究假设；当时“最优”只限于自己的实验口径。运行请用上面的新命令，以下旧文件路径仅作为出处，不表示新目录仍有旧结构。

### 原始说明：direct_xgb_spo_policy_report.md

### 不预测需求的直接补货决策：XGBoost–SPO

## 1. 先说结论

本项目已经实现两种“不先预测需求、直接产生补货决策”的模型：

| 模型 | 测试总体 Fill Rate | 平均总库存 | 总周转率 | 满足 95% |
|---|---:|---:|---:|---:|
| Direct XGB–SPO（一步成本） | 96.4354% | 4,871,152 | 17.9097 | 是 |
| Direct XGB–SPO（六步滚动成本） | 95.0384% | 2,128,309 | 40.3968 | 是 |
| 当前最佳 LightGBM（先预测再决策） | 96.9577% | 585,283 | 149.8646 | 是 |
| 均值—方差 baseline | 98.7838% | 11,949,377 | 7.4787 | 是 |

六步滚动版证明了直接决策路线是可行的：它没有需求预测值，仍在测试集满足 95%
总体 Fill Rate，并比 baseline 减少约 82.2% 的平均库存。不过，它的库存仍约为当前
最佳 LightGBM 的 3.64 倍，所以现在应把它视为一个有效的新研究分支，而不是替换
LightGBM 的最终模型。

## 2. “不预测需求”究竟是什么意思

传统流程是：

$$
\text{历史信息}
\longrightarrow \widehat d_{i,t+1}
\longrightarrow B_{it}
\longrightarrow Q_{it}.
$$

本模型的流程是：

$$
\text{历史信息和当前库存状态}
\longrightarrow \widehat{\boldsymbol c}_{it}
\longrightarrow A_{it}
\longrightarrow Q_{it}.
$$

其中，$\widehat{\boldsymbol c}_{it}$ 是所有候选补货动作的预测成本，而不是需求预测。
模型直接选择预测成本最低的动作。因此，结果目录没有 `test_forecasts.csv`。

模型仍使用过去 12 期需求均值作为“单位换算尺”$r_{it}$。它只是把无量纲动作换成
实际库存数量，并不是另一个训练出来的需求预测模型。

## 3. 决策时间顺序

对 SKU $i$ 和周期 $t$：

1. 上一期订单在本期期初到货；
2. 本期需求 $d_{it}$ 发生；
3. 计算销售、缺货和期末库存 $I_{it}$；
4. 此时才能使用截至周期 $t$ 已经观察到的信息；
5. 模型选择目标库存 $B_{it}$，下单 $Q_{it}$；
6. 订单在周期 $t+1$ 的需求发生前到货。

所以模型从来没有用 $t+1$ 期需求或特征来决定周期 $t$ 的订单。

## 4. 数学模型

### 4.1 符号

| 符号 | 含义 |
|---|---|
| $i\in\mathcal I$ | SKU |
| $t\in\mathcal T$ | 决策周期 |
| $k\in\mathcal K$ | 离散候选动作 |
| $d_{it}$ | 实际需求 |
| $I_{it}$ | 满足本期需求后的期末库存 |
| $r_{it}$ | 截至 $t$ 的最近 12 期需求均值 |
| $a_k$ | 第 $k$ 个库存倍率动作 |
| $B_{itk}$ | 选择动作 $k$ 后的目标库存 |
| $Q_{it}$ | 实际订货量 |
| $h$ | 单位期末持有成本，本实验为 1 |
| $\lambda$ | 单位缺货惩罚，也是服务约束的拉格朗日乘子 |
| $H$ | 成本标签的滚动期数 |

### 4.2 状态和动作

决策状态 $x_{it}$ 包括：

- 严格滞后的需求、滚动统计和 `x1`–`x20`；
- 当期期初可用库存、期末库存、需求和缺货；
- 上述库存量与历史需求尺度 $r_{it}$ 的比值；
- 品类和周期索引。

动作集合为：

$$
\mathcal A=\{0,0.5,1.0,\ldots,5.0\}.
$$

动作 $a_k$ 对应的实际目标库存是：

$$
B_{itk}=\left\lceil a_k r_{it}\right\rceil.
$$

模型选定 $A_{it}$ 后，下单量为：

$$
Q_{it}=\max\{0,B_{it,A_{it}}-I_{it}\}.
$$

### 4.3 一步真实动作成本

若选择候选动作 $k$，下一期期初可用库存为：

$$
Y_{i,t+1,k}=\max\{I_{it},B_{itk}\}.
$$

下一期的期末库存和丢失销售分别为：

$$
I_{i,t+1,k}^{(k)}=\max\{Y_{i,t+1,k}-d_{i,t+1},0\},
$$

$$
L_{i,t+1,k}^{(k)}=\max\{d_{i,t+1}-Y_{i,t+1,k},0\}.
$$

一步成本标签为：

$$
c_{itk}^{(1)}
=hI_{i,t+1,k}^{(k)}+\lambda L_{i,t+1,k}^{(k)}.
$$

$\lambda$ 越大，模型越害怕缺货，通常会选择更高的目标库存。

### 4.4 多步滚动成本

一步成本只看下一期，容易产生“眼前便宜、后面昂贵”的动作。滚动版让同一个候选
倍率在未来 $H$ 期连续作用，并使用训练期已经发生的真实需求计算成本标签：

$$
c_{itk}^{(H)}
=\sum_{\tau=1}^{H}\gamma^{\tau-1}
\left(hI_{i,t+\tau,k}^{(k)}
+\lambda L_{i,t+\tau,k}^{(k)}\right).
$$

本次候选 $H\in\{3,6\}$，折扣因子 $\gamma=1$。最终验证集选择了 $H=6$。
这里的未来真实需求只用于构造训练标签，测试决策时不可见。

### 4.5 XGBoost 输出什么

XGBoost 同时输出 $K$ 个动作成本：

$$
f_\theta(x_{it})=
\left(\widehat c_{it1},\ldots,\widehat c_{itK}\right).
$$

最终动作是：

$$
A_{it}=\arg\min_{k\in\mathcal K}\widehat c_{itk}.
$$

因此它不是“预测一个连续订货量”，而是“比较一组合法动作，直接挑一个”。这样更
容易保证输出非负、动作范围明确，也能把业务成本直接写进训练目标。

## 5. SPO+混合损失

真实成本向量记为 $\boldsymbol c$，预测成本向量记为
$\widehat{\boldsymbol c}$。模型使用成本均方误差和平滑 SPO+ 的混合：

$$
\mathcal L_\alpha
=(1-\alpha)\mathcal L_{\mathrm{MSE}}
+\alpha\mathcal L_{\mathrm{SPO+}},
$$

其中 $\alpha=0.75$。MSE 帮助模型大致预测成本大小；SPO+ 更关心最终是否选对动作。
完整梯度推导见
`tree_boosting_spo_derivation_and_literature.md`（历史出处：`tree_boosting_spo_derivation_and_literature.md`）。

需要准确表述：这是“决策导向的端到端策略学习”，但不是神经网络意义下从仿真器到
树模型完全可微的端到端系统。离散 `argmin`、库存仿真和树分裂仍然不可微，当前实现
通过动作成本标签和平滑 SPO+ 把决策质量传回树模型。

## 6. 服务率约束如何处理

业务问题为：

$$
\min_\pi\quad \operatorname{AvgInventory}(\pi)
$$

$$
\text{s.t.}\quad
\operatorname{FillRate}(\pi)\ge 0.95.
$$

代码用 $\lambda$ 对缺货进行拉格朗日惩罚，并在验证集搜索：

$$
\lambda\in\{0.75,1.00,1.25,1.50,1.75,2.00\}.
$$

为了应对验证集到测试集的分布变化，当前工程版本要求候选的验证 Fill Rate 至少为
97%，再从中选平均库存最低的方案；最终业务合格线仍为 95%。

必须说明：最初使用 95% 验证门槛的版本在当前测试集上失守，项目在诊断这一现象后
才加入 2 个百分点的安全裕度。因此，对这两个新模型而言，当前测试集已经成为开发
反馈的一部分，表中的测试结果适合做工程迭代比较，但不能再当作完全无偏的最终泛化
估计。若用于论文或正式上线验收，应增加一个从未查看过的新时间窗口作为最终留出集。

## 7. 训练、验证和测试

- 训练文件：`data/replenishment_train.xlsx`，60 期、6,000 个 SKU；
- 测试文件：`data/replenishment_test.xlsx`，28 期、6,000 个 SKU；
- 训练期前段生成动作成本标签并拟合候选模型；
- 训练期第 43–60 期只用于选择 $H$ 和 $\lambda$；
- 选好参数后，用全部可用训练期重新拟合；
- 当前测试集只参与滚动评价；但由于安全裕度是在观察早期测试失守后加入的，它对新
  模型已不再是严格意义上的最终盲测集。

每次训练包含固定随机种子 42。当前正式版本调参和最终训练都使用 40 棵树，避免
调参流程与最终训练流程不一致。

## 8. 结果解释

一步版更强调紧接着的一期，测试 Fill Rate 较高，但库存较大。六步版能看到较长的
持有成本累积，因此显著减少库存，并恰好在 95% 约束之上。

与 baseline 相比，六步版同时保住服务约束并显著降低库存，说明直接决策模型学到了
有用策略。与 LightGBM 相比仍有较大差距，主要可能来自：

1. 动作只有 11 个离散倍率，粒度较粗；
2. 所有 SKU 共用一个历史均值尺度，对趋势和间歇需求不够灵活；
3. 训练标签是假设同一倍率在滚动窗口内重复执行的近似成本；
4. 当前未进行多轮 on-policy 状态再采样；
5. 40 棵树偏保守，尚未联合调节树深、学习率、SPO权重和温度。

## 9. 运行命令

从项目根目录直接复制：

```bash
cd "<PROJECT_ROOT>"

PYTHONPYCACHEPREFIX=/private/tmp/replenishment_direct_pycache \
MPLCONFIGDIR=/private/tmp/matplotlib \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
python \
  -u -m experiments.run_direct_xgb_spo --n-jobs 8 --item-id 1
```

如果验证候选表已经存在，只想重训正式模型：

```bash
cd "<PROJECT_ROOT>"

PYTHONPYCACHEPREFIX=/private/tmp/replenishment_direct_pycache \
MPLCONFIGDIR=/private/tmp/matplotlib \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
python \
  -u -m experiments.run_direct_xgb_spo \
  --reuse-validation --n-jobs 8 --item-id 1
```

## 10. 输出文件

两个模型分别保存在：

- `outputs/spo/direct_xgb_spo_one_step/`；
- `outputs/spo/direct_xgb_spo_rollout/`。

每个目录包含：

- `summary.json`：参数、验证选择和测试总体结果；
- `test_per_sku_metrics.csv`：6,000 个 SKU 各自的 Fill Rate、库存和周转率；
- `test_trajectory.csv`：逐 SKU、逐期的到货、需求、销售、缺货、库存和订单；
- `test_order_up_to.csv`：模型直接给出的动态目标库存；
- `test_selected_actions.csv`：模型选择的倍率动作；
- `cost_xgb_spo_model.json`：训练后的 XGBoost 成本模型；
- `model_metadata.json`：动作集合、特征名和所选参数。

全部候选参数的验证结果位于
`outputs/spo/direct_xgb_spo/validation_candidates.csv`，与主要模型的比较位于
`outputs/spo/direct_xgb_spo/comparison.csv`。

## 11. 复现检查

当前自动测试共 20 项，覆盖动态补货时序、特征无泄漏、滚动参考尺度、候选动作、
一步与多步动作成本，以及 SPO+ 梯度的数值差分检查。

```bash
cd "<PROJECT_ROOT>"
python -m unittest discover -s tests -v
```


## 6. 限制

不跨A/B/C组混排；总量周转和加权周转分开；软服务罚项不等于95%硬约束。多轮开发使用过同一个测试集，不是独立确认性结果。有限搜索、首阶近似、容量不足等限制按原报告保留。此次代码重构以语法、导入、小型逻辑与数值一致性校验为主，真实全量重训不是此次任务。
