# direct_lightgbm：模型实现与说明

[总览](../README.md) · [实际代码](../models/direct_lightgbm.py) · [全部变体记录](../results/variants.json)

## 1. 文件如何对应

本模型所有定义、关键决策/损失函数或估计器构造在`models/direct_lightgbm.py`，不是转发到旧experiments目录的脚本。数据/评价/公共训练流程位于common/。参数、窗口、种子和消融不另建模型文件。历史125项仅为结果记录，完整索引见results/variants.json。

## 2. 信息时序与模型原理

### A组

不预测需求，展开11个候选动作；三期滚动真实成本标签，联合成本MSE、平滑SPO+、后悔加权排序损失。验证选择缺货惩罚及权重；当前选中SPO权重0.4、排序权重0.35。

SKU是商品编号；x1–x20是外部属性/特征，不是需求本身。历史滞后表示预测目标日前已观察值。库存状态是今天卖完后的库存，不是明天已实现的库存。预测层误差小不自动意味着决策服务达标。

### 核心公式

$$
\mathcal L=w_0\mathcal L_{\mathrm{MSE}}+w_1\mathcal L_{\mathrm{smooth\ SPO+}}+w_2\mathcal L_{\mathrm{rank}}
$$

d为需求，I为期末库存，s为实际销量，q为下单量；预测目标用S表示，C为公共容量，z为风险余量系数；a_i为训练期商品尺度，theta为学习参数。上式是核心计算关系/目标概要，不把代理损失写成业务硬约束。完整推导和实现细节见后面的完整推导；共享符号、Fill Rate与Ready Rate区别见[评价标准](evaluation.md)。


## 3. 运行命令

从本仓库根目录运行；默认只查看历史结果，不训练：

```bash
python -m models.direct_lightgbm
python -m models.direct_lightgbm --help
python -m models.direct_lightgbm --train
```

普通ML与baseline可用`--protocol legacy`、`--protocol ready_immediate`、`--protocol lead_one`切换已存在实验口径。lead_one为旧冻结预测的安全库存移除/固定z4对照，不是重新在该口径调参。移动平均用`--window 3/6/12`；MPC用`--round all/1/2/3/4`。高级模型默认沿用原批次的验证/消融，但实现全在同一文件，不会调用旧脚本树。

原始Excel放data/。运行只写results/generated/，不覆盖results/variants.json中的历史快照。改变输入请新目录重跑，不允许旧缓存自动作为新数据证据。

## 4. 历史结果（所有配置集中列在这里）

|原标签|组|角色|Fill Rate|Ready Rate达标比例|平均总库存|原始周转|加权周转|
|---|---|---|---:|---:|---:|---:|---:|
|legacy_direct_lgbm_spo_rank|A|主方案|0.959206|未记录|2.19854e+06|39.4693|未记录|

小数比例0.95就是95%。参数与来源SHA256逐项保存在results/variants.json。历史业务失败没有改成通过；没有因为此次重构而重训全部真实数据。

## 5. 完整数学与训练细节

以下为本模型相关历史说明的完整正文，记录旧研究假设；当时“最优”只限于自己的实验口径。运行请用上面的新命令，以下旧文件路径仅作为出处，不表示新目录仍有旧结构。

### 原始说明：direct_lgbm_spo_rank_report.md

### LightGBM与SPO+直接补货决策：成本回归、SPO+与排序的混合模型

> 实验日期：2026-09-19  
> 模型名称：`direct_lgbm_spo_rank`  
> 核心区别：不预测未来需求，直接选择补货动作

## 1. 结论

新模型已经完成训练、调参和测试。正式工程结果为：

| 模型 | 测试总体 Fill Rate | 平均总库存 | 总周转率 | 满足95% |
|---|---:|---:|---:|---:|
| 普通LightGBM：先预测再决策 | 96.9577% | 585,283 | 149.8646 | 是 |
| Direct XGBoost–SPO Rollout | 95.0384% | 2,128,309 | 40.3968 | 是 |
| **Direct LightGBM–SPO–Rank** | **95.9206%** | **2,198,542** | **39.4693** | **是** |
| Direct XGBoost–SPO One-step | 96.4354% | 4,871,152 | 17.9097 | 是 |
| 均值—方差 baseline | 98.7838% | 11,949,377 | 7.4787 | 是 |

Direct LightGBM将baseline的平均库存降低约81.6%，同时保持95%以上的总体Fill
Rate。它明显优于直接XGBoost的一步版本，与直接XGBoost滚动版接近，但仍没有超过
普通LightGBM“先预测、再决策”的方案。

因此，本实验的主要价值是证明LightGBM可以绕过需求点预测，直接学习补货动作；它
目前是一个可行的研究模型，而不是项目的最终最优模型。

## 2. 与普通LightGBM的区别

普通LightGBM：

$$
x_{it}\longrightarrow \widehat d_{i,t+1}
\longrightarrow B_{it}\longrightarrow Q_{it}.
$$

Direct LightGBM：

$$
x_{it},I_{it}\longrightarrow
(\widehat c_{it1},\ldots,\widehat c_{itK})
\longrightarrow A_{it}\longrightarrow Q_{it}.
$$

其中$\widehat c_{itk}$是选择动作$k$的预测运营成本。模型不输出
$\widehat d_{i,t+1}$，而是直接选择预测成本最低的补货动作。

## 3. 文献依据与本项目的重新设计

损失函数不是来自某一篇论文的原样复制，而是把三条文献思路组合成适合本项目和
LightGBM接口的工程模型：

1. Elmachtoub与Grigas提出SPO与SPO+，核心是用决策后悔代替单纯预测误差；
2. Mandi等人指出决策导向学习可以看成对可行解的learning-to-rank问题，并研究了
   pointwise、pairwise和listwise损失；
3. SPO Trees说明树模型的分裂也可以由下游决策质量驱动，而不是只由预测误差驱动。

主要文献：

- [Smart “Predict, then Optimize”, Management Science](https://doi.org/10.1287/mnsc.2020.3922)
- [Decision-Focused Learning: Through the Lens of Learning to Rank, ICML 2022](https://arxiv.org/abs/2112.03609)
- [Decision Trees for Decision-Making under the Predict-then-Optimize Framework, ICML 2020](https://proceedings.mlr.press/v119/elmachtoub20a.html)
- [Risk Bounds and Calibration for a Smart Predict-then-Optimize Method, NeurIPS 2021](https://papers.neurips.cc/paper_files/paper/2021/hash/b943325cc7b7422d2871b345bf9b067f-Abstract.html)
- [Learning with Fenchel–Young Losses, JMLR 2020](https://www.jmlr.org/papers/v21/19-021.html)

没有查到一篇已发表论文与本项目的组合完全相同，即“LightGBM + 有限补货倍率动作 +
多期库存仿真成本 + 平滑SPO+ + 后悔加权排序损失”。因此应称为受上述文献启发的
项目实现，不能在未做系统文献检索的情况下声称是世界首创。

## 4. 状态、动作与库存流转

### 4.1 决策状态

周期$t$需求发生后，模型能够看到：

- 第$t$期及更早的需求历史；
- `x1-x20`截至$t$期的滞后值和变化量；
- 当期期初可用库存、期末库存和当期缺货；
- 最近12期需求均值$r_{it}$；
- 库存、需求与$r_{it}$的比例；
- SKU品类和周期索引。

未来需求和未来匿名特征不进入状态。

### 4.2 动作

动作集合为：

$$
\mathcal A=\{0,0.5,1.0,\ldots,5.0\}.
$$

选择$a_k$表示把下一期库存目标设为：

$$
B_{itk}=\left\lceil a_k r_{it}\right\rceil.
$$

实际订货量是：

$$
Q_{it}=\max\{0,B_{it,A_{it}}-I_{it}\}.
$$

订单在$t+1$期需求发生前到货。

### 4.3 三期滚动成本标签

对每个候选动作$k$，使用训练数据中已经实现的未来需求进行短期仿真：

$$
c_{itk}^{(H)}=
\sum_{\tau=1}^{H}
\left(hI_{i,t+\tau,k}+\lambda L_{i,t+\tau,k}\right).
$$

其中：

- $I_{i,t+\tau,k}$是期末剩余库存；
- $L_{i,t+\tau,k}$是丢失销售；
- $h=1$是单位持有成本；
- $\lambda$是单位缺货惩罚；
- 最终选择$H=3$、$\lambda=4$。

训练标签可以使用训练期后来实现的需求，但线上决策不能看到这些需求。这和监督学习
用后来实现的标签训练、预测时只输入当时可用特征的逻辑相同。

## 5. 新的三部分损失函数

真实动作成本为$\boldsymbol c$，LightGBM预测成本为
$\widehat{\boldsymbol c}$。最终损失为：

$$
\mathcal L
=\omega_{\mathrm{MSE}}\mathcal L_{\mathrm{MSE}}
+\omega_{\mathrm{SPO}}\mathcal L_{\mathrm{SSPO+}}
+\omega_{\mathrm{rank}}\mathcal L_{\mathrm{rank}}.
$$

最终权重为：

$$
(\omega_{\mathrm{MSE}},\omega_{\mathrm{SPO}},\omega_{\mathrm{rank}})
=(0.25,0.40,0.35).
$$

### 5.1 成本回归部分

$$
\mathcal L_{\mathrm{MSE}}
=\frac{1}{2K}\sum_{k=1}^{K}
(\widehat c_k-c_k)^2.
$$

它给模型提供稳定的数值尺度，避免模型只知道动作排序、不知道动作之间的成本差距。

### 5.2 平滑SPO+部分

令$k^*=\arg\min_k c_k$，则平滑SPO+为：

$$
\mathcal L_{\mathrm{SSPO+}}
=\tau_s\log\sum_{k=1}^{K}
\exp\left(\frac{c_k-2\widehat c_k}{\tau_s}\right)
+2\widehat c_{k^*}-c_{k^*}.
$$

本项目$\tau_s=0.5$。这一项重点惩罚会导致错误补货动作的成本预测。

### 5.3 后悔加权的成对排序部分

每个错误动作相对真实最优动作的后悔为：

$$
\Delta_k=c_k-c_{k^*}\ge 0.
$$

归一化权重为：

$$
q_k=\frac{\Delta_k}{\sum_j\Delta_j}.
$$

排序损失为：

$$
\mathcal L_{\mathrm{rank}}
=\sum_{k\ne k^*}q_k
\log\left[
1+\exp\left(
\frac{\widehat c_{k^*}-\widehat c_k}{\tau_r}
\right)
\right].
$$

本项目$\tau_r=1$。如果真实最优动作的预测成本低于错误动作，即
$\widehat c_{k^*}<\widehat c_k$，损失就较小。错误动作的真实后悔越大，$q_k$越大，
排序错误受到的惩罚也越大。

这比普通多分类交叉熵多利用了一层信息：把第二名和最差动作分错，后果不同。

### 5.4 梯度与Hessian

令：

$$
p_k=\sigma\left(
\frac{\widehat c_{k^*}-\widehat c_k}{\tau_r}
\right),
$$

其中$\sigma$是Sigmoid函数。对错误动作$k$：

$$
\frac{\partial\mathcal L_{\mathrm{rank}}}
{\partial\widehat c_k}
=-\frac{q_kp_k}{\tau_r},
$$

$$
\frac{\partial^2\mathcal L_{\mathrm{rank}}}
{\partial\widehat c_k^2}
=\frac{q_kp_k(1-p_k)}{\tau_r^2}.
$$

真实最优动作的梯度是所有配对梯度之和。完整Hessian包含动作之间的交叉项，但
LightGBM自定义目标接口只接收逐行二阶项，因此代码使用正的对角近似。这是一个工程
近似，不应描述为完整Newton Hessian。

梯度已经通过有限差分自动测试验证。

## 6. LightGBM如何训练多个动作

LightGBM标准回归器每行只输出一个值。程序把一个状态样本展开为11行：

```text
同一个状态 + 动作0.0 + 对应目标库存
同一个状态 + 动作0.5 + 对应目标库存
...
同一个状态 + 动作5.0 + 对应目标库存
```

自定义目标函数再把连续11行还原成一个动作组，联合计算MSE、SPO+和排序梯度。预测
时同样展开11行，再选择预测成本最低的动作：

$$
A_{it}=\arg\min_k\widehat c_{itk}.
$$

## 7. 损失函数消融

为避免在不可行服务区间比较损失，消融统一使用$H=3$、$\lambda=4$：

| 损失 | 验证 Fill Rate | 验证平均库存 | 结论 |
|---|---:|---:|---|
| 成本MSE | 94.8566% | 1,178,303 | 库存低，但不满足95% |
| 平滑SPO+ | 98.2779% | 1,883,562 | 合格但偏保守 |
| 后悔加权排序 | 97.7502% | 1,738,271 | 合格 |
| **MSE + SPO+ + 排序** | **97.8220%** | **1,555,817** | 合格方案中库存最低 |

混合损失同时利用成本数值、决策后悔和动作次序，在这组验证实验中取得最好平衡。

## 8. 参数选择与正式结果

运营调参比较：

$$
H\in\{1,3,6\},
$$

$$
\lambda\in\{1.5,1.75,2,2.5,3,4\}.
$$

验证选择要求Fill Rate达到97%，为最终95%业务约束保留安全裕度。最终选择：

$$
H=3,\qquad \lambda=4.
$$

该候选验证Fill Rate为97.8220%，验证平均库存为1,555,817。最终测试结果：

- 总体Fill Rate：95.9206%；
- 平均总库存：2,198,542；
- 总周转率：39.4693；
- 总损失销售：3,690,488；
- 单SKU Fill Rate达到95%的比例：77.0167%。

需要强调：当前`replenishment_test.xlsx`已经被项目多轮实验查看，应视为开发基准，
而不是论文意义上的全新盲测集。正式论文或上线验收需要新的未来时间窗口。

## 9. 为什么没有超过普通LightGBM

直接策略省去了需求预测，但学习难度更高：

1. 一个需求数值标签被转换成11维动作成本排序，标签结构更复杂；
2. 训练状态来自行为策略，和模型上线后的状态分布不完全一致；
3. 候选动作只有0.5倍的粒度；
4. 三期滚动标签假设候选倍率在短期内保持不变；
5. LightGBM使用的是对角Hessian近似；
6. 当前只调了损失形式、滚动期数和缺货惩罚，没有全面搜索树容量和动作网格。

普通LightGBM先预测需求，再用每个SKU的误差尺度增加安全库存，分工更简单，因此目前
仍然更有效。

## 10. 运行命令

```bash
cd "<PROJECT_ROOT>"

PYTHONPYCACHEPREFIX=/private/tmp/replenishment_lgbm_direct_pycache \
MPLCONFIGDIR=/private/tmp/matplotlib \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
python \
  -u -m experiments.run_direct_lgbm_spo --n-jobs 8 --item-id 1
```

## 11. 输出文件

结果位于`outputs/spo/direct_lgbm_spo_rank/`：

- `summary.json`：配置、验证选择和测试总体结果；
- `validation_candidates.csv`：4种损失消融和18组运营参数；
- `test_per_sku_metrics.csv`：每个SKU的Fill Rate、库存和周转率；
- `test_trajectory.csv`：逐SKU逐期库存流转；
- `test_order_up_to.csv`：每期目标库存；
- `test_selected_actions.csv`：模型直接选择的动作；
- `direct_lgbm_spo_model.txt`：LightGBM模型；
- `feature_importance.csv`：状态、动作和目标库存特征重要性；
- `model_metadata.json`：动作集合、特征名和最终参数。


## 6. 限制

不跨A/B/C组混排；总量周转和加权周转分开；软服务罚项不等于95%硬约束。多轮开发使用过同一个测试集，不是独立确认性结果。有限搜索、首阶近似、容量不足等限制按原报告保留。此次代码重构以语法、导入、小型逻辑与数值一致性校验为主，真实全量重训不是此次任务。
