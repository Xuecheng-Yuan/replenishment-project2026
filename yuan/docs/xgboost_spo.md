# xgboost_spo：模型实现与说明

[总览](../README.md) · [实际代码](../models/xgboost_spo.py) · [全部变体记录](../results/variants.json)

## 1. 文件如何对应

本模型所有定义、关键决策/损失函数或估计器构造在`models/xgboost_spo.py`，不是转发到旧experiments目录的脚本。数据/评价/公共训练流程位于common/。参数、窗口、种子和消融不另建模型文件。历史125项仅为结果记录，完整索引见results/variants.json。

## 2. 信息时序与模型原理

### A组

时间外基础需求预测产生9个候选库存目标，第二层树学习动作成本。XGBoost多输出，LightGBM展开样本—动作行；联合成本MSE和平滑SPO+，现存选中alpha=1。加入期末库存；自定义Hessian使用对角近似，不是完整联合Hessian。

SKU是商品编号；x1–x20是外部属性/特征，不是需求本身。历史滞后表示预测目标日前已观察值。库存状态是今天卖完后的库存，不是明天已实现的库存。预测层误差小不自动意味着决策服务达标。

### 核心公式

$$
\mathcal L=(1-\alpha)\mathcal L_{\mathrm{MSE}}+\alpha\mathcal L_{\mathrm{smooth\ SPO+}}
$$

d为需求，I为期末库存，s为实际销量，q为下单量；预测目标用S表示，C为公共容量，z为风险余量系数；a_i为训练期商品尺度，theta为学习参数。上式是核心计算关系/目标概要，不把代理损失写成业务硬约束。完整推导和实现细节见后面的完整推导；共享符号、Fill Rate与Ready Rate区别见[评价标准](evaluation.md)。


## 3. 运行命令

从本仓库根目录运行；默认只查看历史结果，不训练：

```bash
python -m models.xgboost_spo
python -m models.xgboost_spo --help
python -m models.xgboost_spo --train
```

普通ML与baseline可用`--protocol legacy`、`--protocol ready_immediate`、`--protocol lead_one`切换已存在实验口径。lead_one为旧冻结预测的安全库存移除/固定z4对照，不是重新在该口径调参。移动平均用`--window 3/6/12`；MPC用`--round all/1/2/3/4`。高级模型默认沿用原批次的验证/消融，但实现全在同一文件，不会调用旧脚本树。

原始Excel放data/。运行只写results/generated/，不覆盖results/variants.json中的历史快照。改变输入请新目录重跑，不允许旧缓存自动作为新数据证据。

## 4. 历史结果（所有配置集中列在这里）

|原标签|组|角色|Fill Rate|Ready Rate达标比例|平均总库存|原始周转|加权周转|
|---|---|---|---:|---:|---:|---:|---:|
|legacy_xgb_spo_hybrid|A|主方案|0.973647|未记录|825619|106.685|未记录|

小数比例0.95就是95%。参数与来源SHA256逐项保存在results/variants.json。历史业务失败没有改成通过；没有因为此次重构而重训全部真实数据。

## 5. 完整数学与训练细节

以下为本模型相关历史说明的完整正文，记录旧研究假设；当时“最优”只限于自己的实验口径。运行请用上面的新命令，以下旧文件路径仅作为出处，不表示新目录仍有旧结构。

### 原始说明：xgb_spo_hybrid_report.md

### XGBoost–SPO+ 决策导向补货模型

## 1. 实验结论

本实验把XGBoost的非线性树模型与平滑SPO+损失结合，用XGBoost直接预测候选补货动作的成本。最终测试结果为：

| 模型 | 总体Fill Rate | 平均总库存 | 总周转率 | SKU达到95%的比例 |
|---|---:|---:|---:|---:|
| XGB–SPO（含当期库存） | **97.3647%** | **825,619** | **106.6853** | **91.7667%** |

它满足95%的总体Fill Rate约束，在当前22个模型中按“合格后库存最低”排名第二，仅次于LightGBM。

与原来的需求预测XGBoost相比，新模型：

- 平均库存减少1.50%；
- Fill Rate下降0.749个百分点，但仍达到97.07%；
- 总周转率由105.57提高到120.65。

与加入库存后的线性SPO+相比，新模型平均库存减少77.59%，同时Fill Rate提高2.341个百分点。这说明前一版的主要问题不是SPO+思想本身，而是线性成本预测器和MA12动作中心的表达能力不足。

## 2. 必须如实说明的结果

实验确实训练了“成本MSE与平滑SPO+的加权损失”：

$$
\mathcal L_{\mathrm{hybrid}}
=(1-\alpha)\mathcal L_{\mathrm{MSE}}
+\alpha\mathcal L_{\mathrm{SmoothSPO+}}.
$$

搜索了

$$
\alpha\in\{0,0.5,0.75,0.9,1\}.
$$

但是最终最优值是

$$
\boxed{\alpha=1}.
$$

因此最终胜出的模型严格来说是“XGBoost树结构 + 纯平滑SPO+损失”，而不是MSE和SPO+都占正权重的混合损失。不能为了让模型名字好听而隐瞒这个消融结果。

正确结论是：

> XGBoost的非线性函数逼近能力与SPO+决策损失结合后效果很好；但在当前数据和动作成本定义下，额外加入成本数值MSE会使库存策略变得更保守，并未带来增益。

## 3. 信息时序与无泄漏设计

预测第 $t$ 期时，模型只能使用第 $t-1$ 期及更早的信息。基础需求预测采用扩展窗口XGBoost：

1. 用目标期13–24训练，预测25–30；
2. 用目标期13–30训练，预测31–36；
3. 继续每6期扩展一次，直到预测55–60。

这些训练期外预测用于构造成本模型训练样本。真实需求没有被同一个基础XGBoost同时用于拟合和生成自身预测值。

最终测试基础预测器只使用完整训练集拟合，然后对测试特征进行滚动一步预测。测试需求不会用于模型参数选择。

## 4. 候选补货动作

对样本 $j$，基础XGBoost给出需求中心预测 $m_j$，历史训练期外预测误差给出SKU尺度 $\sigma_i$。设置9个动作：

$$
z_k\in\{0,0.5,1.0,\ldots,4.0\}.
$$

动作 $k$ 对应：

$$
B_{jk}=\left\lceil\max(0,m_j+z_k\sigma_i)\right\rceil.
$$

训练期已知下一期真实需求 $D_j$，所以每个动作的真实成本为：

$$
c_{jk}
=h(B_{jk}-D_j)^+
+p(D_j-B_{jk})^+.
$$

本次最终选择 $h=1,p=20$。

## 5. 平滑SPO+损失

模型一次输出9个预测动作成本：

$$
\hat{\boldsymbol c}_j=F_{\Theta}(x_j),
$$

其中 $F_{\Theta}$ 是多输出XGBoost。

真实最优动作是

$$
k_j^*=\arg\min_k c_{jk}.
$$

有限动作下的平滑SPO+写为：

$$
\mathcal L_{\mathrm{SmoothSPO+}}
=\tau\log\sum_k
\exp\left(\frac{c_{jk}-2\hat c_{jk}}{\tau}\right)
+2\hat c_{jk_j^*}
-c_{jk_j^*}.
$$

本次温度 $\tau=0.5$。温度越小越接近原始非光滑SPO+，但训练也会更容易跳变。

令

$$
\pi_{jk}
=\frac{\exp((c_{jk}-2\hat c_{jk})/\tau)}
{\sum_l\exp((c_{jl}-2\hat c_{jl})/\tau)}.
$$

混合损失梯度为：

$$
g_{jk}
=\frac{1-\alpha}{K}(\hat c_{jk}-c_{jk})
+2\alpha[\mathbf 1(k=k_j^*)-\pi_{jk}].
$$

提供给XGBoost的对角Hessian近似为：

$$
h_{jk}
=\frac{1-\alpha}{K}
+\frac{4\alpha}{\tau}\pi_{jk}(1-\pi_{jk})
+\varepsilon.
$$

程序通过有限差分自动测试核对了梯度，并检查所有Hessian为正。

## 6. 消融实验

三类损失都进行了比较：

| 模型类型 | $\alpha$ | 含义 |
|---|---:|---|
| XGB-Cost-MSE | 0 | 只要求准确预测动作成本数值 |
| XGB-SPO-Hybrid | 0.5、0.75、0.9 | 同时考虑成本数值和动作决策 |
| XGB-Smooth-SPO+ | 1 | 只关注能否选到合适动作 |

每个权重与缺货惩罚 `20, 40, 80, 160` 组合，共20个验证候选。

各类最有代表性的验证结果如下：

| 类型 | $\alpha$ | 缺货惩罚 | 验证Fill Rate | 平均总库存 |
|---|---:|---:|---:|---:|
| 纯成本MSE | 0 | 20 | 98.7286% | 1,423,616 |
| 50%混合 | 0.5 | 20 | 98.5310% | 1,668,293 |
| 75% SPO+ | 0.75 | 20 | 98.6163% | 1,870,631 |
| 90% SPO+ | 0.9 | 20 | 98.3978% | 1,500,932 |
| **纯平滑SPO+** | **1** | **20** | **96.2634%** | **662,953** |

加入MSE后，模型更努力拟合所有动作成本的绝对数值，但项目最终只需要正确排序动作。成本MSE把模型推向了更高服务率、更高库存的保守区域。纯SPO+忽略对决策无关的数值误差，因此验证库存最低。

## 7. 与主要模型比较

| 排名 | 模型 | 测试Fill Rate | 平均总库存 | 总周转率 | 约束 |
|---:|---|---:|---:|---:|:---:|
| 1 | LightGBM | 96.9577% | 585,283 | 149.8646 | 是 |
| 2 | **XGB–SPO** | **97.3647%** | **825,619** | **106.6853** | 是 |
| 3 | 原XGBoost | 97.8206% | 838,215 | 105.5741 | 是 |
| 4 | CatBoost | 97.5982% | 995,719 | 88.6722 | 是 |
| 5 | Moving Average 12 | 95.8390% | 2,937,675 | 29.5135 | 是 |
| 11 | 线性SPO+ | 95.0235% | 3,684,607 | 23.3304 | 是 |
| 10 | Baseline | 98.7838% | 11,949,377 | 7.4787 | 是 |

XGB–SPO比LightGBM多使用41.06%的库存，但Fill Rate高0.407个百分点；相对普通XGBoost少用1.50%的库存。因此它是当前很有竞争力的第二方案，但按现有严格目标仍不能取代LightGBM。

## 8. 工程文件

| 文件 | 内容 |
|---|---|
| `src/models/spo/xgb_spo_hybrid.py` | 平滑混合损失、梯度、Hessian和多输出XGBoost模型 |
| `experiments/run_xgb_spo_hybrid.py` | 扩展窗口预测、20组消融调参、最终重训和测试 |
| `tests/test_spo_plus.py` | 平滑损失有限差分梯度测试 |
| `outputs/spo/xgb_spo_hybrid/validation_candidates.csv` | 20个候选验证结果 |
| `outputs/spo/xgb_spo_hybrid/base_demand_xgboost.json` | 最终基础需求XGBoost |
| `outputs/spo/xgb_spo_hybrid/cost_xgb_spo_model.json` | 最终多输出成本模型 |
| `outputs/spo/xgb_spo_hybrid/model_metadata.json` | 动作、成本尺度、特征和参数 |
| `outputs/spo/xgb_spo_hybrid/test_per_sku_metrics.csv` | 6,000个SKU的测试指标 |
| `outputs/spo/xgb_spo_hybrid/test_trajectory.csv` | 168,000行动态库存轨迹 |
| `outputs/spo/xgb_spo_hybrid/summary.json` | 正式结果摘要 |

## 9. 完整复现命令

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

本次运行约271秒。`--quick`只用于检查代码链路，不能作为正式结果。

## 10. 仍然存在的限制

1. 动作成本仍是单期newsvendor代理，没有在训练损失内展开完整28期库存状态；
2. 当前只验证了一个平滑温度 `0.5`；
3. 只使用一个训练后段验证窗口，尚未做多窗口rolling-origin验证；
4. 测试集已经被多轮模型开发反复查看，正式研究应保留新的最终留出集；
5. 最优权重落在边界 $\alpha=1$，所以当前证据支持“XGBoost函数类 + SPO+”，不支持“加入MSE一定更好”。

这个模型最重要的研究发现不是“复杂损失必然更强”，而是：对于当前补货问题，动作成本的正确排序比动作成本数值的精确拟合更有价值。


## 6. 限制

不跨A/B/C组混排；总量周转和加权周转分开；软服务罚项不等于95%硬约束。多轮开发使用过同一个测试集，不是独立确认性结果。有限搜索、首阶近似、容量不足等限制按原报告保留。此次代码重构以语法、导入、小型逻辑与数值一致性校验为主，真实全量重训不是此次任务。
