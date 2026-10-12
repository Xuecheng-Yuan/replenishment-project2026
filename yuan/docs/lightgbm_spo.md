# lightgbm_spo：模型实现与说明

[总览](../README.md) · [实际代码](../models/lightgbm_spo.py) · [全部变体记录](../results/variants.json)

## 1. 文件如何对应

本模型所有定义、关键决策/损失函数或估计器构造在`models/lightgbm_spo.py`，不是转发到旧experiments目录的脚本。数据/评价/公共训练流程位于common/。参数、窗口、种子和消融不另建模型文件。历史125项仅为结果记录，完整索引见results/variants.json。

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
python -m models.lightgbm_spo
python -m models.lightgbm_spo --help
python -m models.lightgbm_spo --train
```

普通ML与baseline可用`--protocol legacy`、`--protocol ready_immediate`、`--protocol lead_one`切换已存在实验口径。lead_one为旧冻结预测的安全库存移除/固定z4对照，不是重新在该口径调参。移动平均用`--window 3/6/12`；MPC用`--round all/1/2/3/4`。高级模型默认沿用原批次的验证/消融，但实现全在同一文件，不会调用旧脚本树。

原始Excel放data/。运行只写results/generated/，不覆盖results/variants.json中的历史快照。改变输入请新目录重跑，不允许旧缓存自动作为新数据证据。

## 4. 历史结果（所有配置集中列在这里）

|原标签|组|角色|Fill Rate|Ready Rate达标比例|平均总库存|原始周转|加权周转|
|---|---|---|---:|---:|---:|---:|---:|
|legacy_lgbm_spo_hybrid|A|主方案|0.976059|未记录|947906|93.1523|未记录|

小数比例0.95就是95%。参数与来源SHA256逐项保存在results/variants.json。历史业务失败没有改成通过；没有因为此次重构而重训全部真实数据。

## 5. 完整数学与训练细节

以下为本模型相关历史说明的完整正文，记录旧研究假设；当时“最优”只限于自己的实验口径。运行请用上面的新命令，以下旧文件路径仅作为出处，不表示新目录仍有旧结构。

### 原始说明：lgbm_spo_hybrid_report.md

### LightGBM–SPO+ 决策导向补货模型

## 1. 实验结论

本实验把LightGBM的非线性树模型与平滑SPO+损失结合，让模型直接学习“哪个补货动作代价最小”。正式测试结果为：

| 模型 | 总体Fill Rate | 平均总库存 | 总周转率 | SKU达到95%的比例 |
|---|---:|---:|---:|---:|
| LightGBM–SPO+（含当期库存） | **97.6059%** | **947,906** | **93.1523** | **95.5167%** |

它满足95%的总体Fill Rate约束，在当前25个模型中按“合格后库存最低”排名第四。它比CatBoost少用约4.8%库存，也远好于线性SPO+；但没有超过普通LightGBM、XGBoost–SPO+和普通XGBoost。

## 2. 模型怎么组合

第一层是基础LightGBM需求预测器。预测第 $t$ 期时，只使用第 $t-1$ 期及更早的需求和匿名特征。为了避免训练目标泄漏，成本学习器使用扩展窗口产生的训练期外预测：

1. 用目标期13–24预测25–30；
2. 用目标期13–30预测31–36；
3. 每6期扩展一次，直到预测55–60。

对样本 $j$，基础预测为 $m_j$，SKU历史训练期外预测误差标准差为 $\sigma_i$。定义9个候选动作：

$$
z_k\in\{0,0.5,1.0,\ldots,4.0\},
\qquad
B_{jk}=\left\lceil\max(0,m_j+z_k\sigma_i)\right\rceil.
$$

已知真实需求 $D_j$ 时，动作成本为：

$$
c_{jk}=h(B_{jk}-D_j)^+ + p(D_j-B_{jk})^+.
$$

本次持有成本 $h=1$，验证集比较缺货惩罚 $p\in\{20,40,80\}$。

## 3. LightGBM如何使用SPO+

XGBoost版可以直接输出9维动作成本，而LightGBM的自定义目标更适合标量输出。因此本实现把每个样本展开成9行：

```text
原样本特征 + 动作z_k + 该动作的目标库存B_jk
```

自定义损失再把连续9行还原为一个动作组，联合计算平滑SPO+梯度和Hessian。损失为：

$$
\mathcal L=(1-\alpha)\mathcal L_{\mathrm{MSE}}
+\alpha\mathcal L_{\mathrm{SmoothSPO+}},
$$

$$
\mathcal L_{\mathrm{SmoothSPO+}}
=\tau\log\sum_k
\exp\left(\frac{c_{jk}-2\hat c_{jk}}{\tau}\right)
+2\hat c_{jk_j^*}-c_{jk_j^*},
\qquad
k_j^*=\arg\min_k c_{jk}.
$$

本次 $\tau=0.5$，消融比较 $\alpha\in\{0,0.5,0.9,1\}$。梯度与XGBoost–SPO+使用同一个已经有限差分检查的实现。

## 4. 12组消融实验

| 损失类型 | $\alpha$ | 缺货惩罚 | 验证Fill Rate | 验证平均总库存 |
|---|---:|---:|---:|---:|
| 纯成本MSE | 0 | 20 | 98.6656% | 1,208,066 |
| 50%混合 | 0.5 | 20 | 98.4416% | 1,170,570 |
| 90% SPO+ | 0.9 | 20 | 98.2373% | 1,268,916 |
| **纯平滑SPO+** | **1** | **20** | **97.1700%** | **755,636** |

所有12个候选都在验证集达到95%。根据“先满足服务率，再最小化库存”的预定规则，最终选中 $\alpha=1,p=20$。

这意味着最终胜出的严格说是“LightGBM函数类 + 纯平滑SPO+”，而不是MSE和SPO+同时占正权重的混合模型。当前证据表明：动作排序比准确拟合每个动作的成本数值更重要。

## 5. 与主要模型比较

| 排名 | 模型 | 测试Fill Rate | 平均总库存 | 总周转率 |
|---:|---|---:|---:|---:|
| 1 | LightGBM | 96.9577% | 585,283 | 149.8646 |
| 2 | XGBoost–SPO+ | 97.3647% | 825,619 | 106.6853 |
| 3 | XGBoost | 97.8206% | 838,215 | 105.5741 |
| 4 | **LightGBM–SPO+** | **97.6059%** | **947,906** | **93.1523** |
| 5 | CatBoost | 97.5982% | 995,719 | 88.6722 |

LightGBM–SPO+相对普通LightGBM多用56.56%库存，Fill Rate只提高0.202个百分点；相对XGBoost–SPO+多用25.89%库存。因此它不是当前推荐模型，但它证明了LightGBM也能与SPO+做成可复现的端到端实验。

一个关键诊断是：LightGBM–SPO+在测试集有95.28%的样本选择 $z=0.5$，动作分布过度集中；XGBoost–SPO+的动作更分散。这表明“样本—动作展开后的单输出LightGBM”可能没有多输出XGBoost那么容易表达动作之间的联动排序。这是根据动作分布作出的工程推断，不是已被理论证明的因果。

## 6. 工程文件

| 文件 | 内容 |
|---|---|
| `src/models/spo/lgbm_spo_hybrid.py` | 样本—动作展开、LightGBM自定义损失与预测 |
| `experiments/run_lgbm_spo_hybrid.py` | 无泄漏预测、12组消融、最终重训和测试 |
| `outputs/spo/lgbm_spo_hybrid/validation_candidates.csv` | 12个验证候选的完整结果 |
| `outputs/spo/lgbm_spo_hybrid/base_demand_lightgbm.txt` | 最终基础需求LightGBM |
| `outputs/spo/lgbm_spo_hybrid/cost_lgbm_spo_model.txt` | 最终动作成本模型 |
| `outputs/spo/lgbm_spo_hybrid/test_selected_actions.csv` | 6,000个SKU、28期的选中动作 |
| `outputs/spo/lgbm_spo_hybrid/test_per_sku_metrics.csv` | 6,000个SKU的测试指标 |
| `outputs/spo/lgbm_spo_hybrid/test_trajectory.csv` | 168,000行动态库存轨迹 |
| `outputs/spo/lgbm_spo_hybrid/summary.json` | 参数、验证和测试摘要 |

## 7. 完整复现命令

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

本次正式运行约154秒。`--quick`只用于检查代码链路，不应作为正式比较结果。

## 8. 局限

1. 训练成本是单期newsvendor代理，没有在损失内完整展开28期库存状态；
2. 目前只验证一个平滑温度和一个验证窗口；
3. LightGBM通过展开动作间接学习多动作排序，不等同于原生多输出树；
4. 最优权重在边界 $\alpha=1$，所以当前实验不支持“MSE与SPO+混合必然优于纯SPO+”；
5. 反复查看同一测试集会产生研究者过拟合，正式论文应保留新的最终留出集。


## 6. 限制

不跨A/B/C组混排；总量周转和加权周转分开；软服务罚项不等于95%硬约束。多轮开发使用过同一个测试集，不是独立确认性结果。有限搜索、首阶近似、容量不足等限制按原报告保留。此次代码重构以语法、导入、小型逻辑与数值一致性校验为主，真实全量重训不是此次任务。
