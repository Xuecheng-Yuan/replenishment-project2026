# LightGBM-SPO（lightgbm_spo）

方向：SPO/有限动作决策学习。

## 1. 方法与数据

基础LightGBM给需求中心；动作展开后增加动作编号与库存目标，用LightGBM学习成本。混合损失与XGBoost-SPO同类，树构建方式不同。

59维历史特征，附加当前期末库存、需求中心、残差尺度。SPO共62列；直接决策增加更多状态共67列。LightGBM动作展开再加动作和目标两列，共64/69列。训练状态来自MA6行为策略回放；测试使用本模型真实运行库存，存在状态分布偏移风险。

## 2. 数学核心

$$
L=(1-\alpha)\frac{\|\widehat c-c\|_2^2}{2K}+\alpha\ell_{\mathrm{smooth\text{-}SPO+}}(\widehat c,c).
$$

训练动作代价为 `期末库存/训练总体平均需求 + λ×1{期末库存≤0}`。直接决策取4期均值，后续固定行为目标使用原点信息。此处不是旧版Fill Rate短缺量代价；精确服务与J仍由完整仿真计算。

## 3. 实现与参数

`replenishment/models/lightgbm_spo.py：LGBMSPOHybrid；models/xgboost_spo.py：hybrid_grad_hess`。

α=.8，τ=.5，80棵树，31叶。动作扩展后为64列。

## 4. 运行

在包根目录先prepare，随后：

```bash
python run.py --workdir runs/main train --models lightgbm_spo
python run.py --workdir runs/main evaluate --models lightgbm_spo
```

## 5. 当前发布版本与真实测试结果

版本：`spo_refinement_20261015/B_refine_only`。

库存需求比 **0.777468**；服务水平 **99.1911%**；Fill Rate **98.7266%**。状态：**高于96%**。

前42期拟合；第13—60期连续库存预热，第43—60期验证评分；最终60期重拟合。

以上为既有正式实验轨迹复算，不是新代码全量重训。细化版没有保存的样本内训练指标不从旧粗网格版本移用。

## 6. 补货决策逐步说明

1. 截至本期期末，使用历史需求、滞后x和当前真实库存构造状态。
2. 线性/直接模型的需求参考来自MA6；树SPO的需求中心来自前缀拟合的基础ML。
3. 模型预测各有限动作的成本，选择一个动作；它不是直接预测下一期真实需求。
4. SPO构造中心＋动作×误差尺度，直接决策构造中心×动作；再加验证集已选风险偏移。
5. 目标库存扣掉当前期末库存，非负部分向上取整下单，下一期到货。

## 7. 对应代码、参数和回放

实现：`replenishment/training/decision.py + replenishment/training/action_grid.py`。

冻结参数：[`configs/models/lightgbm_spo.json`](../../configs/models/lightgbm_spo.json)。

```bash
python run.py --workdir runs/main train --models lightgbm_spo
python run.py --workdir runs/main evaluate --models lightgbm_spo
```

查看已保存订单，不需要重新训练：

```bash
python run.py replay --model lightgbm_spo --sku 1 --through 28 --output runs/replay/lightgbm_spo_sku1.csv
```

训练与评价不读取历史指标作预测。测试不达标与失败候选如实保留；当前实现为有限计算预算，不声称理论全局最优。

### 当前实际执行的动作

`A = [0.0, 0.3125, 0.625, 0.9375, 1.25, 1.5625, 1.875, 2.1875, 4.0]`，`z = 1.3333333333333333`，`λ = 10.0`。

新训练入口比较粗网格与训练成本分布细化网格，仅在验证选择，不再硬编码所有模型执行0、0.5、…、4。执行对齐C/D为历史诊断，详见results/research，不冒充本入口又完整重跑了四版本实验。
