# 直接决策XGBoost（direct_xgboost）

方向：SPO/有限动作决策学习。

## 1. 方法与数据

用MA6作动作参考，不要求先准确预测需求；XGBoost直接对动作成本排序。仍需历史参考和尺度，不是完全没有需求信息。代价用4期固定行为回放构造。

59维历史特征，附加当前期末库存、需求中心、残差尺度。SPO共62列；直接决策增加更多状态共67列。LightGBM动作展开再加动作和目标两列，共64/69列。训练状态来自MA6行为策略回放；测试使用本模型真实运行库存，存在状态分布偏移风险。

## 2. 数学核心

$$
S_a=[a\widehat d]^+,\quad \widehat a=\arg\min_a\widehat c_a(h_t,B_t),\quad q_t=[S_{\widehat a}+z\sigma-B_t]^+.
$$

训练动作代价为 `期末库存/训练总体平均需求 + λ×1{期末库存≤0}`。直接决策取4期均值，后续固定行为目标使用原点信息。此处不是旧版Fill Rate短缺量代价；精确服务与J仍由完整仿真计算。

## 3. 实现与参数

`replenishment/training/decision.py：targets、build_training、estimator；models/xgboost_spo.py`。

复用XGBSPOHybrid，α=1，即纯平滑SPO+；70棵深度4树；z扩展到0—64验证。

## 4. 运行

在包根目录先prepare，随后：

```bash
python run.py --workdir runs/main train --models direct_xgboost
python run.py --workdir runs/main evaluate --models direct_xgboost
```

## 5. 当前发布版本与真实测试结果

版本：`spo_refinement_20261015/B_refine_only`。

库存需求比 **1.783877**；服务水平 **98.3714%**；Fill Rate **97.2632%**。状态：**高于96%**。

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

冻结参数：[`configs/models/direct_xgboost.json`](../../configs/models/direct_xgboost.json)。

```bash
python run.py --workdir runs/main train --models direct_xgboost
python run.py --workdir runs/main evaluate --models direct_xgboost
```

查看已保存订单，不需要重新训练：

```bash
python run.py replay --model direct_xgboost --sku 1 --through 28 --output runs/replay/direct_xgboost_sku1.csv
```

训练与评价不读取历史指标作预测。测试不达标与失败候选如实保留；当前实现为有限计算预算，不声称理论全局最优。

### 当前实际执行的动作

`A = [0.0, 1.3125, 1.625, 1.9375, 2.25, 2.5625, 2.875, 3.1875, 4.0]`，`z = 2.3703703703703702`，`λ = 10.0`。

新训练入口比较粗网格与训练成本分布细化网格，仅在验证选择，不再硬编码所有模型执行0、0.5、…、4。执行对齐C/D为历史诊断，详见results/research，不冒充本入口又完整重跑了四版本实验。
