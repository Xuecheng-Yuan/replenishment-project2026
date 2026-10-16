# XGBoost（xgboost）

方向：机器学习。

## 1. 方法与数据

直方图梯度提升树，拟合对数相对需求的平方误差。不是SPO方案。

共享的59维严格滞后特征。目标先按SKU归一化：`y=log(1+d)−log(1+μ_i)`，μ_i仅用拟合前缀；预测后反变换并截断为非负。当前库存不输入需求回归器，而由补货策略使用。

## 2. 数学核心

$$
\widehat y(x)=\sum_m\eta T_m(x),\quad L=\sum_j(y_j-\widehat y_j)^2+\sum_m\Omega(T_m).
$$

需求预测本身不等于补货决策。最终统一按下式订货，尺度可为固定残差标准差或相对尺度；baseline只用固定尺度。

$$
q_{it}=\left[\widehat d_{i,t+1}+z\sigma_{i,t+1}-B_{it}\right]^+.
$$

## 3. 实现与参数

`replenishment/models/xgboost.py：build；training/forecast.py：ml_fit、ml_bundle`。

具体候选见training/forecast.py的GRID；有限网格按库存验证选优，不按需求训练误差挑选。

## 4. 运行

在包根目录先prepare，随后：

```bash
python run.py --workdir runs/main train --models xgboost
python run.py --workdir runs/main evaluate --models xgboost
```

## 5. 当前发布版本与真实测试结果

版本：`service_retrained_20261013`。

库存需求比 **0.328633**；服务水平 **90.1423%**；Fill Rate **98.3773%**。状态：**低于95%**。

前42期拟合；第13—60期连续库存预热，第43—60期验证评分；最终60期重拟合。

以上为既有正式实验轨迹复算，不是新代码全量重训。细化版没有保存的样本内训练指标不从旧粗网格版本移用。

## 6. 补货决策逐步说明

1. 只使用截至本期已观测的需求；ML还使用滞后一期及更早的x。
2. 历史统计或ML预测下一期需求；普通ML训练不直接优化服务约束。
3. 加上验证集校准的风险缓冲，得到下一期目标库存。
4. 扣除当前期末库存，非负部分向上取整下单；提前期一期。

## 7. 对应代码、参数和回放

实现：`replenishment/training/forecast.py`。

冻结参数：[`configs/models/xgboost.json`](../../configs/models/xgboost.json)。

```bash
python run.py --workdir runs/main train --models xgboost
python run.py --workdir runs/main evaluate --models xgboost
```

查看已保存订单，不需要重新训练：

```bash
python run.py replay --model xgboost --sku 1 --through 28 --output runs/replay/xgboost_sku1.csv
```

训练与评价不读取历史指标作预测。测试不达标与失败候选如实保留；当前实现为有限计算预算，不声称理论全局最优。
