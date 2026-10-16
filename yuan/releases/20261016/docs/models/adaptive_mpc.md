# 自适应情景MPC（adaptive_mpc）

方向：滚动优化。

## 1. 方法与数据

仅用拟合前缀分3组，每组训练4个步长的LightGBM。当天需求揭示后修正个体/组偏差和尺度，再生成4期情景，优化4个订单但只执行第一个。

59维历史特征；聚类用训练需求统计和x均值PCA；实时可见需求误差、当前库存。未来匿名x不可用，四步预测使用同一个原点的历史信息。

## 2. 数学核心

$$
\min_{q_{1:4}\ge0}\frac{\sum_{k,h,i}B_{khi}}{\sum_{k,h,i}D_{khi}}+\rho\Psi(\widetilde R),\qquad \text{只执行 }q_1.
$$

服务代理用 `sigmoid((可用库存−需求−1)/(0.05×SKU尺度))`；双侧罚项为 `Ψ(R̃)=[.95−R̃]²₊+[R̃−.96]²₊`。平滑代理不等于精确二值服务，最终结果不使用代理冒充R。

## 3. 实现与参数

`replenishment/training/mpc.py：fit_predict、cluster、mpc_policy`。

3组，4期，8情景，12步投影；(ρ,α)∈{(80,.1),(240,.2)}；无商业求解器。

## 4. 运行

在包根目录先prepare，随后：

```bash
python run.py --workdir runs/main train --models adaptive_mpc
python run.py --workdir runs/main evaluate --models adaptive_mpc
```

## 5. 当前发布版本与真实测试结果

版本：`service_retrained_20261013`。

库存需求比 **0.262907**；服务水平 **92.1845%**；Fill Rate **90.4545%**。状态：**低于95%**。

前42期拟合；第13—60期连续库存预热，第43—60期验证评分；最终60期重拟合。

以上为既有正式实验轨迹复算，不是新代码全量重训。细化版没有保存的样本内训练指标不从旧粗网格版本移用。

## 6. 补货决策逐步说明

1. 从历史特征滚动预测四期需求与不确定性。
2. 结合训练期PCA/Kmeans分组、当前库存和情景优化四期订单。
3. 只执行第一期，观察新需求后重新规划；验证集选风险参数，测试不调参。

## 7. 对应代码、参数和回放

实现：`replenishment/training/mpc.py`。

冻结参数：[`configs/models/adaptive_mpc.json`](../../configs/models/adaptive_mpc.json)。

```bash
python run.py --workdir runs/main train --models adaptive_mpc
python run.py --workdir runs/main evaluate --models adaptive_mpc
```

查看已保存订单，不需要重新训练：

```bash
python run.py replay --model adaptive_mpc --sku 1 --through 28 --output runs/replay/adaptive_mpc_sku1.csv
```

训练与评价不读取历史指标作预测。测试不达标与失败候选如实保留；当前实现为有限计算预算，不声称理论全局最优。
