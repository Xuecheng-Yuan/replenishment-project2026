# 逐SKU收缩Stacking（sku_stacking）

方向：融合。

## 1. 方法与数据

每个SKU学习自己的14维凸权重，同时向全局权重收缩，缓解24个元期的过拟合；不保证测试胜过最佳单模型。

前5种统计加9种ML共14个需求专家。二层输入是按训练尺度归一化的OOF预测，需求是真实训练目标。SKU路由另用训练需求/x分组与训练库存代价，不含SPO/DFL订单。

## 2. 数学核心

$$
\min_{w_i\in\Delta^{13}}\frac1{2T}\|P_iw_i-y_i\|_2^2+\frac{\tau}{2T}\|w_i-w_0\|_2^2.
$$

## 3. 实现与参数

`replenishment/training/ensemble.py：sku_weights、projection、fit_meta`。

τ∈{6,24,96}，FISTA最多800步并检查投影梯度；按验证选择。

## 4. 运行

在包根目录先prepare，随后：

```bash
python run.py --workdir runs/main train --models sku_stacking
python run.py --workdir runs/main evaluate --models sku_stacking
```

## 5. 当前发布版本与真实测试结果

版本：`service_retrained_20261013`。

库存需求比 **0.449704**；服务水平 **96.6262%**；Fill Rate **98.7259%**。状态：**高于96%**。

时间顺序OOF训练融合权重，训练内部验证校准，最终专家用60期拟合。

以上为既有正式实验轨迹复算，不是新代码全量重训。细化版没有保存的样本内训练指标不从旧粗网格版本移用。

## 6. 补货决策逐步说明

1. 用时间顺序OOF预测学习融合/选择规则，不按测试误差挑专家。
2. 将14个历史统计与ML专家的下一期预测按已冻结权重组合，硬路由只选一个专家。
3. 使用验证集校准的风险尺度和系数形成目标库存。
4. 扣掉当前库存、非负取整，形成下一期订单；不是把专家订单直接相加。

## 7. 对应代码、参数和回放

实现：`replenishment/training/ensemble.py`。

冻结参数：[`configs/models/sku_stacking.json`](../../configs/models/sku_stacking.json)。

```bash
python run.py --workdir runs/main train --models sku_stacking
python run.py --workdir runs/main evaluate --models sku_stacking
```

查看已保存订单，不需要重新训练：

```bash
python run.py replay --model sku_stacking --sku 1 --through 28 --output runs/replay/sku_stacking_sku1.csv
```

训练与评价不读取历史指标作预测。测试不达标与失败候选如实保留；当前实现为有限计算预算，不声称理论全局最优。
