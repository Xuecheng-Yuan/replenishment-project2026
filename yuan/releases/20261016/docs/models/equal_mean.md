# 等权融合（equal_mean）

方向：融合。

## 1. 方法与数据

14个需求专家等权平均，再进入同一库存策略；不是平均14条已经独立运行的库存轨迹。

前5种统计加9种ML共14个需求专家。二层输入是按训练尺度归一化的OOF预测，需求是真实训练目标。SKU路由另用训练需求/x分组与训练库存代价，不含SPO/DFL订单。

## 2. 数学核心

$$
\widehat d_{it}=\frac1{14}\sum_{m=1}^{14}\widehat d^{(m)}_{it}.
$$

## 3. 实现与参数

`replenishment/training/ensemble.py：fit_meta的equal_mean候选`。

各专家固定第一参数候选；固定/相对尺度及z按验证选择。

## 4. 运行

在包根目录先prepare，随后：

```bash
python run.py --workdir runs/main train --models equal_mean
python run.py --workdir runs/main evaluate --models equal_mean
```

## 5. 当前发布版本与真实测试结果

版本：`service_retrained_20261013`。

库存需求比 **0.627861**；服务水平 **96.4167%**；Fill Rate **98.5458%**。状态：**高于96%**。

时间顺序OOF训练融合权重，训练内部验证校准，最终专家用60期拟合。

以上为既有正式实验轨迹复算，不是新代码全量重训。细化版没有保存的样本内训练指标不从旧粗网格版本移用。

## 6. 补货决策逐步说明

1. 用时间顺序OOF预测学习融合/选择规则，不按测试误差挑专家。
2. 将14个历史统计与ML专家的下一期预测按已冻结权重组合，硬路由只选一个专家。
3. 使用验证集校准的风险尺度和系数形成目标库存。
4. 扣掉当前库存、非负取整，形成下一期订单；不是把专家订单直接相加。

## 7. 对应代码、参数和回放

实现：`replenishment/training/ensemble.py`。

冻结参数：[`configs/models/equal_mean.json`](../../configs/models/equal_mean.json)。

```bash
python run.py --workdir runs/main train --models equal_mean
python run.py --workdir runs/main evaluate --models equal_mean
```

查看已保存订单，不需要重新训练：

```bash
python run.py replay --model equal_mean --sku 1 --through 28 --output runs/replay/equal_mean_sku1.csv
```

训练与评价不读取历史指标作预测。测试不达标与失败候选如实保留；当前实现为有限计算预算，不声称理论全局最优。
