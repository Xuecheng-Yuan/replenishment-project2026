# 逐SKU硬路由（sku_router）

方向：融合。

## 1. 方法与数据

每个SKU只选一个需求专家；根据OOF库存成本和无库存比例，向6组商品平均代价收缩。硬选择发生在训练/验证阶段，不是测试后逐SKU挑冠军。

前5种统计加9种ML共14个需求专家。二层输入是按训练尺度归一化的OOF预测，需求是真实训练目标。SKU路由另用训练需求/x分组与训练库存代价，不含SPO/DFL订单。

## 2. 数学核心

$$
m_i=\arg\min_m\left[\alpha C_{im}+(1-\alpha)\overline C_{g(i),m}\right],\quad \alpha=\frac{24}{24+\tau}.
$$

## 3. 实现与参数

`replenishment/training/ensemble.py：fit_meta的sku_router候选`。

C为训练库存代理成本+λ无库存比例；λ∈{.25,1,4,16}，τ∈{0,24}；不是精确总体约束解。

## 4. 运行

在包根目录先prepare，随后：

```bash
python run.py --workdir runs/main train --models sku_router
python run.py --workdir runs/main evaluate --models sku_router
```

## 5. 当前发布版本与真实测试结果

版本：`service_retrained_20261013`。

库存需求比 **1.404620**；服务水平 **98.9536%**；Fill Rate **97.6095%**。状态：**高于96%**。

时间顺序OOF训练融合权重，训练内部验证校准，最终专家用60期拟合。

以上为既有正式实验轨迹复算，不是新代码全量重训。细化版没有保存的样本内训练指标不从旧粗网格版本移用。

## 6. 补货决策逐步说明

1. 用时间顺序OOF预测学习融合/选择规则，不按测试误差挑专家。
2. 将14个历史统计与ML专家的下一期预测按已冻结权重组合，硬路由只选一个专家。
3. 使用验证集校准的风险尺度和系数形成目标库存。
4. 扣掉当前库存、非负取整，形成下一期订单；不是把专家订单直接相加。

## 7. 对应代码、参数和回放

实现：`replenishment/training/ensemble.py`。

冻结参数：[`configs/models/sku_router.json`](../../configs/models/sku_router.json)。

```bash
python run.py --workdir runs/main train --models sku_router
python run.py --workdir runs/main evaluate --models sku_router
```

查看已保存订单，不需要重新训练：

```bash
python run.py replay --model sku_router --sku 1 --through 28 --output runs/replay/sku_router_sku1.csv
```

训练与评价不读取历史指标作预测。测试不达标与失败候选如实保留；当前实现为有限计算预算，不声称理论全局最优。
