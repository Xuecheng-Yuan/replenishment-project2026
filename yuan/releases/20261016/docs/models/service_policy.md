# 直接策略网络（service_policy）

方向：神经网络与DFL。

## 1. 方法与数据

网络直接输出非负目标库存，相对于MA3参考进行乘性调整。输入历史、商品嵌入、当前库存及需求参考，训练目标直接来自4期库存回放。

38维历史输入、商品嵌入。概率网络输出对数需求的位置/尺度；当前库存进入内层优化，直接策略网络则把库存显式输入网络。标准化及残差相关基只在拟合前缀计算。

## 2. 数学核心

$$
S_t=\mathrm{ref}_t\exp\left(\operatorname{clip}(g_\theta(h_t,i,B_t/\mu_i,\mathrm{ref}_t/\mu_i),-3,2)\right).
$$

服务代理用 `sigmoid((可用库存−需求−1)/(0.05×SKU尺度))`；双侧罚项为 `Ψ(R̃)=[.95−R̃]²₊+[R̃−.96]²₊`。平滑代理不等于精确二值服务，最终结果不使用代理冒充R。

## 3. 实现与参数

`replenishment/training/neural.py：DirectNet、fine_tune的service_policy分支`。

4轮微调，学习率4e−4；4维嵌入，单隐层64维Tanh；没有NLL正则项。

## 4. 运行

在包根目录先prepare，随后：

```bash
python run.py --workdir runs/main train --models service_policy
python run.py --workdir runs/main evaluate --models service_policy
```

## 5. 当前发布版本与真实测试结果

版本：`service_retrained_20261013`。

库存需求比 **0.488769**；服务水平 **97.5292%**；Fill Rate **97.4202%**。状态：**高于96%**。

前42期拟合；第13—60期连续库存预热，第43—60期验证评分；最终60期重拟合。

以上为既有正式实验轨迹复算，不是新代码全量重训。细化版没有保存的样本内训练指标不从旧粗网格版本移用。

## 6. 补货决策逐步说明

1. 历史特征、SKU嵌入、当前库存和历史参考量进入直接策略网络。
2. 网络输出目标库存，加验证集选择的风险修正。
3. 减去当前库存，非负取整后形成下一期订单；测试不反向训练。

## 7. 对应代码、参数和回放

实现：`replenishment/training/neural.py`。

冻结参数：[`configs/models/service_policy.json`](../../configs/models/service_policy.json)。

```bash
python run.py --workdir runs/main train --models service_policy
python run.py --workdir runs/main evaluate --models service_policy
```

查看已保存订单，不需要重新训练：

```bash
python run.py replay --model service_policy --sku 1 --through 28 --output runs/replay/service_policy_sku1.csv
```

训练与评价不读取历史指标作预测。测试不达标与失败候选如实保留；当前实现为有限计算预算，不声称理论全局最优。
