# 在线概率网络与内层优化（online_learning）

方向：神经网络与DFL。

## 1. 方法与数据

先离线训练概率网络，运行时每看到当天真实需求才更新参数和偏差，再预测下一期。每次回放从同一离线模型副本开始，不能把验证学到的状态带入测试。

38维历史输入、商品嵌入。概率网络输出对数需求的位置/尺度；当前库存进入内层优化，直接策略网络则把库存显式输入网络。标准化及残差相关基只在拟合前缀计算。

## 2. 数学核心

$$
b_t=(1-\eta)b_{t-1}+\eta[\log(1+d_t)-\mu_t],\quad \theta_{t+1}=\theta_t-\gamma\nabla_\theta L_{\mathrm{NLL},t}.
$$

服务代理用 `sigmoid((可用库存−需求−1)/(0.05×SKU尺度))`；双侧罚项为 `Ψ(R̃)=[.95−R̃]²₊+[R̃−.96]²₊`。平滑代理不等于精确二值服务，最终结果不使用代理冒充R。

## 3. 实现与参数

`replenishment/training/neural.py：runtime_policy的online_learning分支`。

η=.1；γ∈{1e−5,5e−5}与ρ候选配对选择；不做离线DFL微调；内部Adam。

## 4. 运行

在包根目录先prepare，随后：

```bash
python run.py --workdir runs/main train --models online_learning
python run.py --workdir runs/main evaluate --models online_learning
```

## 5. 当前发布版本与真实测试结果

版本：`service_retrained_20261013`。

库存需求比 **0.227781**；服务水平 **92.6518%**；Fill Rate **92.9519%**。状态：**低于95%**。

前42期拟合；第13—60期连续库存预热，第43—60期验证评分；最终60期重拟合。

以上为既有正式实验轨迹复算，不是新代码全量重训。细化版没有保存的样本内训练指标不从旧粗网格版本移用。

## 6. 补货决策逐步说明

1. 使用38维历史特征和SKU嵌入，输出对数需求的位置与尺度。
2. 依据拟合前缀的相关结构生成8个需求情景，当前库存进入内层。
3. 执行有限步内层优化，得到下一期订单；并非精确全局最优或商业求解器结果。
4. nn_scenario只训练概率预测；DFL用库存与服务代理损失微调，4期/7期指训练回放窗口。
5. 测试执行整数订单，重新计算真实服务水平；仅online_learning在需求揭示后更新模型。

## 7. 对应代码、参数和回放

实现：`replenishment/training/neural.py`。

冻结参数：[`configs/models/online_learning.json`](../../configs/models/online_learning.json)。

```bash
python run.py --workdir runs/main train --models online_learning
python run.py --workdir runs/main evaluate --models online_learning
```

查看已保存订单，不需要重新训练：

```bash
python run.py replay --model online_learning --sku 1 --through 28 --output runs/replay/online_learning_sku1.csv
```

训练与评价不读取历史指标作预测。测试不达标与失败候选如实保留；当前实现为有限计算预算，不声称理论全局最优。
