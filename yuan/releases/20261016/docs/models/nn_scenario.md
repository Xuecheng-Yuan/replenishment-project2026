# 概率网络与内层优化（nn_scenario）

方向：神经网络与DFL。

## 1. 方法与数据

概率需求网络预测log(1+d)的位置和尺度，生成8个非负情景。内层优化订单；该版本不做DFL微调，作为预测+内层优化对照。

38维历史输入、商品嵌入。概率网络输出对数需求的位置/尺度；当前库存进入内层优化，直接策略网络则把库存显式输入网络。标准化及残差相关基只在拟合前缀计算。

## 2. 数学核心

$$
\mu,\sigma=f_\theta(h,i),\quad \sigma>0,\qquad D^{(k)}=[e^{\mu+\sigma\epsilon^{(k)}}-1]^+.
$$

服务代理用 `sigmoid((可用库存−需求−1)/(0.05×SKU尺度))`；双侧罚项为 `Ψ(R̃)=[.95−R̃]²₊+[R̃−.96]²₊`。平滑代理不等于精确二值服务，最终结果不使用代理冒充R。

## 3. 实现与参数

`replenishment/training/neural.py：DemandNet、pretrained、inner(logits=True)`。

10轮NLL预训练，38维输入+8维嵌入；softmax加松弛变量仅为内部参数化，不是外加容量。

## 4. 运行

在包根目录先prepare，随后：

```bash
python run.py --workdir runs/main train --models nn_scenario
python run.py --workdir runs/main evaluate --models nn_scenario
```

## 5. 当前发布版本与真实测试结果

版本：`service_retrained_20261013`。

库存需求比 **0.334352**；服务水平 **96.0851%**；Fill Rate **93.5051%**。状态：**高于96%**。

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

冻结参数：[`configs/models/nn_scenario.json`](../../configs/models/nn_scenario.json)。

```bash
python run.py --workdir runs/main train --models nn_scenario
python run.py --workdir runs/main evaluate --models nn_scenario
```

查看已保存订单，不需要重新训练：

```bash
python run.py replay --model nn_scenario --sku 1 --through 28 --output runs/replay/nn_scenario_sku1.csv
```

训练与评价不读取历史指标作预测。测试不达标与失败候选如实保留；当前实现为有限计算预算，不声称理论全局最优。
