# 历史均值Baseline（baseline）

方向：历史统计。

## 1. 方法与数据

逐SKU用拟合前缀的历史均值和标准差。没有使用20个x，也没有独立的s<S触发区间；s=S。

只使用逐SKU需求历史与当前策略库存；不使用x1—x20。

## 2. 数学核心

$$
S_i=\max(\overline d_i+z\sigma_i,0),\quad q_{it}=[S_i-B_{it}]^+.
$$

需求预测本身不等于补货决策。最终统一按下式订货，尺度可为固定残差标准差或相对尺度；baseline只用固定尺度。

$$
q_{it}=\left[\widehat d_{i,t+1}+z\sigma_{i,t+1}-B_{it}\right]^+.
$$

## 3. 实现与参数

`replenishment/training/forecast.py：history_prediction的baseline分支`。

z验证选择后冻结；均值标准差在60期最终重算。

## 4. 运行

在包根目录先prepare，随后：

```bash
python run.py --workdir runs/main train --models baseline
python run.py --workdir runs/main evaluate --models baseline
```

## 5. 当前发布版本与真实测试结果

版本：`service_retrained_20261013`。

库存需求比 **5.417928**；服务水平 **99.4595%**；Fill Rate **99.0866%**。状态：**高于96%**。

前42期拟合；第13—60期连续库存预热，第43—60期验证评分；最终60期重拟合。

以上为既有正式实验轨迹复算，不是新代码全量重训。细化版没有保存的样本内训练指标不从旧粗网格版本移用。

## 6. 补货决策逐步说明

1. 只使用截至本期已观测的需求；ML还使用滞后一期及更早的x。
2. 历史统计或ML预测下一期需求；普通ML训练不直接优化服务约束。
3. 加上验证集校准的风险缓冲，得到下一期目标库存。
4. 扣除当前期末库存，非负部分向上取整下单；提前期一期。

## 7. 对应代码、参数和回放

实现：`replenishment/training/forecast.py`。

冻结参数：[`configs/models/baseline.json`](../../configs/models/baseline.json)。

```bash
python run.py --workdir runs/main train --models baseline
python run.py --workdir runs/main evaluate --models baseline
```

查看已保存订单，不需要重新训练：

```bash
python run.py replay --model baseline --sku 1 --through 28 --output runs/replay/baseline_sku1.csv
```

训练与评价不读取历史指标作预测。测试不达标与失败候选如实保留；当前实现为有限计算预算，不声称理论全局最优。
