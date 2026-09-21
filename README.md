# Replenishment Project 2026

本项目研究单级库存系统中的需求预测与补货决策。当前已经完成均值–方差 baseline，并开始重建滚动时间序列强基准。所有方法都只用 60 期 train 选择模型和参数，再在策略冻结后使用 28 期 test 评价服务水平和周转率。

## 当前数据

- `data/raw/replenishment_train.xlsx`：6,000 个 SKU、60 期，仅用于模型估计、train 内部验证和最终参数重估。
- `data/raw/replenishment_test.xlsx`：6,000 个 SKU、28 期，仅在模型冻结后用于最终评价。
- 两个工作簿还含有 `x1` 至 `x20` 等字段，但当前 baseline 完全不使用这些特征。

数据文件由 `.gitignore` 和 Git LFS 规则管理。代码不会修改原始 Excel。

## Baseline 方法

对 SKU `i`，用训练需求均值和标准差生成静态目标库存：

```text
B_i = ceil(mean_i × L + z × std_i × sqrt(L))
s_i = S_i = B_i
```

当前提前期 `L=1`。60 期 train 按时间顺序切分：

- 第 1–42 期：估计校准阶段的均值和标准差；
- 第 43–60 期：只用于从小到大选择满足 95% 总体服务水平的安全系数 `z`；
- `z` 固定后：用完整 60 期重新估计每个 SKU 的最终 `s=S`；
- 模型完全冻结后：才读取 test，并运行一次最终评价。

完整定义、公式和防泄漏说明见 [docs/baseline_method.md](docs/baseline_method.md)。

## 时间序列方法

时间序列阶段只使用历史需求和商品类别，不使用 `x1` 至 `x20`。候选包括 Naive、移动平均、漂移、局部线性趋势、SES、阻尼 Holt、ARIMA(1,1,0) 和分层共同趋势。

train 使用三个扩展窗口滚动验证：

- 第1–30期拟合，第31–40期验证；
- 第1–40期拟合，第41–50期验证；
- 第1–50期拟合，第51–60期验证。

每个候选都分别校准爆品和长尾品安全系数，但硬约束仍是总体服务水平。候选必须在每个验证折达到95%，然后选择综合周转率最高者。完整公式和时序见 [docs/time_series_method.md](docs/time_series_method.md)。

## 评价指标

总体服务水平采用按需求量加权的 Fill Rate：

```text
总体服务水平 = 所有 SKU 的总销售量 / 所有 SKU 的总需求量
```

总体周转率定义为：

```text
总体周转率 = 测试窗口总销售量 / 测试期平均总期末库存
```

周转率表示完整测试窗口内的周转次数，目前不年化。服务水平不合格时，不能仅凭较高周转率认定策略更好。

爆品和长尾品使用相同口径分别聚合：分组服务率等于该组总销售除以该组总需求，分组周转率等于该组测试窗口总销售除以该组平均总期末库存。代码不会对 SKU 服务率或周转率做简单平均。

## 环境安装

```bash
cd replenishment_project2026
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## 运行测试

```bash
pytest
```

## 运行 Baseline

```bash
python -m experiments.run_baseline
```

默认输出到 `outputs/tables/`：

- `baseline_parameters.csv`：每个 SKU 的 `s`、`S` 和安全系数；
- `baseline_train_statistics.csv`：完整 60 期 train 的需求统计量；
- `baseline_test_per_item_metrics.csv`：test 中每个 SKU 的服务水平、库存和周转率；
- `baseline_summary.json`：数据指纹、使用边界、模型参数、总体指标以及爆品/长尾品分组指标。

分组字段直接加入现有 SKU 明细，分组汇总直接加入现有 JSON，不额外生成文件。

## 运行时间序列实验

```bash
python -m experiments.run_time_series
```

默认只生成三份文件：

- `time_series_comparison.csv`：所有冻结候选的 train 验证和 test 汇总，test 不反向参与选择；
- `time_series_test_per_item_metrics.csv`：最终模型的SKU测试指标；
- `time_series_summary.json`：最终模型、数据指纹、总体及分类测试指标。

不会为每个候选重复保存大规模预测表和逐期轨迹。

### 时间序列首次结果

train 和 test 按两个任意、不相邻的自然时间窗口处理。train 第60期的需求水平和模型状态不会传入 test；test 第1期使用给定初始库存，观察第1期需求后才冷启动预测第2期。原先按连续时期得到的结果已撤销，不再作为研究结论。

冷启动实验中，train 滚动验证选择了 Naive，验证综合服务水平95.7228%，但 test 服务水平只有91.3073%，未达到约束。十个冻结候选均未达到95%，其中12期漂移模型最高，为93.2138%。该结果说明任意窗口的需求水平可以重置，但 test 的增长规律仍没有被 train 稳定覆盖；当前时间序列方法只能作为未达约束的诊断基准。

## 本地复现结果

2026-09-21 使用当前原始数据与 `configs/baseline.yaml` 运行得到：

| 指标 | 结果 |
|---|---:|
| train 拟合期数 | 42 |
| train 内部验证期数 | 18 |
| 选定安全系数 `z` | 3.20 |
| train 验证总体服务水平 | 95.1619% |
| test 总体服务水平 | 98.7838% |
| test 总体周转率 | 7.4787 |
| test 平均总期末库存 | 11,949,377.29 |
| test SKU 95% 达标比例 | 98.7500% |
| 爆品服务率 | 98.5612% |
| 爆品周转率 | 7.4561 |
| 长尾品服务率 | 99.6739% |
| 长尾品周转率 | 7.5692 |

该结果与旧项目 baseline 一致，说明重写保留了原方法含义。高 test 服务水平同时伴随较高库存，因此它只应作为透明、保守的比较基准。

## 防止数据泄漏的实现约束

1. baseline 的 `fit()` 只接受 train；时间序列候选也只在 train 滚动窗口中比较。
2. 流水线完成 train 拟合并取得冻结参数后，才打开 test 文件。
3. test 只用于最终库存评价；本期真实需求只允许更新下一期预测。
4. test 第一时期初库存严格使用 test 文件的 `initial_inventory`，模型不会补写期初到货。
5. 输出 JSON 保存 train/test SHA-256 和各自用途，便于复现实验。

## Git 规则

在你检查本地代码和结果前，不提交、不推送。确认后再执行：

```bash
git status
git add src tests experiments configs docs README.md PROJECT_STRUCTURE.md pyproject.toml
git commit -m "feat: add leakage-safe time-series models"
git push
```
