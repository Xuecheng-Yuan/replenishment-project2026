# yuan：补货决策研究

## 当前版本：2026-10-16

[打开新版工程包](releases/20261016/README.md) · [模型索引](releases/20261016/docs/模型索引.md) · [测试结果](releases/20261016/results/benchmark/README.md) · [下载完整离线网站及回放附件](https://github.com/Xuecheng-Yuan/replenishment-project2026/releases/tag/yuan-v2026.10.16)

新版包含个人29方案、Kristina独立6方案对照、每个模型的中文补货说明、统一评价与可编辑网站源码。后者独立署名，不归入个人算法贡献。大数据、完整离线HTML在Release，不混入Git历史。以下原28模型包和旧指标保留供追溯，不是当前统一评价结果。

---

# 历史版本：28个主要模型

本目录汇总此前完成的28个主要模型家族：28份实际实现代码、28份同名说明，以及公共特征工程、评价和测试代码。原125项历史结果集中在results/variants.json中。不包含后来新增的原截图直接求解诊断，不包含原始数据、训练模型或许可证。

上一版是实验归档，这一版是模型项目。没有src/、experiments/、schemes/或“outputs里面再放程序”的套娃结构，没有依赖上一版文件。公共的数据、特征、库存评价和训练记账在common/；模型的估计器构造、预测方法、成本损失、内层决策及专属训练位于models/。

## 1. 只需要先看这几个位置

```text
README.md                  总说明与所有模型索引（先读这个）
models/lightgbm.py          模型实现：build、fit及统一运行接口
docs/lightgbm.md            对应原理、公式、训练、结果和限制
common/                    公共读数据、特征、评价、训练与命令处理
configs/                   模型索引与搜索参数
results/variants.json      125项旧实验成绩与来源，不放125份代码
data/                      自备授权Excel；原始数据不上传
tests/                     库存、时序、损失和重构等价性测试
verify.py                  整理验收，不训练全部模型
requirements.txt           环境依赖
```

这是仓库中的独立yuan目录；在本目录执行下方命令，不依赖仓库根目录的旧代码。大数组、checkpoint、逐SKU轨迹不是此上传包的一部分。

## 2. 所有模型与文档一一对应

|类别|实际代码|对应说明|
|---|---|---|
|历史均值方差Baseline|[baseline.py](models/baseline.py)|[baseline.md](docs/baseline.md)|
|最近一期Naive|[naive.py](models/naive.py)|[naive.md](docs/naive.md)|
|移动平均（窗口3/6/12）|[moving_average.py](models/moving_average.py)|[moving_average.md](docs/moving_average.md)|
|指数平滑|[exponential_smoothing.py](models/exponential_smoothing.py)|[exponential_smoothing.md](docs/exponential_smoothing.md)|
|ARIMA|[arima.py](models/arima.py)|[arima.md](docs/arima.md)|
|线性回归|[linear_regression.py](models/linear_regression.py)|[linear_regression.md](docs/linear_regression.md)|
|Ridge|[ridge.py](models/ridge.py)|[ridge.md](docs/ridge.md)|
|Lasso|[lasso.py](models/lasso.py)|[lasso.md](docs/lasso.md)|
|弹性网络|[elastic_net.py](models/elastic_net.py)|[elastic_net.md](docs/elastic_net.md)|
|SVR（PCA）|[svr.py](models/svr.py)|[svr.md](docs/svr.md)|
|随机森林|[random_forest.py](models/random_forest.py)|[random_forest.md](docs/random_forest.md)|
|XGBoost|[xgboost.py](models/xgboost.py)|[xgboost.md](docs/xgboost.md)|
|LightGBM|[lightgbm.py](models/lightgbm.py)|[lightgbm.md](docs/lightgbm.md)|
|CatBoost|[catboost.py](models/catboost.py)|[catboost.md](docs/catboost.md)|
|矩信息DRO|[dro_moment.py](models/dro_moment.py)|[dro_moment.md](docs/dro_moment.md)|
|Wasserstein DRO|[dro_wasserstein.py](models/dro_wasserstein.py)|[dro_wasserstein.md](docs/dro_wasserstein.md)|
|卡方DRO|[dro_chi_square.py](models/dro_chi_square.py)|[dro_chi_square.md](docs/dro_chi_square.md)|
|线性有限动作SPO+|[spo_linear.py](models/spo_linear.py)|[spo_linear.md](docs/spo_linear.md)|
|XGBoost＋SPO成本学习|[xgboost_spo.py](models/xgboost_spo.py)|[xgboost_spo.md](docs/xgboost_spo.md)|
|LightGBM＋SPO成本学习|[lightgbm_spo.py](models/lightgbm_spo.py)|[lightgbm_spo.md](docs/lightgbm_spo.md)|
|XGBoost直接动作（一步/滚动）|[direct_xgboost.py](models/direct_xgboost.py)|[direct_xgboost.md](docs/direct_xgboost.md)|
|LightGBM直接动作＋排序|[direct_lightgbm.py](models/direct_lightgbm.py)|[direct_lightgbm.md](docs/direct_lightgbm.md)|
|原始—对偶服务约束策略|[service_policy.py](models/service_policy.py)|[service_policy.md](docs/service_policy.md)|
|原版概率NN与场景DFL|[nn_scenario.py](models/nn_scenario.py)|[nn_scenario.md](docs/nn_scenario.md)|
|库存缺口改进DFL|[dfl_improved.py](models/dfl_improved.py)|[dfl_improved.md](docs/dfl_improved.md)|
|分群＋场景MPC（四轮开发）|[adaptive_mpc.py](models/adaptive_mpc.py)|[adaptive_mpc.md](docs/adaptive_mpc.md)|
|原内层＋在线学习|[online_learning.py](models/online_learning.py)|[online_learning.md](docs/online_learning.md)|
|原始总量周转驱动DFL|[turnover_dfl.py](models/turnover_dfl.py)|[turnover_dfl.md](docs/turnover_dfl.md)|

普通ML文件直接构造所用库的估计器/标准化/PCA，并提供fit；公共训练器统一时间划分、搜索和保存。高级模型包含其损失、梯度、决策与专属训练。共享方法会互相import，例如LightGBM-SPO复用XGBoost-SPO的联合损失梯度；这不是隐藏到旧目录中再运行脚本。

## 3. 评价标准不能混在一起

|口径/参数|约束|到货|x可用性|公共容量|
|---|---|---|---|---|
|A / legacy|总体Fill Rate≥95%|一期提前期|仅历史x|不限制|
|B / ready_immediate|每SKU Ready Rate≥95%|立即到货|当期x已知|有限|
|C / lead_one|每SKU Ready Rate≥95%|一期提前期|未来x未知|有限|

完整定义、特征说明、库存与周转公式见[evaluation.md](docs/evaluation.md)。先看口径再比较成绩。总量周转不是加权周转，SKU Fill Rate达标不是SKU Ready Rate达标。

最近同标准的对照仍为：

|方案|Ready Rate达标SKU|平均库存（万件）|原始总量周转|
|---|---:|---:|---:|
|原内层、不微调|5864/6000|238.25|37.9242|
|原总量周转DFL|5868/6000|263.61|34.2165|
|仅额外NLL训练|5875/6000|237.12|38.1071|

因此最新DFL没有取得整体改进，三个版本都没全SKU达标。旧组普通LightGBM的96.96%总体Fill Rate不能据此与C组排统一名次。每个模型的所有历史成绩在对应文档表格中，完整参数/来源SHA256在results/variants.json；本次没有重新训练全部真实数据。

## 4. 安装和运行

建议Python 3.11。在本目录打开终端：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify.py
python -m unittest discover -s tests -v
```

Windows激活用`.venv\Scripts\Activate.ps1`。依赖使用兼容范围，跨版本/平台数值可能有差异，不声称bitwise复现。

以LightGBM为例：

```bash
python -m models.lightgbm
python -m models.lightgbm --help
python -m models.lightgbm --train --protocol legacy
python -m models.lightgbm --train --protocol ready_immediate
python -m models.lightgbm --train --protocol lead_one
```

默认只读历史成绩；--train才运行。lead_one复用/重现旧模型预测进行无安全库存与固定z4参考，并非该标准重新联合调优。复杂模型有它自己的默认口径，不支持时会明确拒绝，不偷偷切换。第一次完整训练可能较久。

```bash
python -m models.moving_average --train --window 12
python -m models.adaptive_mpc --train --round all
python -m models.online_learning --train
python -m models.turnover_dfl --train
```

MPC第二至四轮需要第一轮的产物，建议第一次用all。最新DFL需要原网络预训练检查点，缺少时先在同目录建立在线模型的离线起点。高级模型的消融/选参仍在其专属训练中一起运行，没有把一个参数变化拆成一个模型。

## 5. 数据与输出

将有权使用的replenishment_train.xlsx、replenishment_test.xlsx放到data/。格式见[data说明](data/README.md)。6000 SKU、训练60期/测试28期，后期程序有严格形状断言。原数据未包含、不会生成假数据。

训练产物只写results/generated/；其中预处理数组、模型、逐SKU与每日轨迹不提交GitHub。results/variants.json是只读历史快照，不会被训练覆盖。改变原始数据需在新克隆中运行，避免旧检查点污染。

## 6. 上传GitHub和局限

只上传此目录。根.gitignore默认排除data内Excel和results/generated内全部产物，因此无需把模型二进制/原始数据上传。聚合成绩也可能敏感，公开前确认协议允许披露；不确定则用私有仓库。本次没有创建远程、推送或擅自选择开源许可证。

默认入口、导入、语法和小型逻辑/重构一致性检查不能代替真实全量训练。历史B组假设不适用于最新口径；验证容量不足、软服务罚项、首阶梯度、测试集反复曝光等限制全部保留。公式使用GitHub原生$与$$，无需本机图片。此次不声称新模型优越、严格DRO或标准SPO+适用于所有DFL实现。
