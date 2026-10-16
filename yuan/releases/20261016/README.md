# 补货决策研究：代码、决策说明与离线网站

GitHub发布版 **2026.10.16**，原工程、旧网站和旧模型保持不动。源码位于本目录，完整离线网页和逐SKU回放附件从[本版Release](https://github.com/Xuecheng-Yuan/replenishment-project2026/releases/tag/yuan-v2026.10.16)下载。

**建议直接下载完整审阅ZIP并解压运行。** Git源码默认不含`artifacts/`与`website/offline/`，只克隆仓库时无法直接打开完整网页或回放订单；可从ZIP内恢复这两个目录。ZIP是上传前冻结的审阅快照，其中“未上传”等文字描述冻结时状态，不代表本次发布未完成。原始Excel及已训练权重不在ZIP中。

个人方向负责人：**yuan xuecheng**。独立对照方向负责人：**Kristina**。

## 先看这四处

1. **网站**：双击 `website/offline/index.html`，无需联网；也可按下方命令开启本机预览。
2. **每个模型怎么补货**：[补货决策总览](docs/补货决策总览.md)。
3. **每个模型的代码、公式、参数**：[模型索引](docs/模型索引.md)，共35份独立中文说明。
4. **真实测试结果**：[结果说明](results/benchmark/README.md)、[完整CSV](results/benchmark/metrics.csv)。

## 这次整理了什么

- Yuan个人29方案：Baseline 1、时间序列4、ML 9、SPO/直接决策5、神经网络/DFL 4、在线/MPC 2、融合4。核心按Baseline→时间序列→ML→SPO→DFL阅读，扩展单列。
- Kristina最新6方案：特征聚类DRO、不分组DRO、爆品/长尾DRO、传统(s,S)、零半径经验模型、传统区间RO，均为**基础水位＋服务预算反馈**。固定Git提交 `aec11c8`，源码与署名单列。
- 今天桌面的DS和Forecast两个ML项目，以及它们的混合对照实验分支，**均不纳入本交付**。
- SPO采用验证预先选中的版本：线性、LightGBM及两种直接决策采用动作细化，XGBoost-SPO保留原版。直接LightGBM测试不达95%如实标出。
- 保留原7期DFL；满足率改进候选未满足库存不升的要求，作为未采纳研究记录保留。
- Kristina今天的约束机制实验是诊断，不替换她已正式交付的六方法结果。

**本次是工程整理与回放核验，不是重新训练全部6000SKU。** 网站成绩来自已保存的正式轨迹。新代码通过的合成测试、真实SPO推理复现，分别记录在`verification/`，不冒充新的模型成绩。

## 目录怎么读

```text
yuan_release_20261016/
├── run.py                    # list / prepare / train / evaluate / reference / replay / serve
├── replenishment/            # 你的算法：数据、特征、模型、训练、决策、统一评价
├── configs/                  # 唯一发布模型目录、逐模型冻结参数
├── docs/                     # 中文指南、数学说明、每个模型一份MD
├── results/benchmark/         # 当前35模型的真实测试指标与来源
├── results/research/          # SPO消融、DFL未采纳候选，不混入主排名
├── partners/kristina/         # 固定Git提交、原始源码及独立复现说明
├── website/app/              # 网站可编辑源码；只改src/content和数据，不改共享运行时
├── website/offline/          # 便携单文件网站；本地审阅附件，默认不进Git
├── artifacts/                # 网站/订单回放的完整轨迹附件，默认不进Git
├── reference/                # 旧版历史记录，只供溯源，不参与新训练
├── tests/                    # 小数据合同与端到端测试
├── verification/             # 本次实际核验记录及文件清单
└── tools/                    # 整理、文档、网站数据、校验、固定提交同步
```

算法共用一套数据和评价器，不复制35份重复仿真代码。9个ML保留独立构造器；SPO保留真实损失与梯度；统计、DFL和融合在各自方法族中明确实现。Kristina上游代码不混入你的训练包。

## 现在直接看网站

在终端复制：

```bash
cd "/Users/xuecheng/Research/02_Operations_Research/Replenishment Problem/yuan_release_20261016"
python3 run.py serve --port 8793
```

然后打开 <http://127.0.0.1:8793/>，终端保持开启。只绑定本机，不是公网网站；运行网站不需要安装ML库，也不联网。

## 安装、训练和查看订单

使用Python 3.11或以上。首次安装依赖需要联网；模型训练与评价本身不联网。不要加载不可信的joblib/pt工件。

```bash
cd "/Users/xuecheng/Research/02_Operations_Research/Replenishment Problem/yuan_release_20261016"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py list
python run.py reference --owner yuan
python run.py replay --model turnover_dfl --sku 1 --through 28 --output runs/replay/dfl_sku1.csv
python run.py --workdir runs/my_experiment prepare --train ../data/replenishment_train.xlsx --test ../data/replenishment_test.xlsx
python run.py --workdir runs/my_experiment train --models baseline lightgbm lightgbm_spo turnover_dfl
python run.py --workdir runs/my_experiment evaluate --models baseline lightgbm lightgbm_spo turnover_dfl
python run.py --workdir runs/my_experiment compare
```

下载到别的电脑后，修改`cd`与Excel路径即可。`reference`读历史正式成绩；`compare`只汇总当前`--workdir`的新运行结果，二者不混用。

详细操作见[运行指南](docs/运行指南.md)、[工程与版本说明](docs/工程与版本说明.md)、[统一评价协议](docs/评价协议.md)。

## 发布范围与数据边界

请查看[审阅清单与上传边界](docs/审阅与GitHub清单.md)。用户确认后，源码、中文文档、参数与小型指标发布到Git；完整轨迹和离线HTML作为同一私有仓库的Release附件提供。原始Excel、权重和运行缓存仍不上传；未改变仓库可见性。Kristina源文件与报告固定版本保留，不修改其原目录。
