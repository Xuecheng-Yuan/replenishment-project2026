# 约束变化原因：2026-10-16重做实验

先看[3页机制探索报告](reports/机制探索报告.pdf)，完整设置、数学模型、108个实验单元和全部核验记录见[19页完整实验报告](reports/完整实验报告.pdf)。报告源文分别为同目录的 `.md`，所有数值从真实Excel、冻结历史轨迹和本轮实际回放生成。

主要结论：服务约束改变可行域，但原新旧版排名反转还涉及反馈与风险包络；在固定反馈策略的对照中，单独开关部署风险包络足以改变分类DRO与(s,S)的数值排序。当前正式版(s,S)与分类DRO的差异区间跨0，稳定优势尚未确认。两者新旧四期预测完全相同；95.5%上限在本轮最优候选处不生效。

本轮为诊断实验，**不替换之前95.28%左右的正式交付结果**。测试已经用于多次修订，训练回溯留出不是向未来预测成绩，全部范围限制在报告中明示。原始业务设置、目标、计数和不达标结果均保留。

## 计算与核验

在仓库根目录运行（Python 3.11或以上）：

```powershell
python -m pip install -r experiments/rolling_dro_v3/requirements_v3.txt
python experiments/constraint_mechanism/verify_experiment.py
python experiments/constraint_mechanism/analyze_results.py
```

若从GitHub新克隆，先用 `git lfs pull --include="data/raw/*.xlsx"` 下载原始Excel，或将用户原始两个同名Excel放入 `data/raw`。输入SHA256不匹配会拒绝使用缓存。已有本地“李佳蔚/项目”文件包含原始文件和校验缓存。

重新计算1620候选与48测试回放请用新目录，保留本次冻结证据：

```powershell
python experiments/constraint_mechanism/run_experiment.py --output experiments/constraint_mechanism/rerun_outputs
python experiments/constraint_mechanism/verify_experiment.py --output experiments/constraint_mechanism/rerun_outputs
```

生成本次报告的命令：

```powershell
python experiments/constraint_mechanism/build_reports.py
python -m pip install -r experiments/constraint_mechanism/requirements_report.txt
python experiments/constraint_mechanism/render_reports.py
```

报告与分析脚本默认读取 `outputs`；新运行目录需明确切换分析输入，不会静默覆盖原证据。模型运行不需要商业求解器或Codex，PDF也可用普通Python加reportlab生成。中文字体优先使用Windows黑体，其他环境回退到内置CJK字体。

## 完整证据索引

- [方案](outputs/protocol.json)、[测试前冻结选择](outputs/frozen_selection.json)：共同库、处理因素、运行脚本散列、全部54条训练选择。
- [全部1620训练候选](outputs/candidate_validation.csv)、[全部108实验单元](outputs/summary.csv)、[训练及测试优胜](outputs/cell_winners.csv)。
- `outputs/shared_arrays.npz`：训练60期需求、测试28期需求、初始库存、预测、残差及两类部署水位。
- `outputs/traces/*.npz`：48次唯一回放，包含 `commencing`、`ending`、`sold`、`orders`（28×6000），`plans`（28×4×6000）。`shared_arrays.npz`中的item_id与每列对应。
- [逐期](outputs/periods.csv)、[全部/爆品/长尾](outputs/segments.csv)、`outputs/sku_summary.csv.gz`：全部48回放的逐商品原始总量。
- [配对差异](outputs/paired_contrasts.csv)、[因素效应](outputs/factor_effects.csv)、[可行集合](outputs/feasible_overlap.csv)、[水位形状](outputs/waterline_shape.csv)。
- [历史新旧重算](outputs/published_summary.csv)、[历史源文件及回放审计](outputs/published_audit.json)、[核验结论](outputs/verification.json)、[实际运行记录](outputs/completed.json)。
- [本实验文件散列清单](SHA256SUMS.json)：由交付脚本生成，可自行复核。

本地 `verify_experiment.py` 会读取并核验原始Excel、完整特征数组并重建聚类与风险水位。云端 `verify_ci.py` 校验交付文件散列，将冻结数组与历史已交付原始轨迹对账，重算1620候选、108单元和806.4万行回放，再重算配对统计；云端不重复下载大体积LFS原始Excel。两种核验的范围分别记录，不将云端结果表述为重新读取了原始工作簿。

导出任一回放为CSV，例如：

```powershell
python experiments/constraint_mechanism/export_trace.py --candidate 806 --envelope envelope --destination trace_806.csv.gz
```

候选号806是本轮分类DRO在P区间反馈库中所选候选，完整条件对应表以summary.csv为准。末期订单在28期后到货，订单有记录，但不计作测试期内销量或库存。
