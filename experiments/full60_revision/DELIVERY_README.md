# 李佳蔚：完整60期训练修正版

本次最终预测器、聚类、风险残差均重新使用全部60期训练数据拟合。选参采用1–32期拟合、33–60期向前验证，取消倒序验证。

**本次新版六种方法在测试集均未达到95%–95.5%的正库存比例。** 这是完整重跑后的结果，不能继续引用原32期新版的达标结论。历史结果保留，但已被替代。

## 先阅读

- [任务完成报告：具体改动、训练证明与新旧结果](任务完成报告.md)
- [单因素实验报告](单因素实验报告.md)
- [新旧及单因素总对照](新旧及单因素总对照.xlsx)：历史页保留旧版和被替代32期新版；单因素页展示本次全60期两组结果。
- [新版全60期六方法Excel](新版_全60期_六方法.xlsx)
- [旧版六方法Excel](旧版_需求满足率_六方法.xlsx)

## 当前结果与代码

|内容|文件夹|
|---|---|
|当前全60期新版完整结果与离线图表|项目/experiments/full60_revision/new_positive|
|本次全60期单因素两组结果、训练候选和模型证明|项目/experiments/full60_revision/outputs|
|旧版需求满足率六方法结果|项目/experiments/demand_fill_v3/with_ro|
|被替代的32期新版（仅作历史追溯）|项目/experiments/rolling_dro_v3/with_ro|
|所有修正代码与核验脚本|项目/experiments/ro_extension|
|原始训练与测试Excel|项目/data/raw|

结果含全部6000商品×28期轨迹、订单、四期预测和计划。CSV.gz可解压阅读，HTML可离线打开。根目录任务完成报告包含数学模型、改动范围、测试未达标状态和局限。

## 重跑

安装Python 3.11或以上，在本文件夹执行：

```sh
python -m pip install -r requirements.txt
cd 项目
python experiments/ro_extension/constraint_ablation.py
python experiments/ro_extension/build_ablation_artifacts.py
python experiments/ro_extension/revision_report.py
python experiments/ro_extension/build_visualizations.py --full60
python experiments/ro_extension/verify_ablation.py
```

数值计算不需要商业求解器。已经交付Excel，可直接使用。若要重建Excel，另需Node.js与@oai/artifact-tool：

```sh
node experiments/ro_extension/build_workbooks.mjs --full60
node experiments/ro_extension/build_ablation_workbook.mjs
```

数值重跑不会自动刷新现有Excel；未执行Excel重建时以最新CSV为准。`constraint_ablation.py`输出位置已改为full60_revision；同一候选库的两组仅外层服务约束不同，最终均重新拟合全部60期。

库存需求比＝累计期末库存/总需求，越低越好；正库存比例按商品－天等权；需求满足率按需求量加权。测试Period_1仍重新起算。训练验证符合目标不保证重新拟合后的测试符合目标，测试不用于回调参数。
