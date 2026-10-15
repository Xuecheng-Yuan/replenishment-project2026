# 李佳蔚：三轮滚动验证与完整60期训练版

六种方法测试正库存比例为79.6554%–83.1958%，均未达到95%–95.5%的目标范围。

选参使用三轮训练内部滚动验证：1–24期拟合/25–36期验证，1–36/37–48，1–48/49–60。三轮36期合并成绩决定参数，随后预测器、聚类和风险残差均用完整60期最终拟合。独立测试Excel的28期从Period_1重新开始。

## 先阅读

- [任务完成报告](任务完成报告.md)：修改、训练证明、结果、数学模型与复现方法。
- [单因素实验报告](单因素实验报告.md)：同一模型与候选库，仅改变服务约束。
- [新旧及单因素总对照](新旧及单因素总对照.xlsx)：历史版本、当前两组及各轮验证证据。
- [新版六方法Excel](新版_全60期_六方法.xlsx)
- [旧版六方法Excel](旧版_需求满足率_六方法.xlsx)

## 文件位置

|内容|文件夹|
|---|---|
|当前六方法完整结果与离线图表|项目/experiments/full60_revision/new_positive|
|当前双约束结果、三轮候选和全60期模型证明|项目/experiments/full60_revision/outputs|
|上一次60期一次划分结果|项目/experiments/full60_revision/history/single_holdout_32_28|
|旧版需求满足率六方法|项目/experiments/demand_fill_v3/with_ro|
|被替代32期新版|项目/experiments/rolling_dro_v3/with_ro|
|模型、导出和核验代码|项目/experiments/ro_extension|
|离线动画入口|动画/index.html；项目/汇报成果/2026-10-15/动画|
|原始训练与测试Excel|项目/data/raw|

结果含6000商品×28期轨迹、订单、四期预测和计划。CSV.gz可解压阅读，HTML可离线打开。动画可切换全部6000种商品、爆品和长尾品。

## 重跑

安装Python 3.11或以上，在本文件夹执行：

```sh
python -m pip install -r requirements.txt
cd 项目
python experiments/ro_extension/constraint_ablation.py
python experiments/ro_extension/build_ablation_artifacts.py
python experiments/ro_extension/revision_report.py
python experiments/ro_extension/build_visualizations.py --full60
```

Excel导出器优先查找本机Codex自带运行环境，本机可直接使用bundled依赖重建；其他电脑需准备Node.js及可用的@oai/artifact-tool包：

```sh
node experiments/ro_extension/build_workbooks.mjs --full60
node experiments/ro_extension/build_ablation_workbook.mjs
python scripts/build_animation.py
python scripts/verify_animation.py
python experiments/ro_extension/verify_ablation.py
```

数值重跑后需执行Excel重建才能同步现有Excel。库存需求比＝累计期末库存/总需求，越低越好；正库存比例按商品天等权，需求满足率按需求量加权。服务范围作用于三轮合并验证，逐轮和测试均如实报告。
