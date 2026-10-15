# 当前三轮滚动验证与完整60期训练版本

六种方法测试正库存比例为79.6554%–83.1958%，均未达到95%–95.5%的目标范围。

[修改与交付报告](CHANGE_REPORT.md) · [服务约束单因素报告](REPORT.md) · [新版六方法Excel](new_positive/outputs_v4/replenishment_comparison.xlsx) · [新旧及单因素Excel](outputs/新旧及单因素总对照.xlsx)

训练内部按1–24→25–36、1–36→37–48、1–48→49–60三轮滚动验证选参，合并36期成绩。最终预测器、聚类和残差使用全部60期拟合，测试使用独立测试Excel的28期，Period_1重新起算。
