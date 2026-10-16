# Kristina：独立方法与固定源码快照

作者：Kristina。来源：`Xuecheng-Yuan/replenishment-project2026`，固定提交`aec11c863b6d8a71a1e0874cfb60594cc034cf51`（2026-10-16）。`SOURCE.json`列出选取的75个文件、Git blob SHA和SHA256；`upstream/`不是完整克隆，不包含原始Excel，也没有运行她的训练代码来生成新成绩。

## 本版六种正式方案

特征聚类DRO、不分组DRO、爆品/长尾分组DRO、传统(s,S)、零半径SAA、传统区间RO。均保留正式版本的**基础风险水位＋服务预算反馈**；RO是本版新增的独立对照，不再沿用旧的五方案清单。

正式结果位于`upstream/experiments/service95_revision/outputs_v4/`。本包读取原始逐SKU逐期轨迹，与共同需求和初始库存逐一对齐后，重算服务水平、Fill Rate和J；没有只照抄汇总分数。六方案服务水平约95.28%，J约0.346—0.458；精确数值看根目录`results/benchmark/metrics.csv`。

## 一期的补货逻辑

先用历史数据确定基础预测及风险水位；在测试第t期需求发生后，累计已经实现的服务计数，并计算剩余时期还需多少服务次数。利用训练残差及已经观察到的误差估计服务响应，二分校准下一期水位倍率；扣掉当前期末库存后下整数订单。提前期一期，只执行四期计划的第一期；传统(s,S)另有触发阈值。

反馈可以使用已经发生的测试需求，不能读未来需求。固定DRO水位经过动态缩放后，不能直接声称仍有原固定模型的分布鲁棒保证。

训练内部回溯校准包含前28期重启检验，并非严格前向留出。目标带为95%—95.5%，个人模型通常为95%—96%；统一评价不等于统一训练协议。

## 原始报告与代码入口

- [正式版本报告](upstream/experiments/service95_revision/REPORT.md)
- [服务预算反馈实现](upstream/experiments/ro_extension/service_band_revision.py)
- [新机制诊断说明](upstream/experiments/constraint_mechanism/README.md)

机制探索是诊断，不替换正式六方法，也不据测试成绩挑一个参数冒充冻结版本。要从原始Excel完整重训，建议在固定提交的完整上游仓库内按其报告运行；本包提供的是可审查的选取快照及已保存轨迹核验，不声称完成上游全量重训。你的个人`run.py train`不会训练或覆盖这些模型。
