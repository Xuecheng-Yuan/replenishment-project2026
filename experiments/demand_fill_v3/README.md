# 需求满足率约束版：独立归档

本目录保存切换到正库存服务约束之前的完整交付版本，来自提交[f569b11](https://github.com/Xuecheng-Yuan/replenishment-project2026/commit/f569b116d7f13bee0cb2d34c5242c8240e9dda88)。模型采用**需求满足率＝销量/需求≥95%**，目标最小化累计期末库存÷累计需求。正库存周期比例是补充指标，未用于本版选参。

- [原数学模型及实验报告](outputs_v4/REPORT.md)
- [原交付报告](docs/DELIVERY_REPORT.md)、[聚类说明](docs/CLUSTERING.md)
- [原Excel决策比较](outputs_v4/replenishment_comparison.xlsx)
- [原图表与离线HTML](visualization/README.md)
- [冻结参数](outputs_v4/frozen_parameters.json)、[汇总结果](outputs_v4/summary.csv)
- [全部商品正库存明细](outputs_v4/positive_stock_by_sku.csv)
- [与当前正库存约束版的比较](../version_comparison/COMPARISON.md)

代码、输出和文档保存在本目录；当前正库存服务版本仍位于`../rolling_dro_v3/`。全部输出文件和Excel保持原始字节及SHA256。仅导航链接与本README适配独立位置；`ARCHIVE_PROVENANCE.json`记录原始和归档文件的哈希。

## 本地运行归档版本

从仓库根目录执行：

```powershell
python -m pip install -r experiments/demand_fill_v3/requirements_v3.txt
python experiments/demand_fill_v3/run_v3.py
python experiments/demand_fill_v3/build_visualizations.py
```

默认读取仓库`data/raw/`的两份Git LFS原始Excel，亦可用`--data-dir`指定现有目录。重跑写入本目录`outputs_v4/`，不会覆盖当前正库存版本。历史报告中的`experiments/rolling_dro_v3`运行路径对应归档前位置，复现时使用上面的新路径。

无需重跑即可核对现有结果：

```powershell
python -m unittest discover -s experiments/demand_fill_v3/tests -p test_v2.py -v
python experiments/demand_fill_v3/verify_published.py
python experiments/demand_fill_v3/verify_workbook_v3.py
node experiments/demand_fill_v3/verify_dashboard.mjs
```

全量轨迹采用`.csv.gz`，Pandas直接读取。原Excel与原可视化已经附带；重新导出Excel需要Codex提供的`@oai/artifact-tool`组件。原模型采用W∞最坏需求满足率约束，允许失销、提前期1、预测4天并执行最近1天，测试Period_1重新起算。
