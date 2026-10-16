# 新旧两版新增传统RO

> 当前新版已修正为全部60期最终拟合。请阅读[全60期修正与交付报告](../full60_revision/CHANGE_REPORT.md)及[当前结果](../full60_revision/README.md)。下文原32期新版及原重置验证仅保留作历史追溯。当前重跑命令为`python experiments/ro_extension/constraint_ablation.py`，输出full60_revision。

两版均增加第六种方法“传统区间RO”，与三种DRO、传统(s,S)、SAA同时比较。原始五方法交付保留，最新六方法交付在各版的 `with_ro` 文件夹。

|版本|报告|Excel|可视化|
|---|---|---|---|
|旧版：需求满足率≥95%|[报告](../demand_fill_v3/with_ro/REPORT.md)|[六方法Excel](../demand_fill_v3/with_ro/outputs_v4/replenishment_comparison.xlsx)|[图表与交互页面](../demand_fill_v3/with_ro/visualization/index.html)|
|新版：正库存商品天比例95%–95.5%|[报告](../rolling_dro_v3/with_ro/REPORT.md)|[六方法Excel](../rolling_dro_v3/with_ro/outputs_v4/replenishment_comparison.xlsx)|[图表与交互页面](../rolling_dro_v3/with_ro/visualization/index.html)|

[新旧两版六方法比较](../version_comparison/COMPARISON.md)。GitHub可直接阅读报告和PNG图片；交互HTML下载后用浏览器打开。

## 本地运行

原始训练和测试工作簿位于 `data/raw`（先安装Git LFS并执行 `git lfs pull`）。在仓库根目录执行：

```sh
python -m pip install -r experiments/rolling_dro_v3/requirements_v3.txt
python experiments/ro_extension/extend_with_ro.py --version all
python experiments/ro_extension/build_visualizations.py
python experiments/ro_extension/extend_with_ro.py --publish
python experiments/ro_extension/verify_ro.py
python experiments/version_comparison/build_comparison.py
```

`--version`也可指定 `demand_fill_v3` 或 `rolling_dro_v3`。重跑读取原五方法已发表轨迹，重新训练RO并重新评估测试，不覆盖历史归档。两版分别冻结Γ，测试不能改变选参。

数值计算使用NumPy/SciPy，箱式集合的鲁棒对应式直接求区间上端点，无需商业求解器。`ro_validation.csv`保存全部训练候选结果；`frozen_parameters.json`保存最终宽度及集合边界；`ro_test_coverage.json`报告集合在测试期的实际覆盖。

## Excel重建

Excel已随仓库交付，无需重新生成即可使用。可视化脚本采用Matplotlib。Excel构建器采用Node.js和 `@oai/artifact-tool`，在已安装该包的环境执行：

```sh
node experiments/ro_extension/build_workbooks.mjs
python experiments/ro_extension/extend_with_ro.py --publish
python experiments/ro_extension/verify_ro.py
```

工作簿包含六方法结论、逐期结果、最近一期6000商品决策、12商品完整28期样本和RO训练搜索。结论页的两种服务比例与库存需求比由原始分子分母通过Excel公式计算。全量历史并列结果在 `decision_comparison_full.csv.gz`，各方法全量四期计划在 `detail_*.csv.gz`，每种方法全部28期订单在 `orders_*.csv`。
