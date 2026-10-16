# ReplenishmentLab · 智能补货决策研究工作台

新版独立目录，旧网站、模型权重、PPT及GitHub均未改动。当前为本地科研展示，没有对外发布。

## 打开网页

启动预览后访问：<http://127.0.0.1:8791/?view=1>。

关闭终端或重启电脑后，可复制以下完整命令重新启动；保持终端窗口开启，再打开上面的链接：

```bash
cd "/Users/xuecheng/Research/02_Operations_Research/Replenishment Problem"
/opt/homebrew/Caskroom/miniforge/base/envs/torch/bin/python -m http.server 8791 --bind 127.0.0.1 --directory "/Users/xuecheng/Research/02_Operations_Research/Replenishment Problem/reports/replenishment_lab_20261015/dist"
```

网页运行不依赖外部CDN或互联网；localhost服务需运行，不代表它已部署到公网。不要只复制`dist/index.html`：本版本的完整数据在同目录的内容哈希JSON中，分享本地文件时需保留整个`dist`。

## 页面与操作

1. **研究概览**：模型在库存、服务水平、满足率上的位置，以及时间序列、PTO、SPO和概率网络DFL四条研究流水线。
2. **算法原理**：34方案可选择；个人25个主要单模型继承既有完整推导，融合和Kristina方案保留方法说明、参数及协议差异。公式使用本地矢量图/原生MathML，不显示未渲染LaTeX作为正文。
3. **逐期实验**：搜索任意6000 SKU，最多叠加4个模型；播放或拖动测试第1—28期。指标累计到所选期，图表与滑块紧邻；可切换全仓汇总、库存/订单/销量/预测，导出当前范围CSV、SVG或保存实验配置。
4. **模型对比**：横纵坐标自由选择服务水平、满足率、库存需求比；显示95%服务参考线、实际帕累托点与可选经验平滑线，支持缩放和平移。服务与满足率的配对点图可聚焦差异，不强制从0%画起。
5. **数据探索**：按爆品/长尾等实际类别查看6000 SKU的训练需求规模、CV及零需求比例分布，查看单SKU完整60+28期需求。

教学推演单独标记，不是新训练模型。它使用所选SKU的真实需求路径作为反事实输入，允许改初始库存、需求倍数与简单均值加波动系数；不会写回真实测试成绩。

## 数据与口径

- 29个个人方案来自2026-10-13统一服务口径训练实验及融合实验；不包含个人历史DRO。
- Kristina的5个方案采用GitHub固定提交`d0f0b13`服务约束版本，恢复冻结设置后核对回放，不重新搜索参数。
- 服务水平＝期末库存大于0的商品—周期比例；满足率＝总实际销量/总需求；库存需求比J＝总期末库存/总需求，越低越好。
- 提前期1：先接收上期订单，再满足当期需求，最后做新订单决策。初始在途为0；末期订单不算期末库存。
- 同一评价口径不表示同一训练协议。Kristina新版29—60期拟合、1—28期重启校准，并非严格前向验证；个人主模型采用42/18前向验证、最终60期重拟合。融合方案采用时间顺序OOF，其协议另行保留。
- 历史测试集曾多次被查看，不能当作全新外部留出集；二维经验前沿不是理论最优或显著性证明。

数据在`src/data.json`中保存完整来源与哈希，矩阵为无损压缩float64小端格式；浏览器只解码所选模型，不绘制所有6000 SKU。主快照约53MB，首次打开需要加载完整数据，后续模型切换使用内存缓存。

当前保存14个时间序列/PTO模型的点预测，以及5个Kristina方案经时点对齐的第一步预测。Kristina首期未保存对应预测，留空不补0；概率网络、SPO和融合中未接入点预测的方案会明确提示，不以销量或库存冒充预测。未内嵌20个原始特征矩阵，原理页说明其使用规则。

## 计算与图表核验

8项自动核验通过，覆盖34模型全部5,712,000个SKU—周期、库存守恒、一期提前期、整数订单、三指标独立重算、12种帕累托组合、教学推演因果性、预测形状与缺失值、来源哈希和公式资源。详见`验收记录.json`。

**浏览器自动检查被本机安全策略阻止，未通过其他浏览器、CDP或截图路径绕过。不能声称已完成渲染、窄屏、悬停或实际下载的视觉验收。** CSS提供响应式布局，但效果仍需用户实际打开确认；HTTP状态200仅证明服务可访问，不证明页面所有控件都正确渲染。

## 代码与重新构建

自定义代码集中在`src/content/dashboard/`：`DashboardContent.jsx`、`Tradeoff.jsx`、`Methods.jsx`、`Replay.jsx`、`lab-data.js`、`lab.css`。散点交互与PCHIP复用已有实现，数学SVG复用已有已渲染资源。不改共享运行时。

```bash
cd "/Users/xuecheng/Research/02_Operations_Research/Replenishment Problem"
/Users/xuecheng/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node tools/test_replenishment_lab.mjs
/Users/xuecheng/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node "/Users/xuecheng/.codex/plugins/cache/openai-curated-remote/data-analytics/1.0.11/scripts/data-app.mjs" build --project-dir "/Users/xuecheng/Research/02_Operations_Research/Replenishment Problem/reports/replenishment_lab_20261015" --separate-data
```

数据再生成器为`tools/build_replenishment_lab_data.py`。它从冻结轨迹重建独立源快照，不重新训练个人模型；Kristina方案恢复冻结模型与参数后回放。更新网页数据时需保留当前应用ID，不要重新创建应用或覆盖旧网站。
