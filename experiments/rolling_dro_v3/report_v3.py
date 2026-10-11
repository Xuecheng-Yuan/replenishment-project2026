"""Pyramid-structured Chinese report matching run_v3.py and its actual outputs."""
from pathlib import Path
import pandas as pd

LABELS={'DRO_cluster':'特征聚类DRO','DRO_global':'不分组DRO','DRO_category':'爆品/长尾DRO',
        'sS':'传统(s,S)','SAA':'零半径经验模型'}
ORDER=list(LABELS)


def build_report(out:Path,runs,settings,forecast_scores,seconds):
    selected=settings['selected']
    baseline=runs['sS']['summary']
    winner=max(ORDER,key=lambda m:runs[m]['summary']['turnover'] if runs[m]['summary']['fill_rate']>=.95 else -1)
    main=runs[winner]['summary']; clustered=runs['DRO_cluster']['summary']
    valid=[m for m in ORDER if runs[m]['summary']['fill_rate']>=.95]
    turnover_gain=main['turnover']/baseline['turnover']-1
    stock_change=main['avg_inventory_per_sku']/baseline['avg_inventory_per_sku']-1
    lines=[
        '# 多周期补货实验报告：三种分组的分布鲁棒策略',
        '',
        '## 一、结论先行',
        '',
        f'**结论：三种分组的分布鲁棒模型均使用相同的需求预测、提前期与缺货处理；完整测试集上，满足95%需求满足率且库存周转率最高的方法为{LABELS[winner]}。**'
        if valid else '**结论：本次所有方法均未达到95%需求满足率。**',
        '',
        f'{LABELS[winner]}的需求满足率为 **{main["fill_rate"]:.2%}**，平均库存为 **{main["avg_inventory_per_sku"]:.2f} 件/品种**，**28期库存周转率为 {main["turnover"]:.3f} 次**。与传统(s,S)相比，库存周转率变化 **{turnover_gain:+.2%}**，平均库存变化 **{stock_change:+.2%}**。六类特征聚类DRO的对应结果为 **{clustered["fill_rate"]:.2%}**、**{clustered["turnover"]:.3f} 次**。各方法实际满足率略有不同，所以该对比是选定策略在本测试集上的表现。',
        '',
        '支撑结论的证据依次是：①五种方法在同一测试路径上的服务与库存对照；②聚类、不分组、爆品/长尾三种DRO只改变风险样本分组；③库存与丢失销售遵守相同的提前期和滚动执行规则。以下先给结果，再解释模型，最后给复现方法。',
        '',
        '## 二、同一数据与时序下的比较',
        '',
        '| 方法 | 需求满足率 | 是否达95% | 平均库存/品种 | 28期库存周转率（次） | 损失需求 |',
        '|---|---:|:---:|---:|---:|---:|',
    ]
    for method in ORDER:
        s=runs[method]['summary']
        lines.append(f'| {LABELS[method]} | {s["fill_rate"]:.2%} | {"是" if s["fill_rate"]>=.95 else "否"} | {s["avg_inventory_per_sku"]:.2f} | {s["turnover"]:.3f} | {s["lost_units"]:,.0f} |')
    lines.extend(['',
        '需求满足率是总销量除以总需求。**库存周转率（28期，数量口径）=28期总销量÷28期平均在手库存**，单位为“次”，越高越好。每期平均在手库存取该期需求前可用库存与需求后库存的均值，再对28期取平均；缺货直接丢失，因此分子用实际销量，不用需求量。本数据没有单位成本，也没有每期对应的日历长度，所以不计算货币口径或年化周转率。完整测试期包含无法预先补货的第1期；`summary.csv` 另列第2–28期结果。',
        '',
        f'第1期起始在途库存设为0，只能销售原表中的初始库存，因此所有方法共享第1期损失 **{(runs["DRO_cluster"]["summary"]["lost_units"]-runs["DRO_cluster"]["controlled"]["lost_units"]):,.0f} 件**。第28期下单在第29期到货，单列 `final_pipeline`，不计入28期销量或在手库存。',
        '',
        '### 三种DRO的单一变化',
        '',
        '| 变体 | 残差分组 | 训练期选定的目标 β | 鲁棒半径 ε | 测试满足率 | 28期库存周转率（次） |',
        '|---|---|---:|---:|---:|---:|',
    ])
    for method,description in [('DRO_cluster','6个特征聚类'),('DRO_global','所有6000品种共用1组'),('DRO_category','爆品/长尾共2组')]:
        s=runs[method]['summary']; p=selected[method]
        lines.append(f'| {LABELS[method]} | {description} | {p["beta"]:.4f} | {p["radius"]:.3f} | {s["fill_rate"]:.2%} | {s["turnover"]:.3f} |')
    lines.extend(['',
        '三种变体使用同一份SKU级点预测、同一需求实现路径和同一库存模拟。各自的 β 与 ε 在训练集内部验证后固定；因此表中的差异同时反映分组方式及各组独立选定的参数。`risk_group_profile.csv` 给出每组样本量与水位，`policy_validation.csv` 保留全部候选结果。',
        '',
        '### 预测跨度',
        '',
        '| 从本期下单后预测的期数 | 可评估单元格数 | WAPE | RMSE |',
        '|---:|---:|---:|---:|',
    ])
    for row in forecast_scores:
        lines.append(f'| {row["horizon"]} | {row["n_cells"]:,} | {row["wape"]:.2%} | {row["rmse"]:.2f} |')
    lines.extend(['',
        '四个预测值都在每次决策时生成；`decision_comparison_full.csv` 给出6000个SKU、28个周期、五种方法的逐格比较。',
        '',
        '## 三、数学模型与逐项解读',
        '',
        '### 1. 时间、状态与丢失销售',
        '',
        '用 $i=1,\\ldots,6000$ 表示品种，$t=1,\\ldots,28$ 表示测试周期。$D_{it}\\ge0$ 是实际需求，$A_{it}$ 是本期需求前可用库存，$B_{it}$ 是需求后在手库存，$L_{it}$ 是丢失的需求，$q_{it}$ 是本期需求实现后下的单。提前期为1，故',
        '',
        '$$A_{i1}=I_i^{\\mathrm{initial}},\\qquad A_{i,t+1}=B_{it}+q_{it},$$',
        '$$Y_{it}=\\min(A_{it},D_{it}),\\quad B_{it}=(A_{it}-D_{it})_+,\\quad L_{it}=(D_{it}-A_{it})_+.$$',
        '',
        '这里 $Y_{it}$ 为销量，$(x)_+=\\max(x,0)$。缺货直接丢失，不结转到下一期；同一周期不会同时出现正的期末库存与正的丢失需求。数据只有周期汇总，连续盘点用“本期需求结束即观察并下单、下期需求前到货”的离散表示。没有额外盘点周期。',
        '',
        '### 2. 利用特征聚类与滚动预测',
        '',
        '训练数据 $y_{it}$ 和 $x_{it1},\\ldots,x_{it,20}$ 分别是历史需求和20个特征。每个SKU提取对数平均需求、变异系数、零需求比例、上尾/平均值、前后时段趋势，并把20个特征的训练期均值标准化后压缩为前三个主成分。把这些量共同输入 $K$-means 得到 $g(i)\\in\\{1,\\ldots,6\\}$。另两组对照分别采用 $g(i)=1$ 和由原始爆品/长尾标签给出的 $g(i)\\in\\{0,1\\}$。',
        '',
        '在训练集对每个SKU拟合线性趋势 $\\hat d_i(s)=a_i+b_i s$；候选模型还包括簇级趋势与SKU趋势的混合，权重在训练内部验证中选择。时点 $t$ 的预测只用已观测需求修正近期偏差：',
        '',
        '$$\\hat D_{i,t+h\\mid t}=\\max\\left\\{1,\\;\\hat d_i(t+h)+\\frac{k}{k+\\lambda}\\,\\frac1k\\sum_{u=t-k+1}^{t}\\bigl(D_{iu}-\\hat d_i(u)\\bigr)\\right\\},\\qquad h=1,2,3,4.$$',
        '',
        f'实际选中权重={settings["forecast"]["blend"]}，$k={settings["forecast"]["window"]}$，$\\lambda={settings["forecast"]["prior"]}$；权重1表示最终点预测取SKU趋势。未来周期的 $x$ 是否可提前得到没有给定，所以20个特征用于训练期聚类，未来需求预测只依赖历史需求和已知的周期序号。',
        '',
        '### 3. Wasserstein 分布鲁棒服务约束',
        '',
        '用训练期扩展窗口的样本外预测生成非负相对需求样本 $r_{gj}$：实际需求除以对应的一步预测，再按组 $g$ 汇总。对每组的经验分布 $\\widehat P_g$ 定义 $W_\\infty$ 半径为 $\\varepsilon_g$ 的模糊集。每个样本可以移动到',
        '',
        '$$[\\ell_{gj},u_{gj}]=[\\max(0,r_{gj}-\\varepsilon_g),\\;r_{gj}+\\varepsilon_g].$$',
        '',
        '用预测均值 $m_{i,t+h}=\\hat D_{i,t+h\\mid t}$ 把无量纲水位 $z_g$ 还原为实物水位 $S_{i,t+h}=m_{i,t+h}z_g$。对组内每个条件需求分布，模型要求最坏情况下的期望丢失量不超过需求的 $1-\\beta_g$：',
        '',
        '$$\\sup_{Q\\in\\mathcal P_g(\\varepsilon_g)}\\mathbb E_Q\\left[(R-z_g)_+-(1-\\beta_g)R\\right]\\le0.$$',
        '',
        '这直接约束期望需求满足率。$\\beta_g$ 是模型内部的目标，最终是否达到用户要求的95%仍以测试集的总销量/总需求检验。由于括号里的函数对 $R$ 是凸的分段线性函数，每个样本区间的最坏值在端点，故约束精确化为',
        '',
        '$$\\frac1{N_g}\\sum_{j=1}^{N_g}\\max\\left\\{(\\ell_{gj}-z_g)_+-(1-\\beta_g)\\ell_{gj},\\;(u_{gj}-z_g)_+-(1-\\beta_g)u_{gj}\\right\\}\\le0.$$',
        '',
        '完整的簇级优化目标可以写成以下按组可分解的问题：',
        '',
        '$$\\min_{z_g\\ge0}\\;\\sum_g w_g\\,\\mathbb E_{\\widehat P_g}[(z_g-R)_+]\\quad\\mathrm{s.t.}\\quad\\sup_{Q\\in\\mathcal P_g(\\varepsilon_g)}\\mathbb E_Q[(R-z_g)_+-(1-\\beta_g)R]\\le0\\quad\\forall g,$$',
        '',
        '其中 $w_g>0$ 是组的正权重；由于组间没有产能等耦合约束，正权重的具体取值不改变各组最优水位。目标是名义残差分布下的期末库存，约束控制最坏分布下的缺货比例。目标随 $z_g$ 单调不减，因此取满足约束的最小水位。约束左端随 $z_g$ 单调不增，代码用二分求根；单元测试用独立的线性规划端点形式核对。半径为零时得到经验模型 SAA。三个DRO只改变样本分组和各自训练选定的 $\\beta,\\varepsilon$，不更换损失函数。',
        '',
        '### 4. 四期滚动决策',
        '',
        '本期需求实现后预测 $t+1$ 至 $t+4$。对当前需求后在手库存 $B_{it}$，实际执行的订单为',
        '',
        '$$q_{it}=\\max\\left(0,\\left\\lceil S_{i,t+1}-B_{it}\\right\\rceil\\right).$$',
        '',
        '代码还沿着四期预测均值路径计算远期的候选订单；那些订单不在本期执行。下一期真实需求出现后，更新状态、重新预测四期并只执行新的第一笔订单。未来状态的策略仍由当时已实现的需求决定。',
        '',
        '本数据没有给出共享产能、预算、固定订货费或采购价格变化，所以各期的最低可行水位是即时补货：提前囤入的额外库存不会提高未来可达到的最低可行水位。四期计划因此可分解，报告不把远期计划描述成独立的周转收益。若增加跨期耦合约束，需重新求解真正耦合的多阶段模型。',
        '',
        '### 5. 传统 (s,S) 基线与指标',
        '',
        '基线在相同预测 $m_{i,t+1}$ 上，用SKU残差与相应聚类残差的收缩经验分位得到两档水位：$s_{it}=m_{i,t+1}Q_i(p_s)$，$S_{it}=m_{i,t+1}Q_i(p_S)$。若需求后库存低于 $s_{it}$，订至 $S_{it}$，否则订单为零。训练验证允许 $p_s=p_S$，即包含每期订货至目标水位的特例。',
        '',
        '$$\\mathrm{Fill}=\\frac{\\sum_{it}Y_{it}}{\\sum_{it}D_{it}},\\qquad \\overline I=\\frac1{28}\\sum_{t=1}^{28}\\sum_i\\frac{A_{it}+B_{it}}2,\\qquad \\mathrm{ITR}_{28}=\\frac{\\sum_{it}Y_{it}}{\\overline I}.$$',
        '',
        f'其中 $\\mathrm{{ITR}}_{{28}}$ 是28期库存周转率。以{LABELS[winner]}为例，销量 **{main["sales_units"]:,.0f} 件**，全仓平均在手库存 **{main["avg_inventory_per_sku"]*main["skus"]:,.2f} 件**，所以库存周转率为 **{main["turnover"]:.3f} 次**。',
        '',
        '## 四、参数选择、可信范围与复现',
        '',
        '预测参数和每种策略的风险参数均由训练集内部两个验证折选择：前向折拟合第1–36期、验证第37–60期；重启折拟合第29–60期、验证第1–28期。重启折是离线分块验证，用来对应测试集从Period_1重新起算的情形。对每个方法，候选需在重启折完整周期达到95%，并在前向折的可控制期达到按训练初始缺货预算修正的目标，然后按重启折平均库存最小选参。确认参数后，全部60期重新拟合；测试期只做评估。',
        '',
        '先前版本已经用测试结果发现早期重启预测偏差，因而这次属于**修订后的重复测试**。此处结果不能当作从未看过测试集的独立泛化保证。训练和测试两份原始Excel的哈希与解析检查记在 `input_audit.json`，库存守恒及实际订单可用 `verify_results_v3.py` 复核。',
        '',
        '### 在本地运行',
        '',
        '默认读取仓库 `data/raw/` 中的两份原始 `.xlsx`，也可用 `--data-dir` 指定目录。安装 Python 3.10+，从仓库根目录执行：',
        '',
        '```powershell',
        'cd experiments/rolling_dro_v3',
        'python -m pip install -r requirements_v3.txt',
        'python run_v3.py',
        'python verify_results_v3.py',
        'python -m unittest discover -s tests -p test_v2.py -v',
        '```',
        '',
        f'首次解析两份大工作簿需要等待；相同源文件的后续运行复用哈希核验的缓存。{f"本次模型运行约 {seconds:.1f} 秒（不含Excel导出）。" if seconds>0 else ""}实验输出在 `outputs_v4`，其中 `decision_comparison_full.csv` 是6000×28行、五种策略并列的完整对照；`forecast_h1.csv` 至 `forecast_h4.csv` 为四个预测跨度的完整矩阵；`orders_*.csv` 为各方法所有实际下单量；`summary.csv` 和 `results_by_period.csv` 给出汇总。Git中的完整并列对照与五种策略明细采用 `.csv.gz` 无损压缩，Pandas可直接读取。Excel `replenishment_comparison.xlsx` 展示主要对照，CSV保留全部逐格结果。',
        '',
        '方法参考：[RSOME多阶段库存示例](https://xiongpengnus.github.io/rsome/example_dro_inv)、[RSOME分布鲁棒建模文档](https://xiongpengnus.github.io/rsome/dro_rsome)。本实现采用该示例的样本邻域思想，并针对本题的丢失销售和需求满足率写出可直接求解的端点等价式；运行时不依赖商业求解器或RSOME安装。',
    ])
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
