"""Explain rolling calibration, final all-60 fitting and the delivered results."""
import json
import pandas as pd
from constraint_ablation import HOME
from extend_with_ro import ROOT, LABELS


def positive_headline(frame):
    valid = frame[frame.positive_stock_ratio.between(.95, .955)].sort_values('inventory_demand_ratio')
    if valid.empty:
        return (f'六种方法测试正库存比例为{frame.positive_stock_ratio.min():.4%}–'
                f'{frame.positive_stock_ratio.max():.4%}，均未达到95%–95.5%的目标范围。')
    best = valid.iloc[0]
    return (f'六种方法中{len(valid)}种在测试集达到95%–95.5%的正库存比例；'
            f'达标方法中库存需求比最低的是{LABELS[best.name]}（{best.inventory_demand_ratio:.6f}）。')


def main():
    out = HOME / 'outputs'
    s = pd.read_csv(out / 'summary.csv')
    current = s[s.arm == 'positive'].set_index('method')
    previous = pd.read_csv(ROOT / 'experiments/rolling_dro_v3/with_ro/outputs_v4/summary.csv').query("scope=='all_28_periods'").set_index('method')
    old = pd.read_csv(ROOT / 'experiments/demand_fill_v3/with_ro/outputs_v4/summary.csv').query("scope=='all_28_periods'").set_index('method')
    history_file = HOME / 'history/single_holdout_32_28/summary.csv'
    single = pd.read_csv(history_file).query("arm=='positive'").set_index('method') if history_file.exists() else None
    design = json.loads((out / 'frozen_design.json').read_text(encoding='utf-8'))
    diagnostic_file = HOME / 'history/reset_validation_diagnostic.json'
    diagnostic = json.loads(diagnostic_file.read_text(encoding='utf-8')) if diagnostic_file.exists() else None
    folds = pd.read_csv(out / 'candidate_validation_by_fold.csv')
    selected_folds = []
    for chosen in design['selected']:
        group = folds[folds.candidate == chosen['candidate']].sort_values('fold')
        selected_folds.extend(dict(arm=chosen['arm'], **r) for r in group.to_dict('records'))
    selected_frame = pd.DataFrame(selected_folds)
    selected_frame.to_csv(out / 'selected_validation_by_fold.csv', index=False, encoding='utf-8-sig')
    rows = []
    for method in LABELS:
        a, b, c = old.loc[method], previous.loc[method], current.loc[method]
        row = dict(method=method, old_fill=a.fill_rate, old_positive=a.positive_stock_ratio,
                   old_inventory_ratio=a.inventory_demand_ratio, superseded32_positive=b.positive_stock_ratio,
                   superseded32_inventory_ratio=b.inventory_demand_ratio, full60_fill=c.fill_rate,
                   full60_positive=c.positive_stock_ratio, full60_inventory_ratio=c.inventory_demand_ratio,
                   full60_test_feasible=bool(.95 <= c.positive_stock_ratio <= .955))
        if single is not None:
            row.update(single_holdout60_positive=single.loc[method].positive_stock_ratio,
                       single_holdout60_inventory_ratio=single.loc[method].inventory_demand_ratio)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / 'old_new_revision_comparison.csv', index=False, encoding='utf-8-sig')
    headline = positive_headline(current)
    lines = ['# 三轮滚动验证与完整60期训练交付报告', '', headline, '',
        '新版已改为训练文件内部三轮滚动验证选参，随后用训练文件全部第1–60期最终拟合，并在独立测试文件28期上评估。六种方法、双服务约束对照、报告、Excel、图表和动画使用同一轮结果。', '',
        '## 本次完成的修改', '',
        '1. 将一次“第1–32期拟合、第33–60期验证”改为三轮扩展窗口验证。前24期提供初始历史，第25–60期依次贡献36期验证成绩。',
        '2. 同一预测参数与补货候选在三轮分别评估，再合并销量、需求、期末库存和正库存商品天等原始总量，计算统一的36期验证指标。',
        '3. 根据合并验证成绩冻结预测参数和六种方法的策略参数，再用全部60期重新估计预测器、特征聚类及拟合窗内固定分组的扩展窗预测残差。',
        '4. 测试文件保持独立。测试Period_1重新起算，提前期1，每次预测4期且只执行最近一期订单，缺货按丢失处理。',
        '5. 重跑六种方法及同一候选库的两种服务约束对照，更新全部6000商品、爆品和长尾品的结果、可视化和交付文件。', '',
        '## 选参数如何使用训练60期', '',
        '|轮次|本轮初始拟合|验证时期|验证长度|', '|---|---|---|---:|',
        '|第1轮|第1–24期|第25–36期|12|', '|第2轮|第1–36期|第37–48期|12|',
        '|第3轮|第1–48期|第49–60期|12|', '',
        '每轮预测只使用本轮拟合历史及验证中已经发生的需求。早期数据在后续轮次仍作为历史使用；在选定参数以后，第1–60期全部进入最终拟合。这样既能让后36期检验参数，也保留前24期建立预测和库存风险估计所需的历史。',
        '每轮在拟合期历史内用本轮预测器和候选策略回放库存，将历史末期库存和在途订单带入验证。拟合期回放用于初始化，不计入36期验证成绩；验证期需求不参与初始化。',
        (f'初始化核查发现：若在第25、37、49期均强行重置为原始初始库存且无在途，提前期1使首期损失无法挽回。即使其余时期全部满足，合并需求满足率最高也只有{diagnostic["fill_rate_upper_bound"]:.4%}，正库存比例最高{diagnostic["positive_stock_ratio_upper_bound"]:.4%}。因此验证采用历史回放继承库存状态，独立测试仍按用户指定从Period_1重置。数学上界记录见history/reset_validation_diagnostic.json。' if diagnostic else ''),
        '三轮合并正库存比例按216,000个商品天计算（6000×36），选择95%≤P≤95.5%的候选并最小化合并库存需求比。该规则约束三轮合并成绩；各轮12期的服务比例单独展示，可以高于或低于目标范围。需求满足率对照组以同样36期合并销量/合并需求≥95%筛选。预测超参数以三轮合计绝对误差/三轮合计需求，即合并WAPE选择。', '',
        f'本次选中的共享预测参数为：blend={design["forecast"]["blend"]}，window={design["forecast"]["window"]}，prior={design["forecast"]["prior"]}。完整36组预测参数及逐轮误差保存于forecast_validation.csv和forecast_validation_by_fold.csv。', '',
        f'本轮库存策略候选共{design["library_size"]:,}个。若原倍率网格在训练合并验证中没有可行候选，程序按固定规则细化该方法倍率网格；新增候选同时提供给两种服务约束。网格细化及参数冻结均在测试回放之前完成，细化记录保存在frozen_design.json的calibration_refinement字段。', '',
        '## 如何核实最终用了全部60期', '',
        f'冻结设计保存final_fit_periods=1,…,60，final_fit_observations={design["final_fit_observations"]:,}（60期×6000商品），并保存三轮时间划分和全部候选编号。',
        f'完整训练需求数组SHA256：`{design["training_y_sha256"]}`。',
        'shared_model.npz保存60×6000训练需求、最终聚类、最终残差和预测器斜率、截距。核验脚本从这60期数据独立重新拟合，并核对保存的模型参数。',
        '残差库采用拟合窗内固定分组的扩展窗预测残差。最早18期作为残差估计的暖启动，第19–60期提供预测残差；前18期同时参与对应预测器和最终60期模型的拟合。选参的24期初始历史与残差估计的18期暖启动用途不同。', '',
        '## 新版独立测试结果', '',
        '|方法|需求满足率|正库存比例|库存需求比↓|95%–95.5%达标|', '|---|---:|---:|---:|---|']
    for method, r in current.iterrows():
        lines.append(f'|{LABELS[method]}|{r.fill_rate:.4%}|{r.positive_stock_ratio:.4%}|{r.inventory_demand_ratio:.6f}|'+('是' if .95 <= r.positive_stock_ratio <= .955 else '否')+'|')
    lines += ['', '## 冻结参数在各轮验证中的表现', '',
        '以下是正库存约束组最终选中的候选；合并值决定选参，逐轮值用于观察稳定性。完整两组逐轮证据见outputs/selected_validation_by_fold.csv。', '',
        '|方法|轮次|拟合时期|验证时期|需求满足率|正库存比例|库存需求比|', '|---|---|---|---|---:|---:|---:|']
    for r in selected_frame[selected_frame.arm == 'positive'].itertuples():
        lines.append(f'|{LABELS[r.method]}|{r.fold}|{r.fit_periods}|{r.validation_periods}|{r.fill_rate:.4%}|{r.positive_stock_ratio:.4%}|{r.inventory_demand_ratio:.6f}|')
    lines += ['', '|方法|36期合并需求满足率|36期合并正库存比例|36期合并库存需求比|', '|---|---:|---:|---:|']
    for r in s[s.arm == 'positive'].itertuples():
        lines.append(f'|{LABELS[r.method]}|{r.train_fill:.4%}|{r.train_positive:.4%}|{r.train_inventory_ratio:.6f}|')
    lines += ['', '## 与上一版选参方式的对比', '']
    if single is not None:
        lines += ['两版最终拟合均使用全部60期。本次将一次32/28内部划分改为三轮滚动验证，同时让验证承接拟合历史回放末库存及在途，并仅依据训练成绩细化SAA倍率网格。以下差异反映整套选参流程调整的结果。', '',
            '|方法|一次划分版正库存比例|三轮滚动版正库存比例|一次划分版库存需求比|三轮滚动版库存需求比|', '|---|---:|---:|---:|---:|']
        for r in rows:
            lines.append(f'|{LABELS[r["method"]]}|{r["single_holdout60_positive"]:.4%}|{r["full60_positive"]:.4%}|{r["single_holdout60_inventory_ratio"]:.6f}|{r["full60_inventory_ratio"]:.6f}|')
        lines += ['', '上一版冻结结果保存在[一次划分历史资料](history/single_holdout_32_28/README.md)。']
    lines += ['', '旧需求满足率版、被替代的32期新版、首次60期一次划分版和当前三轮滚动版在总对照Excel中分别标注。完整方案比较同时包含训练和选参变化；只检验服务约束影响时，使用本次共同预测、候选库和最终拟合的[单因素报告](REPORT.md)。', '',
        '## 数学模型与解读', '',
        r'目标为 $\min J=\sum_{i,t}I_{it}/\sum_{i,t}d_{it}$，即累计期末库存相对于累计需求，越低越好。服务指标 $P=\sum\mathbf1\{I_{it}>0\}/(NT)$，要求三轮合并验证值在[0.95,0.955]内。补充报告 $F=\sum s_{it}/\sum d_{it}$。',
        r'到货可用量$a_{it}=I_{i,t-1}+q_{i,t-1}$，销量$s_{it}=\min(a_{it},d_{it})$，期末库存$I_{it}=\max(a_{it}-d_{it},0)$，丢失需求为$d_{it}-s_{it}$。提前期为1；滚动四期计划只执行最近一期订单，之后按新信息重新求计划。',
        r'DRO用训练拟合窗内固定分组的扩展窗相对预测残差$r_j$构造W∞半径ε的分布集合，保护水位$z=r_{(\lceil\beta n\rceil)}+\epsilon$，目标库存$S_h=\lfloor m_h\kappa z\rfloor+1$。特征聚类DRO按特征相似商品池估计风险，不分组DRO共用一池，爆品/长尾DRO分两池。SAA令ε=0；(s,S)用经验分位数确定触发和补足水位。',
        r'传统RO采用需求相对风险区间$[\max(0,\mu-\Gamma\sigma),\mu+\Gamma\sigma]$，上端点给出目标$S_h=\lfloor m_h(\mu+\Gamma\sigma)\rfloor+1$，Γ在训练验证中选择。',
        '倍率κ是训练选择的库存控制参数。κ<1时，内部β对应的概率保护需由缩放后策略的实际服务结果重新评价。本实验在预先声明的有限控制器库内选优。', '',
        '## 结果适用范围', '',
        '验证服务目标作用于36期合并成绩。最终60期重拟合会改变预测器、聚类和残差水位；独立测试又从Period_1重新起算，因此测试服务以实际回放结果为准。测试已用于此前多轮修订，本次属于同一测试集上的重复评估；新的未使用数据可进一步检验泛化能力。', '',
        '## 复现与交付内容', '', '```sh', 'python -m pip install -r experiments/rolling_dro_v3/requirements_v3.txt',
        'python experiments/ro_extension/constraint_ablation.py', 'python experiments/ro_extension/build_ablation_artifacts.py',
        'python experiments/ro_extension/revision_report.py', 'python experiments/ro_extension/build_visualizations.py --full60',
        'node experiments/ro_extension/build_workbooks.mjs --full60', 'node experiments/ro_extension/build_ablation_workbook.mjs',
        'python scripts/build_animation.py', 'python scripts/verify_animation.py',
        'python experiments/ro_extension/verify_ablation.py', '```', '',
        'Python数值计算可按requirements安装后独立运行。Excel导出器优先查找本机Codex自带的artifact-tool运行环境；其他电脑需准备Node.js及可用的@oai/artifact-tool包。已交付的Excel可直接打开，重新运行Python不会自动更新Excel。', '',
        '当前六方法结果位于new_positive/outputs_v4。outputs保存双服务约束全量轨迹、三轮候选成绩、冻结设计、最终模型和统计区间。总对照Excel新增逐轮验证证据，三种商品范围的动画使用全部商品实际轨迹。']
    text = '\n'.join(lines) + '\n'
    (HOME / 'CHANGE_REPORT.md').write_text(text, encoding='utf-8')
    (HOME / 'new_positive/REPORT.md').write_text(text.replace('(REPORT.md)', '(../REPORT.md)').replace('(history/', '(../history/'), encoding='utf-8')
    (HOME / 'README.md').write_text('# 当前三轮滚动验证与完整60期训练版本\n\n' + headline + '\n\n'
        '[修改与交付报告](CHANGE_REPORT.md) · [服务约束单因素报告](REPORT.md) · [新版六方法Excel](new_positive/outputs_v4/replenishment_comparison.xlsx) · [新旧及单因素Excel](outputs/新旧及单因素总对照.xlsx)\n\n'
        '训练内部按1–24→25–36、1–36→37–48、1–48→49–60三轮滚动验证选参，合并36期成绩。最终预测器、聚类和残差使用全部60期拟合，测试使用独立测试Excel的28期，Period_1重新起算。\n', encoding='utf-8')
    (HOME / 'DELIVERY_README.md').write_text(f'''# 李佳蔚：三轮滚动验证与完整60期训练版

{headline}

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
''', encoding='utf-8')
    print(headline)


if __name__ == '__main__':
    main()
