"""Produce current report and a concrete account of the full-60 correction."""
import json
import pandas as pd
from constraint_ablation import HOME
from extend_with_ro import ROOT,LABELS

def main():
    out=HOME/'outputs';new=HOME/'new_positive/outputs_v4'
    s=pd.read_csv(out/'summary.csv');current=s[s.arm=='positive'].set_index('method')
    previous=pd.read_csv(ROOT/'experiments/rolling_dro_v3/with_ro/outputs_v4/summary.csv').query("scope=='all_28_periods'").set_index('method')
    old=pd.read_csv(ROOT/'experiments/demand_fill_v3/with_ro/outputs_v4/summary.csv').query("scope=='all_28_periods'").set_index('method')
    rows=[]
    for method in LABELS:
        a,b,c=old.loc[method],previous.loc[method],current.loc[method]
        rows.append(dict(method=method,old_fill=a.fill_rate,old_positive=a.positive_stock_ratio,old_inventory_ratio=a.inventory_demand_ratio,
            superseded32_positive=b.positive_stock_ratio,superseded32_inventory_ratio=b.inventory_demand_ratio,
            full60_fill=c.fill_rate,full60_positive=c.positive_stock_ratio,full60_inventory_ratio=c.inventory_demand_ratio,
            full60_test_feasible=bool(.95<=c.positive_stock_ratio<=.955)))
    pd.DataFrame(rows).to_csv(out/'old_new_revision_comparison.csv',index=False,encoding='utf-8-sig')
    feasible=current[current.positive_stock_ratio.between(.95,.955)].sort_values('inventory_demand_ratio')
    headline=('测试达标策略中，库存需求比最低为'+LABELS[feasible.index[0]]+f'（{feasible.iloc[0].inventory_demand_ratio:.6f}）。') if len(feasible) else '本次六种方法均未在测试集达到95%–95.5%的正库存服务范围，不能宣布达标最优方法。'
    design=json.loads((out/'frozen_design.json').read_text(encoding='utf-8'))
    lines=['# 新版完整60期训练修正报告','',headline,'',
        '已修正最终模型只拟合第29–60期的问题。新版全部六种方法及单因素实验的两组模型均使用训练集第1–60期完成最终拟合，测试集只作冻结后的滚动评估。','',
        '## 本次具体完成的修改','',
        '1. 取消“29–60期拟合、1–28期倒序验证”的当前流程。改为1–32期拟合、33–60期按时间顺序验证，前28期明确参与模型拟合。',
        '2. 预测超参数通过同一向前验证选择。每个验证时点只吸收已经实现的需求，未来验证需求不进入当前预测。',
        '3. 策略参数在训练验证集选定后冻结，再用全部60期重新估计预测器、特征聚类与训练样本外相对需求残差。重新估计改变风险水位，但不改变已冻结的β、半径、分位数、倍率或RO宽度。',
        '4. 重跑特征聚类DRO、不分组DRO、爆品/长尾DRO、传统(s,S)、SAA与传统RO；重跑相同候选库仅改变服务约束的两组对照。',
        '5. 更新结果明细、全部商品订单、四期预测计划、统计区间、图表、Excel、版本对比及本地文件包。历史旧版与被替代的32期新版保留作追溯，明确不作为当前60期结果。','',
        '## 如何证明用了全部60期','',
        f'冻结设计记录final_fit_periods=1,…,60，final_fit_observations={design["final_fit_observations"]:,}（60期×6000商品）。',
        f'完整训练需求数组SHA256：`{design["training_y_sha256"]}`。',
        'shared_model.npz保存60×6000训练需求、最终聚类、最终残差及最终预测器斜率与截距。verify_ablation.py从这60期数据重新拟合，逐项核对保存的模型参数，防止只有文字说明改为60期。','',
        '样本外残差需要先有历史数据拟合预测器：前18期作为最早暖启动，随后在训练期19–60构造扩展窗口预测残差。前18期参与预测器拟合，前28期也参与最终60期拟合；不会伪造没有可用历史的样本外残差。残差样本数少于60期不等于最终模型只训练这些残差对应的时期。','',
        '## 新版测试结果','',
        '|方法|需求满足率|正库存比例|库存需求比↓|95%–95.5%达标|','|---|---:|---:|---:|---|']
    for method,r in current.iterrows():lines.append(f'|{LABELS[method]}|{r.fill_rate:.4%}|{r.positive_stock_ratio:.4%}|{r.inventory_demand_ratio:.6f}|'+('是' if .95<=r.positive_stock_ratio<=.955 else '否')+'|')
    lines+=['','## 与被替代32期新版的对比','',
        '|方法|原32期正库存比例|全60期正库存比例|原32期库存需求比|全60期库存需求比|','|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f'|{LABELS[r["method"]]}|{r["superseded32_positive"]:.4%}|{r["full60_positive"]:.4%}|{r["superseded32_inventory_ratio"]:.6f}|{r["full60_inventory_ratio"]:.6f}|')
    lines+=['','这次同时修正时间划分、最终拟合数据量及统一策略搜索范围，因此上述修正前后差异不能全部归因于“多使用28期”这一项。真正的服务约束单因素比较使用本次同一套全60期最终模型，见[单因素报告](REPORT.md)。','',
        '## 模型与服务口径','',
        r'目标为 $\min J=\sum_{i,t}I_{it}/\sum_{i,t}d_{it}$，越低越好。服务指标 $P=\sum\mathbf1\{I_{it}>0\}/(NT)$，目标区间[0.95,0.955]。补充报告 $F=\sum s_{it}/\sum d_{it}$。',
        r'$a_{it}=I_{i,t-1}+q_{i,t-1}$，$s_{it}=\min(a_{it},d_{it})$，$I_{it}=\max(a_{it}-d_{it},0)$，缺货损失为$d_{it}-s_{it}$。提前期1；四期计划只执行最近一期。',
        r'DRO以训练样本外相对需求$r_j$构造W∞半径ε的分布集合，水位$z=r_{(\lceil\beta n\rceil)}+\epsilon$；目标库存$S_h=\lfloor m_h\kappa z\rfloor+1$。特征聚类、全局及爆品/长尾分别确定残差池。SAA令ε=0。sS以分位数确定上下水位。',
        r'RO使用区间$[\max(0,\mu-\Gamma\sigma),\mu+\Gamma\sigma]$，上端点目标$S_h=\lfloor m_h(\mu+\Gamma\sigma)\rfloor+1$。',
        '倍率κ是训练选取的库存控制参数。κ<1时，原始z对应的内部β概率保护不能直接转移到缩放后的目标，因此最终服务达标与否以外层验证和实际测试指标为准。此六方法实验优化的是固定有限库存控制器库，未声称得到全局多期最优解。',
        '', '## 结果解释与局限','',
        '训练验证符合服务范围，不保证用60期重新拟合后的测试也符合范围。完整训练改变了残差、聚类和预测器，且测试Period_1重新起算与向前验证的时间位置不同。所有偏离均如实报告，不用测试集回调参数。',
        '测试集已用于此前多轮修订，当前结果属于同一测试集上的重复评估。进一步评估泛化能力应使用新的未使用测试数据。',
        '', '## 复现与核验','', '```sh','python -m pip install -r experiments/rolling_dro_v3/requirements_v3.txt',
        'python experiments/ro_extension/constraint_ablation.py','python experiments/ro_extension/build_ablation_artifacts.py',
        'python experiments/ro_extension/revision_report.py','python experiments/ro_extension/verify_ablation.py','```',
        '新版全部结果位于new_positive/outputs_v4，受控两组全量轨迹位于outputs。新旧及单因素总对照.xlsx保留历史页，并在当前对照页展示全60期结果。']
    (HOME/'CHANGE_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (HOME/'new_positive/REPORT.md').write_text('\n'.join(lines).replace('(REPORT.md)','(../REPORT.md)')+'\n',encoding='utf-8')
    (HOME/'README.md').write_text('# 当前完整60期训练版本\n\n[修正与交付报告](CHANGE_REPORT.md) · [服务约束单因素报告](REPORT.md) · [新版六方法Excel](new_positive/outputs_v4/replenishment_comparison.xlsx) · [新旧及单因素Excel](outputs/新旧及单因素总对照.xlsx)\n\n历史32期拟合版本已被本版替代。\n',encoding='utf-8')
    (HOME/'DELIVERY_README.md').write_text('''# 李佳蔚：完整60期训练修正版

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
''',encoding='utf-8')
    print(headline)

if __name__=='__main__':main()
