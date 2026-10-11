"""Create portable charts and a self-contained HTML dashboard from saved results."""
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs_v4'
VIS = ROOT/'visualization'
METHODS = ['DRO_cluster', 'DRO_global', 'DRO_category', 'sS', 'SAA']
LABELS = ['Cluster DRO', 'Global DRO', 'Hot / long-tail DRO', '(s,S)', 'SAA']
CN = dict(zip(METHODS, ['特征聚类DRO', '不分组DRO', '爆品/长尾DRO', '传统(s,S)', '零半径经验模型']))
COLORS = ['#287c8e', '#64748b', '#c6781d', '#4c956c', '#9a78ad']


def save(fig, name):
    fig.savefig(VIS/(name+'.png'), dpi=170, bbox_inches='tight')
    fig.savefig(VIS/(name+'.svg'), bbox_inches='tight')
    plt.close(fig)


def svg(name):
    text = (VIS/(name+'.svg')).read_text(encoding='utf-8')
    return text[text.index('<svg'):]


def main():
    VIS.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.hashsalt': 'rolling-dro-v3'})
    summary = pd.read_csv(OUT/'summary.csv').query("scope == 'all_28_periods'").set_index('method').loc[METHODS]
    periods = pd.read_csv(OUT/'results_by_period.csv')
    profile = pd.read_csv(OUT/'risk_group_profile.csv').query("method == 'DRO_cluster'").sort_values('group')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, field, title in zip(axes, ['fill_rate', 'inventory_demand_ratio', 'avg_inventory_per_sku'],
            ['Demand fill rate (%)', 'Inventory / demand ratio (lower is better)', 'Average inventory per SKU (units)']):
        values = summary[field]*(100 if field == 'fill_rate' else 1)
        ax.barh(LABELS, values, color=COLORS)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, values.max()*1.2)
        for row, value in enumerate(values):
            ax.text(value+values.max()*.015, row, f'{value:.6f}' if field == 'inventory_demand_ratio' else f'{value:.2f}', va='center', fontsize=9)
        if field == 'fill_rate':
            ax.axvline(95, color='#a11', linestyle='--', linewidth=1, label='95% target')
        ax.grid(axis='x', alpha=.15)
    save(fig, 'strategy_comparison')
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    for method, label, color in zip(METHODS, LABELS, COLORS):
        rows = periods[periods.method == method].sort_values('period')
        axes[0].plot(rows.period, rows.fill_rate*100, label=label, color=color, linewidth=1.6)
        axes[1].plot(rows.period, rows.avg_inventory_per_sku, color=color, linewidth=1.6)
    axes[0].axhline(95, color='#a11', linestyle='--', linewidth=1)
    axes[0].set_ylabel('Period demand fill rate (%)')
    axes[0].set_title('Rolling test replay: all 28 periods (period 1 uses initial stock only)')
    axes[0].legend(loc='lower right', ncol=2, fontsize=9)
    axes[1].set_ylabel('Average inventory per SKU (units)')
    axes[1].set_xlabel('Test period')
    axes[1].set_xticks(range(1, 29, 3))
    for ax in axes:
        ax.grid(alpha=.15)
    save(fig, 'period_comparison')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, field, title in zip(axes, ['skus', 'train_mean', 'level'],
            ['Number of SKUs', 'Mean training demand (units / SKU / period)', 'Robust inventory multiplier']):
        values = profile[field]
        axes[0].set_xlabel('Cluster ID (arbitrary numbering)')
        ax.bar(profile.group.astype(str), values, color=COLORS[0])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, values.max()*1.2)
        for index, value in enumerate(values):
            ax.text(index, value+values.max()*.02, f'{value:.3f}' if field == 'level' else f'{value:,.0f}', ha='center', fontsize=9)
        ax.grid(axis='y', alpha=.15)
    save(fig, 'cluster_profile')
    header = '<tr><th>策略</th><th>需求满足率</th><th>28期库存需求比</th><th>平均库存/品种</th><th>损失需求</th></tr>'
    rows = ''.join(f'<tr><td>{CN[m]}</td><td>{summary.loc[m,"fill_rate"]:.2%}</td><td>{summary.loc[m,"inventory_demand_ratio"]:.6f}</td><td>{summary.loc[m,"avg_inventory_per_sku"]:.2f}</td><td>{summary.loc[m,"lost_units"]:,.0f}</td></tr>' for m in METHODS)
    options = ''.join(f'<option value="{m}">{html.escape(CN[m])}</option>' for m in METHODS)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>滚动分布鲁棒补货：最终结果</title><style>
body{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#17324d;background:#f2f6f8;margin:0}main{max-width:1240px;margin:auto;padding:32px 24px}h1{font-size:30px}h2{font-size:22px;margin-top:0}.card{background:white;padding:24px;margin:20px 0;border-radius:12px;box-shadow:0 2px 8px #17324d0c}p{max-width:1000px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:right;padding:9px;border-bottom:1px solid #e5eaee}td:first-child,th:first-child{text-align:left}th{background:#e9f3f4}svg{width:100%;height:auto}select{padding:8px;font-size:15px}.tag{background:#e9f3f4;padding:5px 10px;border-radius:6px;margin-right:8px;display:inline-block}.scroll{overflow:auto}.note{color:#546775;font-size:14px}a{color:#0f6b77}
</style><main><h1>滚动分布鲁棒补货：最终结果</h1>
<p><span class="tag">6,000个SKU × 28期</span><span class="tag">预测4期，执行最近1期</span><span class="tag">提前期1 · 缺货丢失</span></p>
<section class="card"><h2>__HEADLINE__</h2><p>__CONCLUSION__</p><p class="note">库存需求比＝所有SKU、28期累计期末库存÷同期总需求，无量纲，越低越好。分母包括未满足的需求。全期满足率达标不表示每期或每个SKU都达标。测试集曾用于早期版本诊断，本次为修订后的重复评估。</p><div class="scroll"><table>__TABLE__</table></div></section>
<section class="card"><h2>策略比较</h2>__STRATEGY__</section>
<section class="card"><h2>28期滚动回放</h2>__PERIOD__<p class="note">测试第1期只能使用初始库存，所有方法共享该期的初始缺货。下方可以选择策略查阅逐期数值。</p><label>策略：<select id="method">__OPTIONS__</select></label><div class="scroll"><table><thead><tr><th>测试周期</th><th>需求满足率</th><th>平均库存/品种</th><th>当期库存需求比</th><th>损失需求</th></tr></thead><tbody id="periodRows"></tbody></table></div></section>
<section class="card"><h2>六类特征聚类的风险画像</h2>__CLUSTER__<p>聚类输入包括5个训练需求形态指标和20个原始特征训练均值的3个主成分。K=6与主成分权重0.5为预设值。聚类主要用于共享相对预测误差分布，形成分组鲁棒补货水位。组号仅用于标识，没有等级顺序。</p></section>
<p><a href="../outputs_v4/REPORT.md">数学模型与实验报告</a> · <a href="../outputs_v4/replenishment_comparison.xlsx">Excel决策对照表</a> · <a href="../../../docs/rolling_dro_v3/DELIVERY_REPORT.md">交付与复现报告</a></p>
<p class="note">图表由已发布CSV生成，离线打开本HTML即可查阅，无需网络或前端依赖。</p></main><script>
const data=__DATA__;
function refresh(){let m=document.getElementById('method').value;document.getElementById('periodRows').innerHTML=data.filter(r=>r.method===m).sort((a,b)=>a.period-b.period).map(r=>`<tr><td>${r.period}</td><td>${(r.fill_rate*100).toFixed(2)}%</td><td>${r.avg_inventory_per_sku.toFixed(2)}</td><td>${r.inventory_demand_ratio.toFixed(6)}</td><td>${r.lost_units.toLocaleString()}</td></tr>`).join('');}
document.getElementById('method').addEventListener('change',refresh);refresh();
</script></html>'''
    feasible = summary[summary.fill_rate >= .95]
    if feasible.empty:
        headline, conclusion = '本次结果中没有达到95%总体需求满足率的策略', '请查阅下表的服务率及库存结果。'
    else:
        winner = feasible.inventory_demand_ratio.idxmin()
        result = summary.loc[winner]
        gain = result.inventory_demand_ratio/summary.loc['sS', 'inventory_demand_ratio']-1
        headline = f'{CN[winner]}在达标策略中的库存需求比最低'
        conclusion = (f'需求满足率{result.fill_rate:.2%}，28期库存需求比{result.inventory_demand_ratio:.6f}；'
                      f'相对传统(s,S)变化{gain:+.2%}。{len(feasible)}种策略达到总体需求满足率95%的目标。')
    for token, value in {'__HEADLINE__': headline, '__CONCLUSION__': conclusion,
            '__TABLE__': header+rows, '__STRATEGY__': svg('strategy_comparison'),
            '__PERIOD__': svg('period_comparison'), '__CLUSTER__': svg('cluster_profile'),
            '__OPTIONS__': options, '__DATA__': periods.to_json(orient='records', double_precision=15)}.items():
        page = page.replace(token, value)
    (VIS/'index.html').write_text(page, encoding='utf-8')
    print(VIS/'index.html')


if __name__ == '__main__':
    main()
