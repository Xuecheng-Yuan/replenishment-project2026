"""Compare preserved demand-fill results with the current full-60 positive arm."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
LABELS={'DRO_cluster':'特征聚类DRO','DRO_global':'不分组DRO','DRO_category':'爆品/长尾DRO','sS':'传统(s,S)','SAA':'SAA','RO':'传统区间RO'}

def main():
    old=pd.read_csv(ROOT.parent/'demand_fill_v3/with_ro/outputs_v4/summary.csv').query("scope=='all_28_periods'").set_index('method')
    new=pd.read_csv(ROOT.parent/'full60_revision/new_positive/outputs_v4/summary.csv').query("scope=='all_28_periods'").set_index('method')
    valid=new[new.positive_stock_ratio.between(.95,.955)].sort_values('inventory_demand_ratio')
    headline=(f'当前新版测试达标{len(valid)}种方法；达标方法中库存需求比最低为{LABELS[valid.index[0]]}。' if len(valid)
        else f'当前新版六方法测试正库存比例为{new.positive_stock_ratio.min():.4%}–{new.positive_stock_ratio.max():.4%}，均未达到95%–95.5%目标。')
    rows=[]
    lines=['# 旧版与当前三轮滚动验证新版结果对比','',headline,'',
        '新版使用训练文件内部三轮验证：1–24期拟合/25–36期验证、1–36/37–48、1–48/49–60，按36期合并成绩选参。随后用全部60期重新拟合预测器、聚类与拟合窗内固定分组的扩展窗预测残差，再评估独立测试文件的28期。','',
        '旧版与新版服务指标和选参流程不同。下表比较完整方案；服务约束的单因素影响另用相同模型和策略库检验。','',
        '|方法|旧版需求满足率|旧版正库存比例|旧版库存需求比|全60期新版需求满足率|全60期新版正库存比例|全60期新版库存需求比|新版达标|',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for method in LABELS:
        a,b=old.loc[method],new.loc[method];valid=.95<=b.positive_stock_ratio<=.955
        rows.append(dict(method=method,old_demand_fill=a.fill_rate,new_demand_fill=b.fill_rate,
            old_positive_stock_service=a.positive_stock_ratio,new_positive_stock_service=b.positive_stock_ratio,
            old_inventory_demand_ratio=a.inventory_demand_ratio,new_inventory_demand_ratio=b.inventory_demand_ratio,new_test_feasible=valid))
        lines.append(f'|{LABELS[method]}|{a.fill_rate:.4%}|{a.positive_stock_ratio:.4%}|{a.inventory_demand_ratio:.6f}|{b.fill_rate:.4%}|{b.positive_stock_ratio:.4%}|{b.inventory_demand_ratio:.6f}|'+('是' if valid else '否')+'|')
    pd.DataFrame(rows).to_csv(ROOT/'results_comparison.csv',index=False,encoding='utf-8-sig')
    single_file=ROOT.parent/'full60_revision/history/single_holdout_32_28/summary.csv'
    if single_file.exists():
        single=pd.read_csv(single_file).query("arm=='positive'").set_index('method')
        lines+=['','## 同为完整60期最终拟合：本次选参变化','',
            '上一版以1–32期拟合、33–60期验证选参。本次改为三轮滚动验证，同时将验证初始化改为承接拟合历史库存及在途，并根据训练可行性细化SAA倍率网格。两版最终均用60期拟合，测试文件和服务目标一致；结果差异体现整套选参流程的调整。','',
            '|方法|上次一次划分正库存比例|本次三轮正库存比例|上次库存需求比|本次库存需求比|','|---|---:|---:|---:|---:|']
        for method in LABELS:
            a,b=single.loc[method],new.loc[method]
            lines.append(f'|{LABELS[method]}|{a.positive_stock_ratio:.4%}|{b.positive_stock_ratio:.4%}|{a.inventory_demand_ratio:.6f}|{b.inventory_demand_ratio:.6f}|')
    lines+=['','库存需求比越低越好，但未满足相应服务约束时，不能据此认定更优。训练验证达标不保证重新拟合后测试达标，本次未用测试调整参数。','',
        '- [本次具体修改和全60期训练证明](../full60_revision/CHANGE_REPORT.md)',
        '- [固定其他条件的服务约束单因素报告](../full60_revision/REPORT.md)',
        '- [当前新版六方法Excel](../full60_revision/new_positive/outputs_v4/replenishment_comparison.xlsx)',
        '- [旧版六方法Excel](../demand_fill_v3/with_ro/outputs_v4/replenishment_comparison.xlsx)',
        '- [被替代32期版本的历史比较](HISTORICAL_COMPARISON_32.md)（仅作追溯）。']
    if single_file.exists():lines.append('- [上一次60期一次划分结果](../full60_revision/history/single_holdout_32_28/README.md)。')
    (ROOT/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(ROOT/'COMPARISON.md')

if __name__=='__main__':main()
