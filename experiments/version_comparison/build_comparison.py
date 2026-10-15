"""Compare preserved demand-fill results with the current full-60 positive arm."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
LABELS={'DRO_cluster':'特征聚类DRO','DRO_global':'不分组DRO','DRO_category':'爆品/长尾DRO','sS':'传统(s,S)','SAA':'SAA','RO':'传统区间RO'}

def main():
    old=pd.read_csv(ROOT.parent/'demand_fill_v3/with_ro/outputs_v4/summary.csv').query("scope=='all_28_periods'").set_index('method')
    new=pd.read_csv(ROOT.parent/'full60_revision/new_positive/outputs_v4/summary.csv').query("scope=='all_28_periods'").set_index('method')
    rows=[]
    lines=['# 旧版与当前全60期新版结果对比','',
        '新版已经修正为训练集1–32期拟合、33–60期向前验证选参，然后用全部60期重新拟合预测器、聚类与风险残差。原29–60期拟合版本被替代。','',
        '**当前新版六种方法测试正库存比例均未达到95%–95.5%，不能宣布达标最优方法。** 旧版与新版服务指标不同，且最终拟合与选参流程不同；以下是完整方案比较，不是单因素因果比较。','',
        '|方法|旧版需求满足率|旧版正库存比例|旧版库存需求比|全60期新版需求满足率|全60期新版正库存比例|全60期新版库存需求比|新版达标|',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for method in LABELS:
        a,b=old.loc[method],new.loc[method];valid=.95<=b.positive_stock_ratio<=.955
        rows.append(dict(method=method,old_demand_fill=a.fill_rate,new_demand_fill=b.fill_rate,
            old_positive_stock_service=a.positive_stock_ratio,new_positive_stock_service=b.positive_stock_ratio,
            old_inventory_demand_ratio=a.inventory_demand_ratio,new_inventory_demand_ratio=b.inventory_demand_ratio,new_test_feasible=valid))
        lines.append(f'|{LABELS[method]}|{a.fill_rate:.4%}|{a.positive_stock_ratio:.4%}|{a.inventory_demand_ratio:.6f}|{b.fill_rate:.4%}|{b.positive_stock_ratio:.4%}|{b.inventory_demand_ratio:.6f}|'+('是' if valid else '否')+'|')
    pd.DataFrame(rows).to_csv(ROOT/'results_comparison.csv',index=False,encoding='utf-8-sig')
    lines+=['','库存需求比越低越好，但未满足相应服务约束时，不能据此认定更优。训练验证达标不保证重新拟合后测试达标，本次未用测试调整参数。','',
        '- [本次具体修改和全60期训练证明](../full60_revision/CHANGE_REPORT.md)',
        '- [固定其他条件的服务约束单因素报告](../full60_revision/REPORT.md)',
        '- [当前新版六方法Excel](../full60_revision/new_positive/outputs_v4/replenishment_comparison.xlsx)',
        '- [旧版六方法Excel](../demand_fill_v3/with_ro/outputs_v4/replenishment_comparison.xlsx)',
        '- [被替代32期版本的历史比较](HISTORICAL_COMPARISON_32.md)（仅作追溯）。']
    (ROOT/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(ROOT/'COMPARISON.md')

if __name__=='__main__':main()
