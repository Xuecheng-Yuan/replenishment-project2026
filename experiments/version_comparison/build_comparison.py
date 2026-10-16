"""Compare preserved demand-fill results with the current full-60 positive arm."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
LABELS={'DRO_cluster':'特征聚类DRO','DRO_global':'不分组DRO','DRO_category':'爆品/长尾DRO','sS':'传统(s,S)','SAA':'SAA','RO':'传统区间RO'}

def main():
    old=pd.read_csv(ROOT.parent/'demand_fill_v3/with_ro/outputs_v4/summary.csv').query("scope=='all_28_periods'").set_index('method')
    new=pd.read_csv(ROOT.parent/'service95_revision/outputs_v4/summary.csv').query("scope=='all_28_periods'").set_index('method')
    valid=new[new.positive_stock_ratio.between(.95,.955)].sort_values('inventory_demand_ratio')
    headline=(f'当前新版测试达标{len(valid)}种方法；达标方法中库存需求比最低为{LABELS[valid.index[0]]}。' if len(valid)
        else f'当前新版六方法测试正库存比例为{new.positive_stock_ratio.min():.4%}–{new.positive_stock_ratio.max():.4%}，均未达到95%–95.5%目标。')
    rows=[]
    lines=['# 旧版与当前服务95%–95.5%新版结果对比','',headline,'',
        '新版用全部60期最终拟合预测器、聚类与扩展窗残差。预测参数来自三轮训练验证，反馈参数用训练29–60拟合、1–28回溯重启留出，要求完整28期P在95%–95.5%。三轮向前/重启12期前缀作为诊断。独立测试文件28期从Period_1重新起算，反馈仅读取已经发生需求。回溯留出不是向前预测性能估计。','',
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
    single_file=ROOT.parent/'full60_revision/new_positive/outputs_v4/summary.csv'
    if single_file.exists():
        single=pd.read_csv(single_file).query("scope=='all_28_periods'").set_index('method')
        lines+=['','## 同为完整60期最终拟合：本次与PR7的变化','',
            'PR7在三轮验证中继承拟合历史库存，并采用静态水位；本次加入周期重启回溯留出与因果服务预算反馈，保留跨训练切片风险包络并取消静态均值储备。两版均用60期最终拟合，测试与服务目标一致；差异来自整套控制器与选参流程。','',
            '|方法|PR7正库存比例|本次正库存比例|PR7库存需求比|本次库存需求比|','|---|---:|---:|---:|---:|']
        for method in LABELS:
            a,b=single.loc[method],new.loc[method]
            lines.append(f'|{LABELS[method]}|{a.positive_stock_ratio:.4%}|{b.positive_stock_ratio:.4%}|{a.inventory_demand_ratio:.6f}|{b.inventory_demand_ratio:.6f}|')
    lines+=['','库存需求比越低越好，但未满足相应服务约束时不能据此认定更优。参数数值按声明的训练规则确定；本测试文件此前已多次用于模型修订，本次属于重复测试。','',
        '- [本次具体修改、数学模型和全60期训练证据](../service95_revision/REPORT.md)',
        '- [固定其他条件的服务约束单因素报告](../full60_revision/REPORT.md)',
        '- [当前新版六方法Excel及三版对比](../service95_revision/outputs_v4/replenishment_comparison.xlsx)',
        '- [旧版六方法Excel](../demand_fill_v3/with_ro/outputs_v4/replenishment_comparison.xlsx)',
        '- [被替代32期版本的历史比较](HISTORICAL_COMPARISON_32.md)（仅作追溯）。']
    if single_file.exists():lines.append('- [上一次60期一次划分结果](../full60_revision/history/single_holdout_32_28/README.md)。')
    (ROOT/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(ROOT/'COMPARISON.md')

if __name__=='__main__':main()
