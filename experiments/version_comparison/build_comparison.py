"""Compare the preserved demand-fill version and current positive-stock version."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
EXPERIMENTS=ROOT.parent
LABELS={'DRO_cluster':'特征聚类DRO','DRO_global':'不分组DRO','DRO_category':'爆品/长尾DRO','sS':'传统(s,S)','SAA':'SAA','RO':'传统区间RO'}


def load(folder):
    out=EXPERIMENTS/folder/'with_ro/outputs_v4'
    s=pd.read_csv(out/'summary.csv').query("scope=='all_28_periods'").set_index('method')
    if 'positive_stock_ratio' not in s:
        stock=pd.read_csv(out/'positive_stock_service.csv').query("scope=='all_28_periods'").set_index('method')
        s['positive_stock_ratio']=stock.positive_stock_ratio
    return s


def main():
    old=load('demand_fill_v3'); new=load('rolling_dro_v3')
    rows=[]
    for method in LABELS:
        a=old.loc[method]; b=new.loc[method]
        rows.append(dict(method=method,old_demand_fill=a.fill_rate,new_demand_fill=b.fill_rate,
            old_positive_stock_service=a.positive_stock_ratio,new_positive_stock_service=b.positive_stock_ratio,
            old_inventory_demand_ratio=a.inventory_demand_ratio,new_inventory_demand_ratio=b.inventory_demand_ratio,
            inventory_ratio_relative_change=b.inventory_demand_ratio/a.inventory_demand_ratio-1,
            old_avg_inventory_per_sku=a.avg_inventory_per_sku,new_avg_inventory_per_sku=b.avg_inventory_per_sku,
            old_lost_units=a.lost_units,new_lost_units=b.lost_units))
    frame=pd.DataFrame(rows)
    frame.to_csv(ROOT/'results_comparison.csv',index=False,encoding='utf-8-sig')
    lines=['# 两版补货模型对比','','## 结论','',
      '旧版要求销量/需求≥95%，新版要求期末正库存商品－天占比在95%–95.5%。本次两版各增加传统区间RO，共六种方法。旧版最优仍为爆品/长尾DRO，新版最优仍为传统(s,S)。旧版较低的库存需求比对应不同的服务口径，不能据此判断它在新服务要求下更优。','',
      '两版均使用相同原始训练集和测试集，6000个商品、完整28天测试、相同初始库存和零初始在途。提前期1、失销不结转、预测4天且仅执行最近1天。共同库存指标为累计期末库存÷累计需求，越低越好。','',
      '## 定义及实现差异','',
      '| 项目 | 旧版：需求满足率约束 | 新版：正库存服务约束 |',
      '|---|---|---|',
      '| 服务定义 | 总销量÷总需求，需求加权 | 期末库存>0的商品－天数÷168,000，商品－天等权 |',
      '| 服务范围 | ≥95%，无上界 | 95%–95.5%，要求靠近95% |',
      '| 期末恰好为0 | 可能仍满足全部需求 | 此商品－天记为0 |',
      '| 共同优化目标 | 库存需求比最小 | 库存需求比最小 |',
      '| DRO约束 | 最坏分布期望失销量≤(1−β)×期望需求 | 最坏分布正库存概率≥β |',
      '| 水位解法 | 样本区间端点stop-loss函数二分求根 | 上端点顺序统计量，整数目标floor(预测×水位)+1 |',
      '| 参数选择 | 重启验证需求满足率≥95%，另有前向服务门槛 | 重启验证正库存服务95%–95.5%，前向折仅诊断 |',
      '| 最终模型拟合 | 校准后用全部60期重新拟合 | 拟合第29–60期，校准第1–28期，保持同一模型 |',
      '| 传统(s,S)和SAA | 原分位水位 | 增加训练选择的安全倍率κ |',
      '| 分组方案 | 特征聚类、全局、爆品/长尾 | 同三种方案，但拟合块改变，标签与残差重新估计 |','',
      '新版同时改变服务定义、鲁棒约束、最终拟合流程和基线校准。因此下面是两套完整实现的对比，不是只改变一个服务指标的消融实验；库存变化不能全部归因于单一因素。','',
      '## 完整测试结果','',
      '| 方法 | 旧需求满足率 | 旧正库存比例 | 旧库存需求比 | 新正库存比例 | 新需求满足率 | 新库存需求比 | 库存比相对变化 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in frame.itertuples():
        lines.append(f'| {LABELS[r.method]} | {r.old_demand_fill:.2%} | {r.old_positive_stock_service:.2%} | {r.old_inventory_demand_ratio:.6f} | {r.new_positive_stock_service:.2%} | {r.new_demand_fill:.2%} | {r.new_inventory_demand_ratio:.6f} | {r.inventory_ratio_relative_change:+.2%} |')
    lines += ['', '相对变化＝新版库存需求比÷旧版库存需求比−1，正值代表库存需求比上升。销量/需求和正库存比例都同时列出，避免把两种服务水平混为一谈。原始精度、平均库存及丢失需求对照见[results_comparison.csv](results_comparison.csv)。','',
      '## 各版内部的策略排序','']
    for title,s,mask in [('旧版',old,old.fill_rate>=.95),('新版',new,new.positive_stock_ratio.between(.95,.955))]:
        eligible=s[mask].sort_values('inventory_demand_ratio')
        lines += [title+'：'+' → '.join(LABELS[m] for m in eligible.index)+'。']
    lines += ['', '同一版内部，在共同数据、共同服务口径下比较更有意义。新版三种DRO中，特征聚类DRO库存需求比最低，但比传统(s,S)高6.80%；旧版爆品/长尾DRO比传统(s,S)低2.18%。','',
      '## 新增传统RO','',
      '传统RO采用由训练样本外相对需求均值μ、标准差σ构造的区间集合[max(0,μ−Γσ),μ+Γσ]。Γ只在训练集选择，旧版Γ=1.098，新版Γ=4.406。旧版对集合内每个需求点限制失销比例，新版要求整数目标库存严格大于区间上界。DRO则约束分布模糊集内的最坏期望或概率。两版采用各自原有的服务校准规则和最终拟合流程。','',
      '旧版RO需求满足率96.7100%，正库存比例74.2548%，库存需求比0.174962；新版RO需求满足率98.9711%，正库存比例95.0077%，库存需求比0.589326。新版RO高于特征聚类DRO和传统(s,S)的库存需求比，低于不分组DRO。六方法的其余五种原始轨迹保持字节一致。','',
      '- [旧版六方法报告](../demand_fill_v3/with_ro/REPORT.md)及[Excel](../demand_fill_v3/with_ro/outputs_v4/replenishment_comparison.xlsx)。',
      '- [新版六方法报告](../rolling_dro_v3/with_ro/REPORT.md)及[Excel](../rolling_dro_v3/with_ro/outputs_v4/replenishment_comparison.xlsx)。',
      '- [RO运行说明与文件导航](../ro_extension/README.md)。','',
      '## 文件与复现','',
      '- [旧版完整独立文件夹](../demand_fill_v3/README.md)：原代码、交付报告、数学报告、Excel、图表、冻结参数和全部结果。','- [新版完整文件夹](../rolling_dro_v3/README.md)：当前正库存服务版本。','- [旧版Excel](../demand_fill_v3/outputs_v4/replenishment_comparison.xlsx)与[新版Excel](../rolling_dro_v3/outputs_v4/replenishment_comparison.xlsx)。','',
      '旧版从提交f569b116d7f13bee0cb2d34c5242c8240e9dda88恢复，原数值报告和Excel保持字节一致。导航链接适配独立目录；归档文件的来源与哈希在ARCHIVE_PROVENANCE.json。','',
      '从仓库根目录执行`python experiments/version_comparison/build_comparison.py`可重建本对比，不重新训练模型。两版代码各自写入本目录的outputs_v4，互不覆盖。历史测试曾参与早期偏差诊断，两版结果均应视为修订后的重复评估。']
    (ROOT/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(ROOT/'COMPARISON.md')


if __name__=='__main__':
    main()
