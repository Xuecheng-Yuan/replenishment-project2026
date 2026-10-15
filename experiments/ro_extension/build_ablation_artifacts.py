"""Plots, SKU block bootstrap and category mechanisms for the paired experiment."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from constraint_ablation import HOME
from extend_with_ro import LABELS

def main():
    out=HOME/'outputs';s=pd.read_csv(out/'summary.csv');sku=pd.read_csv(out/'sku_summary.csv')
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    methods=list(LABELS);fig,axs=plt.subplots(1,2,figsize=(13,5))
    for arm,color,offset,title in [('fill','#34698e',-.18,'需求满足率约束'),('positive','#ad694c',.18,'正库存比例约束')]:
        sub=s[s.arm==arm].set_index('method').loc[methods]
        axs[0].bar(np.arange(6)+offset,sub.inventory_demand_ratio,.36,label=title,color=color)
        axs[1].bar(np.arange(6)+offset,sub.train_inventory_ratio,.36,label=title,color=color)
    for ax,title in zip(axs,['冻结后测试：库存需求比','训练内部验证：库存需求比']):
        ax.set_xticks(np.arange(6),[LABELS[m] for m in methods],rotation=23);ax.set_title(title);ax.legend(fontsize=9);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(out/'constraint_only_comparison.png',dpi=170);plt.close(fig)
    mechanism=[]
    for (arm,method,cat),f in sku.groupby(['arm','method','category']):
        mechanism.append(dict(arm=arm,method=method,category=cat,skus=len(f),fill_rate=f.sales.sum()/f.demand.sum(),
            positive_stock_ratio=f.positive_days.sum()/(28*len(f)),ending_units=f.ending.sum(),demand_units=f.demand.sum(),
            inventory_demand_ratio=f.ending.sum()/f.demand.sum()))
    m=pd.DataFrame(mechanism);m.to_csv(out/'category_mechanism.csv',index=False,encoding='utf-8-sig')
    # SKU-block resampling retains all 28 periods per SKU; same draws for all methods.
    rng=np.random.default_rng(20261015);ids=sku.item_id.unique();n=len(ids)
    draws=rng.integers(0,n,size=(1000,n));boot=[];paired={}
    for arm in ('fill','positive'):
        reference=sku[(sku.arm==arm)&(sku.method=='sS')].set_index('item_id').loc[ids]
        denom=reference.demand.to_numpy()[draws].sum(axis=1)
        for method in methods:
            f=sku[(sku.arm==arm)&(sku.method==method)].set_index('item_id').loc[ids]
            diff=(f.ending-reference.ending).to_numpy()[draws].sum(axis=1)/denom
            paired[(arm,method)]=diff
            lo,hi=np.quantile(diff,[.025,.975])
            boot.append(dict(arm=arm,method=method,delta_vs_sS=(f.ending.sum()-reference.ending.sum())/reference.demand.sum(),
                ci95_lower=float(lo),ci95_upper=float(hi),replicates=1000,unit='SKU block (all 28 periods)'))
    b=pd.DataFrame(boot);b.to_csv(out/'paired_bootstrap.csv',index=False,encoding='utf-8-sig')
    delta=paired[('positive','DRO_category')]-paired[('fill','DRO_category')]
    effect=(b[(b.arm=='positive')&(b.method=='DRO_category')].iloc[0].delta_vs_sS-
            b[(b.arm=='fill')&(b.method=='DRO_category')].iloc[0].delta_vs_sS)
    ci=np.quantile(delta,[.025,.975])
    pd.DataFrame([dict(contrast='(category-sS)_positive - (category-sS)_fill',effect=effect,
        ci95_lower=ci[0],ci95_upper=ci[1],replicates=1000)]).to_csv(out/'interaction_effect.csv',index=False,encoding='utf-8-sig')
    appendix=['','## 原排名变化是否在单因素实验中复现','',
        '预先关注原旧版最优的“爆品/长尾DRO”与原新版最优的“传统(s,S)”。以下用相同SKU抽样作1000次配对块bootstrap，保留每个SKU完整28期，计算ΔJ＝类别DRO库存需求比−sS库存需求比。负数表示类别DRO库存更少。',
        '|约束|ΔJ|95% SKU块bootstrap区间|','|---|---:|---:|']
    for arm in ('fill','positive'):
        r=b[(b.arm==arm)&(b.method=='DRO_category')].iloc[0]
        appendix.append(f'|{arm}|{r.delta_vs_sS:+.6f}|[{r.ci95_lower:+.6f}, {r.ci95_upper:+.6f}]|')
    appendix+=['',f'约束切换对两方法相对库存差异的影响为 ΔJ正库存−ΔJ需求＝{effect:+.6f}，95%配对区间[{ci[0]:+.6f}, {ci[1]:+.6f}]。负值表示切换至正库存约束后，类别DRO相对(s,S)的库存表现改善；正值表示变差。',
        '训练约束达标不保证测试达标。所有测试未达标的方法都保留原始结果，并从该约束下“达标最优”比较中排除。','',
        '区间反映在当前商品样本上、假设商品块可交换时的抽样不确定性。它不包含重新训练或重新选参的不确定性，也不能替代未来新测试集。若区间跨0，不能据此确定两方法库存差异的方向。测试服务未达标的方法即使J较低，也不能称为该约束下最优。','',
        '### 爆品与长尾品的服务和库存变化','',
        '|约束|方法|类别|需求满足率|正库存比例|库存需求比|','|---|---|---|---:|---:|---:|']
    for r in m[m.method.isin(['DRO_category','sS'])].itertuples():appendix.append(f'|{r.arm}|{LABELS[r.method]}|{r.category}|{r.fill_rate:.3%}|{r.positive_stock_ratio:.3%}|{r.inventory_demand_ratio:.6f}|')
    (HOME/'REPORT.md').write_text((HOME/'REPORT.md').read_text(encoding='utf-8').split('\n## 原排名变化是否在单因素实验中复现')[0]+'\n'.join(appendix)+'\n',encoding='utf-8')
    historical=[]
    for version,name in [('demand_fill_v3','旧版：需求满足率'),('rolling_dro_v3','被替代32期新版：正库存比例')]:
        frame=pd.read_csv(HOME.parent/version/'with_ro/outputs_v4/summary.csv').query("scope=='all_28_periods'")
        for r in frame.to_dict('records'):historical.append(dict(version=name,**r))
    design=json.loads((out/'frozen_design.json').read_text(encoding='utf-8'))
    selected=[dict(**r,recipe=design['recipe'][r['candidate']]) for r in design['selected']]
    payload=dict(historical=historical,controlled=s.to_dict('records'),category=m.to_dict('records'),
        bootstrap=b.to_dict('records'),selected=selected,labels=LABELS,effect=dict(value=effect,lower=ci[0],upper=ci[1]))
    (out/'workbook_data.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(HOME/'REPORT.md')

if __name__=='__main__':main()
