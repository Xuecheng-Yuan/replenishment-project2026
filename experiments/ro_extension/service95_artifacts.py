"""Build the service>=95% report, comparison CSVs and plots from frozen runs."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from service95_revision import ROOT, HOME, LABELS


def main():
    out=HOME/'outputs_v4';viz=HOME/'visualization';viz.mkdir(exist_ok=True)
    current=pd.read_csv(out/'summary.csv').query("scope=='all_28_periods'")
    prior=pd.read_csv(ROOT/'experiments/full60_revision/new_positive/outputs_v4/summary.csv').query("scope=='all_28_periods'")
    old=pd.read_csv(ROOT/'experiments/demand_fill_v3/with_ro/outputs_v4/summary.csv').query("scope=='all_28_periods'")
    versions=pd.concat([old.assign(version='历史需求满足率版'),prior.assign(version='上一轮正库存版_PR7'),current.assign(version='当前服务95%–95.5%版')],ignore_index=True)
    versions.to_csv(out/'version_comparison.csv',index=False,encoding='utf-8-sig')
    config=json.loads((out/'frozen_parameters.json').read_text(encoding='utf-8'))
    audit=json.loads((out/'test_audit.json').read_text(encoding='utf-8'))
    folds=pd.read_csv(out/'candidate_validation_by_fold.csv')
    selected=folds[folds.apply(lambda r:r.candidate==config['selected'][r.method]['candidate'],axis=1)]
    selected.to_csv(out/'selected_validation_by_fold.csv',index=False,encoding='utf-8-sig')
    binding=[]
    for m in LABELS:
        f=pd.read_csv(out/f'detail_{m}.csv.gz');p=np.load(out/f'policy_{m}.npz')
        fc=f.forecast_h1.to_numpy().reshape(28,6000)
        h=p['high'] if 'high' in p.files else p['low']
        binding.append(dict(method=m,reserve_binding_share=float(np.mean(p['reserve']>=fc*h)),
            reserve_sum=float(p['reserve'].sum()),level_min=float(p['low'].min()),level_max=float(p['low'].max())))
    pd.DataFrame(binding).to_csv(out/'protection_diagnostics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(binding).to_csv(out/'risk_profiles.csv',index=False,encoding='utf-8-sig')
    from service95_revision import model
    bank=np.load(out/'fitted_model.npz');features,descriptors,xmean=model.feature_matrix(bank['train_y_all60'],bank['train_x_mean'][None])
    items=pd.read_csv(out/'detail_DRO_cluster.csv.gz').query('period==1')[['item_id','category','cluster']].reset_index(drop=True)
    for j,name in enumerate(('log_mean','log_cv','zero_fraction','log_q95_over_mean','log_last8_over_first8')):items[name]=descriptors[:,j]
    for j in range(20):items[f'feature_mean_{j+1}']=xmean[:,j]
    for j in range(8):items[f'clustering_coordinate_{j+1}']=features[:,j]
    items.to_csv(out/'cluster_assignment.csv',index=False,encoding='utf-8-sig')
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    methods=list(LABELS);cur=current.set_index('method').loc[methods]
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    names=[LABELS[m] for m in methods]
    axes[0].bar(names,cur.positive_stock_ratio*100,color='#24785f');axes[0].axhspan(95,95.5,color='#24785f',alpha=.15,label='要求95%–95.5%');axes[0].axhline(95,color='#b44c32',ls='--');axes[0].axhline(95.5,color='#b44c32',ls='--');axes[0].set_ylim(90,100);axes[0].set_ylabel('完整28期正库存比例（%）');axes[0].legend()
    axes[1].bar(names,cur.inventory_demand_ratio,color='#588eae');axes[1].set_ylabel('库存需求比（越低越好）')
    for ax in axes:ax.tick_params(axis='x',rotation=24);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('全部6000商品：先满足95%服务下限，再比较库存需求比')
    fig.savefig(viz/'six_method_comparison.png',dpi=160);plt.close(fig)
    segments=pd.read_csv(out/'summary_by_segment.csv')
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,segment in zip(axes,('hot','long_tail')):
        names_available=segments.segment.unique().tolist()
        actual=segment if segment in names_available else ('爆品' if segment=='hot' else '长尾品')
        block=segments[segments.segment==actual].set_index('method').reindex(methods)
        if block.positive_stock_ratio.isna().all():
            actual=names_available[0 if segment=='hot' else 1];block=segments[segments.segment==actual].set_index('method').reindex(methods)
        ax.bar(names,block.positive_stock_ratio*100,color='#24785f');ax.axhspan(95,95.5,color='#24785f',alpha=.15);ax.axhline(95,color='#b44c32',ls='--');ax.set_ylim(max(0,min(90,float(block.positive_stock_ratio.min()*100)-2)),100);ax.set_title(str(actual));ax.tick_params(axis='x',rotation=24);ax.set_ylabel('正库存比例（%）')
    fig.savefig(viz/'hot_and_long_tail.png',dpi=160);plt.close(fig)
    period=pd.read_csv(out/'results_by_period.csv')
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    for m in methods:
        f=period[period.method==m];ax.plot(f.period,f.positive_stock_ratio*100,label=LABELS[m])
    ax.axhline(95,color='#b44c32',ls='--');ax.set_xlabel('测试周期');ax.set_ylabel('当期正库存商品比例（%）');ax.legend(ncol=3);ax.set_title('当期服务比例：完整28期累计约束与单日指标分别解释')
    fig.savefig(viz/'service_by_period.png',dpi=160);plt.close(fig)
    from service_band_report import write_report
    write_report()
    # The figure dashboard links to the full offline interactive view.
    (viz/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>服务95%–95.5%重跑结果</title><style>body{max-width:1200px;margin:30px auto;font:17px Microsoft YaHei;color:#19342f}img{width:100%}a{color:#176a53}</style><h1>服务95%–95.5%：六方法重跑</h1><p><a href="../../../汇报成果/2026-10-15/动画/index.html">打开全部商品、爆品、长尾的六策略互动动画</a></p><p>总体服务要求为95%–95.5%。当期、类别与单商品统计用于解释，总体达标以全部6000商品完整28期计数为准。</p>'+''.join(f'<img src="{name}">' for name in ('six_method_comparison.png','hot_and_long_tail.png','service_by_period.png'))+'</html>',encoding='utf-8')
    print('Built service95 report, 18-row version comparison, 42 selected validation records and full-scope plots.')


if __name__=='__main__':main()
