"""Generate portable six-policy plots and a self-contained HTML view."""
from pathlib import Path
import sys
import html, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extend_with_ro import ROOT, VERSIONS, LABELS

plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
full60='--full60' in sys.argv
for version in (('rolling_dro_v3',) if full60 else VERSIONS):
    home=ROOT/'experiments/full60_revision/new_positive' if full60 else ROOT/'experiments'/version/'with_ro'; out=home/'outputs_v4'
    vis=home/'visualization'; vis.mkdir(exist_ok=True)
    old=version==VERSIONS[0]
    data=pd.read_csv(out/'summary.csv').query("scope=='all_28_periods'")
    metric='fill_rate' if old else 'positive_stock_ratio'
    title='需求满足率版' if old else ('正库存天数版（完整60期训练）' if full60 else '正库存天数版')
    names=[LABELS[m] for m in data.method]
    colors=['#34698e','#7595ac','#3b8576','#a97a38','#a4a5a7','#a34c46']
    fig,axs=plt.subplots(1,2,figsize=(13,4.8))
    axs[0].bar(names,data.inventory_demand_ratio,color=colors)
    axs[0].set_title('库存需求比（越低越好）')
    for i,v in enumerate(data.inventory_demand_ratio): axs[0].text(i,v+.004,f'{v:.6f}',ha='center',fontsize=9)
    axs[0].set_ylim(0,data.inventory_demand_ratio.max()*1.16)
    axs[1].bar(names,data[metric]*100,color=colors)
    axs[1].axhline(95,color='#a34c46',linestyle='--',label='95%')
    if not old: axs[1].axhspan(95,95.5,color='#87ac91',alpha=.3,label='95%–95.5%')
    axs[1].set_ylim(min(94,data[metric].min()*100-1), max(100,data[metric].max()*100+1))
    axs[1].set_title(('需求满足率' if old else '正库存天数比例')+'（%）')
    for i,v in enumerate(data[metric]*100): axs[1].text(i,v+.1,f'{v:.3f}%',ha='center',fontsize=9)
    for ax in axs:
        ax.tick_params(axis='x',labelrotation=22); ax.spines[['top','right']].set_visible(False)
    fig.suptitle(title+'：三种DRO、传统(s,S)、SAA及传统区间RO')
    fig.tight_layout(); fig.savefig(vis/'six_method_comparison.png',dpi=160); plt.close(fig)
    periods=pd.read_csv(out/'results_by_period.csv')
    fig,axs=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for method,color in zip(data.method,colors):
        p=periods[periods.method==method]
        axs[0].plot(p.period,p[metric]*100,label=LABELS[method],color=color)
        axs[1].plot(p.period,p.inventory_demand_ratio,label=LABELS[method],color=color)
    axs[0].axhline(95,linestyle='--',color='gray');axs[0].set_ylabel('当期服务比例（%）')
    axs[1].set_ylabel('当期库存需求比');axs[1].set_xlabel('测试周期');axs[0].legend(ncol=3,fontsize=9)
    fig.tight_layout();fig.savefig(vis/'period_comparison.png',dpi=160);plt.close(fig)
    table=data[['method','fill_rate','positive_stock_ratio','inventory_demand_ratio','lost_units']].copy()
    table.method=table.method.map(LABELS)
    for col in ('fill_rate','positive_stock_ratio'):table[col]=table[col].map(lambda x:f'{x:.4%}')
    table.inventory_demand_ratio=table.inventory_demand_ratio.map(lambda x:f'{x:.6f}')
    table.columns=['方法','需求满足率','正库存天数比例','库存需求比↓','损失需求']
    records=[dict(method=LABELS[r.method],fill_rate=r.fill_rate,positive_stock_ratio=r.positive_stock_ratio,inventory_demand_ratio=r.inventory_demand_ratio) for r in data.itertuples()]
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>六种补货方法比较</title>
<style>body{font:16px "Microsoft YaHei",sans-serif;color:#17324d;max-width:1200px;margin:40px auto;padding:0 24px}table{border-collapse:collapse;width:100%}td,th{text-align:right;padding:12px;border-bottom:1px solid #ddd}td:first-child,th:first-child{text-align:left}img{width:100%}a{color:#34698e}.bar{background:#34698e;height:24px;margin:6px 0;color:white;min-width:100px;padding:4px;white-space:nowrap}select{font:inherit;padding:5px}</style>'''
    page+=f'<h1>{title}：六种方法结果</h1><p>完整6000商品×28天测试。服务要求：'+('需求满足率≥95%。' if old else '正库存商品天比例95%–95.5%。')+'</p>'
    page+='<p>库存需求比＝累计期末库存/累计需求，越低越好。RO采用需求区间最坏情形，Γ仅在训练集选取。</p>'+table.to_html(index=False,border=0)
    page+='<p><a href="../REPORT.md">数学模型与报告</a>　<a href="../outputs_v4/replenishment_comparison.xlsx">下载Excel</a>　<a href="../../../../version_comparison/COMPARISON.md">新旧版比较</a></p>'
    page+='<h2>切换指标</h2><select id="metric"><option value="inventory_demand_ratio">库存需求比</option><option value="fill_rate">需求满足率</option><option value="positive_stock_ratio">正库存天数比例</option></select><div id="bars"></div>'
    page+='<img alt="六种方法库存与服务比较" src="six_method_comparison.png"><img alt="逐期比较" src="period_comparison.png">'
    page+='<script>const data='+json.dumps(records,ensure_ascii=False)+''';function draw(){let k=document.getElementById('metric').value,max=Math.max(...data.map(r=>r[k]));document.getElementById('bars').replaceChildren(...data.map(r=>{let e=document.createElement('div');e.className='bar';e.style.width=(r[k]/max*90)+'%';e.textContent=r.method+' '+(k==='inventory_demand_ratio'?r[k].toFixed(6):(r[k]*100).toFixed(4)+'%');return e}));}document.getElementById('metric').addEventListener('change',draw);draw();</script></html>'''
    page=page.replace('../../../../version_comparison/','../../../version_comparison/')
    (vis/'index.html').write_text(page,encoding='utf-8')
    print(vis)
