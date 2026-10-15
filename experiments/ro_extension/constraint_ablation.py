"""Paired finite-policy experiment: ONLY the outer service feasibility rule changes.

The candidate controller library, forecast, groups, integer guard, fit window,
validation data, objective, tie breaking and test replay are common to both arms.
This tests the effect within this library, not a universal optimality theorem.
"""
from pathlib import Path
import gzip, hashlib, json, sys
import numpy as np
import pandas as pd
from extend_with_ro import ROOT, LABELS, box

HOME=ROOT/'experiments/full60_revision'

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def pick(frame, arm):
    mask=frame.fill_rate>=.95 if arm=='fill' else frame.positive_stock_ratio.between(.95,.955)
    eligible=frame[mask]
    if eligible.empty:raise RuntimeError(f'No training-feasible candidate: {arm}')
    return eligible.sort_values(['inventory_demand_ratio','candidate'],kind='stable').iloc[0]

def main():
    sys.path.insert(0,str(ROOT/'experiments/rolling_dro_v3'))
    import run_v3 as pipeline
    import model_v2 as model
    from run_v2 import ss_quantiles
    out=HOME/'outputs';out.mkdir(parents=True,exist_ok=True)
    data=pipeline.verified_data(ROOT/'data/raw',out,False)
    y,x=data['train_y'].astype(float),data['train_x'].astype(float)
    init=data['initial_inventory'].astype(float)
    # Chronological split: no validation demand or feature enters fitting.
    cut=32
    labels=model.cluster(y[:cut],x[:cut])[0]
    forecast_validation=[]
    for blend in (0.,.5,1.):
        for window in (3,6,12):
            for prior in (0.,.5,1.,3.):
                candidate=dict(blend=blend,window=window,prior=prior)
                fm=model.Forecaster(y[:cut],labels,**candidate)
                pred=model.rolling_predictions(fm,y[cut:],cut,history=y[:cut])[:,0]
                forecast_validation.append(dict(**candidate,wape=float(abs(pred-y[cut:]).sum()/y[cut:].sum())))
    settings={k:min(forecast_validation,key=lambda r:r['wape'])[k] for k in ('blend','window','prior')}
    pd.DataFrame(forecast_validation).to_csv(out/'forecast_validation.csv',index=False,encoding='utf-8-sig')
    fitted=model.Forecaster(y[:cut],labels,**settings)
    fc=model.make_forecasts(fitted,y[cut:],cut,history=y[:cut])
    residual=model.residual_oof(y[:cut],labels,**settings)[0]
    print('[chronological] fit 1-32; validate 33-60; forecast',settings,flush=True)
    groups=pipeline.variant_labels(labels,data['is_hot'])
    recipes=[]; rows=[];lowbank=[];highbank=[]
    def add(method,low,high=None,**recipe):
        index=len(rows)
        r=model.simulate(y[cut:],init,fc,low,high,positive_guard=True)['summary']
        rows.append(dict(candidate=index,method=method,**recipe,**r))
        recipes.append(dict(candidate=index,method=method,**recipe))
        lowbank.append(np.asarray(low,float));highbank.append(np.asarray(low if high is None else high,float))
    # Exactly the same union of controllers is offered to both service arms.
    # Fixed chance-DRO controller generator: the outer service rule is the treatment.
    for method in ('DRO_cluster','DRO_global','DRO_category','SAA'):
        for beta in (.5,.6,.7,.75,.8,.85,.9,.95,.96,.97,.98,.99,.995,.999):
            for radius in ((0.,) if method=='SAA' else (.005,.01,.025,.05,.075,.1,.15,.2,.3,.5)):
                level=model.risk_levels(residual,labels if method=='SAA' else groups[method],beta,radius)[0]
                for scale in ((.75,1.,1.125,1.25) if method=='SAA' else (.75,1.,1.25)):
                    add(method,level*scale,beta=beta,radius=radius,scale=scale)
        print('[library]',method,len(rows),flush=True)
    for ps in (.5,.6,.7,.75,.8,.85,.9,.95,.975,.99,.999):
        for gap in (0.,.025,.05,.1,.15):
            pS=min(ps+gap,.999)
            lo=ss_quantiles(residual,labels,ps);hi=ss_quantiles(residual,labels,pS)
            for scale in np.arange(.75,1.501,.025):
                add('sS',lo*scale,hi*scale,p_s=ps,p_S=pS,scale=float(scale))
    for gamma in np.arange(0,8.0001,.01):
        add('RO',np.full(6000,box(residual,gamma)[1]),gamma=float(gamma))
    validation=pd.DataFrame(rows)
    validation.to_csv(out/'candidate_validation.csv',index=False,encoding='utf-8-sig')
    frozen=[]
    for method in LABELS:
        subset=validation[validation.method==method]
        for arm in ('fill','positive'):
            chosen=pick(subset,arm)
            frozen.append(dict(arm=arm,method=method,candidate=int(chosen.candidate),
                train_fill=float(chosen.fill_rate),train_positive=float(chosen.positive_stock_ratio),
                train_inventory_ratio=float(chosen.inventory_demand_ratio)))
    control=dict(forecast=settings,fit_periods='1-32 (calibration), 1-60 (final)',validation_periods='33-60',
        final_fit_periods=list(range(1,61)),final_fit_observations=int(y.size),training_y_sha256=digest(y),
        refit=True,positive_integer_guard=True,library_size=len(rows),
        residual_sha256=digest(residual),forecast_validation_sha256=digest(fc),labels_sha256=digest(labels),
        initial_sha256=digest(init),objective='min sum ending / sum demand',
        treatment={'fill':'training fill >= .95','positive':'training positive-stock ratio in [.95,.955]'},
        common_tie_break='inventory_demand_ratio then candidate index',selected=frozen,
        interpretation='Controlled effect within one fixed finite controller library; not a universal method ranking.',
        recipe=recipes)
    # Refit every learned component on ALL 60 training periods; recipes frozen.
    full_labels=model.cluster(y,x)[0]
    full_residual=model.residual_oof(y,full_labels,**settings)[0]
    final=model.Forecaster(y,full_labels,**settings)
    full_groups=pipeline.variant_labels(full_labels,data['is_hot'])
    control.update(final_labels_sha256=digest(full_labels),final_residual_sha256=digest(full_residual))
    (out/'frozen_design.json').write_text(json.dumps(control,ensure_ascii=False,indent=2),encoding='utf-8')
    level_cache={}
    def final_levels(recipe):
        method=recipe['method']
        if method=='RO':return np.full(6000,box(full_residual,recipe['gamma'])[1]),None
        if method=='sS':
            return (ss_quantiles(full_residual,full_labels,recipe['p_s'])*recipe['scale'],
                    ss_quantiles(full_residual,full_labels,recipe['p_S'])*recipe['scale'])
        key=(method,recipe['beta'],recipe['radius'])
        if key not in level_cache:
            level_cache[key]=model.risk_levels(full_residual,full_labels if method=='SAA' else full_groups[method],recipe['beta'],recipe['radius'])[0]
        return level_cache[key]*recipe['scale'],None
    # No test outcomes participate in library construction or policy selection.
    test=data['test_y'].astype(float)
    ft=model.make_forecasts(final,test,0)
    result=[];per_sku=[];periods=[];positive_runs={}
    for r in frozen:
        idx=r['candidate']; method=r['method']
        low,high=final_levels(recipes[idx])
        run=model.simulate(test,init,ft,low,high,keep=True,positive_guard=True)
        if r['arm']=='positive':positive_runs[method]=run
        result.append(dict(**r,**run['summary']))
        trace=pd.DataFrame(dict(period=np.repeat(np.arange(1,29),6000),item_id=np.tile(data['item_id'],28),
            demand=test.ravel(),available=run['commencing'].ravel(),sales=run['sold'].ravel(),
            lost=(test-run['sold']).ravel(),ending=run['ending'].ravel(),order=run['orders'].ravel()))
        for h in range(4):
            trace[f'forecast_h{h+1}']=ft[:,h].ravel();trace[f'planned_order_h{h+1}']=run['plans'][:,h].ravel()
        trace.to_csv(out/f'detail_{r["arm"]}_{method}.csv.gz',index=False,encoding='utf-8-sig',compression={'method':'gzip','mtime':0})
        for t in range(28):
            periods.append(dict(arm=r['arm'],method=method,period=t+1,**model.summarize(test[t:t+1],run['sold'][t:t+1],run['ending'][t:t+1],run['commencing'][t:t+1],run['orders'][t:t+1])))
        for i,item in enumerate(data['item_id']):
            per_sku.append(dict(arm=r['arm'],method=method,item_id=item,category='爆品' if data['is_hot'][i] else '长尾品',
                demand=test[:,i].sum(),sales=run['sold'][:,i].sum(),ending=run['ending'][:,i].sum(),
                positive_days=int((run['ending'][:,i]>0).sum()),order=run['orders'][:,i].sum()))
    pd.DataFrame(result).to_csv(out/'summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(periods).to_csv(out/'periods.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(per_sku).to_csv(out/'sku_summary.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(out/'shared_model.npz',labels=labels,relative_residual=residual,forecasts_validation=fc,forecasts_test=ft,initial=init,
        train_y_all60=y,final_labels=full_labels,final_relative_residual=full_residual,
        final_slope=final.slope,final_intercept=final.intercept)
    latest=HOME/'new_positive/outputs_v4';latest.mkdir(parents=True,exist_ok=True)
    pipeline.METHODS=tuple(LABELS)
    pipeline.output_results(latest,positive_runs,test,ft,data['item_id'],full_labels,data['is_hot'])
    positive_frozen=dict(service_metric='positive_stock_ratio',service_lower=.95,service_upper=.955,
        model_fit_periods='1-60',calibration_fit_periods='1-32',calibration_periods='33-60',refit_after_calibration=True,
        forecast=settings,lead_time=1,horizon=4,selected={r['method']:recipes[r['candidate']] for r in frozen if r['arm']=='positive'})
    (latest/'frozen_parameters.json').write_text(json.dumps(positive_frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    wide=pd.read_csv(latest/'decision_comparison_full.csv')
    wide[wide.period==28].to_csv(latest/'latest_decisions.csv',index=False,encoding='utf-8-sig')
    wide[wide.item_id.isin(wide.item_id.unique()[:12])].to_csv(latest/'sample_decisions.csv',index=False,encoding='utf-8-sig')
    for source in [*latest.glob('detail_*.csv'),latest/'decision_comparison_full.csv']:
        with source.open('rb') as f,gzip.GzipFile(filename=str(source)+'.gz',mode='wb',mtime=0) as z:
            import shutil
            shutil.copyfileobj(f,z)
    (HOME/'.gitignore').write_text('new_positive/outputs_v4/detail_*.csv\nnew_positive/outputs_v4/decision_comparison_full.csv\n**/*.inspect.ndjson\n',encoding='utf-8')
    make_report()
    print(pd.DataFrame(result)[['arm','method','fill_rate','positive_stock_ratio','inventory_demand_ratio']].to_string(index=False),flush=True)

def make_report():
    out=HOME/'outputs';s=pd.read_csv(out/'summary.csv')
    lines=['# 只改变服务约束的对照实验','', '## 实验结论','']
    for arm,title in [('fill','需求满足率≥95%'),('positive','正库存天数比例95%–95.5%')]:
        a=s[s.arm==arm].copy();a['test_feasible']=a.fill_rate>=.95 if arm=='fill' else a.positive_stock_ratio.between(.95,.955)
        train=a.sort_values('train_inventory_ratio');test=a[a.test_feasible].sort_values('inventory_demand_ratio')
        lines.append(f'{title}：训练验证最优为{LABELS[train.iloc[0].method]}；测试达标方法中库存需求比最低为'+(LABELS[test.iloc[0].method] if len(test) else '无达标方法')+'。')
    winners=[]
    for arm in ('fill','positive'):
        a=s[s.arm==arm];valid=a[a.fill_rate>=.95] if arm=='fill' else a[a.positive_stock_ratio.between(.95,.955)]
        winners.append(None if valid.empty else valid.sort_values('inventory_demand_ratio').iloc[0].method)
    conclusion=('至少一组测试没有达标方法，无法比较两组的达标最优方法。' if None in winners else
        ('本次在固定候选策略库内，单独切换服务约束后，测试最优方法发生变化。' if winners[0]!=winners[1] else '本次固定其他条件后，两种服务约束下测试最优方法相同。原新旧两版排名变化不能仅由服务约束解释，拟合与策略实现变化也需要考虑。'))
    lines+=['',conclusion,
        '这是对当前数据和预先声明的有限策略库的受控实证检验。它检验约束变化在本实验内的作用，不能证明某种方法对所有数据都最优。测试已被此前多次评估，本实验属于既有测试集上的事后机制检验。','',
        '|约束|方法|训练库存需求比|测试需求满足率|测试正库存比例|测试库存需求比↓|', '|---|---|---:|---:|---:|---:|']
    for r in s.itertuples():lines.append(f'|{r.arm}|{LABELS[r.method]}|{r.train_inventory_ratio:.6f}|{r.fill_rate:.4%}|{r.positive_stock_ratio:.4%}|{r.inventory_demand_ratio:.6f}|')
    lines+=['','## 如何排除混杂因素','',
        '两组共用第1–32期拟合、第33–60期按时间向前验证；预测超参数也仅按此向前验证误差选取。选参结束后，预测器、特征聚类和样本外残差库全部用训练集第1–60期重新拟合，再评估测试集。两组使用相同初始库存、整数处理、L=1、H=4、失销处理、库存需求比目标及排序规则。',
        '每种方法的候选参数和候选补货控制器完全相同，同一候选在训练集只计算一次，结果同时交给两种约束筛选。DRO控制器的构造公式保持一致（正库存机会约束水位），这里只改变外层实际服务可行性规则。因此这是外层服务指标的单因素检验，而不是把旧版stop-loss DRO公式与新版chance-DRO公式再次混在一起比较。',
        '需求满足率组选择训练F≥95%的候选；正库存组选择训练95%≤P≤95.5%的候选。两组随后都最小化训练库存需求比，平局按同一候选编号排序。约束改变后最优参数发生变化，是处理作用的结果，不是另一个被人为改变的因素。',
        '参数选择只用训练数据。frozen_design.json在测试回放前写入，保存共享数组SHA256、候选库和最终编号。candidate_validation.csv保存同一个候选在两种服务指标下的表现。',
        '', '## 数学表达与作用机制','',
        r'固定每种方法的策略集合 $\Pi_m$，数据和预测器也固定。令训练验证指标 $J(\pi)=\sum I/\sum d$，$F(\pi)=\sum s/\sum d$，$P(\pi)=\sum\mathbf1\{I>0\}/(NT)$。',
        r'需求约束组：$\pi_m^F=\arg\min_{\pi\in\Pi_m}\{J(\pi):F(\pi)\ge0.95\}$。',
        r'正库存约束组：$\pi_m^P=\arg\min_{\pi\in\Pi_m}\{J(\pi):0.95\le P(\pi)\le0.955\}$。',
        '两式唯一差别是可行域。F按需求量加权，而P按商品天等权，并把恰好售罄记为0。可行域变化使同一策略库中的最优控制器、各方法库存需求比以及方法排序可能改变。',
        '补货仍在本期需求观察后形成未来四期计划，仅执行最近一期订单。库内DRO为分组W∞机会水位，sS为双水位，SAA为零半径水位，RO为均值±Γ标准差箱式集合。候选β、半径、倍率、分位数或Γ均在两组完全一致。',
        '', '## 商品结构证据','',
        'sku_summary.csv逐商品保存销量、需求、累计库存、正库存天数和订单量，可分别按爆品、长尾品核对服务变化来自哪些商品。训练与测试达标状态均如实报告，测试不达标的方法不参与“达标最优”排序。',
        '', '## 复现','', '```sh','python experiments/ro_extension/constraint_ablation.py','python experiments/ro_extension/verify_ablation.py','```',
        '完整双组×六方法轨迹为outputs/detail_*.csv.gz。原始新旧两版六方法交付仍独立保留，此受控实验不覆盖历史结果。']
    (HOME/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
