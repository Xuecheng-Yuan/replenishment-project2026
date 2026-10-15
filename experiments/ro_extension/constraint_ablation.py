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
FOLDS=((24,36),(36,48),(48,60))

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def pick(frame, arm):
    mask=frame.fill_rate>=.95 if arm=='fill' else frame.positive_stock_ratio.between(.95,.955)
    eligible=frame[mask]
    if eligible.empty:raise RuntimeError(f'No training-feasible candidate: {arm}')
    return eligible.sort_values(['inventory_demand_ratio','candidate'],kind='stable').iloc[0]

def aggregate_summaries(records):
    """Pool primitive totals across validation episodes, rather than average ratios."""
    totals={k:sum(r[k] for r in records) for k in (
        'positive_stock_sku_periods','total_sku_periods','ending_inventory_units',
        'demand_units','sales_units','lost_units','order_units','periods')}
    n=records[0]['skus'];T=totals['periods'];denom=totals['demand_units']
    return dict(**totals,skus=n,fill_rate=totals['sales_units']/denom,
        positive_stock_ratio=totals['positive_stock_sku_periods']/totals['total_sku_periods'],
        inventory_demand_ratio=totals['ending_inventory_units']/denom,
        cycle_service=sum(r['cycle_service']*r['total_sku_periods'] for r in records)/totals['total_sku_periods'],
        avg_inventory_per_sku=sum(r['avg_inventory_per_sku']*r['periods'] for r in records)/T,
        avg_ending_inventory_per_sku=totals['ending_inventory_units']/(T*n),
        final_stock=records[-1]['final_stock'],final_pipeline=records[-1]['final_pipeline'])

def validation_initial_stock(y,initial,forecast,low,high=None):
    """State after a fit-window replay; only the next arrival joins inventory.

    The historical replay establishes state, and contributes no validation score.
    Only the first planned order is executed, as in the four-horizon simulator.
    """
    stock=initial.copy();pipeline=np.zeros_like(stock)
    for t in range(len(y)):
        stock=np.maximum(stock+pipeline-y[t],0.)
        upper=np.floor(forecast[t,0]*(low if high is None else high))+1
        pipeline=np.maximum(0.,np.ceil(upper-stock))
        if high is not None:pipeline=np.where(stock<np.floor(forecast[t,0]*low)+1,pipeline,0.)
    return stock+pipeline

def main():
    sys.path.insert(0,str(ROOT/'experiments/rolling_dro_v3'))
    import run_v3 as pipeline
    import model_v2 as model
    from run_v2 import ss_quantiles
    out=HOME/'outputs';out.mkdir(parents=True,exist_ok=True)
    data=pipeline.verified_data(ROOT/'data/raw',out,False)
    y,x=data['train_y'].astype(float),data['train_x'].astype(float)
    init=data['initial_inventory'].astype(float)
    if y.shape!=(60,6000) or data['test_y'].shape!=(28,6000):
        raise ValueError('Expected the original train60/test28 workbooks with 6000 items')
    fold_data=[]
    for fold,(cut,end) in enumerate(FOLDS,1):
        labels=model.cluster(y[:cut],x[:cut])[0]
        fold_data.append(dict(fold=fold,cut=cut,end=end,labels=labels))
    forecast_validation=[];forecast_by_fold=[]
    for blend in (0.,.5,1.):
        for window in (3,6,12):
            for prior in (0.,.5,1.,3.):
                candidate=dict(blend=blend,window=window,prior=prior)
                errors=[];demands=[]
                for f in fold_data:
                    cut,end=f['cut'],f['end']
                    fm=model.Forecaster(y[:cut],f['labels'],**candidate)
                    pred=model.rolling_predictions(fm,y[cut:end],cut,history=y[:cut])[:,0]
                    error=float(abs(pred-y[cut:end]).sum());demand=float(y[cut:end].sum())
                    forecast_by_fold.append(dict(**candidate,fold=f['fold'],fit_periods=f'1-{cut}',
                        validation_periods=f'{cut+1}-{end}',absolute_error=error,demand_units=demand,wape=error/demand))
                    errors.append(error);demands.append(demand)
                forecast_validation.append(dict(**candidate,absolute_error=sum(errors),demand_units=sum(demands),wape=sum(errors)/sum(demands)))
    settings={k:min(forecast_validation,key=lambda r:r['wape'])[k] for k in ('blend','window','prior')}
    pd.DataFrame(forecast_validation).to_csv(out/'forecast_validation.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(forecast_by_fold).to_csv(out/'forecast_validation_by_fold.csv',index=False,encoding='utf-8-sig')
    for f in fold_data:
        cut,end,labels=f['cut'],f['end'],f['labels']
        fitted=model.Forecaster(y[:cut],labels,**settings)
        f['fc']=model.make_forecasts(fitted,y[cut:end],cut,history=y[:cut])
        f['warm_fc']=model.make_forecasts(fitted,y[:cut],0)
        f['residual']=model.residual_oof(y[:cut],labels,**settings)[0]
        f['groups']=pipeline.variant_labels(labels,data['is_hot'])
        f['level_cache']={};f['quantile_cache']={}
    print('[rolling calibration] 1-24 -> 25-36; 1-36 -> 37-48; 1-48 -> 49-60; forecast',settings,flush=True)
    recipes=[];rows=[];rows_by_fold=[]
    def levels(f,recipe):
        method=recipe['method'];residual=f['residual'];labels=f['labels']
        if method=='RO':return np.full(6000,box(residual,recipe['gamma'])[1]),None
        if method=='sS':
            def quantile(p):
                if p not in f['quantile_cache']:f['quantile_cache'][p]=ss_quantiles(residual,labels,p)
                return f['quantile_cache'][p]
            return quantile(recipe['p_s'])*recipe['scale'],quantile(recipe['p_S'])*recipe['scale']
        key=(method,recipe['beta'])
        if key not in f['level_cache']:
            f['level_cache'][key]=model.risk_levels(residual,labels if method=='SAA' else f['groups'][method],recipe['beta'],0.)[0]
        return (f['level_cache'][key]+recipe['radius'])*recipe['scale'],None
    def add(method,**recipe):
        index=len(rows);entry=dict(candidate=index,method=method,**recipe);parts=[]
        for f in fold_data:
            low,high=levels(f,entry)
            validation_initial=validation_initial_stock(y[:f['cut']],init,f['warm_fc'],low,high)
            r=model.simulate(y[f['cut']:f['end']],validation_initial,f['fc'],low,high,positive_guard=True)['summary']
            rows_by_fold.append(dict(**entry,fold=f['fold'],fit_periods=f"1-{f['cut']}",
                validation_periods=f"{f['cut']+1}-{f['end']}",validation_initial_available_units=float(validation_initial.sum()),**r));parts.append(r)
        rows.append(dict(**entry,**aggregate_summaries(parts)));recipes.append(entry)
    # Exactly the same union of controllers is offered to both service arms.
    # Fixed chance-DRO controller generator: the outer service rule is the treatment.
    for method in ('DRO_cluster','DRO_global','DRO_category','SAA'):
        for beta in (.5,.6,.7,.75,.8,.85,.9,.95,.96,.97,.98,.99,.995,.999):
            for radius in ((0.,) if method=='SAA' else (.005,.01,.025,.05,.075,.1,.15,.2,.3,.5)):
                for scale in ((.75,1.,1.125,1.25) if method=='SAA' else (.75,1.,1.25)):
                    add(method,beta=beta,radius=radius,scale=scale)
        print('[library]',method,len(rows),flush=True)
    for ps in (.5,.6,.7,.75,.8,.85,.9,.95,.975,.99,.999):
        for gap in (0.,.025,.05,.1,.15):
            pS=min(ps+gap,.999)
            for scale in np.arange(.75,1.501,.025):
                add('sS',p_s=ps,p_S=pS,scale=float(scale))
    print('[library] sS',len(rows),flush=True)
    for gamma in np.arange(0,8.0001,.01):
        add('RO',gamma=float(gamma))
    # Training-only feasibility refinement, applied identically to both arms.
    # Original candidate indices remain stable; test data are not read here.
    refinement=[]
    for step in (.025,.005):
        frame=pd.DataFrame(rows)
        for method in LABELS:
            subset=frame[frame.method==method]
            if subset.positive_stock_ratio.between(.95,.955).any():continue
            if method not in ('DRO_cluster','DRO_global','DRO_category','SAA'):
                continue
            before=len(rows)
            prototypes={ (r['beta'],r['radius']) for r in recipes if r['method']==method }
            for beta,radius in sorted(prototypes):
                known=[r['scale'] for r in recipes if r['method']==method and r['beta']==beta and r['radius']==radius]
                for scale in np.round(np.arange(.75,1.250001,step),6):
                    if any(abs(float(scale)-v)<1e-8 for v in known):continue
                    add(method,beta=beta,radius=radius,scale=float(scale))
            refinement.append(dict(method=method,trigger='no positive-service training-feasible candidate',
                scale_lower=.75,scale_upper=1.25,scale_step=step,added_candidates=len(rows)-before))
            print('[training-only refinement]',method,'scale step',step,'added',len(rows)-before,flush=True)
    validation=pd.DataFrame(rows)
    validation.to_csv(out/'candidate_validation.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(rows_by_fold).to_csv(out/'candidate_validation_by_fold.csv',index=False,encoding='utf-8-sig')
    frozen=[]
    for method in LABELS:
        subset=validation[validation.method==method]
        for arm in ('fill','positive'):
            chosen=pick(subset,arm)
            frozen.append(dict(arm=arm,method=method,candidate=int(chosen.candidate),
                train_fill=float(chosen.fill_rate),train_positive=float(chosen.positive_stock_ratio),
                train_inventory_ratio=float(chosen.inventory_demand_ratio)))
    calibration_folds=[dict(fold=f['fold'],fit_periods=list(range(1,f['cut']+1)),validation_periods=list(range(f['cut']+1,f['end']+1)),
        fit_y_sha256=digest(y[:f['cut']]),fit_x_sha256=digest(x[:f['cut']]),
        validation_y_sha256=digest(y[f['cut']:f['end']]),labels_sha256=digest(f['labels']),
        residual_sha256=digest(f['residual']),forecast_sha256=digest(f['fc']),warm_forecast_sha256=digest(f['warm_fc'])) for f in fold_data]
    control=dict(forecast=settings,fit_periods='1-24 / 1-36 / 1-48 (calibration), 1-60 (final)',validation_periods='25-60 (3 rolling folds)',
        calibration_folds=calibration_folds,validation_aggregation='sum primitive totals across 3 separate 12-period episodes; recompute ratios',
        calibration_refinement=refinement,
        validation_initialization='carry candidate-specific stock and next arrival from a fit-window replay; score only the following 12 validation periods',
        test_initialization='Period_1 restart with original initial stock, empty pipeline and empty test observation history',
        residual_method='expanding-window forecast residuals with groups fixed within each outer fit slice',
        final_fit_periods=list(range(1,61)),final_fit_observations=int(y.size),training_y_sha256=digest(y),
        refit=True,positive_integer_guard=True,library_size=len(rows),
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
    fold_arrays={}
    for f in fold_data:
        for key in ('labels','residual','fc','warm_fc'):fold_arrays[f"fold{f['fold']}_{key}"]=f[key]
    np.savez_compressed(out/'shared_model.npz',**fold_arrays,forecasts_validation=np.concatenate([f['fc'] for f in fold_data]),forecasts_test=ft,initial=init,
        train_y_all60=y,final_labels=full_labels,final_relative_residual=full_residual,
        final_slope=final.slope,final_intercept=final.intercept)
    latest=HOME/'new_positive/outputs_v4';latest.mkdir(parents=True,exist_ok=True)
    pipeline.METHODS=tuple(LABELS)
    pipeline.output_results(latest,positive_runs,test,ft,data['item_id'],full_labels,data['is_hot'])
    positive_frozen=dict(service_metric='positive_stock_ratio',service_lower=.95,service_upper=.955,
        model_fit_periods='1-60',calibration_fit_periods='1-24 / 1-36 / 1-48',calibration_periods='25-60 (3 rolling folds)',
        calibration_folds=calibration_folds,refit_after_calibration=True,
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
        '两组共用三轮滚动验证：1–24期拟合、25–36期验证；1–36期拟合、37–48期验证；1–48期拟合、49–60期验证。每轮只使用更早的训练数据拟合聚类、预测器与残差，验证时依次吸收已经观察到的需求。每个候选在该轮拟合期内回放库存，以拟合末端库存和在途到货承接验证首期，历史回放只用于状态初始化，不计入验证成绩，也不使用未来验证需求。三轮共36期的销量、需求、库存与正库存计数相加后重算F、P和J，据此选参，不直接平均三轮比率。P范围针对三轮合并成绩，逐轮成绩也单独报告。选参结束后，预测器、特征聚类和固定分组下的扩展窗预测残差库使用训练集第1–60期最终拟合，再评估独立测试文件。测试仍从原始初始库存、零在途和Period_1开始。两组使用相同整数处理、L=1、H=4、失销处理、库存需求比目标及排序规则。',
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
