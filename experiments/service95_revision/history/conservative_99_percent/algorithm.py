"""Train-only service-first calibration, with reset and forward stress episodes.

Run: python experiments/ro_extension/service95_revision.py
No test outcomes enter candidate construction or selection.
"""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
from extend_with_ro import ROOT, LABELS, box
from constraint_ablation import FOLDS, digest

HOME = ROOT / 'experiments/service95_revision'
CONTROL_TARGET = .99
sys.path.insert(0, str(ROOT/'experiments/rolling_dro_v3'))
import model_v2 as model
import run_v3 as pipeline
from run_v2 import ss_quantiles


def replay(y, initial, fc, low, high=None, reserve=None, keep=False):
    """L=1 lost sales. Orders after d[t]; four plans, only plan[0] executed."""
    reserve = np.zeros(y.shape[1]) if reserve is None else reserve
    stock=initial.copy(); pipe=np.zeros_like(stock)
    c=[];e=[];s=[];q=[];plans=[];targets=[]
    for t in range(len(y)):
        available=stock+pipe; sold=np.minimum(available,y[t]);stock=available-sold
        upper=np.floor(np.maximum(fc[t]*(low if high is None else high),reserve))+1
        lower=np.floor(np.maximum(fc[t]*low,reserve))+1
        order=np.maximum(0,np.ceil(upper[0]-stock))
        if high is not None:order=np.where(stock<lower[0],order,0.)
        if keep:
            projected=stock.copy();plan=[]
            for h in range(4):
                v=np.maximum(0,np.ceil(upper[h]-projected))
                if high is not None:v=np.where(projected<lower[h],v,0.)
                plan.append(v);projected=np.maximum(0,projected+v-fc[t,h])
            np.testing.assert_array_equal(plan[0],order)
            plans.append(plan);targets.append(upper)
        c.append(available);e.append(stock.copy());s.append(sold);q.append(order);pipe=order
    c,e,s,q=map(np.asarray,(c,e,s,q))
    ans=dict(summary=model.summarize(y,s,e,c,q),controlled=model.summarize(y[1:],s[1:],e[1:],c[1:],q[1:]))
    if keep:ans.update(commencing=c,ending=e,sold=s,orders=q,plans=np.asarray(plans),targets=np.asarray(targets))
    return ans


def levels(residual, labels, hot, recipe, cache=None):
    method=recipe['method'];cache={} if cache is None else cache
    if method=='RO':return np.full(len(labels),box(residual,recipe['gamma'])[1]),None
    if method=='sS':
        def quantile(p):
            key=('ss',p)
            if key not in cache:cache[key]=ss_quantiles(residual,labels,p)
            return cache[key]*recipe['scale']
        return quantile(recipe['p_s']),quantile(recipe['p_S'])
    key=(method,recipe['beta'])
    if key not in cache:
        groups=labels if method=='SAA' else pipeline.variant_labels(labels,hot)[method]
        cache[key]=model.risk_levels(residual,groups,recipe['beta'],0.)[0]
    return (cache[key]+recipe['radius'])*recipe['scale'],None


def candidates():
    beta=(.95,.975,.99,.995,.999,.9999,1.)
    for method in ('DRO_cluster','DRO_global','DRO_category','SAA'):
        for b in beta:
            for r in ((0.,) if method=='SAA' else (.005,.05,.1,.2,.4,.8,1.6)):
                for scale in (1.,1.25,1.5,2.,3.):
                    for reserve in (0.,.5,1.):
                        yield dict(method=method,beta=b,radius=r,scale=scale,reserve_scale=reserve)
    for ps in (.95,.975,.99,.995,.999,.9999,1.):
        for gap in (0.,.01,.05):
            for scale in (1.,1.25,1.5,2.,3.):
                for reserve in (0.,.5,1.):
                    yield dict(method='sS',p_s=ps,p_S=min(1.,ps+gap),scale=scale,reserve_scale=reserve)
    for gamma in (0.,.5,1.,1.5,2.,3.,4.,6.,8.,12.,16.,24.,32.):
        for reserve in (0.,.5,1.):yield dict(method='RO',gamma=gamma,reserve_scale=reserve)


def main():
    out=HOME/'outputs_v4';out.mkdir(parents=True,exist_ok=True)
    data=pipeline.verified_data(ROOT/'data/raw',out,False)
    y,x,initial=data['train_y'].astype(float),data['train_x'].astype(float),data['initial_inventory'].astype(float)
    previous=ROOT/'experiments/full60_revision/outputs'
    settings=json.loads((previous/'frozen_design.json').read_text(encoding='utf-8'))['forecast']
    # Forecast hyperparameters were selected on the same three training folds.
    # Re-estimate each learned fold component from its fit slice.
    folds=[]
    for i,(cut,end) in enumerate(FOLDS,1):
        labels=model.cluster(y[:cut],x[:cut])[0]
        fit=model.Forecaster(y[:cut],labels,**settings)
        folds.append(dict(fold=i,cut=cut,end=end,fit_start=0,fit_end=cut,val_start=cut,cases=('forward','reset'),labels=labels,fit=fit,
            residual=model.residual_oof(y[:cut],labels,**settings)[0],mean=y[:cut].mean(0),cache={},
            forward=model.make_forecasts(fit,y[cut:end],cut,history=y[:cut]),
            reset=model.make_forecasts(fit,y[cut:end],0)))
    # Disjoint retrospective episode holdout, solely inside the TRAIN workbook.
    # It tests an early-calendar restart. It is explicitly NOT a forward-time
    # performance estimate: the fit block follows the held-out episode.
    reverse_labels=model.cluster(y[28:],x[28:])[0]
    reverse_fit=model.Forecaster(y[28:],reverse_labels,**settings,time_start=28)
    folds.append(dict(fold=4,cut=28,end=28,fit_start=28,fit_end=60,val_start=0,cases=('reset',),labels=reverse_labels,fit=reverse_fit,
        residual=model.residual_oof(y[28:],reverse_labels,**settings,time_start=28)[0],mean=y[28:].mean(0),cache={},
        reset=model.make_forecasts(reverse_fit,y[:28],0)))
    recipes=list(candidates());rows=[];details=[]
    for idx,recipe in enumerate(recipes):
        parts=[]
        for f in folds:
            low,high=levels(f['residual'],f['labels'],data['is_hot'],recipe,f['cache'])
            for case in f['cases']:
                run=replay(y[f['val_start']:f['end']],initial,f[case],low,high,f['mean']*recipe['reserve_scale'])
                r=run['controlled'];parts.append(r)
                details.append(dict(candidate=idx,method=recipe['method'],fold=f['fold'],case=case,
                    fit_periods=f"{f['fit_start']+1}-{f['fit_end']}",validation_periods=f"{f['val_start']+1}-{f['end']}",
                    score_periods=f"{f['val_start']+2}-{f['end']}",**r))
        demand=sum(p['demand_units'] for p in parts);ending=sum(p['ending_inventory_units'] for p in parts)
        rows.append(dict(candidate=idx,**recipe,min_controlled_service=min(p['positive_stock_ratio'] for p in parts),
            controlled_service=sum(p['positive_stock_sku_periods'] for p in parts)/sum(p['total_sku_periods'] for p in parts),
            inventory_demand_ratio=ending/demand,ending_inventory_units=ending,demand_units=demand,
            projected_28_service_lower=min(p['positive_stock_ratio'] for p in parts)*27/28))
        if idx%300==0:print('[train-only candidates]',idx,'/',len(recipes),flush=True)
    table=pd.DataFrame(rows);table.to_csv(out/'candidate_validation.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(details).to_csv(out/'candidate_validation_by_fold.csv',index=False,encoding='utf-8-sig')
    selected={}
    for method in LABELS:
        feasible=table[(table.method==method)&(table.min_controlled_service>=CONTROL_TARGET)]
        if feasible.empty:raise RuntimeError(f'No training-feasible {method}; test has not been evaluated')
        chosen=feasible.sort_values(['inventory_demand_ratio','candidate'],kind='stable').iloc[0]
        selected[method]=dict(**recipes[int(chosen.candidate)],candidate=int(chosen.candidate),
            train_min_controlled_service=float(chosen.min_controlled_service),train_inventory_ratio=float(chosen.inventory_demand_ratio))
    labels=model.cluster(y,x)[0];residual=model.residual_oof(y,labels,**settings)[0]
    final=model.Forecaster(y,labels,**settings)
    design=dict(service_metric='positive_stock_ratio',service_lower=.95,service_upper=None,
        controlled_training_target=CONTROL_TARGET,conservative_28_service_projection=CONTROL_TARGET*27/28,
        projection_assumption='First period contributes zero; remaining 27 periods retain training controlled-service level. A validation projection, not an unconditional future guarantee.',
        validation_initialization='original initial stock and empty pipeline in BOTH cases; exclude only uncontrollable first period from candidate scoring',
        cases={'forward':'calendar=cut, observed history=fit slice','reset':'calendar=0, no observed episode history'},
        final_fit_periods=list(range(1,61)),model_fit_periods='1-60',final_fit_observations=int(y.size),training_y_sha256=digest(y),training_x_sha256=digest(x),
        initial_sha256=digest(initial),forecast=settings,selected=selected,lead_time=1,horizon=4,
        test_initialization='Period_1 restarts; original stock; zero pipeline; empty episode history',
        objective='min pooled controlled validation ending inventory / demand subject to each of seven episode services >= .99',
        reserve='optional floor reserve_scale * per-SKU fit-slice mean; common controller family across all six methods',
        candidate_count=len(recipes),candidate_grid_sha256=digest(np.frombuffer(json.dumps(recipes,sort_keys=True).encode(),dtype=np.uint8)),
        previous_version='full60_revision (PR7); retained for historical comparison',
        final_risk_rule='SKU-wise maximum of full60 waterline and four disjoint-slice calibration waterlines; reserve uses maximum training fit-slice mean. No test-dependent updates.',
        reset_holdout=dict(fold=4,fit_periods=list(range(29,61)),validation_periods=list(range(1,29)),
            interpretation='disjoint retrospective training episode holdout; not a forward chronological backtest'),
        calibration_folds=[dict(fold=f['fold'],fit_periods=list(range(1,f['cut']+1)),validation_periods=list(range(f['cut']+1,f['end']+1)),
            fit_y_sha256=digest(y[:f['cut']]),fit_x_sha256=digest(x[:f['cut']]),labels_sha256=digest(f['labels']),
            residual_sha256=digest(f['residual']),forward_sha256=digest(f['forward']),reset_sha256=digest(f['reset'])) for f in folds[:3]])
    # Freeze BEFORE constructing the independent test forecasts or evaluating policies.
    frozen_path=out/'frozen_parameters.json'
    frozen_path.write_text(json.dumps(design,ensure_ascii=False,indent=2),encoding='utf-8')
    frozen_sha=hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    print('[FROZEN before test]',frozen_sha,selected,flush=True)
    test=data['test_y'].astype(float);fc=model.make_forecasts(final,test,0)
    runs={};risk=[]
    for method,recipe in selected.items():
        low,high=levels(residual,labels,data['is_hot'],recipe)
        reserve=y.mean(0)*recipe['reserve_scale']
        for f in folds:
            fl,fh=levels(f['residual'],f['labels'],data['is_hot'],recipe,f['cache'])
            low=np.maximum(low,fl)
            if high is not None:high=np.maximum(high,fh)
            reserve=np.maximum(reserve,f['mean']*recipe['reserve_scale'])
        runs[method]=replay(test,initial,fc,low,high,reserve,keep=True)
        risk.append(dict(method=method,level_min=float(low.min()),level_max=float(low.max()),
            reserve_scale=recipe['reserve_scale'],reserve_sum=float(reserve.sum())))
        arrays_for_method=dict(low=low,reserve=reserve)
        if high is not None:arrays_for_method['high']=high
        np.savez_compressed(out/f'policy_{method}.npz',**arrays_for_method)
    pipeline.METHODS=tuple(LABELS)
    pipeline.output_results(out,runs,test,fc,data['item_id'],labels,data['is_hot'])
    for path in [*out.glob('detail_*.csv'),out/'decision_comparison_full.csv']:
        import gzip,shutil
        with path.open('rb') as source,gzip.GzipFile(filename=str(path)+'.gz',mode='wb',mtime=0) as dest:shutil.copyfileobj(source,dest)
    wide=pd.read_csv(out/'decision_comparison_full.csv')
    wide[wide.period==28].to_csv(out/'latest_decisions.csv',index=False,encoding='utf-8-sig')
    wide[wide.item_id.isin(data['item_id'][:12])].to_csv(out/'sample_decisions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(risk).to_csv(out/'risk_profiles.csv',index=False,encoding='utf-8-sig')
    arrays=dict(train_y_all60=y,train_x_mean=x.mean(0),initial=initial,final_labels=labels,final_relative_residual=residual,
        final_slope=final.slope,final_intercept=final.intercept,forecasts_test=fc)
    for f in folds:
        arrays[f"fold{f['fold']}_x_mean"]=x[f['fit_start']:f['fit_end']].mean(0)
        for k in ('labels','residual','forward','reset'):
            if k in f:arrays[f"fold{f['fold']}_{k}"]=f[k]
    np.savez_compressed(out/'fitted_model.npz',**arrays)
    bound=dict(first_positive=int((initial>test[0]).sum()),sku_periods=int(test.size),first_stock_controllable=False,
        max_possible_service=float(((initial>test[0]).sum()+(len(test)-1)*test.shape[1])/test.size),
        frozen_parameters_sha256=frozen_sha,test_results_used_for_selection=False)
    (out/'test_audit.json').write_text(json.dumps(bound,indent=2),encoding='utf-8')
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest()==frozen_sha
    summary=pd.DataFrame([dict(method=m,**r['summary']) for m,r in runs.items()])
    print(summary[['method','positive_stock_ratio','fill_rate','inventory_demand_ratio']].to_string(index=False),flush=True)
    print('All methods >=95%:',bool((summary.positive_stock_ratio>=.95).all()),flush=True)


if __name__=='__main__':main()
