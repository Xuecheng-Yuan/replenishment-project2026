"""Causal service-budget feedback; train-only tuning for P in [95%,95.5%]."""
import hashlib, json
import numpy as np
import pandas as pd
from service95_revision import ROOT, HOME, LABELS, FOLDS, model, pipeline, levels, digest

TARGET=.9525
PROBS=(np.arange(128)+.5)/128


def response_groups(method,labels,hot):
    # The same pool definitions as the corresponding demand-risk estimator.
    if method=='DRO_global' or method=='RO':return np.zeros(len(labels),int)
    if method=='DRO_category':return hot.astype(int)
    return labels


def response_banks(y,fc,residual,groups,window):
    """At decision t only realized next-day errors 1..t are available.

    y[0] has no forecast made in this episode and never becomes a false
    one-step residual. Quantile sketches approximate empirical CDFs.
    """
    banks=[]
    for t in range(len(y)):
        per_group={}
        first=max(1,t-window+1)
        past=y[first:t+1]/np.maximum(fc[first-1:t,0],1.) if t else None
        for g in np.unique(groups):
            mask=groups==g
            prior=np.quantile(residual[:,mask],PROBS)
            recent=np.quantile(past[:,mask],PROBS) if t else prior
            per_group[int(g)]=(mask,prior,recent,min(t,window))
        banks.append(per_group)
    return banks


def run_budget(y,initial,fc,low,high,banks,prior,bias,target=TARGET,keep=False):
    stock=initial.copy();pipe=np.zeros_like(stock);count=0
    cs=[];es=[];ss=[];qs=[];plans=[];targets=[];logs=[]
    for t in range(len(y)):
        available=stock+pipe;sold=np.minimum(available,y[t]);stock=available-sold
        count+=int((stock>0).sum())
        # Forecast budget over the fixed 28-day business episode; validation
        # prefixes use the same horizon. Actual test denominator stays 28*N.
        remaining=27-t
        requested=np.clip((target*28*y.shape[1]-count)/(max(remaining,1)*y.shape[1])+bias,0.,1.)
        base=fc[t]*(low if high is None else high)
        trigger=fc[t]*low
        def predict(scale,return_order=False):
            upper=np.floor(scale*base[0])+1
            q=np.maximum(0,np.ceil(upper-stock))
            if high is not None:q=np.where(stock<np.floor(scale*trigger[0])+1,q,0.)
            next_available=stock+q;value=0.
            for mask,old,recent,n in banks[t].values():
                threshold=next_available[mask]/np.maximum(fc[t,0,mask],1.)
                weight=1. if n==0 else prior/(prior+n)
                old_cdf=np.searchsorted(old,threshold,side='left')/len(old)
                recent_cdf=np.searchsorted(recent,threshold,side='left')/len(recent)
                value+=float((weight*old_cdf+(1-weight)*recent_cdf).sum())
            p=value/y.shape[1]
            return (p,q,upper) if return_order else p
        lo=0.;hi=1.
        while predict(hi)<requested and hi<128:hi*=2
        for _ in range(16):
            mid=(lo+hi)/2
            if predict(mid)>=requested:hi=mid
            else:lo=mid
        scale=hi;estimated,order,upper=predict(scale,True)
        logs.append(dict(period=t+1,scale=float(scale),requested_next_service=float(requested),
            estimated_next_service=float(estimated),cumulative_positive=count,
            cumulative_service=count/((t+1)*y.shape[1]),feedback_target=target,bias=bias,
            irreducible_predicted_service=float(predict(0.)),lambda_cap_reached=bool(hi>=128)))
        if keep:
            projected=stock.copy();plan=[];target4=[]
            for h in range(4):
                S=np.floor(scale*base[h])+1;v=np.maximum(0,np.ceil(S-projected))
                if high is not None:v=np.where(projected<np.floor(scale*trigger[h])+1,v,0.)
                plan.append(v);target4.append(S)
                projected=np.maximum(0,projected+v-fc[t,h])
            np.testing.assert_array_equal(order,plan[0]);plans.append(plan);targets.append(target4)
        cs.append(available);es.append(stock.copy());ss.append(sold);qs.append(order);pipe=order
    c,e,s,q=map(np.asarray,(cs,es,ss,qs))
    answer=dict(summary=model.summarize(y,s,e,c,q),controlled=model.summarize(y[1:],s[1:],e[1:],c[1:],q[1:]),feedback=logs)
    if keep:answer.update(commencing=c,ending=e,sold=s,orders=q,plans=np.asarray(plans),targets=np.asarray(targets))
    return answer


def main():
    out=HOME/'outputs_v4';out.mkdir(parents=True,exist_ok=True)
    data=pipeline.verified_data(ROOT/'data/raw',out,False)
    y,x,initial=data['train_y'].astype(float),data['train_x'].astype(float),data['initial_inventory'].astype(float)
    base_design=json.loads((HOME/'history/conservative_99_percent/frozen_parameters.json').read_text(encoding='utf-8'))
    settings=base_design['forecast'];folds=[]
    for i,(cut,end) in enumerate(FOLDS,1):
        lab=model.cluster(y[:cut],x[:cut])[0];fit=model.Forecaster(y[:cut],lab,**settings)
        folds.append(dict(fold=i,fit_start=0,fit_end=cut,val_start=cut,end=end,labels=lab,
            residual=model.residual_oof(y[:cut],lab,**settings)[0],
            forward=model.make_forecasts(fit,y[cut:end],cut,history=y[:cut]),reset=model.make_forecasts(fit,y[cut:end],0),
            cases=('forward','reset')))
    lab=model.cluster(y[28:],x[28:])[0];fit=model.Forecaster(y[28:],lab,**settings,time_start=28)
    folds.append(dict(fold=4,fit_start=28,fit_end=60,val_start=0,end=28,labels=lab,
        residual=model.residual_oof(y[28:],lab,**settings,time_start=28)[0],reset=model.make_forecasts(fit,y[:28],0),cases=('reset',)))
    # First select feedback on the disjoint 28-day training episode. Three
    # chronological folds then provide prefix diagnostics under a 28-day budget.
    records=[];recipes=[];by_fold=[];chosen={}
    f=folds[3];vy=y[:28]
    for method in LABELS:
        base={k:v for k,v in base_design['selected'][method].items() if k in ('method','beta','radius','scale','gamma','p_s','p_S')}
        lo,hi=levels(f['residual'],f['labels'],data['is_hot'],base)
        groups=response_groups(method,f['labels'],data['is_hot'])
        for window in (1,3,7):
            banks=response_banks(vy,f['reset'],f['residual'],groups,window)
            for prior in (0.,3.):
                for bias in (-.01,0.,.01):
                    idx=len(recipes);recipe=dict(**base,candidate=idx,response_window=window,response_prior=prior,response_bias=bias)
                    run=run_budget(vy,initial,f['reset'],lo,hi,banks,prior,bias)
                    recipes.append(recipe);records.append(dict(**recipe,**run['summary']))
                    print('[training reset feedback]',method,idx,round(run['summary']['positive_stock_ratio'],6),flush=True)
        subset=pd.DataFrame(records);subset=subset[(subset.method==method)&subset.positive_stock_ratio.between(.95,.955)]
        if subset.empty:raise RuntimeError(f'No train service-band feasible feedback for {method}; no test evaluation')
        best=subset.sort_values(['inventory_demand_ratio','candidate'],kind='stable').iloc[0]
        recipe=recipes[int(best.candidate)]
        chosen[method]=dict(**recipe,train_service=float(best.positive_stock_ratio),train_inventory_ratio=float(best.inventory_demand_ratio))
        for f2 in folds:
            lo2,hi2=levels(f2['residual'],f2['labels'],data['is_hot'],recipe)
            groups2=response_groups(method,f2['labels'],data['is_hot'])
            for case in f2['cases']:
                dy=y[f2['val_start']:f2['end']]
                banks=response_banks(dy,f2[case],f2['residual'],groups2,recipe['response_window'])
                run=run_budget(dy,initial,f2[case],lo2,hi2,banks,recipe['response_prior'],recipe['response_bias'])
                by_fold.append(dict(candidate=recipe['candidate'],method=method,fold=f2['fold'],case=case,
                    fit_periods=f"{f2['fit_start']+1}-{f2['fit_end']}",validation_periods=f"{f2['val_start']+1}-{f2['end']}",
                    score_periods=f"{f2['val_start']+1}-{f2['end']}",validation_role='28-day held-out reset selection' if f2['fold']==4 else '12-day forward/reset prefix diagnostic',**run['summary']))
    pd.DataFrame(records).to_csv(out/'candidate_validation.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(by_fold).to_csv(out/'candidate_validation_by_fold.csv',index=False,encoding='utf-8-sig')
    labels=model.cluster(y,x)[0];residual=model.residual_oof(y,labels,**settings)[0];final=model.Forecaster(y,labels,**settings)
    design=dict(service_metric='positive_stock_ratio',service_lower=.95,service_upper=.955,feedback_target=TARGET,
        final_fit_periods=list(range(1,61)),model_fit_periods='1-60',final_fit_observations=int(y.size),
        training_y_sha256=digest(y),training_x_sha256=digest(x),initial_sha256=digest(initial),forecast=settings,
        lead_time=1,horizon=4,candidate_count=len(recipes),selected=chosen,
        calibration_folds=[dict(fold=f['fold'],fit_periods=list(range(f['fit_start']+1,f['fit_end']+1)),validation_periods=list(range(f['val_start']+1,f['end']+1))) for f in folds[:3]],
        reset_holdout=dict(fit_periods=list(range(29,61)),validation_periods=list(range(1,29)),interpretation='disjoint retrospective training episode; not chronological future performance'),
        selection_rule='28-day held-out training reset service in [.95,.955]; min inventory/demand, then candidate id; three forward folds are prefix diagnostics',
        final_risk_rule='max full60 and four training-slice risk waterlines; dynamic common scalar lambda chosen causally from remaining service budget and empirical response CDF',
        response_rule='128 midpoint quantiles per group; prior/(prior+observed periods), latest window one-step residuals; episode first demand never counted as one-step error',
        reserve_scale=0.,test_initialization='original initial inventory, zero pipeline, Period_1 restart; all28 periods scored',
        audit_note='Same test file has been evaluated in prior model revisions. Feedback uses only already realized demand, never future demand. Narrow-band guarantee is empirical, not for arbitrary unbounded future demand.')
    path=out/'frozen_parameters.json';path.write_text(json.dumps(design,ensure_ascii=False,indent=2),encoding='utf-8');frozen=hashlib.sha256(path.read_bytes()).hexdigest()
    print('[FROZEN before test]',frozen,chosen,flush=True)
    test=data['test_y'].astype(float);fc=model.make_forecasts(final,test,0);runs={};feedback=[]
    arrays=dict(train_y_all60=y,train_x_mean=x.mean(0),initial=initial,final_labels=labels,final_relative_residual=residual,
        final_slope=final.slope,final_intercept=final.intercept,forecasts_test=fc)
    for f in folds:
        arrays[f"fold{f['fold']}_x_mean"]=x[f['fit_start']:f['fit_end']].mean(0)
        for k in ('labels','residual','forward','reset'):
            if k in f:arrays[f"fold{f['fold']}_{k}"]=f[k]
    for method,recipe in chosen.items():
        lo,hi=levels(residual,labels,data['is_hot'],recipe)
        for f in folds:
            fl,fh=levels(f['residual'],f['labels'],data['is_hot'],recipe);lo=np.maximum(lo,fl)
            if hi is not None:hi=np.maximum(hi,fh)
        groups=response_groups(method,labels,data['is_hot'])
        banks=response_banks(test,fc,residual,groups,recipe['response_window'])
        run=run_budget(test,initial,fc,lo,hi,banks,recipe['response_prior'],recipe['response_bias'],keep=True)
        runs[method]=run;feedback.extend(dict(method=method,**r) for r in run['feedback'])
        policy=dict(low=lo,reserve=np.zeros(6000),response_groups=groups)
        if hi is not None:policy['high']=hi
        np.savez_compressed(out/f'policy_{method}.npz',**policy)
    pipeline.METHODS=tuple(LABELS);pipeline.output_results(out,runs,test,fc,data['item_id'],labels,data['is_hot'])
    import gzip,shutil
    for path2 in [*out.glob('detail_*.csv'),out/'decision_comparison_full.csv']:
        with path2.open('rb') as src,gzip.GzipFile(filename=str(path2)+'.gz',mode='wb',mtime=0) as dest:shutil.copyfileobj(src,dest)
    wide=pd.read_csv(out/'decision_comparison_full.csv');wide[wide.period==28].to_csv(out/'latest_decisions.csv',index=False,encoding='utf-8-sig');wide[wide.item_id.isin(data['item_id'][:12])].to_csv(out/'sample_decisions.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(out/'fitted_model.npz',**arrays)
    pd.DataFrame(feedback).to_csv(out/'service_feedback.csv',index=False,encoding='utf-8-sig')
    (out/'test_audit.json').write_text(json.dumps(dict(first_positive=int((initial>test[0]).sum()),sku_periods=int(test.size),
        max_possible_service=float(((initial>test[0]).sum()+27*6000)/test.size),first_stock_controllable=False,
        frozen_parameters_sha256=frozen,test_results_used_for_selection=False),indent=2),encoding='utf-8')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==frozen
    summary=pd.DataFrame([dict(method=m,**r['summary']) for m,r in runs.items()])
    print(summary[['method','positive_stock_ratio','fill_rate','inventory_demand_ratio']].to_string(index=False),flush=True)
    print('All service in [95%,95.5%]:',summary.positive_stock_ratio.between(.95,.955).all(),flush=True)


if __name__=='__main__':main()
