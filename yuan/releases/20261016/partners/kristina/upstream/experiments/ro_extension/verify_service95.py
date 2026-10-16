"""Reconcile full60 fitting, training selection, causal feedback and all outputs."""
import gzip, hashlib,json,sys,zipfile,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from service_band_revision import ROOT,HOME,LABELS,FOLDS,model,levels,response_groups,response_banks,run_budget,digest


def independent_levels(residual,labels,hot,recipe):
    method=recipe['method'];N=len(labels)
    if method=='RO':return np.full(N,residual.mean()+recipe['gamma']*residual.std()),None
    if method=='sS':
        def q(p):
            pooled=np.zeros(N)
            for g in np.unique(labels):pooled[labels==g]=np.quantile(residual[:,labels==g],p)
            w=len(residual)/(len(residual)+24)
            return recipe['scale']*(w*np.quantile(residual,p,axis=0)+(1-w)*pooled)
        return q(recipe['p_s']),q(recipe['p_S'])
    groups=np.zeros(N,int) if method=='DRO_global' else hot.astype(int) if method=='DRO_category' else labels
    risk=np.zeros(N)
    for g in np.unique(groups):
        samples=np.sort(residual[:,groups==g].ravel());k=int(np.ceil(recipe['beta']*len(samples)))-1
        risk[groups==g]=(samples[k]+recipe['radius'])*recipe['scale']
    return risk,None


def main():
    out=HOME/'outputs_v4';d=json.loads((out/'frozen_parameters.json').read_text(encoding='utf-8'))
    audit=json.loads((out/'test_audit.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((out/'frozen_parameters.json').read_bytes()).hexdigest()==audit['frozen_parameters_sha256']
    assert d['final_fit_periods']==list(range(1,61)) and d['final_fit_observations']==360000
    assert d['service_lower']==.95 and d['service_upper']==.955 and d['feedback_target']==.9525
    b=np.load(out/'fitted_model.npz');y=b['train_y_all60'];assert y.shape==(60,6000) and digest(y)==d['training_y_sha256']
    previous=np.load(ROOT/'experiments/full60_revision/outputs/shared_model.npz')
    np.testing.assert_array_equal(y,previous['train_y_all60'])
    inputs=json.loads((out/'input_audit.json').read_text(encoding='utf-8'))
    prior_inputs=json.loads((ROOT/'experiments/full60_revision/outputs/input_audit.json').read_text(encoding='utf-8'))
    assert inputs['inputs']==prior_inputs['inputs']
    first=pd.read_csv(out/'detail_DRO_cluster.csv.gz');hot=first.loc[first.period==1,'category'].eq('爆品').to_numpy()
    assert hot.sum()==480 and len(hot)==6000
    # PCA uses time means of x; all supplied features retain their exact means.
    lab=model.cluster(y,b['train_x_mean'][None])[0];np.testing.assert_array_equal(lab,b['final_labels'])
    fit=model.Forecaster(y,lab,**d['forecast'])
    np.testing.assert_allclose(fit.slope,b['final_slope'],atol=1e-10,rtol=1e-10);np.testing.assert_allclose(fit.intercept,b['final_intercept'],atol=1e-8,rtol=1e-10)
    res=model.residual_oof(y,lab,**d['forecast'])[0];np.testing.assert_allclose(res,b['final_relative_residual'],rtol=1e-9,atol=1e-9)
    slices=[]
    for i,(cut,end) in enumerate(FOLDS,1):slices.append((i,0,cut,cut,end,('forward','reset')))
    slices.append((4,28,60,0,28,('reset',)))
    folds={}
    for i,start,end,vstart,vend,cases in slices:
        labels=model.cluster(y[start:end],b[f'fold{i}_x_mean'][None])[0]
        np.testing.assert_array_equal(labels,b[f'fold{i}_labels'])
        fitted=model.Forecaster(y[start:end],labels,**d['forecast'],time_start=start)
        residual=model.residual_oof(y[start:end],labels,**d['forecast'],time_start=start)[0]
        np.testing.assert_allclose(residual,b[f'fold{i}_residual'],atol=1e-9,rtol=1e-9)
        forecasts={}
        for case in cases:
            forecasts[case]=model.make_forecasts(fitted,y[vstart:vend],vstart if case=='forward' else 0,history=y[:vstart] if case=='forward' else None)
            np.testing.assert_allclose(forecasts[case],b[f'fold{i}_{case}'],atol=1e-7,rtol=1e-9)
        folds[i]=dict(y=y[vstart:vend],labels=labels,residual=residual,fc=forecasts)
    table=pd.read_csv(out/'candidate_validation.csv');assert len(table)==108
    base=json.loads((HOME/'history/conservative_99_percent/frozen_parameters.json').read_text(encoding='utf-8'))
    # Replay all 108 training-only candidates, not merely the chosen six.
    for method in LABELS:
        subset=table[table.method==method];assert len(subset)==18
        feasible=subset[subset.positive_stock_ratio.between(.95,.955)]
        selected=feasible.sort_values(['inventory_demand_ratio','candidate'],kind='stable').iloc[0]
        assert int(selected.candidate)==d['selected'][method]['candidate']
        f=folds[4];groups=response_groups(method,f['labels'],hot)
        base_recipe=base['selected'][method];lo,hi=independent_levels(f['residual'],f['labels'],hot,base_recipe)
        for _,r in subset.iterrows():
            banks=response_banks(f['y'],f['fc']['reset'],f['residual'],groups,int(r.response_window))
            run=run_budget(f['y'],b['initial'],f['fc']['reset'],lo,hi,banks,r.response_prior,r.response_bias)
            for k in ('positive_stock_sku_periods','total_sku_periods','ending_inventory_units','sales_units','demand_units'):
                assert run['summary'][k]==r[k],(method,r.candidate,k)
        print(method,'18 train-only feedback candidates replayed',flush=True)
    selected_records=pd.read_csv(out/'candidate_validation_by_fold.csv');assert len(selected_records)==42
    for r in selected_records.itertuples():
        f=folds[r.fold];recipe=d['selected'][r.method]
        lo,hi=independent_levels(f['residual'],f['labels'],hot,recipe)
        banks=response_banks(f['y'],f['fc'][r.case],f['residual'],response_groups(r.method,f['labels'],hot),recipe['response_window'])
        run=run_budget(f['y'],b['initial'],f['fc'][r.case],lo,hi,banks,recipe['response_prior'],recipe['response_bias'])
        for k in ('positive_stock_sku_periods','total_sku_periods','ending_inventory_units','sales_units','demand_units'):assert run['summary'][k]==getattr(r,k)
    summary=pd.read_csv(out/'summary.csv').query("scope=='all_28_periods'").set_index('method')
    feedback=pd.read_csv(out/'service_feedback.csv');assert len(feedback)==168
    original=pd.read_csv(ROOT/'experiments/demand_fill_v3/with_ro/outputs_v4/detail_DRO_cluster.csv.gz').demand.to_numpy().reshape(28,6000)
    forecasts=model.make_forecasts(fit,original,0);np.testing.assert_allclose(forecasts,b['forecasts_test'],atol=1e-7,rtol=1e-9)
    changed=original.copy();changed[12:]*=7
    np.testing.assert_allclose(model.make_forecasts(fit,changed,0)[:12],forecasts[:12],atol=1e-7,rtol=1e-9)
    for method,recipe in d['selected'].items():
        lo,hi=independent_levels(res,lab,hot,recipe)
        for f in folds.values():
            fl,fh=independent_levels(f['residual'],f['labels'],hot,recipe);lo=np.maximum(lo,fl)
            if hi is not None:hi=np.maximum(hi,fh)
        policy=np.load(out/f'policy_{method}.npz');np.testing.assert_allclose(policy['low'],lo,atol=1e-9,rtol=1e-9)
        if hi is not None:np.testing.assert_allclose(policy['high'],hi,atol=1e-9,rtol=1e-9)
        assert not policy['reserve'].any()
        f=pd.read_csv(out/f'detail_{method}.csv.gz');assert len(f)==168000 and not f.duplicated(['period','item_id']).any()
        arrays={k:f[k].to_numpy().reshape(28,6000) for k in ('demand','available','sales','ending','order','lost')}
        np.testing.assert_array_equal(arrays['demand'],original);np.testing.assert_array_equal(arrays['available'][0],b['initial'])
        np.testing.assert_array_equal(arrays['sales'],np.minimum(arrays['available'],original));np.testing.assert_array_equal(arrays['ending'],arrays['available']-arrays['sales'])
        np.testing.assert_array_equal(arrays['lost'],original-arrays['sales']);np.testing.assert_array_equal(arrays['available'][1:],arrays['ending'][:-1]+arrays['order'][:-1])
        assert (arrays['order']>=0).all() and (arrays['order']==np.floor(arrays['order'])).all()
        logs=feedback[feedback.method==method].reset_index(drop=True);counts=(arrays['ending']>0).sum(1).cumsum()
        np.testing.assert_array_equal(logs.cumulative_positive,counts)
        for t in range(28):
            request=np.clip((.9525*168000-counts[t])/(max(27-t,1)*6000)+recipe['response_bias'],0.,1.)
            np.testing.assert_allclose(request,logs.requested_next_service[t],atol=1e-12)
            projected=arrays['ending'][t].copy();scale=logs.scale[t]
            for h in range(4):
                S=np.floor(scale*forecasts[t,h]*(lo if hi is None else hi))+1
                q=np.maximum(0,np.ceil(S-projected))
                if hi is not None:q=np.where(projected<np.floor(scale*forecasts[t,h]*lo)+1,q,0.)
                np.testing.assert_array_equal(f[f'planned_order_h{h+1}'].to_numpy().reshape(28,6000)[t],q)
                np.testing.assert_array_equal(f[f'target_h{h+1}'].to_numpy().reshape(28,6000)[t],S)
                projected=np.maximum(0,projected+q-forecasts[t,h])
        groups=response_groups(method,lab,hot);banks=response_banks(original,forecasts,res,groups,recipe['response_window'])
        replay=run_budget(original,b['initial'],forecasts,lo,hi,banks,recipe['response_prior'],recipe['response_bias'],keep=True)
        for source,dest in [('order','orders'),('ending','ending'),('sales','sold')]:np.testing.assert_array_equal(arrays[source],replay[dest])
        # Perturb all FUTURE observed demands; the earlier orders must match.
        banks_changed=response_banks(changed,forecasts,res,groups,recipe['response_window'])
        changed_run=run_budget(changed,b['initial'],forecasts,lo,hi,banks_changed,recipe['response_prior'],recipe['response_bias'],keep=True)
        np.testing.assert_array_equal(changed_run['orders'][:12],replay['orders'][:12])
        row=summary.loc[method];actual=np.mean(arrays['ending']>0)
        assert .95<=actual<=.955 and int((arrays['ending']>0).sum())==int(row.positive_stock_sku_periods)
        np.testing.assert_allclose([actual,arrays['sales'].sum()/original.sum(),arrays['ending'].sum()/original.sum()],
            [row.positive_stock_ratio,row.fill_rate,row.inventory_demand_ratio],rtol=1e-12)
        print(method,'all168000 rows, four plans, causal feedback and 95%-95.5% verified',flush=True)
    # Verify exported workbook formulas and cached headline results.
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(out/'replenishment_comparison.xlsx') as z:
        xml=ET.fromstring(z.read('xl/worksheets/sheet1.xml'));cells={c.attrib['r']:c for c in xml.findall('.//s:c',ns)}
        for r,method in enumerate(LABELS,5):
            for column,key in [('D','fill_rate'),('G','positive_stock_ratio'),('I','inventory_demand_ratio')]:
                c=cells[f'{column}{r}'];assert c.find('s:f',ns) is not None
                np.testing.assert_allclose(float(c.find('s:v',ns).text),summary.loc[method,key],rtol=1e-12)
        for name in z.namelist():
            if name.startswith('xl/worksheets/sheet') and name.endswith('.xml'):
                assert not ET.fromstring(z.read(name)).findall('.//s:c[@t="e"]',ns),name
    print('Verified full60 final fit,108 feedback candidates,42 selected scenarios,1008000 test rows, causal no-future orders, all service in band, workbook formulas and caches.')


if __name__=='__main__':main()
