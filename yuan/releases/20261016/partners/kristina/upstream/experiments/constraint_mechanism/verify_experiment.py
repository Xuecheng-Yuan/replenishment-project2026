"""Independent inventory identities plus full finite-library and causality checks."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from run_experiment import *

def independent_water(residual,labels,hot,recipe):
    N=len(labels);method=recipe['method']
    if method=='RO':return np.full(N,residual.mean()+recipe['gamma']*residual.std()),None
    if method=='sS':
        def quant(p):
            pool=np.zeros(N)
            for g in np.unique(labels):pool[labels==g]=np.quantile(residual[:,labels==g],p)
            w=len(residual)/(len(residual)+24)
            return recipe['scale']*(w*np.quantile(residual,p,axis=0)+(1-w)*pool)
        return quant(recipe['p_s']),quant(recipe['p_S'])
    groups=np.zeros(N,int) if method=='DRO_global' else hot.astype(int) if method=='DRO_category' else labels
    z=np.zeros(N)
    for g in np.unique(groups):
        sample=np.sort(residual[:,groups==g].ravel())
        z[groups==g]=(sample[int(np.ceil(recipe['beta']*len(sample)))-1]+recipe['radius'])*recipe['scale']
    return z,None

def identities(y,initial,fc,trace):
    c,e,s,q=(trace[k].astype(float) for k in ('commencing','ending','sold','orders'))
    np.testing.assert_array_equal(c[0],initial)
    np.testing.assert_array_equal(c[1:],e[:-1]+q[:-1])
    np.testing.assert_array_equal(s,np.minimum(c,y));np.testing.assert_array_equal(e,c-s)
    assert np.all(q>=0) and np.all(q==np.floor(q))
    np.testing.assert_array_equal(trace['plans'][:,0],q)
    assert trace['plans'].shape==(28,4,6000) and (trace['plans']>=0).all()
    return model.summarize(y,s,e,c,q)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'outputs');a=p.parse_args();out=a.output
    complete=json.loads((out/'completed.json').read_text());protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    design=json.loads((out/'frozen_selection.json').read_text(encoding='utf-8'))
    assert sha(out/'protocol.json')==complete['protocol_sha256']==design['protocol_sha256']
    assert sha(out/'frozen_selection.json')==complete['selection_sha256']
    assert sha(HERE/'run_experiment.py')==complete['code_sha256']==protocol['run_code_sha256']
    data=pipeline.verified_data(ROOT/'data/raw',out,False)
    b=np.load(out/'shared_arrays.npz');y=b['train_y'];x=data['train_x'].astype(float);initial=b['initial'];hot=b['hot'];test=b['test_y'];fc=b['final_forecast']
    np.testing.assert_array_equal(y,data['train_y']);np.testing.assert_array_equal(test,data['test_y']);np.testing.assert_array_equal(initial,data['initial_inventory'])
    assert y.shape==(60,6000) and test.shape==(28,6000) and y.size==design['final_fit_observations']==360000
    assert digest(y)==design['training_y_sha256'] and digest(x)==design['training_x_sha256']
    assert set(protocol['validation']['fit_periods']).isdisjoint(protocol['validation']['score_periods'])
    settings=protocol['forecast'];vl=model.cluster(y[28:],x[28:])[0];fl=model.cluster(y,x)[0]
    np.testing.assert_array_equal(vl,b['validation_labels']);np.testing.assert_array_equal(fl,b['final_labels'])
    for yy,ll,start,key in [(y[28:],vl,28,'validation_residual'),(y,fl,0,'final_residual')]:
        np.testing.assert_allclose(model.residual_oof(yy,ll,**settings,time_start=start)[0],b[key],rtol=1e-12,atol=1e-10)
    vfc=model.make_forecasts(model.Forecaster(y[28:],vl,**settings,time_start=28),y[:28],0)
    np.testing.assert_allclose(vfc,b['validation_forecast'],rtol=1e-12,atol=1e-9)
    fit=model.Forecaster(y,fl,**settings);np.testing.assert_allclose(model.make_forecasts(fit,test,0),fc,rtol=1e-12,atol=1e-9)
    slices=[]
    for start,end in ((0,24),(0,36),(0,48),(28,60)):
        lab=model.cluster(y[start:end],x[start:end])[0]
        slices.append((lab,model.residual_oof(y[start:end],lab,**settings,time_start=start)[0]))
    for method in LABELS:
        r=next(r for r in protocol['library'] if r['method']==method)
        lo,hi=independent_water(b['final_residual'],fl,hot,r);el=lo.copy();eh=None if hi is None else hi.copy()
        for lab,res in slices:
            ll,hh=independent_water(res,lab,hot,r);el=np.maximum(el,ll)
            if eh is not None:eh=np.maximum(eh,hh)
        for envelope,(low,high) in [('full60',(lo,hi)),('envelope',(el,eh))]:
            np.testing.assert_allclose(b[f'{method}_{envelope}_low'],low,rtol=1e-12)
            if high is not None:np.testing.assert_allclose(b[f'{method}_{envelope}_high'],high,rtol=1e-12)
    print('[VERIFIED] original inputs, all60 fit, forecasts, disjoint validation, independent waterlines',flush=True)
    candidates=pd.read_csv(out/'candidate_validation.csv');assert len(candidates)==1620 and candidates.candidate.tolist()==list(range(1620))
    cache={};water={}
    keys=('sales_units','ending_inventory_units','positive_stock_sku_periods','demand_units','order_units')
    for r in protocol['library']:
        method=r['method'];banks=None
        if method not in water:water[method]=independent_water(b['validation_residual'],vl,hot,r)
        if r['family']=='budget':
            key=(method,r['response_window'])
            if key not in cache:cache[key]=response_banks(y[:28],vfc,b['validation_residual'],response_groups(method,vl,hot),key[1])
            banks=cache[key]
        actual=execute(y[:28],initial,vfc,*water[method],r,banks)['summary'];saved=candidates.iloc[r['candidate']]
        np.testing.assert_allclose([actual[k] for k in keys],[saved[k] for k in keys],rtol=0,atol=0)
    summary=pd.read_csv(out/'summary.csv');assert len(summary)==108
    for selected in design['selected']:
        best=choose(candidates,selected['method'],selected['family'],selected['arm'])
        assert selected['candidate']==(None if best is None else int(best.candidate))
    print('[VERIFIED] all 1620 training candidates, all 54 selections and both deployment arms',flush=True)
    bank={};count=0
    for path in sorted((out/'traces').glob('*.npz')):
        cid,env=path.stem.split('_',1);idx=int(cid[1:]);r=protocol['library'][idx];method=r['method'];banks=None
        tr=np.load(path);actual=identities(test,initial,fc,tr)
        selected=summary[(summary.candidate==idx)&summary.envelope.eq(env)]
        for row in selected.itertuples():
            np.testing.assert_allclose([actual[k] for k in keys],[getattr(row,k) for k in keys],rtol=0,atol=0)
            assert bool(feasible(pd.DataFrame([actual]),row.arm).iloc[0])==row.test_feasible
        if r['family']=='budget':
            key=(method,r['response_window'])
            if key not in bank:bank[key]=response_banks(test,fc,b['final_residual'],response_groups(method,fl,hot),key[1])
            banks=bank[key]
        lo=b[f'{method}_{env}_low'];hi=b[f'{method}_{env}_high'] if method=='sS' else None
        replay=execute(test,initial,fc,lo,hi,r,banks,keep=True)
        for k in tr.files:np.testing.assert_array_equal(replay[k],tr[k])
        count+=test.size
    assert count==complete['test_trace_rows']
    # Alter FUTURE demands and rebuild forecasts and response CDFs. All orders
    # before period 13 must be identical. This checks the actual controller.
    altered=test.copy();altered[12:]=altered[12:]*7+123
    changed_fc=model.make_forecasts(fit,altered,0);np.testing.assert_array_equal(changed_fc[:12],fc[:12])
    for method,family in itertools.product(LABELS,('static','budget')):
        row=summary[summary.method.eq(method)&summary.family.eq(family)&summary.arm.eq('positive_band')&summary.envelope.eq('envelope')].iloc[0]
        if not row.training_feasible:continue
        r=protocol['library'][int(row.candidate)];banks=None
        if family=='budget':banks=response_banks(altered,changed_fc,b['final_residual'],response_groups(method,fl,hot),r['response_window'])
        lo=b[f'{method}_envelope_low'];hi=b[f'{method}_envelope_high'] if method=='sS' else None
        changed=execute(altered,initial,changed_fc,lo,hi,r,banks,keep=True)
        saved=np.load(out/'traces'/f'c{int(row.candidate)}_envelope.npz')
        np.testing.assert_array_equal(changed['orders'][:12],saved['orders'][:12])
    result=dict(status='passed',training_candidates=1620,selection_cells=108,test_trace_rows=count,
                independent_inventory_flow=True,independent_waterline_rebuild=True,future_perturbation=True,
                protocol_sha256=sha(out/'protocol.json'),selection_sha256=sha(out/'frozen_selection.json'))
    dump(out/'verification.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
