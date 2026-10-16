"""Reproduce three DRO groupings and (s,S) from the two source Excel files.

    python -m pip install -r requirements_v3.txt
    python run_v3.py

The first run streams all sheets into a verified cache. Later runs check SHA256
of both source workbooks and the cache before reusing it. Test demands are used
only in the final policy replay, after parameters are written to disk.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from config import Config
from data_io import load_period_data
from model_v2 import cluster, Forecaster, rolling_predictions, residual_oof, risk_levels, simulate, make_forecasts, summarize
from run_v2 import ss_quantiles, feature_diagnostics

ROOT=Path(__file__).resolve().parent
METHODS=('DRO_cluster','DRO_global','DRO_category','sS','SAA')
METHOD_LABELS={'DRO_cluster':'特征聚类DRO','DRO_global':'不分组DRO','DRO_category':'爆品/长尾DRO','sS':'传统(s,S)','SAA':'零半径经验模型'}


def csv(frame,path):
    frame.to_csv(path,index=False,encoding='utf-8-sig')


def digest(path):
    sha=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''): sha.update(block)
    return sha.hexdigest()


def verified_data(source,out,refresh):
    cache_dir=ROOT/'.cache'; cache=cache_dir/'period_data.npz'
    manifest=cache_dir/'source_manifest_v3.json'
    signatures=[]
    for name in ('replenishment_train.xlsx','replenishment_test.xlsx'):
        path=source/name
        if not path.is_file(): raise FileNotFoundError(path)
        signatures.append(dict(file=name,sha256=digest(path)))
    old=json.loads(manifest.read_text(encoding='utf-8')) if manifest.is_file() else None
    valid=old is not None and old.get('inputs')==signatures and cache.is_file() and old.get('cache_sha256')==digest(cache)
    # Adopt the previously completed cell-by-cell audit when exactly the same
    # source and cache hashes match; no need to reparse 88 large worksheets.
    if not valid and not refresh and cache.is_file():
        prior=ROOT/'outputs_v2/input_audit.json'
        if prior.is_file():
            earlier=json.loads(prior.read_text(encoding='utf-8'))
            if earlier.get('inputs')==signatures and earlier.get('cache_sha256')==digest(cache):
                valid=True
                old=dict(inputs=signatures,cache_sha256=earlier['cache_sha256'],validation='all source cells matched; see outputs_v2/input_audit.json')
                manifest.write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding='utf-8')
    if refresh or not valid:
        print('[data] Reading both workbooks into a fresh cache.',flush=True)
        load_period_data(Config(data_dir=source,cache_dir=cache_dir,output_dir=out),refresh_cache=True)
        old=dict(inputs=signatures,cache_sha256=digest(cache),validation='streamed all worksheets; checked keys, categories, initial stock, finite nonnegative demand/features')
        manifest.write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'input_audit.json').write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding='utf-8')
    return np.load(cache,allow_pickle=False)


def variant_labels(cluster_labels,is_hot):
    return {'DRO_cluster':np.asarray(cluster_labels,dtype=int),
            'DRO_global':np.zeros(len(is_hot),dtype=int),
            'DRO_category':np.asarray(is_hot,dtype=int)}


def selected(candidates,forward_target=None,reset_target=.95):
    # The reset fold matches the test's Period_1 restart. Forward fold is
    # a separate temporal-transfer diagnostic, not a hidden second objective.
    eligible=[r for r in candidates if .95<=r['reset_service']<=.955]
    if not eligible: return None
    return min(eligible,key=lambda r:(r['reset_inventory_ratio'],abs(r['reset_service']-.95)))


def main():
    ap=argparse.ArgumentParser(description='Cluster/global/category DRO rolling inventory benchmark')
    ap.add_argument('--data-dir',type=Path,default=(ROOT.parents[1]/'data'/'raw' if (ROOT.parents[1]/'data'/'raw').is_dir() else ROOT.parent))
    ap.add_argument('--output-dir',type=Path,default=ROOT/'outputs_v4')
    ap.add_argument('--refresh-cache',action='store_true')
    args=ap.parse_args(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    started=time.time(); data=verified_data(args.data_dir.resolve(),out,args.refresh_cache)
    y=data['train_y'].astype(float); x=data['train_x'].astype(float)
    ids=data['item_id']; hot=data['is_hot']; initial=data['initial_inventory'].astype(float)
    if y.shape!=(60,6000) or data['test_y'].shape!=(28,6000):
        raise ValueError(f'Expected train (60,6000), test (28,6000); got {y.shape}, {data["test_y"].shape}')
    forward_end=36
    group_f,_,_=cluster(y[:forward_end],x[:forward_end])
    group_r,_,_=cluster(y[28:],x[28:])
    groups_f=variant_labels(group_f,hot); groups_r=variant_labels(group_r,hot)

    forecast_rows=[]
    for blend in (0.,.5,1.):
        for window in (3,6,12):
            for prior in (0.,.5,1.,3.):
                mf=Forecaster(y[:36],group_f,blend,window,prior)
                pf=rolling_predictions(mf,y[36:],36,history=y[:36])[:,0]
                mr=Forecaster(y[28:],group_r,blend,window,prior,time_start=28)
                pr=rolling_predictions(mr,y[:28],0)[:,0]
                fw=float(abs(pf-y[36:]).sum()/y[36:].sum())
                rw=float(abs(pr-y[:28]).sum()/y[:28].sum())
                forecast_rows.append(dict(blend=blend,window=window,prior=prior,forward_wape=fw,reset_wape=rw,selection_wape=(fw+rw)/2))
    best=min(forecast_rows,key=lambda r:r['selection_wape'])
    settings={k:best[k] for k in ('blend','window','prior')}
    csv(pd.DataFrame(forecast_rows),out/'forecast_validation.csv')
    print('[forecast]',best,flush=True)
    mf=Forecaster(y[:36],group_f,**settings)
    mr=Forecaster(y[28:],group_r,**settings,time_start=28)
    fc_f=make_forecasts(mf,y[36:],36,history=y[:36])
    fc_r=make_forecasts(mr,y[:28],0)
    residual_f,_=residual_oof(y[:36],group_f,**settings)
    residual_r,_=residual_oof(y[28:],group_r,**settings,time_start=28)
    target=.95
    forward_target=.95
    beta_grid=(.95,.96,.97,.98,.99,.995,.999)
    radius_grid=(0.,.005,.01,.025,.05,.075,.1,.15,.2,.3,.5)
    candidates=[]; chosen={}
    for method in ('DRO_cluster','DRO_global','DRO_category'):
        method_candidates=[]
        for beta in beta_grid:
            for radius in radius_grid[1:]:
                zf,_=risk_levels(residual_f,groups_f[method],beta,radius)
                zr,_=risk_levels(residual_r,groups_r[method],beta,radius)
                sf=simulate(y[36:],initial,fc_f,zf,positive_guard=True)['controlled']
                sr=simulate(y[:28],initial,fc_r,zr,positive_guard=True)['summary']
                row=dict(method=method,beta=beta,radius=radius,forward_service=sf['positive_stock_ratio'],forward_fill=sf['fill_rate'],
                         forward_inventory_ratio=sf['inventory_demand_ratio'],reset_service=sr['positive_stock_ratio'],reset_fill=sr['fill_rate'],
                         reset_inventory_ratio=sr['inventory_demand_ratio'])
                candidates.append(row); method_candidates.append(row)
        chosen[method]=selected(method_candidates,forward_target)
        if chosen[method] is None:
            csv(pd.DataFrame(candidates),out/'policy_validation.csv')
            raise RuntimeError(f'{method} has no training-feasible parameter; inspect policy_validation.csv')
        print('[selected]',method,chosen[method],flush=True)
    ss_rows=[]
    for p in (.95,.975,.99,.999):
      for scale in np.arange(1.,2.001,.025):
        for gap in (0.,.025,.05):
            pS=min(p+gap,.999)
            lf=ss_quantiles(residual_f,group_f,p); uf=ss_quantiles(residual_f,group_f,pS)
            lr=ss_quantiles(residual_r,group_r,p); ur=ss_quantiles(residual_r,group_r,pS)
            lf*=scale; uf*=scale; lr*=scale; ur*=scale
            sf=simulate(y[36:],initial,fc_f,lf,uf,positive_guard=True)['controlled']
            sr=simulate(y[:28],initial,fc_r,lr,ur,positive_guard=True)['summary']
            row=dict(method='sS',p_s=p,p_S=pS,scale=float(scale),forward_service=sf['positive_stock_ratio'],forward_fill=sf['fill_rate'],
                     forward_inventory_ratio=sf['inventory_demand_ratio'],reset_service=sr['positive_stock_ratio'],reset_fill=sr['fill_rate'],
                     reset_inventory_ratio=sr['inventory_demand_ratio'])
            candidates.append(row); ss_rows.append(row)
    chosen['sS']=selected(ss_rows,forward_target)
    if chosen['sS'] is None: raise RuntimeError('(s,S) has no training-feasible parameter')
    saa_rows=[]
    for beta in beta_grid:
      for scale in np.arange(1.,2.001,.025):
        zf,_=risk_levels(residual_f,group_f,beta,0.)
        zr,_=risk_levels(residual_r,group_r,beta,0.)
        zf*=scale; zr*=scale
        sf=simulate(y[36:],initial,fc_f,zf,positive_guard=True)['controlled']
        sr=simulate(y[:28],initial,fc_r,zr,positive_guard=True)['summary']
        row=dict(method='SAA',beta=beta,radius=0.,scale=float(scale),forward_service=sf['positive_stock_ratio'],forward_fill=sf['fill_rate'],
                 forward_inventory_ratio=sf['inventory_demand_ratio'],reset_service=sr['positive_stock_ratio'],reset_fill=sr['fill_rate'],
                 reset_inventory_ratio=sr['inventory_demand_ratio'])
        candidates.append(row); saa_rows.append(row)
    chosen['SAA']=selected(saa_rows,forward_target)
    if chosen['SAA'] is None: raise RuntimeError('SAA has no training-feasible positive-stock parameter')
    csv(pd.DataFrame(candidates),out/'policy_validation.csv')

    # Freeze all selected parameters before touching the test outcomes.
    # Keep the exact fitted model and residual bank whose policies were
    # calibrated on the disjoint restart block; refitting invalidates the band.
    labels=group_r
    xmean=x[28:].mean(0)
    groups=variant_labels(labels,hot)
    final=mr
    relative=residual_r
    multipliers={}; risk_profiles={}
    for method in ('DRO_cluster','DRO_global','DRO_category'):
        row=chosen[method]
        multipliers[method],risk_profiles[method]=risk_levels(relative,groups[method],row['beta'],row['radius'])
    multipliers['SAA'],_=risk_levels(relative,labels,chosen['SAA']['beta'],0.)
    multipliers['SAA']*=chosen['SAA']['scale']
    sslo=ss_quantiles(relative,labels,chosen['sS']['p_s'])*chosen['sS']['scale']
    sshi=ss_quantiles(relative,labels,chosen['sS']['p_S'])*chosen['sS']['scale']
    frozen=dict(target_service=target,service_lower=.95,service_upper=.955,service_metric="positive_stock_ratio",
                service_definition="mean of period-ending inventory > 0 across all SKU-periods",
                calibration_scope="training reset fold all 28 periods; forward fold diagnostic only",
                model_fit_periods="29-60",calibration_periods="1-28",refit_after_calibration=False,
                positive_integer_guard=True,forecast=settings,selected=chosen,
                lead_time=1,review_period=0,horizon=4,test_periods_restart=True,
                inventory_rule='after demand t, order; receive before demand t+1',
                objective='minimize sum of period-ending inventory / sum of demand',
                objective_metric='inventory_demand_ratio',objective_direction='min',
                ambiguity='rectangular W-infinity balls around nonnegative relative residual samples',
                selection='training-only forward and early-reset blocked validation',
                test_used_in_prior_revision_diagnosis=True,training_periods=60,test_periods=28,skus=len(ids))
    (out/'frozen_parameters.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(out/'fitted_model.npz',labels=labels,category_labels=groups['DRO_category'],
                        relative_residual=relative,item_id=ids,ss_lower=sslo,ss_upper=sshi,
                        **{f'level_{k}':v for k,v in multipliers.items()})
    profiles=[]
    for method in ('DRO_cluster','DRO_global','DRO_category'):
        for group,values in risk_profiles[method].items():
            mask=groups[method]==group
            profiles.append(dict(method=method,group=group,skus=int(mask.sum()),train_mean=float(y[:,mask].mean()),
                                 **values))
    csv(pd.DataFrame(profiles),out/'risk_group_profile.csv')
    cluster_table=pd.DataFrame(dict(item_id=ids,cluster=labels,category=np.where(hot,'爆品','长尾品')))
    for j in range(20): cluster_table[f'train_mean_x{j+1}']=xmean[:,j]
    csv(cluster_table,out/'cluster_assignment.csv')
    feature_diagnostics(y,x,out)
    print('[test] All model choices frozen; replaying held-out test.',flush=True)
    test=data['test_y'].astype(float)
    forecasts=make_forecasts(final,test,start=0)
    runs={}
    for method in METHODS:
        if method=='sS': result=simulate(test,initial,forecasts,sslo,sshi,keep=True,positive_guard=True)
        else: result=simulate(test,initial,forecasts,multipliers[method],keep=True,positive_guard=True)
        runs[method]=result
        print('[test]',method,'positive_stock_service',round(result['summary']['positive_stock_ratio'],6),'fill',round(result['summary']['fill_rate'],6),'inventory_demand_ratio',round(result['summary']['inventory_demand_ratio'],6),flush=True)
    output_results(out,runs,test,forecasts,ids,labels,hot)
    forecast_scores=[]
    for h in range(4):
        actual=test[h+1:]; predicted=forecasts[:28-h-1,h]
        forecast_scores.append(dict(horizon=h+1,n_cells=actual.size,
                                    wape=float(abs(actual-predicted).sum()/actual.sum()),
                                    rmse=float(np.sqrt(np.mean((actual-predicted)**2)))))
    csv(pd.DataFrame(forecast_scores),out/'forecast_test_metrics.csv')
    from report_v3 import build_report
    build_report(out,runs,frozen,forecast_scores,time.time()-started)
    print('[done]',out,flush=True)


def output_results(out,runs,test,forecasts,ids,labels,hot):
    T,n=test.shape
    summary=[]; periods=[]; segments=[]
    for method in METHODS:
        r=runs[method]
        summary.extend([dict(method=method,scope='all_28_periods',**r['summary']),
                        dict(method=method,scope='controllable_periods_2_to_28',**r['controlled'])])
        for t in range(T):
            result=summarize(test[t:t+1],r['sold'][t:t+1],r['ending'][t:t+1],r['commencing'][t:t+1],r['orders'][t:t+1])
            periods.append(dict(method=method,period=t+1,**result))
        for category in ('爆品','长尾品'):
            mask=hot if category=='爆品' else ~hot
            result=summarize(test[:,mask],r['sold'][:,mask],r['ending'][:,mask],r['commencing'][:,mask],r['orders'][:,mask])
            segments.append(dict(method=method,segment=category,**result))
        frame=pd.DataFrame(dict(period=np.repeat(np.arange(1,T+1),n),item_id=np.tile(ids,T),
                                cluster=np.tile(labels,T),category=np.tile(np.where(hot,'爆品','长尾品'),T),
                                demand=test.ravel(),available=r['commencing'].ravel(),
                                sales=r['sold'].ravel(),lost=(test-r['sold']).ravel(),
                                ending=r['ending'].ravel(),order=r['orders'].ravel(),
                                arrival_period=np.repeat(np.arange(2,T+2),n)))
        for h in range(4):
            frame[f'forecast_h{h+1}']=forecasts[:,h,:].ravel()
            frame[f'target_h{h+1}']=r['targets'][:,h,:].ravel()
            frame[f'planned_order_h{h+1}']=r['plans'][:,h,:].ravel()
        csv(frame,out/f'detail_{method}.csv')
        orders=pd.DataFrame(r['orders'].T,columns=[f'order_after_period_{t+1}' for t in range(T)])
        orders.insert(0,'item_id',ids)
        csv(orders,out/f'orders_{method}.csv')
    csv(pd.DataFrame(summary),out/'summary.csv')
    csv(pd.DataFrame(periods),out/'results_by_period.csv')
    csv(pd.DataFrame(segments),out/'summary_by_segment.csv')
    for h in range(4):
        matrix=pd.DataFrame(forecasts[:,h,:].T,columns=[f'issued_after_period_{t+1}' for t in range(T)])
        matrix.insert(0,'item_id',ids)
        csv(matrix,out/f'forecast_h{h+1}.csv')
    # One row per SKU and decision period, all methods side by side. This is
    # the full-grain input for the presentation workbook and independent audit.
    compare=pd.DataFrame(dict(period=np.repeat(np.arange(1,T+1),n),item_id=np.tile(ids,T),
                              category=np.tile(np.where(hot,'爆品','长尾品'),T),cluster=np.tile(labels,T),
                              actual_demand=test.ravel(),forecast_h1=forecasts[:,0,:].ravel(),
                              forecast_h2=forecasts[:,1,:].ravel(),forecast_h3=forecasts[:,2,:].ravel(),
                              forecast_h4=forecasts[:,3,:].ravel()))
    for method in METHODS:
        r=runs[method]
        compare[f'order_{method}']=r['orders'].ravel()
        compare[f'lost_{method}']=(test-r['sold']).ravel()
        compare[f'ending_{method}']=r['ending'].ravel()
    csv(compare,out/'decision_comparison_full.csv')


if __name__=='__main__': main()
