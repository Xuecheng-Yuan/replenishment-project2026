"""Declared, paired mechanism experiment; no historical output is overwritten.

Run from the repository root with the existing requirements_v3.txt environment.
The same candidate scores are filtered by three service rules. Final deployment
risk envelopes are an additional intervention, never a validation-data shortcut.
"""
from pathlib import Path
import argparse, hashlib, itertools, json, platform, subprocess, sys, time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/ro_extension'))
from service_band_revision import (model,pipeline,levels,response_banks,response_groups,run_budget,digest,LABELS)

ARMS=('fill95','positive95','positive_band')
FAMILIES=('static','budget','union')
ENVELOPES=('full60','envelope')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def csv(frame,path):pd.DataFrame(frame).to_csv(path,index=False,encoding='utf-8-sig')

def source_revision():
    try:
        result=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True)
        if result.returncode==0:return result.stdout.strip()
    except FileNotFoundError:pass
    stamp=ROOT/'SOURCE_COMMIT.txt'
    return stamp.read_text(encoding='utf-8').strip() if stamp.exists() else 'local snapshot without Git metadata; verify source hashes'

def feasible(frame,arm):
    if arm=='fill95':return frame.fill_rate>=.95
    if arm=='positive95':return frame.positive_stock_ratio>=.95
    return frame.positive_stock_ratio.between(.95,.955)

def choose(frame,method,family,arm):
    rows=frame[frame.method.eq(method)]
    if family!='union':rows=rows[rows.family.eq(family)]
    rows=rows[feasible(rows,arm)]
    return None if rows.empty else rows.sort_values(['inventory_demand_ratio','candidate'],kind='stable').iloc[0]

def recipes(base):
    ans=[]
    for method in LABELS:
        fixed={k:v for k,v in base['selected'][method].items() if k in ('method','beta','radius','scale','gamma','p_s','p_S')}
        for multiplier in np.round(np.arange(.20,2.50001,.01),6):
            ans.append(dict(**fixed,candidate=len(ans),family='static',multiplier=float(multiplier)))
        # Full current feedback grid plus broader service targets with a fixed
        # window/prior. All targets belong to ALL three constraint arms.
        params=list(itertools.product((1,3,7),(0.,3.),(-.01,0.,.01),(.9525,)))
        params+=list(itertools.product((3,),(3.,),(-.01,0.,.01),(.70,.80,.90,.95,.955,.975,.99)))
        for window,prior,bias,target in params:
            ans.append(dict(**fixed,candidate=len(ans),family='budget',response_window=window,
                            response_prior=prior,response_bias=bias,feedback_target=target))
    return ans

def execute(y,initial,fc,low,high,recipe,banks=None,keep=False):
    if recipe['family']=='static':
        scale=recipe['multiplier']
        return model.simulate(y,initial,fc,low*scale,None if high is None else high*scale,keep=keep,positive_guard=True)
    return run_budget(y,initial,fc,low,high,banks,recipe['response_prior'],recipe['response_bias'],
                      target=recipe['feedback_target'],keep=keep)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'outputs');args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():raise RuntimeError('Use a new output directory for another run; completed evidence is immutable.')
    start=time.time()
    base_path=ROOT/'experiments/service95_revision/outputs_v4/frozen_parameters.json'
    base=json.loads(base_path.read_text(encoding='utf-8'));settings=base['forecast'];library=recipes(base)
    protocol=dict(experiment='constraint_mechanism_20261016',declared_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        source_commit=source_revision(),
        run_code_sha256=sha(__file__),base_policy_sha256=sha(base_path),
        service_arms={'fill95':'F >= .95','positive95':'P >= .95','positive_band':'.95 <= P <= .955'},
        families=list(FAMILIES),deployment_risk=list(ENVELOPES),library=library,candidate_count=len(library),
        validation=dict(fit_periods=list(range(29,61)),score_periods=list(range(1,29)),time_start=28,
                        initial='original stock; zero pipeline; score all 28 periods',
                        interpretation='retrospective disjoint training holdout, not a future chronological estimate'),
        final_fit_periods=list(range(1,61)),test_periods=list(range(1,29)),forecast=settings,
        selection='Identical training candidate scores, constrain service then minimize J and candidate id; no test selection',
        envelope_intervention='Final deployment only; no extra risk slices enter validation. Both deployment arms share selected recipe.',
        limitations=['Post-hoc mechanism study on an already repeatedly examined test file.',
                     'Finite restricted controller library; no universal or global policy optimality claim.',
                     'Both static and budget libraries are generated from the same current base recipes, not a full historical algorithm reconstruction.'],
        seed=20261016,bootstrap_replicates=1000,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)
    dump(out/'protocol.json',protocol);protocol_sha=sha(out/'protocol.json')
    print('[PROTOCOL FROZEN]',protocol_sha,len(library),flush=True)
    data=pipeline.verified_data(ROOT/'data/raw',out,False)
    y=data['train_y'].astype(float);x=data['train_x'].astype(float);initial=data['initial_inventory'].astype(float);hot=data['is_hot'].astype(bool)
    assert y.shape==(60,6000)
    labels=model.cluster(y[28:],x[28:])[0]
    residual=model.residual_oof(y[28:],labels,**settings,time_start=28)[0]
    forecast=model.make_forecasts(model.Forecaster(y[28:],labels,**settings,time_start=28),y[:28],0)
    validation=[];bank_cache={};water_cache={}
    for recipe in library:
        method=recipe['method']
        if method not in water_cache:water_cache[method]=levels(residual,labels,hot,recipe)
        lo,hi=water_cache[method];banks=None
        if recipe['family']=='budget':
            key=(method,recipe['response_window'])
            if key not in bank_cache:bank_cache[key]=response_banks(y[:28],forecast,residual,response_groups(method,labels,hot),key[1])
            banks=bank_cache[key]
        run=execute(y[:28],initial,forecast,lo,hi,recipe,banks)
        validation.append(dict(**recipe,**run['summary']))
        if (recipe['candidate']+1)%50==0:print('[TRAIN]',recipe['candidate']+1,'/',len(library),flush=True)
    frame=pd.DataFrame(validation);csv(frame,out/'candidate_validation.csv')
    selected=[]
    for method,family,arm in itertools.product(LABELS,FAMILIES,ARMS):
        best=choose(frame,method,family,arm)
        selected.append(dict(method=method,family=family,arm=arm,candidate=None if best is None else int(best.candidate),
                             training_feasible=best is not None))
    # Refit ALL sixty periods; envelopes use only training data. Four slice
    # arrays align by item; their differing cluster labels do not align by ID.
    lab=model.cluster(y,x)[0];res=model.residual_oof(y,lab,**settings)[0]
    fit=model.Forecaster(y,lab,**settings)
    slices=[]
    for a,b in ((0,24),(0,36),(0,48),(28,60)):
        g=model.cluster(y[a:b],x[a:b])[0]
        slices.append((g,model.residual_oof(y[a:b],g,**settings,time_start=a)[0]))
    policies={};policy_arrays={}
    for method in LABELS:
        recipe=next(r for r in library if r['method']==method);low,high=levels(res,lab,hot,recipe)
        env=low.copy();eh=None if high is None else high.copy()
        for gl,rr in slices:
            ll,hh=levels(rr,gl,hot,recipe);env=np.maximum(env,ll)
            if eh is not None:eh=np.maximum(eh,hh)
        policies[(method,'full60')]=(low,high);policies[(method,'envelope')]=(env,eh)
        for name,(ll,hh) in [('full60',(low,high)),('envelope',(env,eh))]:
            policy_arrays[f'{method}_{name}_low']=ll
            if hh is not None:policy_arrays[f'{method}_{name}_high']=hh
    design=dict(protocol_sha256=protocol_sha,selected=selected,training_y_sha256=digest(y),training_x_sha256=digest(x),
                validation_forecast_sha256=digest(forecast),validation_labels_sha256=digest(labels),
                validation_residual_sha256=digest(residual),final_labels_sha256=digest(lab),final_residual_sha256=digest(res),
                final_fit_observations=y.size,initial_sha256=digest(initial),
                policy_hashes={k:digest(v) for k,v in policy_arrays.items()})
    dump(out/'frozen_selection.json',design);selection_sha=sha(out/'frozen_selection.json')
    print('[SELECTION FROZEN BEFORE TEST]',selection_sha,flush=True)
    test=data['test_y'].astype(float);fc=model.make_forecasts(fit,test,0)
    np.savez_compressed(out/'shared_arrays.npz',train_y=y,train_x_mean=x.mean(0),test_y=test,initial=initial,hot=hot,item_id=data['item_id'],
                        validation_labels=labels,validation_residual=residual,validation_forecast=forecast,
                        final_labels=lab,final_residual=res,final_forecast=fc,**policy_arrays)
    unique=sorted({r['candidate'] for r in selected if r['candidate'] is not None})
    results={};bc={};primitive=[];periods=[];segments=[]
    trace_dir=out/'traces';trace_dir.mkdir(exist_ok=True)
    for candidate,envelope in itertools.product(unique,ENVELOPES):
        recipe=library[candidate];method=recipe['method'];lo,hi=policies[(method,envelope)];banks=None
        if recipe['family']=='budget':
            key=(method,recipe['response_window'])
            if key not in bc:bc[key]=response_banks(test,fc,res,response_groups(method,lab,hot),key[1])
            banks=bc[key]
        run=execute(test,initial,fc,lo,hi,recipe,banks,keep=True)
        name=f'c{candidate}_{envelope}'
        arrays={k:run[k].astype(np.uint32) for k in ('commencing','ending','sold','orders','plans')}
        for k in arrays:np.testing.assert_array_equal(arrays[k],run[k])
        np.savez_compressed(trace_dir/(name+'.npz'),**arrays)
        results[(candidate,envelope)]=run['summary']
        for i in range(6000):primitive.append(dict(candidate=candidate,envelope=envelope,method=method,item_id=data['item_id'][i],
            category='hot' if hot[i] else 'tail',demand=float(test[:,i].sum()),sales=float(run['sold'][:,i].sum()),
            ending=float(run['ending'][:,i].sum()),positive_days=int((run['ending'][:,i]>0).sum()),order=float(run['orders'][:,i].sum())))
        for t in range(28):periods.append(dict(candidate=candidate,envelope=envelope,method=method,period=t+1,
            **model.summarize(test[t:t+1],run['sold'][t:t+1],run['ending'][t:t+1],run['commencing'][t:t+1],run['orders'][t:t+1])))
        for category,mask in [('all',np.ones(6000,bool)),('hot',hot),('tail',~hot)]:
            segments.append(dict(candidate=candidate,envelope=envelope,method=method,category=category,
                **model.summarize(test[:,mask],run['sold'][:,mask],run['ending'][:,mask],run['commencing'][:,mask],run['orders'][:,mask])))
        print('[TEST REPLAY]',name,method,flush=True)
    summary=[]
    for entry,envelope in itertools.product(selected,ENVELOPES):
        candidate=entry['candidate'];row=dict(**entry,envelope=envelope)
        if candidate is not None:
            val=frame.loc[frame.candidate.eq(candidate)].iloc[0]
            row.update(train_F=float(val.fill_rate),train_P=float(val.positive_stock_ratio),train_J=float(val.inventory_demand_ratio),
                       **results[(candidate,envelope)])
            row['test_feasible']=bool(feasible(pd.DataFrame([row]),entry['arm']).iloc[0])
        else:row['test_feasible']=False
        summary.append(row)
    csv(summary,out/'summary.csv');csv(periods,out/'periods.csv');csv(segments,out/'segments.csv')
    pd.DataFrame(primitive).to_csv(out/'sku_summary.csv.gz',index=False,encoding='utf-8-sig',compression={'method':'gzip','mtime':0})
    assert sha(out/'protocol.json')==protocol_sha and sha(out/'frozen_selection.json')==selection_sha
    dump(out/'completed.json',dict(protocol_sha256=protocol_sha,selection_sha256=selection_sha,
        elapsed_seconds=time.time()-start,candidate_rows=len(frame),selected_cells=len(summary),unique_test_replays=len(results),
        test_trace_rows=len(results)*test.size,code_sha256=sha(__file__),
        note='Raw test file loaded for hash audit at start; no numerical test score was used for selection in this run. Existing test was previously examined.'))
    print('[COMPLETED]',len(frame),'candidates;',len(results),'unique replays;',time.time()-start,'seconds',flush=True)

if __name__=='__main__':main()
