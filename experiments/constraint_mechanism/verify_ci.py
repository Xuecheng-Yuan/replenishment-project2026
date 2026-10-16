"""Portable frozen-evidence audit (no LFS workbook download required).

Full raw Excel and feature-array reconstruction was performed locally by
verify_experiment.py. Here we reconcile committed arrays against historical
artifacts, replay all training candidates, and audit all selected test paths.
"""
from verify_experiment import *

def main():
    out=HERE/'outputs';manifest=json.loads((HERE/'SHA256SUMS.json').read_text(encoding='utf-8'))
    for name,signature in manifest.items():assert sha(HERE/name)==signature,name
    complete=json.loads((out/'completed.json').read_text());p=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    d=json.loads((out/'frozen_selection.json').read_text(encoding='utf-8'))
    assert sha(HERE/'run_experiment.py')==complete['code_sha256']==p['run_code_sha256']
    assert sha(out/'protocol.json')==complete['protocol_sha256']==d['protocol_sha256']
    assert sha(out/'frozen_selection.json')==complete['selection_sha256']
    source=ROOT/'experiments/service95_revision/outputs_v4'
    original=np.load(source/'fitted_model.npz');b=np.load(out/'shared_arrays.npz')
    for ours,other in [('train_y','train_y_all60'),('train_x_mean','train_x_mean'),('initial','initial'),('final_labels','final_labels'),('final_residual','final_relative_residual'),('final_forecast','forecasts_test')]:
        np.testing.assert_array_equal(b[ours],original[other])
    assert digest(b['train_y'])==d['training_y_sha256'] and b['train_y'].shape==(60,6000)
    assert json.loads((source/'input_audit.json').read_text(encoding='utf-8'))['inputs']==json.loads((out/'input_audit.json').read_text(encoding='utf-8'))['inputs']
    data=pd.read_csv(source/'detail_DRO_category.csv.gz',usecols=['demand','category'],float_precision='round_trip')
    np.testing.assert_array_equal(b['test_y'],data.demand.to_numpy().reshape(28,6000))
    np.testing.assert_array_equal(b['hot'],data.category.to_numpy().reshape(28,6000)[0]=='爆品')
    candidates=pd.read_csv(out/'candidate_validation.csv');assert len(candidates)==1620
    train=b['train_y'];vl=b['validation_labels'];vr=b['validation_residual'];vfc=b['validation_forecast'];init=b['initial'];hot=b['hot']
    expected_fc=model.make_forecasts(model.Forecaster(train[28:],vl,**p['forecast'],time_start=28),train[:28],0)
    np.testing.assert_allclose(expected_fc,vfc,rtol=1e-12,atol=1e-9)
    cache={};water={};keys=('sales_units','demand_units','ending_inventory_units','positive_stock_sku_periods','order_units')
    for r in p['library']:
        method=r['method'];banks=None
        if method not in water:water[method]=independent_water(vr,vl,hot,r)
        if r['family']=='budget':
            key=(method,r['response_window'])
            if key not in cache:cache[key]=response_banks(train[:28],vfc,vr,response_groups(method,vl,hot),key[1])
            banks=cache[key]
        actual=execute(train[:28],init,vfc,*water[method],r,banks)['summary'];saved=candidates.iloc[r['candidate']]
        np.testing.assert_allclose([actual[k] for k in keys],[saved[k] for k in keys],rtol=0,atol=0)
    summary=pd.read_csv(out/'summary.csv');assert len(summary)==108
    for chosen in d['selected']:
        best=choose(candidates,chosen['method'],chosen['family'],chosen['arm'])
        assert chosen['candidate']==(None if best is None else int(best.candidate))
    for method,family in itertools.product(LABELS,FAMILIES):
        a=summary[summary.method.eq(method)&summary.family.eq(family)&summary.envelope.eq('full60')].set_index('arm')
        assert a.loc['positive95','candidate']==a.loc['positive_band','candidate']
    bank={};count=0
    for path in (out/'traces').glob('*.npz'):
        cid,env=path.stem.split('_',1);idx=int(cid[1:]);r=p['library'][idx];method=r['method'];trace=np.load(path)
        actual=identities(b['test_y'],init,b['final_forecast'],trace)
        for row in summary[(summary.candidate==idx)&summary.envelope.eq(env)].itertuples():
            np.testing.assert_allclose([actual[k] for k in keys],[getattr(row,k) for k in keys],rtol=0,atol=0)
            assert bool(feasible(pd.DataFrame([actual]),row.arm).iloc[0])==row.test_feasible
        lo=b[f'{method}_{env}_low'];hi=b[f'{method}_{env}_high'] if method=='sS' else None;banks=None
        if r['family']=='budget':
            key=(method,r['response_window'])
            if key not in bank:bank[key]=response_banks(b['test_y'],b['final_forecast'],b['final_residual'],response_groups(method,b['final_labels'],hot),key[1])
            banks=bank[key]
        replay=execute(b['test_y'],init,b['final_forecast'],lo,hi,r,banks,keep=True)
        for k in trace.files:np.testing.assert_array_equal(replay[k],trace[k])
        count+=b['test_y'].size
    assert count==8064000==complete['test_trace_rows']
    # Reproduce historical audit and paired statistics; compare regenerated
    # numeric tables to the committed report evidence at numerical tolerance.
    tables=('paired_contrasts.csv','factor_effects.csv','published_summary.csv','published_segments.csv','waterline_shape.csv','feasible_overlap.csv','cell_winners.csv')
    before={name:pd.read_csv(out/name) for name in tables}
    from analyze_results import main as analyze
    sys.argv=[sys.argv[0]];analyze()
    for name in tables:
        after=pd.read_csv(out/name);pd.testing.assert_frame_equal(before[name],after,check_exact=False,rtol=1e-11,atol=1e-11)
    print('CI frozen-evidence audit PASSED: complete manifest, original artifact reconciliation, 1620 candidates, 108 cells, 8064000 trace rows and reproduced statistics.')

if __name__=='__main__':main()
