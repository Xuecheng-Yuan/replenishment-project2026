"""Audit frozen historical trajectories and calculate all evidence for reports."""
from run_experiment import *

def audit_published(out,b):
    rows=[];segments=[];files=[];sku={};fcmap={};test=b['test_y'];initial=b['initial'];hot=b['hot']
    for version,folder in [('old','demand_fill_v3/with_ro'),('current','service95_revision')]:
        home=ROOT/'experiments'/folder/'outputs_v4';declared=pd.read_csv(home/'summary.csv').query("scope=='all_28_periods'")
        frozen=json.loads((home/'frozen_parameters.json').read_text(encoding='utf-8'))
        old_bank=np.load(ROOT/'experiments/demand_fill_v3/outputs_v4/fitted_model.npz') if version=='old' else None
        for method in LABELS:
            path=home/f'detail_{method}.csv.gz';f=pd.read_csv(path,float_precision='round_trip');shape=test.shape
            files.append(dict(version=version,method=method,path=str(path.relative_to(ROOT)).replace('\\','/'),sha256=sha(path)))
            c,e,s,q=(f[k].to_numpy().reshape(shape) for k in ('available','ending','sales','order'))
            np.testing.assert_array_equal(f.demand.to_numpy().reshape(shape),test)
            np.testing.assert_array_equal(c[0],initial);np.testing.assert_array_equal(c[1:],e[:-1]+q[:-1])
            np.testing.assert_array_equal(s,np.minimum(c,test));np.testing.assert_array_equal(e,c-s)
            fc=np.stack([f[f'forecast_h{h}'].to_numpy().reshape(shape) for h in range(1,5)],axis=1)
            fcmap[(version,method)]=digest(fc)
            if version=='old':
                low=old_bank['ss_lower'] if method=='sS' else np.full(6000,frozen['selected']['RO']['beta']*frozen['selected']['RO']['upper_relative']) if method=='RO' else old_bank['level_'+method]
                high=old_bank['ss_upper'] if method=='sS' else None
                replay=model.simulate(test,initial,fc,low,high,keep=True,positive_guard=False)
            else:
                policy=np.load(home/f'policy_{method}.npz');recipe=frozen['selected'][method]
                lo,hi=policy['low'],policy['high'] if method=='sS' else None
                np.testing.assert_allclose(lo,b[f'{method}_envelope_low'],rtol=1e-12)
                groups=response_groups(method,b['final_labels'],hot)
                banks=response_banks(test,fc,b['final_residual'],groups,recipe['response_window'])
                replay=run_budget(test,initial,fc,lo,hi,banks,recipe['response_prior'],recipe['response_bias'],target=frozen['feedback_target'],keep=True)
            for name,array in [('commencing',c),('ending',e),('sold',s),('orders',q)]:np.testing.assert_array_equal(replay[name],array)
            for h in range(4):np.testing.assert_array_equal(replay['plans'][:,h],f[f'planned_order_h{h+1}'].to_numpy().reshape(shape))
            result=model.summarize(test,s,e,c,q);d=declared[declared.method.eq(method)].iloc[0]
            for k in ('fill_rate','positive_stock_ratio','inventory_demand_ratio'):np.testing.assert_allclose(result[k],d[k],rtol=1e-12)
            rows.append(dict(version=version,method=method,**result,forecast_max_abs_vs_full60=float(np.abs(fc-b['final_forecast']).max()),
                             exact_sellout_fraction=float(((e==0)&(s==test)).mean()),
                             lost_day_fraction=float((s<test).mean()),forecast_sha256=digest(fc),source_trace_sha256=sha(path)))
            sku[(version,method)]=e.sum(0)
            for cat,mask in [('all',np.ones(6000,bool)),('hot',hot),('tail',~hot)]:
                segments.append(dict(version=version,method=method,category=cat,sku_weight=float(mask.mean()),
                    demand_weight=float(test[:,mask].sum()/test.sum()),**model.summarize(test[:,mask],s[:,mask],e[:,mask],c[:,mask],q[:,mask])))
        files.append(dict(version=version,path=str((home/'frozen_parameters.json').relative_to(ROOT)).replace('\\','/'),sha256=sha(home/'frozen_parameters.json')))
    audit=dict(files=files,all_inventory_identities_passed=True,all_frozen_policy_replays_passed=True,
               max_forecast_difference_vs_full60=max(r['forecast_max_abs_vs_full60'] for r in rows),
               forecasts_equivalent_at_1e_9=all(r['forecast_max_abs_vs_full60']<1e-9 for r in rows),
               same_forecasts_old_current=len(set(fcmap.values()))==1,forecast_hashes={a+'_'+m:h for (a,m),h in fcmap.items()})
    dump(out/'published_audit.json',audit);csv(rows,out/'published_summary.csv');csv(segments,out/'published_segments.csv')
    print('[PUBLISHED AUDIT] old/current frozen six-policy replay passed; identical forecasts:',audit['same_forecasts_old_current'],flush=True)
    return sku

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'outputs');args=p.parse_args();out=args.output
    complete=json.loads((out/'completed.json').read_text());assert complete['candidate_rows']==1620
    b=np.load(out/'shared_arrays.npz');s=pd.read_csv(out/'summary.csv');v=pd.read_csv(out/'candidate_validation.csv');hot=b['hot'];test=b['test_y']
    published_sku=audit_published(out,b)
    winners=[]
    for arm,family,envelope in itertools.product(ARMS,FAMILIES,ENVELOPES):
        group=s[s.arm.eq(arm)&s.family.eq(family)&s.envelope.eq(envelope)]
        train=group[group.training_feasible].sort_values(['train_J','candidate'],kind='stable')
        valid=group[group.test_feasible].sort_values(['inventory_demand_ratio','candidate'],kind='stable')
        train_winner=train.iloc[0] if len(train) else None;tw=valid.iloc[0] if len(valid) else None
        winners.append(dict(arm=arm,family=family,envelope=envelope,
            training_winner=None if train_winner is None else train_winner.method,
            training_winner_test_feasible=False if train_winner is None else bool(train_winner.test_feasible),
            test_feasible_methods=len(valid),descriptive_test_winner=None if tw is None else tw.method,
            test_winner_J=None if tw is None else float(tw.inventory_demand_ratio)))
    csv(winners,out/'cell_winners.csv')
    overlap=[]
    for method,family in itertools.product(LABELS,('static','budget','union')):
        subset=v[v.method.eq(method)]
        if family!='union':subset=subset[subset.family.eq(family)]
        f=feasible(subset,'fill95');p95=feasible(subset,'positive95');band=feasible(subset,'positive_band')
        overlap.append(dict(method=method,family=family,candidates=len(subset),fill_feasible=int(f.sum()),
            positive_lower_feasible=int(p95.sum()),positive_band_feasible=int(band.sum()),
            fill_and_band=int((f&band).sum()),fill_only=int((f&~band).sum()),band_only=int((band&~f).sum())))
    csv(overlap,out/'feasible_overlap.csv')
    # Shared, stratified item blocks: preserve 480 hot + 5520 tail items, each
    # with its complete 28-day path. These CIs condition on training and policies.
    rng=np.random.default_rng(20261016);weights=np.empty((1000,6000))
    for mask in (hot,~hot):
        n=int(mask.sum());weights[:,mask]=rng.multinomial(n,np.full(n,1/n),size=1000)
    denominators=weights@test.sum(0);contrasts=[];pair_vectors={}
    for family,envelope,arm in itertools.product(FAMILIES,ENVELOPES,ARMS):
        group=s[s.family.eq(family)&s.envelope.eq(envelope)&s.arm.eq(arm)].set_index('method')
        if not group.loc[['DRO_category','sS'],'training_feasible'].all():continue
        ca,ss=(int(group.loc[m,'candidate']) for m in ('DRO_category','sS'))
        ea=np.load(out/'traces'/f'c{ca}_{envelope}.npz')['ending'].sum(0).astype(float)
        es=np.load(out/'traces'/f'c{ss}_{envelope}.npz')['ending'].sum(0).astype(float)
        delta=ea-es;pair_vectors[(family,envelope,arm)]=delta
        boot=(weights@delta)/denominators;lower,upper=np.quantile(boot,[.025,.975])
        contrasts.append(dict(family=family,envelope=envelope,arm=arm,delta_J=float(delta.sum()/test.sum()),
            ci_low=float(lower),ci_high=float(upper),both_test_feasible=bool(group.loc[['DRO_category','sS'],'test_feasible'].all()),
            category_test_feasible=bool(group.loc['DRO_category','test_feasible']),ss_test_feasible=bool(group.loc['sS','test_feasible'])))
    for version in ('old','current'):
        delta=published_sku[(version,'DRO_category')]-published_sku[(version,'sS')]
        boot=(weights@delta)/denominators;lower,upper=np.quantile(boot,[.025,.975])
        contrasts.append(dict(family='published',envelope=version,arm='historical',delta_J=float(delta.sum()/test.sum()),
                              ci_low=float(lower),ci_high=float(upper),both_test_feasible=True))
    csv(contrasts,out/'paired_contrasts.csv')
    effects=[]
    def effect(name,case,a,c):
        delta=c-a;boot=(weights@delta)/denominators;lower,upper=np.quantile(boot,[.025,.975])
        effects.append(dict(effect=name,case=case,delta_relative_J=float(delta.sum()/test.sum()),ci_low=float(lower),ci_high=float(upper)))
    for family,envelope in itertools.product(FAMILIES,ENVELOPES):
        effect('metric_and_band',family+'/'+envelope,pair_vectors[(family,envelope,'fill95')],pair_vectors[(family,envelope,'positive_band')])
        effect('upper_bound_only',family+'/'+envelope,pair_vectors[(family,envelope,'positive95')],pair_vectors[(family,envelope,'positive_band')])
    for family,arm in itertools.product(FAMILIES,ARMS):
        effect('deployment_envelope',family+'/'+arm,pair_vectors[(family,'full60',arm)],pair_vectors[(family,'envelope',arm)])
    for envelope,arm in itertools.product(ENVELOPES,ARMS):
        effect('feedback_library',envelope+'/'+arm,pair_vectors[('static',envelope,arm)],pair_vectors[('budget',envelope,arm)])
    effect('historical_total_change','old_to_current',published_sku[('old','DRO_category')]-published_sku[('old','sS')],published_sku[('current','DRO_category')]-published_sku[('current','sS')])
    csv(effects,out/'factor_effects.csv')
    water=[]
    for method in LABELS:
        lo=b[f'{method}_full60_low'];env=b[f'{method}_envelope_low'];ratio=env/lo
        water.append(dict(method=method,full_mean=float(lo.mean()),envelope_mean=float(env.mean()),
            full_cv=float(lo.std()/lo.mean()),envelope_cv=float(env.std()/env.mean()),
            changed_items=int((env>lo+1e-12).sum()),min_multiplier=float(ratio.min()),max_multiplier=float(ratio.max()),
            multiplier_std=float(ratio.std())))
    csv(water,out/'waterline_shape.csv')
    # Logical counterexample, separate from all real-data performance results.
    d=np.r_[1000.,np.ones(19)];aa=np.r_[1000.,np.zeros(19)];ab=np.r_[1001.,np.full(18,2.),0.]
    toy=[]
    for name,a in [('A',aa),('B',ab)]:
        sale=np.minimum(a,d);end=a-sale
        toy.append(dict(policy=name,demand=float(d.sum()),sales=float(sale.sum()),ending=float(end.sum()),
                        F=float(sale.sum()/d.sum()),P=float((end>0).mean()),J=float(end.sum()/d.sum())))
    assert toy[0]['F']>=.95 and toy[0]['J']<toy[1]['J'] and toy[1]['P']==.95
    csv(toy,out/'logical_example.csv')
    dump(out/'analysis_audit.json',dict(bootstrap_replicates=1000,seed=20261016,block='paired full28-day SKU, stratified hot/tail',
        counts=dict(hot=int(hot.sum()),tail=int((~hot).sum())),denominator_min=float(denominators.min()),
        denominator_max=float(denominators.max()),logical_example_verified=True,
        qualification='CIs condition on the fitted policies; exploratory contrasts are not multiplicity adjusted.'))
    print(pd.DataFrame(winners).to_string(index=False),flush=True)
    print(pd.DataFrame(contrasts).to_string(index=False),flush=True)

if __name__=='__main__':main()
