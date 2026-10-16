"""Reconcile all five persisted policy decisions against the original test array."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from run_v3 import ROOT,METHODS
out=ROOT/'outputs_v4'
source=np.load(ROOT/'.cache/period_data.npz',allow_pickle=False)
y=source['test_y'].astype(float); T,n=y.shape
results=pd.read_csv(out/'summary.csv').query("scope == 'all_28_periods'").set_index('method')
comparisons=pd.read_csv(out/'decision_comparison_full.csv')
assert len(comparisons)==T*n and np.array_equal(comparisons.actual_demand.to_numpy().reshape(T,n),y)
report=[]
for method in METHODS:
    frame=pd.read_csv(out/f'detail_{method}.csv')
    if len(frame)!=T*n: raise AssertionError((method,len(frame)))
    col=lambda c:frame[c].to_numpy().reshape(T,n)
    available=col('available'); sales=col('sales'); ending=col('ending'); lost=col('lost'); orders=col('order')
    np.testing.assert_array_equal(col('demand'),y)
    np.testing.assert_array_equal(available[0],source['initial_inventory'])
    np.testing.assert_array_equal(available[1:],ending[:-1]+orders[:-1])
    np.testing.assert_array_equal(available,sales+ending)
    np.testing.assert_array_equal(y,sales+lost)
    np.testing.assert_array_equal(orders,col('planned_order_h1'))
    np.testing.assert_array_equal(orders,comparisons[f'order_{method}'].to_numpy().reshape(T,n))
    if not ((orders>=0).all() and (orders==np.ceil(orders)).all() and (ending*lost==0).all()):
        raise AssertionError(f'{method}: order integrality or lost-sales rule failed')
    assert np.isclose(sales.sum()/y.sum(),results.loc[method,'fill_rate'])
    inventory_ratio=ending.sum()/y.sum()
    assert np.isclose(inventory_ratio,results.loc[method,'inventory_demand_ratio'])
    assert np.isclose(ending.sum(),results.loc[method,'ending_inventory_units'])
    assert np.isclose(np.mean(ending>0),results.loc[method,'positive_stock_ratio'])
    assert np.count_nonzero(ending>0)==results.loc[method,'positive_stock_sku_periods']
    for h in range(1,5):
        np.testing.assert_array_equal(col(f'forecast_h{h}'),comparisons[f'forecast_h{h}'].to_numpy().reshape(T,n))
    report.append(dict(method=method,rows=len(frame),balances='pass',lead_time='pass',
                       integer_orders='pass',summary='pass',comparison='pass'))
(out/'result_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(pd.DataFrame(report).to_string(index=False))
