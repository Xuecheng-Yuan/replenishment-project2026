"""Validate downloaded results without rebuilding the training cache."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs_v4'
METHODS = ('DRO_cluster', 'DRO_global', 'DRO_category', 'sS', 'SAA')


def read_csv(name):
    path = OUT/name
    return pd.read_csv(path if path.exists() else path.with_suffix('.csv.gz'))


def main():
    manifest = json.loads((OUT/'artifact_manifest.json').read_text(encoding='utf-8'))
    for file in manifest['files']:
        path = OUT/file['file']
        assert path.stat().st_size == file['bytes'], path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == file['sha256'], path
    full = read_csv('decision_comparison_full.csv')
    assert len(full) == 6000*28
    assert not full.duplicated(['period', 'item_id']).any()
    summary = read_csv('summary.csv').query("scope == 'all_28_periods'").set_index('method')
    positive=read_csv('positive_stock_service.csv')
    sku_positive=read_csv('positive_stock_by_sku.csv')
    assert len(sku_positive)==30000
    assert not sku_positive.duplicated(['item_id','method']).any()
    check = []
    for method in METHODS:
        detail = read_csv(f'detail_{method}.csv')
        assert len(detail) == len(full)
        for key in ('period', 'item_id'):
            np.testing.assert_array_equal(detail[key], full[key])
        col = lambda name: detail[name].to_numpy().reshape(28, 6000)
        demand, available, sales = col('demand'), col('available'), col('sales')
        ending, lost, order = col('ending'), col('lost'), col('order')
        np.testing.assert_array_equal(available[1:], ending[:-1]+order[:-1])
        np.testing.assert_array_equal(available, sales+ending)
        np.testing.assert_array_equal(demand, sales+lost)
        np.testing.assert_array_equal(demand.ravel(), full.actual_demand)
        assert np.all(ending*lost == 0)
        assert np.all(order >= 0) and np.all(order == np.ceil(order))
        np.testing.assert_array_equal(order, col('planned_order_h1'))
        for source, target in [('order', 'order'), ('lost', 'lost'), ('ending', 'ending')]:
            np.testing.assert_array_equal(col(source).ravel(), full[f'{target}_{method}'])
        for h in range(1, 5):
            np.testing.assert_array_equal(col(f'forecast_h{h}').ravel(), full[f'forecast_h{h}'])
        assert np.isclose(sales.sum()/demand.sum(), summary.loc[method, 'fill_rate'])
        assert np.isclose(ending.sum()/demand.sum(),
                          summary.loc[method, 'inventory_demand_ratio'])
        assert np.isclose(ending.sum(), summary.loc[method, 'ending_inventory_units'])
        assert np.isclose(np.mean(ending>0),summary.loc[method,'positive_stock_ratio'])
        assert np.count_nonzero(ending>0)==summary.loc[method,'positive_stock_sku_periods']
        for scope,values in [('all_28_periods',ending),('controllable_periods_2_to_28',ending[1:])]:
            row=positive[(positive.method==method)&(positive.scope==scope)].iloc[0]
            assert row.positive_stock_sku_periods==np.count_nonzero(values>0)
            assert row.total_sku_periods==values.size
            assert np.isclose(row.positive_stock_ratio,np.mean(values>0))
        per_sku=sku_positive[sku_positive.method==method].set_index('item_id')
        expected=detail.assign(positive=detail.ending>0).groupby('item_id').positive.agg(['sum','count'])
        np.testing.assert_array_equal(per_sku.loc[expected.index,'positive_stock_periods'],expected['sum'])
        np.testing.assert_array_equal(per_sku.loc[expected.index,'total_periods'],expected['count'])
        np.testing.assert_allclose(per_sku.loc[expected.index,'positive_stock_ratio'],expected['sum']/expected['count'])
        check.append(dict(method=method, rows=len(detail), balances='pass',
                          lead_time='pass', integer_orders='pass', summary='pass'))
    parameters=json.loads((OUT/'frozen_parameters.json').read_text(encoding='utf-8'))
    assert parameters['service_metric']=='positive_stock_ratio'
    for method in METHODS:
        assert .95<=parameters['selected'][method]['reset_service']<=.955
    print(json.dumps(dict(artifact_hashes=len(manifest['files']), methods=check,
                          status='pass'), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
