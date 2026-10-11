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
        assert np.isclose(sales.sum()/((available+ending)/2).sum()*28,
                          summary.loc[method, 'turnover'])
        assert summary.loc[method, 'fill_rate'] >= .95
        check.append(dict(method=method, rows=len(detail), balances='pass',
                          lead_time='pass', integer_orders='pass', summary='pass'))
    print(json.dumps(dict(artifact_hashes=len(manifest['files']), methods=check,
                          status='pass'), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
