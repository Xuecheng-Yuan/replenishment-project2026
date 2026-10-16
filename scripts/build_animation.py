"""Build the offline six-policy animation from complete saved test traces.

Run from any directory: python scripts/build_animation.py
Only pandas/numpy (the project's existing dependencies) are required. No model
fitting, test-set calibration or representative-SKU extrapolation occurs here.
"""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '汇报成果/2026-10-15'
DEST = OUT / '动画'
TEMPLATE = ROOT / 'scripts/animation'
METHODS = {'sS': '传统 (s,S)', 'SAA': '经验 SAA', 'DRO_global': '不分组 DRO',
           'DRO_category': '爆品/长尾 DRO', 'DRO_cluster': '特征聚类 DRO', 'RO': '传统区间 RO'}
SKU_IDS = (705, 1804, 2620, 3711, 4905, 928, 2140, 3026, 3918, 5118)
FOLDS = ((24, 25, 36), (36, 37, 48), (48, 49, 60))
NUMERIC = ['start', 'arrival', 'available', 'demand', 'sales', 'lost', 'ending', 'order']
NUMERIC += [f'{kind}_h{h}' for h in range(1, 5) for kind in ('forecast', 'target', 'planned_order')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows):
    f = pd.DataFrame(rows)
    demand = float(f.demand.sum())
    return dict(P=float(f.positive.sum() / f.sku_days.sum()),
                FR=float(f.sales.sum() / demand), J=float(f.ending.sum() / demand),
                average=float(f.ending.mean()), demand=demand, sales=float(f.sales.sum()),
                lost=float(f.lost.sum()), positive=int(f.positive.sum()), sku_days=int(f.sku_days.sum()))


def load_trace(folder, method):
    path = folder / f'detail_{method}.csv.gz'
    if not path.exists():
        path = folder / f'detail_{method}.csv'
    f = pd.read_csv(path).sort_values(['period', 'item_id']).reset_index(drop=True)
    assert len(f) == 6000 * 28 and not f.duplicated(['period', 'item_id']).any(), path
    assert np.array_equal(f.period, np.repeat(np.arange(1, 29), 6000)), path
    assert all(f.groupby('period').item_id.nunique() == 6000), path
    a, d, s, e, q = [f[c].to_numpy().reshape(28, 6000) for c in ('available', 'demand', 'sales', 'ending', 'order')]
    np.testing.assert_array_equal(s, np.minimum(a, d))
    np.testing.assert_array_equal(e, a - s)
    np.testing.assert_array_equal(a[1:], e[:-1] + q[:-1])
    np.testing.assert_array_equal(q.ravel(), f.planned_order_h1)
    assert np.all(q >= 0) and np.all(q == np.floor(q))
    for h in range(1,5):
        planned=f[f'planned_order_h{h}'].to_numpy()
        assert np.all(planned>=0) and np.all(planned==np.floor(planned)), (method,h)
    start = np.vstack([a[0], e[:-1]])
    arrival = np.vstack([np.zeros(6000), q[:-1]])
    f['start'], f['arrival'] = start.ravel(), arrival.ravel()
    f['positive'] = (f.ending > 0).astype(int)
    f['sku_days'] = 1
    return f, path


def rows_for(f, ids):
    g = f[f.item_id.isin(ids)].groupby('period', sort=True)[NUMERIC + ['positive', 'sku_days']].sum()
    return [dict(period=int(t), **{k: float(v) if k not in ('positive', 'sku_days') else int(v)
                                  for k, v in r.items()}) for t, r in g.iterrows()]


def build_bank(version):
    current = version == 'current'
    folder = ROOT / ('experiments/service95_revision/outputs_v4' if current else
                     'experiments/demand_fill_v3/with_ro/outputs_v4')
    # The immutable historical RO file uses NaN for two inapplicable diagnostic
    # service fields. Represent those as JSON null; do not alter the archive.
    params = json.loads((folder / 'frozen_parameters.json').read_text(encoding='utf-8'), parse_constant=lambda _: None)
    if current:
        design = params
        assert design['final_fit_periods'] == list(range(1, 61)), 'Final fitting must use all 60 periods.'
        folds = design.get('validation_folds', design.get('calibration_folds'))
        assert folds, 'Run the approved three-fold calibration before rebuilding the animation.'
        assert len(folds) == 3, folds
        for fold, (cut, first, last) in zip(folds, FOLDS):
            assert fold['fit_periods'] == list(range(1, cut+1)), fold
            assert fold['validation_periods'] == list(range(first, last+1)), fold
        with np.load(folder / 'fitted_model.npz') as shared:
            residual = shared['final_relative_residual']
            mu, sigma = float(residual.mean()), float(residual.std())
        gamma = params['selected']['RO']['gamma']
        with np.load(folder/'policy_RO.npz') as policy:
            envelope_upper=float(policy['low'].max())
        ro = dict(mu=mu, sigma=sigma, gamma=gamma, upper=envelope_upper, full60_upper=mu+gamma*sigma,
                  lower=max(0., mu-gamma*sigma), beta=1.)
    else:
        old_ro = params['selected']['RO']
        ro = dict(mu=old_ro['mu'], sigma=old_ro['sigma'], gamma=old_ro['gamma'],
                  upper=old_ro['upper_relative'], lower=old_ro['lower_relative'], beta=old_ro['beta'])
    methods, scopes, source, csv_rows = [], None, {(folder/'frozen_parameters.json').relative_to(ROOT).as_posix(): sha(folder/'frozen_parameters.json')}, []
    if current:
        design_path = folder/'frozen_parameters.json'
        source[design_path.relative_to(ROOT).as_posix()] = sha(design_path)
    first_demand = None
    for method, label in METHODS.items():
        f, path = load_trace(folder, method)
        source[path.relative_to(ROOT).as_posix()] = sha(path)
        if first_demand is None:
            first_demand = f[['period', 'item_id', 'category', 'demand', 'start']].copy()
            hot = f.loc[(f.period == 1) & (f.category == '爆品'), 'item_id'].tolist()
            tail = f.loc[(f.period == 1) & (f.category != '爆品'), 'item_id'].tolist()
            assert len(hot) == 480 and len(tail) == 5520
            scopes = [dict(id='all', label='全部 6,000 种商品', n=6000, ids=hot + tail, kind='aggregate'),
                      dict(id='hot', label='全部爆品 · 480 种', n=480, ids=hot, kind='aggregate'),
                      dict(id='tail', label='全部长尾品 · 5,520 种', n=5520, ids=tail, kind='aggregate')]
            for sku in SKU_IDS:
                category = '爆品' if sku in hot else '长尾品'
                scopes.append(dict(id=f'sku_{sku}', label=f'{category} SKU {sku}', n=1, ids=[sku], kind='sku'))
        else:
            np.testing.assert_array_equal(f.demand, first_demand.demand)
            np.testing.assert_array_equal(f.loc[f.period == 1, 'start'], first_demand.loc[first_demand.period == 1, 'start'])
        by_scope = {}
        for scope in scopes:
            rows = rows_for(f, scope['ids'])
            cumulative = [metrics(rows[:t]) for t in range(1, 29)]
            by_scope[scope['id']] = dict(rows=rows, cumulative=cumulative, summary=metrics(rows))
            for row, c in zip(rows, cumulative):
                csv_rows.append(dict(version=version, method=method, scope=scope['id'],
                                     scope_label=scope['label'], **row,
                                     period_P=row['positive']/row['sku_days'],
                                     period_FR=row['sales']/row['demand'], period_J=row['ending']/row['demand'],
                                     cumulative_P=c['P'], cumulative_FR=c['FR'], cumulative_J=c['J']))
        method_data=dict(id=method,label=label,scopes=by_scope)
        if current:
            feedback_path=folder/'service_feedback.csv'
            feedback=pd.read_csv(feedback_path)
            method_data['feedback']=feedback[feedback.method==method].drop(columns='method').to_dict('records')
            source[feedback_path.relative_to(ROOT).as_posix()]=sha(feedback_path)
        methods.append(method_data)
        print(version, method, 'complete 168,000-row trace aggregated', flush=True)
    summary = pd.read_csv(folder / 'summary.csv').query("scope == 'all_28_periods'").set_index('method')
    for m in methods:
        s = m['scopes']['all']['summary']
        r = summary.loc[m['id']]
        np.testing.assert_allclose([s['P'], s['FR'], s['J']],
                                  [r.positive_stock_ratio, r.fill_rate, r.inventory_demand_ratio], rtol=1e-12)
    for scope in scopes:
        scope.pop('ids')
    return dict(version=version, label='新版 · 服务95%–95.5%、全60期最终拟合' if current else '历史版 · 需求满足率约束',
                source_folder=folder.relative_to(ROOT).as_posix(), source_sha256=source, methods=methods, scopes=scopes,
                parameters=params, ro=ro, validation_folds=folds if current else [],
                final_training_periods=60, test_periods=28, test_file='replenishment_test.xlsx',
                service_metric='P' if current else 'FR', service_lower=.95, service_upper=.955 if current else None), csv_rows


def manifest():
    files = sorted(p for p in OUT.rglob('*') if p.is_file() and p.name not in ('SHA256SUMS', '最近启动状态.json'))
    (OUT / 'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(OUT).as_posix()+'\n' for p in files), encoding='utf-8')


def main():
    bank, records = {}, []
    for version in ('current', 'historical'):
        bank[version], rows = build_bank(version)
        records.extend(rows)
    DEST.mkdir(parents=True, exist_ok=True)
    data = json.dumps(bank, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    (DEST / 'data.json').write_text(data+'\n', encoding='utf-8')
    for name in ('index.html', '公式互动.html'):
        template = (TEMPLATE / name).read_text(encoding='utf-8')
        assert template.count('__BANK_JSON__') == 1
        (DEST / name).write_text(template.replace('__BANK_JSON__', data.replace('</', '<\\/')), encoding='utf-8')
    pd.DataFrame(records).to_csv(DEST / '新旧六策略真实轨迹.csv', index=False, encoding='utf-8-sig')
    (OUT / 'README.md').write_text((TEMPLATE / 'README.md').read_text(encoding='utf-8'), encoding='utf-8')
    manifest()
    print('Built complete-scope animation, two versions, 13 views × 6 methods × 28 periods each.')


if __name__ == '__main__':
    main()
