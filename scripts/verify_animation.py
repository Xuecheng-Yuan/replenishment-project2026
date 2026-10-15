"""Independently reconcile the offline animation to complete source traces."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '汇报成果/2026-10-15'


def main():
    path = BASE / '动画/data.json'
    bank = json.loads(path.read_text(encoding='utf-8'), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    for name in ('index.html', '公式互动.html'):
        html = (BASE / '动画' / name).read_text(encoding='utf-8')
        embedded = json.loads(re.search(r'const BANK=(.*);\n', html)[1])
        assert embedded == bank, name
        assert 'rolling_dro_v3/with_ro/outputs_v4' not in html, 'Superseded 32-period source'
    csv = pd.read_csv(BASE / '动画/新旧六策略真实轨迹.csv')
    assert len(csv) == 2 * 6 * 13 * 28 and not csv.duplicated(['version', 'method', 'scope', 'period']).any()
    for version, data in bank.items():
        assert len(data['methods']) == 6 and len(data['scopes']) == 13
        assert [(s['id'], s['n']) for s in data['scopes'][:3]] == [('all', 6000), ('hot', 480), ('tail', 5520)]
        if version == 'current':
            assert data['final_training_periods'] == 60 and len(data['validation_folds']) == 3
            assert data['parameters']['model_fit_periods'] == '1-60'
        for rel, expected in data['source_sha256'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == expected, rel
        for method in data['methods']:
            source = ROOT / data['source_folder'] / ('detail_'+method['id']+'.csv.gz')
            if not source.exists():
                source = source.with_suffix('')
            full = pd.read_csv(source).sort_values(['period', 'item_id'])
            assert len(full) == 168000
            for scope in data['scopes']:
                if scope['id'] == 'all':
                    f = full
                elif scope['id'] == 'hot':
                    f = full[full.category == '爆品']
                elif scope['id'] == 'tail':
                    f = full[full.category != '爆品']
                else:
                    f = full[full.item_id == int(scope['id'].split('_')[1])]
                assert len(f) == scope['n'] * 28
                got = method['scopes'][scope['id']]
                accumulated = {'demand': 0., 'sales': 0., 'ending': 0., 'positive': 0}
                for t, r in enumerate(got['rows'], 1):
                    block = f[f.period == t]
                    columns = ['demand', 'available', 'sales', 'lost', 'ending', 'order']
                    columns += [f'{k}_h{h}' for h in range(1, 5) for k in ('forecast', 'target', 'planned_order')]
                    np.testing.assert_allclose([r[k] for k in columns], block[columns].sum(), rtol=1e-12, atol=1e-7)
                    assert r['positive'] == int((block.ending > 0).sum())
                    assert r['sku_days'] == scope['n']
                    assert r['order'] == r['planned_order_h1']
                    prev = got['rows'][t-2] if t > 1 else None
                    np.testing.assert_allclose(r['arrival'], prev['order'] if prev else 0.)
                    np.testing.assert_allclose(r['start'], prev['ending'] if prev else r['available'])
                    np.testing.assert_allclose(r['available'], r['arrival']+r['start'])
                    for k in accumulated:
                        accumulated[k] += r[k]
                    c = got['cumulative'][t-1]
                    expected = [accumulated['positive']/(scope['n']*t),
                                accumulated['sales']/accumulated['demand'],
                                accumulated['ending']/accumulated['demand']]
                    np.testing.assert_allclose([c['P'], c['FR'], c['J']], expected, rtol=1e-12)
                    saved = csv[(csv.version == version) & (csv.method == method['id']) &
                                (csv.scope == scope['id']) & (csv.period == t)].iloc[0]
                    np.testing.assert_allclose([saved.cumulative_P, saved.cumulative_FR, saved.cumulative_J], expected, rtol=1e-12)
                assert got['summary'] == got['cumulative'][-1]
            # The two categories must reproduce all-SKU counts and quantities.
            for t in range(28):
                all_row, hot, tail = [method['scopes'][s]['rows'][t] for s in ('all', 'hot', 'tail')]
                keys = [k for k in all_row if k != 'period']
                np.testing.assert_allclose([all_row[k] for k in keys], [hot[k]+tail[k] for k in keys], rtol=1e-12, atol=1e-7)
            print(version, method['id'], 'all, hot, tail and ten fixed SKUs reconciled', flush=True)
    for line in (BASE / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        assert hashlib.sha256((BASE/relative).read_bytes()).hexdigest() == expected, relative
    for path in BASE.rglob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if not target.startswith(('https://', 'http://', '#')):
                assert (path.parent/target).exists(), (path, target)
    print('Animation verified: complete 2,016,000 source rows, strict positive-stock counts, all quantities/plans, cumulative metrics, source hashes, offline embeds and delivery manifest.')


if __name__ == '__main__':
    main()
