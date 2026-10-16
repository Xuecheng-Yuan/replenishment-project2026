"""Check report values against official sources and the frozen file manifest."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    manifest=json.loads((HERE/'SHA256SUMS.json').read_text(encoding='utf-8'))
    for rel,h in manifest.items():assert sha(HERE/rel)==h,rel
    e=HERE/'evidence'
    audit=json.loads((e/'published_audit.json').read_text())
    assert audit['all_inventory_identities_passed'] and audit['all_frozen_policy_replays_passed']
    for row in audit['files']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
    sources=json.loads((e/'source_manifest.json').read_text())
    for row in sources['official_sources']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
    pub=pd.read_csv(e/'published_summary.csv')
    for version,folder in [('old','demand_fill_v3/with_ro'),('current','service95_revision')]:
        home=ROOT/'experiments'/folder/'outputs_v4'
        s=pub[pub.version.eq(version)].set_index('method')
        declared=pd.read_csv(home/'summary.csv').query("scope=='all_28_periods'").set_index('method').loc[s.index]
        for col in ['fill_rate','positive_stock_ratio','inventory_demand_ratio']:
            np.testing.assert_allclose(s[col],declared[col],rtol=1e-12)
        assert len(s)==6 and (s.total_sku_periods==168000).all()
        assert s.fill_rate.ge(.95).all() if version=='old' else s.positive_stock_ratio.between(.95,.955).all()
        report=next(r for r in (HERE/'reports').glob('*.json') if json.loads(r.read_text(encoding='utf-8'))['version']==version)
        content=json.loads(report.read_text(encoding='utf-8'));assert content['official_results_only']
        table=next(b for b in content['blocks'] if b['kind']=='table' and b['headers']==['方法','F：销量/需求','P：正库存比例','J：库存/需求','无缺货比例C'])
        for row,(_,r) in zip(table['rows'],s.iterrows()):
            assert row[1:]==[f'{r.fill_rate:.4%}',f'{r.positive_stock_ratio:.4%}',f'{r.inventory_demand_ratio:.6f}',f'{r.cycle_service:.4%}']
        periods=pd.read_csv(e/f'{version}_official_periods.csv')
        assert len(periods)==168
        for method,r in s.iterrows():
            q=periods[periods.method.eq(method)]
            assert q.positive_stock_sku_periods.sum()==r.positive_stock_sku_periods
            assert q.ending_inventory_units.sum()==r.ending_inventory_units
            assert q.demand_units.sum()==r.demand_units
    print(f'PASS: {len(manifest)} report/evidence files; 12 official method summaries; 336 period rows; source hashes and report tables.')

if __name__=='__main__':main()
