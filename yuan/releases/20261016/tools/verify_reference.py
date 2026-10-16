"""核验本目录全部冻结轨迹、指标、网站记录及上游文件身份，不训练模型。"""
from pathlib import Path
import hashlib
import json
import sys
import zlib
import base64

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from replenishment.evaluation import metrics, validate


def read(path):
    return json.loads((ROOT/path).read_text())


def main():
    catalog=read('configs/model_catalog.json')
    records={r['model']:r for r in read('results/benchmark/metrics.json')}
    website=read('website/app/src/data.json')
    assert len(catalog)==len(records)==35
    assert sum(r['owner']=='kristina' for r in catalog)==6
    with np.load(ROOT/'artifacts/panel.npz') as panel:
        d,initial=panel['test_demand'],panel['initial']
    assert d.shape==(6000,28)
    for row in catalog:
        name=row['model']
        for field in ['document','parameters','trajectory']:
            assert (ROOT/row[field]).is_file(),(name,field)
        # 代码字段可附函数定位，文件部分必须真实存在。
        for code in row['code'].split(' + '):
            assert (ROOT/code.split(':')[0]).is_file(),row['code']
        with np.load(ROOT/row['trajectory']) as trace:
            b,q=trace['ending'],trace['orders']
        a=np.column_stack([initial,b[:,:-1]+q[:,:-1]])
        s=np.minimum(a,d)
        validate(d,initial,a,b,s,q)
        computed=metrics(d,a,b,s,q)
        for key in ['ready_rate','fill_rate','inventory_demand_ratio','average_ending_total_inventory']:
            np.testing.assert_allclose(computed[key],records[name][key],rtol=1e-12)
        web=next(r for r in website['queries']['trajectories']['rows'] if r['model']==name)
        decoded=np.frombuffer(zlib.decompress(base64.b64decode(web['payload'])),dtype='<f8').reshape(web['shape'])
        np.testing.assert_array_equal(decoded,np.stack([b,q]))
        for period in website['queries']['periods']['rows']:
            if period['model']!=name: continue
            t=period['period']
            expected=[100*(b[:,:t]>0).mean(),100*s[:,:t].sum()/d[:,:t].sum(),b[:,:t].sum()/d[:,:t].sum()]
            np.testing.assert_allclose([period['service'],period['fill'],period['J']],expected,rtol=1e-12)
        dossier=read(f'artifacts/methods/{name}.json')
        assert dossier.get('decision_steps') and dossier.get('markdown'),name
    assert len(website['queries']['periods']['rows'])==980
    for query in website['queries'].values():
        for record in query['source'].get('files',[]):
            file=ROOT/record['file']
            assert hashlib.sha256(file.read_bytes()).hexdigest()==record['sha256'],str(file)
    upstream=read('partners/kristina/SOURCE.json')
    for record in upstream['files']:
        file=ROOT/'partners/kristina/upstream'/record['path']
        assert hashlib.sha256(file.read_bytes()).hexdigest()==record['sha256'],str(file)
    result=dict(status='passed',models=35,personal=29,kristina=6,skus=6000,periods=28,
                period_records=980,upstream_files=len(upstream['files']),
                checks=['inventory_conservation','integer_orders','metrics','lossless_web_payload','cumulative_periods','local_paths','source_sha256'],
                limitation='冻结轨迹及网站数据核验，不是从零重训。')
    (ROOT/'verification/reference_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
