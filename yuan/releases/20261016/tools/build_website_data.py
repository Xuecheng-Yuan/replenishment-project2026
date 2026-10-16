"""从本交付目录内的统一目录、参数和已核验轨迹构建网站数据，不重新训练。"""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import zlib
from datetime import datetime, timezone

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def read(path): return json.loads(Path(path).read_text())


def pack(array):
    return base64.b64encode(zlib.compress(np.asarray(array,dtype='<f8').tobytes(),6)).decode()


def file_record(path):
    with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(file=str(path.relative_to(ROOT)),sha256=digest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--complete',action='store_true')
    args=parser.parse_args()
    catalog=read(ROOT/'configs/model_catalog.json')
    records=read(ROOT/'results/benchmark/metrics.json')
    by_id={r['model']:r for r in records}
    panel=np.load(ROOT/'artifacts/panel.npz',allow_pickle=False)
    d=panel['test_demand']; n,T=d.shape
    models,trajectories,periods,methods=[],[],[],[]
    for entry in catalog:
        name=entry['model']; r=by_id[name]
        trajectory=np.load(ROOT/entry['trajectory'],allow_pickle=False)
        b,q,p=trajectory['ending'],trajectory['orders'],trajectory['forecast']
        a=np.column_stack([panel['initial'],b[:,:-1]+q[:,:-1]])
        s=np.minimum(a,d)
        selected=r['selection']; validation=selected.get('validation',{})
        models.append(dict(model=name,name=r['name'],family=r['family'],owner=r['owner'],
            service=100*r['ready_rate'],fill=100*r['fill_rate'],J=r['inventory_demand_ratio'],
            average_inventory=r['average_ending_total_inventory'],sales=r['sales_units'],
            demand=r['demand_units'],ending=r['ending_inventory_units'],version=r['version'],
            test_status=r['test_status'],code=r['code'],
            val_service=100*validation['ready_rate'] if 'ready_rate' in validation else None,
            protocol=r['protocol'],forecast_available=bool(p.size),
            parameters=json.dumps(selected,ensure_ascii=False)))
        trajectories.append(dict(model=name,name=r['name'],shape=[2,n,T],dtype='float64-le',
            fields='ending,orders',payload=pack(np.stack([b,q])),forecast_payload=pack(p) if p.size else '',
            forecast_shape=[n,T] if p.size else [],forecast_note='冻结的一步需求预测；Kristina首期无预先预测，留空' if p.size else '未保存对应需求预测，不用库存或销量冒充'))
        sd=np.cumsum(d.sum(0));ss=np.cumsum(s.sum(0));sb=np.cumsum(b.sum(0));sp=np.cumsum((b>0).sum(0))
        for t in range(T):
            periods.append(dict(model=name,name=r['name'],period=t+1,service=100*sp[t]/(n*(t+1)),
                fill=100*ss[t]/sd[t],J=sb[t]/sd[t],demand=float(d[:,t].sum()),sales=float(s[:,t].sum()),
                ending=float(b[:,t].sum()),orders=float(q[:,t].sum()),arrivals=0. if t==0 else float(q[:,t-1].sum()),
                period_service=100*float((b[:,t]>0).mean()),period_fill=100*float(s[:,t].sum()/d[:,t].sum())))
        methods.append(dict(model=name,payload=json.dumps(read(ROOT/'artifacts/methods'/f'{name}.json'),ensure_ascii=False)))
    definitions=[dict(label='服务水平',definition='期末库存严格大于0的SKU-期数 / 全部SKU-期数，恰好售罄记为0。'),
        dict(label='Fill Rate',definition='总实际销量 / 总需求。'),
        dict(label='库存需求比J',definition='总期末库存 / 总需求，越低越好，不是通常销量/平均库存的周转次数。')]
    source=dict(label='统一冻结测试轨迹',type='local-experiment',metricDefinitions=definitions,
        files=[file_record(ROOT/'configs/model_catalog.json'),file_record(ROOT/'results/benchmark/metrics.json')]
               +[file_record(ROOT/r['trajectory']) for r in catalog],
        caveats=['同一测试指标，不等于相同训练/验证协议：个人主要42/18前向验证，Kristina包含回溯校准和服务预算反馈。',
                 '已有测试多次用于历史研究，不是新的外部盲测。',
                 '当前数据来自既有正式轨迹的重新核验，不宣称这次整理已重新训练全部模型。',
                 '今日桌面的DS和Forecast两个ML项目没有纳入；也没有混入其36/24冷启动协议。'],
        evidenceFlow=[dict(kind='validation',title='指标与时点核验',detail='全部35方案、6000SKU、28期，核对需求、初始库存、一期到货、整数订单、库存守恒；指标从轨迹重算。')])
    def query(rows, payloads=None, src=None):
        out=dict(rows=rows,source=src or source)
        if payloads: out['payloadColumns']=payloads
        return out
    queries=dict(models=query(models,['parameters']),periods=query(periods),
        trajectories=query(trajectories,['payload','forecast_payload']),
        skus=query(read(ROOT/'artifacts/sku_rows.json'),['payload'],dict(source,files=[file_record(ROOT/'artifacts/panel.npz')])),
        methods=query(methods,['payload']),experiments=query(read(ROOT/'artifacts/experiments.json')),
        release=query([dict(version='2026.10.16',personal_models=29,partner_models=6,
            partner_commit='aec11c8',source='既有正式实验轨迹重新核验',desktop_ml_included=False)]))
    snapshot=dict(id='yuan-kristina-replenishment-release-20261016',title='补货决策研究工作台',surface='dashboard',
        status='reviewed-frozen-experiments',generatedAt=datetime.now(timezone.utc).isoformat(),
        buildStatus='complete' if args.complete else 'creating',filters=[],queries=queries)
    out=ROOT/'website/app/src/data.json'
    out.write_text(json.dumps(snapshot,ensure_ascii=False,separators=(',',':'),allow_nan=False))
    print('WEBSITE DATA',len(models),'models;',len(periods),'period records;',round(out.stat().st_size/1e6,2),'MB')


if __name__=='__main__': main()
