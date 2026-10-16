"""只读浏览已核验的历史成绩与订单；和新训练的runs结果明确分开。"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]


def compare(owner='all'):
    path=PACKAGE/'results/benchmark/metrics.csv'
    frame=pd.read_csv(path)
    if owner!='all':
        frame=frame[frame.owner.eq('yuan xuecheng' if owner=='yuan' else 'kristina')]
    view=frame[['model','name','ready_rate','fill_rate','inventory_demand_ratio','test_status']].copy()
    for key in ['ready_rate','fill_rate']: view[key]=view[key].map(lambda v:f'{v:.2%}')
    print('已有正式实验轨迹，不是本次重构后全量重训；不同验证协议不可混作同协议优劣。')
    print(view.to_string(index=False))


def replay(model, sku, through, output):
    catalog=json.loads((PACKAGE/'configs/model_catalog.json').read_text())
    row=next((r for r in catalog if r['model']==model),None)
    if row is None: raise ValueError('未知模型，请先运行 reference。')
    path=PACKAGE/row['trajectory']
    if not path.exists():
        raise FileNotFoundError('完整轨迹在本地审阅包artifacts中，默认不上传Git；请恢复审阅附件后再回放。')
    data=np.load(PACKAGE/'artifacts/panel.npz',allow_pickle=False)
    trace=np.load(path,allow_pickle=False)
    ids=data['item_ids'].astype(str)
    indices=np.flatnonzero(ids==str(sku))
    if len(indices)!=1: raise ValueError(f'找不到SKU {sku}。')
    i=int(indices[0]); demand=data['test_demand'][i,:through]
    end,orders=trace['ending'][i,:through],trace['orders'][i,:through]
    opening=np.r_[data['initial'][i],end[:-1]]
    arrivals=np.r_[0.,orders[:-1]]
    available=opening+arrivals; sales=np.minimum(available,demand)
    frame=pd.DataFrame(dict(model=model,item_id=str(sku),period=np.arange(1,through+1),
        opening=opening,arrivals=arrivals,available=available,demand=demand,sales=sales,
        lost=demand-sales,ending=end,order=orders,next_available=end+orders))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output,index=False,encoding='utf-8-sig')
    print('已输出逐期决策：',output)
    print(f'累计服务水平={(end>0).mean():.2%}，库存需求比={end.sum()/demand.sum():.6f}，'
          f'Fill Rate={sales.sum()/demand.sum():.2%}')
