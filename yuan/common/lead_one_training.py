"""Evaluate one selected historical forecaster under true lead-one timing."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from common.lead_one_evaluation import LeadOneReadyRateEvaluator
from common import prepare
from common.cli import call

ROOT=Path(__file__).resolve().parents[1]

def train(name,n_jobs=2):
    prepare.current()
    raw=np.load(ROOT/'results/generated/ready_immediate/data.npz',allow_pickle=False)
    demand,ids=raw['demand'],raw['item_ids']
    evaluator=LeadOneReadyRateEvaluator(demand[:,:60],ids)
    if name=='baseline':forecast=np.repeat(evaluator.mean[:,None],28,axis=1)
    else:
        # Reproduce the legacy selected prediction model; never train siblings.
        from common.ml_training import train as train_legacy
        call(train_legacy,['--models',name,'--n-jobs',str(n_jobs)])
        table=pd.read_csv(ROOT/'results/generated/machine_learning'/name/'test_forecasts.csv')
        assert np.array_equal(table.item_id.to_numpy(),ids)
        forecast=table[[f'period_{k}' for k in range(1,29)]].to_numpy(float)
    records=[]
    for z,variant in [(0.,'no_safety'),(4.,'fixed_z4_reference')]:
        output=ROOT/'results/generated/lead_one'/variant/name
        m=evaluator.evaluate_forecasts(demand[:,60:],forecast,z,output)
        records.append(dict(model=name,variant=variant,**m))
    directory=ROOT/'results/generated/lead_one';directory.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(directory/(name+'_comparison.csv'),index=False)
    (directory/(name+'_summary.json')).write_text(json.dumps(records,indent=2))
    print(pd.DataFrame(records).to_string(index=False))
