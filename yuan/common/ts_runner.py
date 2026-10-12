"""Chronological training of exactly one requested time-series model."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common.data import load_historical_demand
from common.evaluation import InventoryEvaluator
from common.ts_training import ExperimentConfig,_period_columns,_calibrate_z,_selection_key,_run_forecast_model

ROOT=Path(__file__).resolve().parents[1]

def train_one(name,window=12,n_jobs=2):
    train_df,_=load_historical_demand(ROOT/'data/replenishment_train.xlsx')
    test_df,info=load_historical_demand(ROOT/'data/replenishment_test.xlsx')
    train=train_df[_period_columns(train_df)].to_numpy(float)
    test=test_df[_period_columns(test_df)].to_numpy(float)
    ids=train_df.item_id.to_numpy()
    split=int(np.floor(train.shape[1]*(1-ExperimentConfig.validation_fraction)))
    config=ExperimentConfig();evaluator=InventoryEvaluator(lead_time=1)
    actual=train[:,split:];params={}
    if name=='naive':
        from models.naive import naive_path
        val,path=naive_path(train[:,:split],actual),naive_path(train,test)
    elif name=='moving_average':
        from models.moving_average import moving_average_path
        val,path=moving_average_path(train[:,:split],actual,window),moving_average_path(train,test,window)
        name+='_'+str(window)
    elif name=='exponential_smoothing':
        from models.exponential_smoothing import exponential_smoothing_path
        candidates=[dict(alpha=a,beta=0.,damping=1.) for a in (.2,.4,.6,.8,1.)]
        candidates += [dict(alpha=a,beta=b,damping=d) for a in (.4,.6,.8,1.) for b in (.05,.2,.5) for d in (.9,1.)]
        best=None
        for params in candidates:
            p=exponential_smoothing_path(train[:,:split],actual,**params)
            z,m=_calibrate_z(evaluator,train_df,ids,split,p,config)
            key=_selection_key(m,z)
            if best is None or key<best[0]:best=(key,params,p)
        _,params,val=best
        path=exponential_smoothing_path(train,test,**params)
    else:
        from models.arima import arima_paths
        val,path,orders,fallback=arima_paths(train,test,split,n_jobs=n_jobs)
        params=dict(order_counts={str(k):v for k,v in Counter(orders).items()},fallback_skus=fallback)
        directory=ROOT/'results/generated/time_series/arima';directory.mkdir(parents=True,exist_ok=True)
        pd.DataFrame({'item_id':ids,'order':[str(o) for o in orders]}).to_csv(directory/'selected_orders.csv',index=False)
    row,_=_run_forecast_model(name,params,val,path,train_df,test_df,info,ids,split,config,evaluator,'1')
    output=ROOT/'results/generated/time_series'/name
    (output/'comparison_row.json').write_text(json.dumps(row,indent=2))
    print(json.dumps(row,ensure_ascii=False,indent=2))
