"""只读加载自己已训练的可信SPO工件，核对新决策代码与正式轨迹逐元素一致。"""
from pathlib import Path
import argparse
import importlib
import json
import sys
import types

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from replenishment.features import common_features,normalized_features
from replenishment.training.forecast import ml_predict,history_prediction
from replenishment.training.decision import policy
from replenishment.evaluation import simulate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-source',required=True,type=Path)
    args=parser.parse_args();project=args.project_source.resolve()
    # 旧可信工件的类路径兼容，仅影响当前核验进程。
    alias=types.ModuleType('models');alias.__path__=[];sys.modules['models']=alias
    for name in ['spo_linear','xgboost_spo','lightgbm_spo','direct_lightgbm']:
        sys.modules['models.'+name]=importlib.import_module('replenishment.models.'+name)
    raw=dict(np.load(project/'outputs/teacher_retrained_20261013/数据/processed/input.npz',allow_pickle=False))
    d,X,_=common_features(raw)
    checks=[]
    parent=project/'outputs/spo_action_refinement_20261015'
    records=json.loads((parent/'图表/全部结果.json').read_text())['results']
    for row in records:
        if not row['validation_selected']:continue
        name,version=row['model'],row['version']
        bundle=joblib.load(parent/'models'/name/(version+'.joblib'))
        pack=dict(bundle['pack']);spec=bundle['spec']
        if name in ['xgboost_spo','lightgbm_spo']:centers=ml_predict(pack['base'],X)
        else:centers=history_prediction(d,'moving_average',{'window':6},60)[1]
        pack.update(centers=centers,features=normalized_features(X,pack['scale']),actions=np.asarray(spec['actions']))
        _,arrays=simulate(raw['test_demand'],raw['initial'],policy(name,bundle['estimator'],pack,48,spec['z']))
        original=np.load(parent/'results'/name/version/'trajectory.npz',allow_pickle=False)
        for field,actual in zip(['available','ending','sales','orders'],arrays):
            np.testing.assert_array_equal(actual,original[field])
        checks.append(dict(model=name,version=version,sku_periods=6000*28,all_four_arrays_equal=True))
        print('EXACT REPLAY',name,version,flush=True)
    out=ROOT/'verification/frozen_spo_replay.json'
    out.write_text(json.dumps(dict(status='passed',checks=checks,
        limitation='核验已有权重的新推理路径，不等于新训练路径全量重训。'),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
