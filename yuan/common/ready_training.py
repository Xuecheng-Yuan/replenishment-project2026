"""common.ready_training: actual model/processing implementation, not a legacy runner alias.

Refactored mechanically from the existing project. See matching docs for
information timing, historical results, variants and limitations.
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path(__file__).resolve().parents[1]
if str(HERE) in sys.path:sys.path.remove(str(HERE))
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
PROJECT_ROOT=ROOT
DATA_DIR=ROOT/'data'
OUTPUT_DIR=ROOT/'results/generated'
OUT=ROOT/'results/generated/ready_immediate'
WORK=OUT
TABLES=OUT/'图表'
OLD=ROOT/'results/generated/ready_immediate'
import json, time, argparse
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from common.regressors import build_regressor
from common.ready_evaluation import ReadyRateEvaluator, selection_key
GRID = {'linear_regression': [{}], 'ridge': [{'alpha': 1.0}, {'alpha': 100.0}], 'lasso': [{'alpha': 0.001}, {'alpha': 0.01}], 'elastic_net': [{'alpha': 0.001, 'l1_ratio': 0.5}, {'alpha': 0.01, 'l1_ratio': 0.5}], 'svr': [{'C': 1.0, 'epsilon': 0.05}, {'C': 10.0, 'epsilon': 0.05}], 'random_forest': [{'n_estimators': 80, 'max_depth': 16, 'min_samples_leaf': 3}, {'n_estimators': 100, 'max_depth': 24, 'min_samples_leaf': 3}], 'xgboost': [{'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.9}, {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.9}], 'lightgbm': [{'n_estimators': 250, 'num_leaves': 31, 'learning_rate': 0.05, 'min_child_samples': 30}, {'n_estimators': 350, 'num_leaves': 63, 'learning_rate': 0.05, 'min_child_samples': 30}], 'catboost': [{'iterations': 200, 'depth': 6, 'learning_rate': 0.05}, {'iterations': 300, 'depth': 8, 'learning_rate': 0.05}]}
Z = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0)

def design(data, split):
    d = data['demand']
    scale = np.maximum(d[:, :split].mean(axis=1), 0.5)
    f = data['features'].copy()
    f[:, :, :13] /= scale[None, :, None]
    f[:, :, 14] /= scale[None, :]
    offset = np.log1p(scale)
    y = np.log1p(d[:, 12:]).T - offset[None, :]
    return (f, y, offset)

def forecast(model, x, offset):
    p = model.predict(x.reshape(-1, x.shape[-1])).reshape(x.shape[:2]) + offset[None, :]
    return np.expm1(np.clip(p, 0, 20)).T

def policy_score(e, d, prediction, z, output=None):
    target = prediction + z * e.std[:, None]
    return e.evaluate(d, lambda t, i: e.order_up_to(target[:, t], i), output)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='*', default=list(GRID))
    args = parser.parse_args()
    threadpool_limits(4)
    data = np.load(OUT / 'data.npz', allow_pickle=False)
    d = data['demand']
    ids = data['item_ids']
    ev = ReadyRateEvaluator(d[:, :42], ids)
    et = ReadyRateEvaluator(d[:, :60], ids)
    rows = []
    search = []
    candidates = []
    for z in Z:
        m = policy_score(ev, d[:, 42:60], np.repeat(ev.mean[:, None], 18, axis=1), z)
        search.append(dict(model='historical_baseline', z=z, parameters='{}', **m))
        candidates.append((selection_key(m), z, m))
    _, z, val = max(candidates, key=lambda a: a[0])
    test = policy_score(et, d[:, 60:], np.repeat(et.mean[:, None], 28, axis=1), z, OUT / 'historical_baseline')
    rows.append(dict(model='historical_baseline', z=z, parameters='{}', validation_ready_rate=val['mean_ready_rate'], test_wape=float(np.abs(d[:, 60:] - et.mean[:, None]).sum() / d[:, 60:].sum()), **test))
    print('baseline', test, flush=True)
    fx, y, offset = design(data, 42)
    finalx, finaly, finaloffset = design(data, 60)
    for name in args.models:
        start = time.time()
        print('Training', name, flush=True)
        options = []
        for params in GRID[name]:
            model = build_regressor(name, params, n_jobs=4)
            trainx = fx[:30].reshape(-1, fx.shape[-1])
            trainy = y[:30].ravel()
            if name == 'svr':
                idx = np.random.default_rng(42).choice(len(trainy), 8000, replace=False)
                trainx, trainy = (trainx[idx], trainy[idx])
            model.fit(trainx, trainy)
            pred = forecast(model, fx[30:48], offset)
            for z in Z:
                m = policy_score(ev, d[:, 42:60], pred, z)
                search.append(dict(model=name, z=z, parameters=json.dumps(params), **m))
                options.append((selection_key(m), params, z, m))
        _, params, z, val = max(options, key=lambda a: a[0])
        model = build_regressor(name, params, n_jobs=4)
        trainx = finalx[:48].reshape(-1, finalx.shape[-1])
        trainy = finaly[:48].ravel()
        if name == 'svr':
            idx = np.random.default_rng(42).choice(len(trainy), 10000, replace=False)
            trainx, trainy = (trainx[idx], trainy[idx])
        model.fit(trainx, trainy)
        pred = forecast(model, finalx[48:], finaloffset)
        m = policy_score(et, d[:, 60:], pred, z, OUT / name)
        row = dict(model=name, z=z, parameters=json.dumps(params), validation_ready_rate=val['mean_ready_rate'], test_wape=float(np.abs(pred - d[:, 60:]).sum() / d[:, 60:].sum()), seconds=time.time() - start, **m)
        rows.append(row)
        (OUT / name).mkdir(exist_ok=True)
        joblib.dump(model, OUT / name / 'model.joblib')
        np.savez_compressed(OUT / name / 'forecast.npz', prediction=pred, offset=finaloffset, feature_names=data['feature_names'])
        pd.DataFrame(rows).to_csv(OUT / 'ml_comparison.csv', index=False)
        pd.DataFrame(search).to_csv(OUT / 'validation_grid.csv', index=False)
        print(name, m, flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'ml_comparison.csv', index=False)
    pd.DataFrame(search).to_csv(OUT / 'validation_grid.csv', index=False)
