"""训练调度：只在训练/验证集上选参，所有产物写入当前运行目录。"""
import json
import time

import joblib
import numpy as np
import pandas as pd

from ..evaluation import calibrate_target, key, simulate
from ..features import common_features
from ..registry import FORECAST_MODELS, DECISION_MODELS, NEURAL_MODELS, ENSEMBLE_MODELS
from ..runtime import ROOT, data, training_data, freeze_selection, sha, source_fingerprint


def search_offset(run, values, rounds=2, points=7):
    """状态相关策略不假定单调；在已验证的候选附近细化。"""
    records = []
    def record(z):
        if all(abs(z - row['z']) > 1e-8 for row in records):
            m = run(float(z))
            records.append(dict(z=float(z), **m))
    for z in values:
        record(z)
    for _ in range(rounds):
        best = min(records, key=key)
        zs = sorted(row['z'] for row in records)
        j = zs.index(best['z'])
        for z in np.linspace(zs[max(0, j - 1)], zs[min(len(zs) - 1, j + 1)], points):
            record(z)
    return min(records, key=key), records


def train_forecast(name, raw):
    """合并原先分两轮进行的拟合和尺度重选，不再重复训练相同候选。"""
    from .forecast import GRID, configurations, ml_bundle, history_prediction, cv_scale
    d, X, _ = common_features(raw)
    out = ROOT / 'models' / name
    out.mkdir(parents=True, exist_ok=True)
    start, records, options = time.perf_counter(), [], []
    for ci, params in enumerate(configurations(name)):
        if name in GRID:
            pack, p, fixed, errors = ml_bundle(name, params, d, X, 42)
        else:
            pack, p, fixed, errors = history_prediction(d, name, params, 42)
        relative, _ = cv_scale(errors, d[:, 42 - errors.shape[1]:42],
                               np.maximum(d[:, :42].mean(1), .5), p)
        modes = [('fixed', fixed)] if name == 'baseline' else [('fixed', fixed), ('relative', relative)]
        for mode, sigma in modes:
            z, val, search = calibrate_target(d[:, 12:60], raw['initial'], p[:, :49],
                                               sigma[:, :49] if sigma.ndim == 2 else sigma, 30)
            options.append((key(val), ci, params, mode, z, val))
            records.extend(dict(candidate=ci, parameters=params, scale_mode=mode, **r) for r in search)
    _, ci, params, mode, z, val = min(options, key=lambda r: r[0])
    if name in GRID:
        pack, p, fixed, errors = ml_bundle(name, params, d, X, 60)
    else:
        pack, p, fixed, errors = history_prediction(d, name, params, 60)
    _, cv = cv_scale(errors, d[:, 60 - errors.shape[1]:60],
                     np.maximum(d[:, :60].mean(1), .5), p)
    joblib.dump(pack, out / 'model.joblib')
    np.savez_compressed(out / 'risk.npz', fixed=fixed, cv=cv)
    freeze_selection(name, dict(parameters=params, candidate=ci, scale_mode=mode, z=z,
                                validation=val, fit_cutoff=42, refit_cutoff=60,
                                training_status='trained', training_seconds=time.perf_counter() - start,
                                model_sha256=sha(out / 'model.joblib')))
    pd.DataFrame(records).to_csv(out / 'validation_search.csv', index=False)


def train_decision(name, raw):
    """五种决策学习只保存必要的最终学习器、尺度和选参日志。"""
    import cloudpickle
    from .decision import ACTIONS, build_training, centers, estimator, policy
    from .action_grid import refine_grid
    d, X, _ = common_features(raw)
    out = ROOT / 'models' / name
    out.mkdir(parents=True, exist_ok=True)
    start, options, records = time.perf_counter(), [], []
    direct = name.startswith('direct_')
    center42 = centers(name, d, X, 42)
    for penalty in [10., 50.]:
        xx, cost, targets, pack, orders = build_training(
            name, d, X, 42, penalty, center_bundle=center42, return_orders=True)
        refined = refine_grid(ACTIONS, cost, orders)
        for recipe, actions in [('coarse', ACTIONS), ('refined', refined)]:
            if recipe == 'refined':
                xx, cost, targets, pack = build_training(
                    name, d, X, 42, penalty, actions, center42)
            model = estimator(name, penalty)
            if name in ['lightgbm_spo', 'direct_lightgbm']:
                model.fit(xx, targets, cost, actions)
            else:
                model.fit(xx, cost)
            def run(z):
                return simulate(raw['train_demand'][:, 12:60], raw['initial'],
                                policy(name, model, pack, 0, z), 30)[0]
            grid = [0., 4., 8., 16., 32., 64.] if direct else [-2., 0., 1., 2., 3., 4., 6., 8.]
            best, search = search_offset(run, grid, rounds=3 if direct else 2)
            options.append((key(best), penalty, best, recipe, actions.copy()))
            records.extend(dict(penalty=penalty, action_recipe=recipe, **row) for row in search)
    _, penalty, val, recipe, actions = min(options, key=lambda r: r[0])
    # 最终拟合沿用验证已选网格，不能在全部训练期再选一个范围或看测试调参。
    xx, cost, targets, pack = build_training(name, d, X, 60, penalty, actions)
    model = estimator(name, penalty)
    if name in ['lightgbm_spo', 'direct_lightgbm']:
        model.fit(xx, targets, cost, actions)
    else:
        model.fit(xx, cost)
    # cloudpickle能保存树模型自定义目标；只加载自己信任的工件。
    with (out / 'model.joblib').open('wb') as stream:
        cloudpickle.dump(dict(estimator=model, pack={k: v for k, v in pack.items()
                                                    if k not in ['features', 'centers']}), stream)
    freeze_selection(name, dict(penalty=penalty, z=val['z'], validation=val,
                                fit_cutoff=42, refit_cutoff=60, actions=actions, action_recipe=recipe,
                                training_status='trained', training_seconds=time.perf_counter() - start,
                                model_sha256=sha(out / 'model.joblib')))
    pd.DataFrame(records).to_csv(out / 'validation_search.csv', index=False)


def train_mpc(raw):
    """同时保留两个前缀模型供验证；最终只执行所选候选的第一期订单。"""
    from .mpc import fit_predict, train
    d, X, _ = common_features(raw)
    out = ROOT / 'models/adaptive_mpc'
    out.mkdir(parents=True, exist_ok=True)
    for ci in range(2):
        for split in [42, 60]:
            pack, prediction = fit_predict(raw, d, X, split, ci)
            _, oof = fit_predict(raw, d, X, split - 12, ci)
            errors = d[:, split - 12:split] - oof[split - 24:split - 12, :, 0].T
            sigma = np.maximum(errors.std(1, ddof=1), .03 * pack['scale'])
            joblib.dump(pack, out / f'forecaster_{split}_{ci}.joblib')
            np.savez_compressed(out / f'forecast_{split}_{ci}.npz', forecast=prediction,
                                sigma=sigma, scale=pack['scale'], groups=pack['groups'])
    train()


def train_ensembles(raw):
    """重新拟合14专家OOF，不再借用原工程candidate0缓存。四种融合共用专家。"""
    from .ensemble import fit, fit_meta, train_raw
    raw = train_raw(raw)
    base = ROOT / 'models/base'
    base.mkdir(parents=True, exist_ok=True)
    (ROOT / 'logs').mkdir(parents=True, exist_ok=True)
    for name in FORECAST_MODELS:
        parts, warm = [], None
        for cut in [18, 24, 30, 36]:
            _, p, _ = fit(name, raw, cut, cut + 6)
            parts.append(p[:, cut - 12:cut - 6])
            if warm is None:
                warm = p[:, :6]
        _, validation_prediction, _ = fit(name, raw, 42, 60)
        pack, _, _ = fit(name, raw, 60, 60)
        joblib.dump(pack, base / f'{name}.joblib')
        np.savez_compressed(base / f'{name}.npz', oof=np.concatenate(parts, 1),
                            warm=warm, validation_prediction=validation_prediction)
    fit_meta()


def train_models(names, resume=False):
    """按依赖调度；已有模型默认不覆盖，显式--resume才跳过已完成模型。"""
    from threadpoolctl import threadpool_limits
    threadpool_limits(2)
    raw = training_data()
    meta_path = ROOT / 'models/meta/selected.json'
    completed_ensemble = {r['family'] for r in json.loads(meta_path.read_text())} if meta_path.exists() else set()
    existing = []
    completed = set(completed_ensemble)
    for name in names:
        path = ROOT / 'models' / name / 'selection.json'
        if path.exists():
            existing.append(name)
            previous=json.loads(path.read_text())
            if resume and previous.get('source_code_sha256')!=source_fingerprint():
                raise ValueError(f'{name}工件源码版本不符；请换一个--workdir，不可混用旧配方。')
            if previous.get('training_status') == 'trained':
                completed.add(name)
        elif name in completed_ensemble:
            existing.append(name)
    if existing and not resume:
        raise FileExistsError(f'已有模型{existing}；请换--workdir或用--resume跳过。')
    ensemble_done = False
    for name in names:
        if resume and name in completed:
            continue
        print('训练：', name, flush=True)
        if name in FORECAST_MODELS:
            train_forecast(name, raw)
        elif name in DECISION_MODELS:
            # 树SPO依赖同轮训练验证所选的基础需求模型参数。
            base = {'xgboost_spo': 'xgboost', 'lightgbm_spo': 'lightgbm'}.get(name)
            if base and not (ROOT / 'models' / base / 'selection.json').exists():
                train_forecast(base, raw)
            train_decision(name, raw)
        elif name in NEURAL_MODELS:
            from .neural import pretrained, with_basis, train_one
            base42, pack42 = pretrained(raw, 42)
            base60, pack60 = pretrained(raw, 60)
            train_one(name, raw, base42, with_basis(pack42, 42), base60, with_basis(pack60, 60))
        elif name == 'adaptive_mpc':
            train_mpc(raw)
        elif name in ENSEMBLE_MODELS and not ensemble_done:
            train_ensembles(raw)
            ensemble_done = True
