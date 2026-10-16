"""加载当前运行工件，统一生成测试策略；此模块不调参。"""
import json
from functools import lru_cache

import joblib
import numpy as np

from .evaluation import InventoryEvaluator, target_policy
from .features import common_features
from .registry import FORECAST_MODELS, DECISION_MODELS, NEURAL_MODELS, ENSEMBLE_MODELS
from .runtime import ROOT, data, sha, source_fingerprint


@lru_cache(maxsize=1)
def ensemble_predictions():
    """四种融合共用14专家的测试预测，避免重复执行昂贵的SVR推理。"""
    from .training.forecast import predict
    raw = data()
    predictions = [predict(name, joblib.load(ROOT / 'models/base' / f'{name}.joblib'), raw, start=48)
                   for name in FORECAST_MODELS]
    return np.stack(predictions, 2)


def load_policy(name, raw):
    """根据冻结的模型与参数生成策略；不根据测试结果重新选择。"""
    out = ROOT / 'models' / name
    if name in ENSEMBLE_MODELS:
        from .training.ensemble import risk_scale
        meta = json.loads((ROOT / 'models/meta/selected.json').read_text())
        selected = next(c for c in meta if c['family'] == name)
        a = dict(np.load(ROOT / 'models/meta' / f'{name}.npz'))
        p = np.einsum('ntm,nm->nt', ensemble_predictions(), a['weights'])
        scale = np.maximum(raw['train_demand'].mean(1), .5)
        sigma = risk_scale(p, scale, a['fixed'], a['cv'], selected['scale_mode'])
        return target_policy(p, sigma, selected['z']), selected
    selected = json.loads((out / 'selection.json').read_text())
    if selected.get('source_code_sha256') and selected['source_code_sha256']!=source_fingerprint():
        raise ValueError(f'{name}源码已改变，请新建运行目录重训，不能借用旧成绩。')
    model_path=out/('model.pt' if name in NEURAL_MODELS else 'model.joblib')
    if name=='adaptive_mpc':
        model_path=out/f"forecaster_60_{selected['candidate']}.joblib"
    expected=selected.get('model_sha256',selected.get('final_model_sha256'))
    if expected and sha(model_path)!=expected:
        raise ValueError(f'{name}模型权重与冻结记录不一致。')
    if name=='adaptive_mpc' and selected.get('policy_sha256') and sha(out/'policy.npz')!=selected['policy_sha256']:
        raise ValueError('MPC策略工件与冻结记录不一致。')
    if name in FORECAST_MODELS:
        from .training.forecast import predict
        pack = joblib.load(out / 'model.joblib')
        p = predict(name, pack, raw, start=48)
        risk = dict(np.load(out / 'risk.npz'))
        sigma = risk['fixed'] if selected['scale_mode'] == 'fixed' else np.maximum(p, .1 * np.maximum(
            raw['train_demand'].mean(1), .5)[:, None]) * risk['cv'][:, None]
        return target_policy(p, sigma, selected['z']), selected
    if name in DECISION_MODELS:
        from .training.decision import policy
        from .training.forecast import ml_predict, history_prediction
        from .features import normalized_features
        model = joblib.load(out / 'model.joblib')
        pack = model['pack']
        d, X, _ = common_features(raw)
        if name in ['xgboost_spo', 'lightgbm_spo']:
            pack['centers'] = ml_predict(pack['base'], X)
        else:
            pack['centers'] = history_prediction(d, 'moving_average', {'window': 6}, 60)[1]
        pack['features'] = normalized_features(X, pack['scale'])
        return policy(name, model['estimator'], pack, 48, selected['z']), selected
    if name in NEURAL_MODELS:
        import torch
        from .training.neural import prepare, DemandNet, DirectNet, runtime_policy
        pack = prepare(raw, 60)
        ck = torch.load(out / 'model.pt', map_location='cpu', weights_only=False)
        np.testing.assert_array_equal(pack['feature_mean'], ck['feature_mean'])
        np.testing.assert_array_equal(pack['feature_std'], ck['feature_std'])
        pack['basis'] = ck['basis']
        model = DirectNet(len(raw['initial'])) if name == 'service_policy' else DemandNet(
            len(raw['initial']),
            pack['offset']
        )
        model.load_state_dict(ck['state_dict'])
        return runtime_policy(name, model, pack, 48, selected['z'], selected['rho'],
                              selected['radius'], lr=selected['online_lr'], alpha=selected['online_alpha']), selected
    if name == 'adaptive_mpc':
        from .training.mpc import mpc_policy
        return mpc_policy(dict(np.load(out / 'policy.npz')), raw, 48,
                          selected['z'], selected['rho'], selected['alpha']), selected
    raise ValueError(name)


def evaluate_models(names):
    """正式评价测试28期；训练/验证历史指标另见reference，不混入新运行。"""
    raw = data()
    evaluator = InventoryEvaluator()
    for name in names:
        policy, selected = load_policy(name, raw)
        evaluator.evaluate(name, raw['test_demand'], raw['initial'], policy,
                           raw['item_ids'], dict(selection=selected))
