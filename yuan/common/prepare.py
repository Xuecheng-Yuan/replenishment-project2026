"""common.prepare: actual model/processing implementation, not a legacy runner alias.

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
import hashlib, json
import numpy as np
from common.features import load_panel, build_supervised_data

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def current():
    OUT = ROOT / 'results/generated/ready_immediate'
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / 'data/replenishment_train.xlsx', ROOT / 'data/replenishment_test.xlsx']
    panels = [load_panel(p) for p in paths]
    a, b = panels
    assert np.array_equal(a.item_ids, b.item_ids)
    assert len(np.unique(a.item_ids)) == len(a.item_ids)
    d = np.concatenate([a.demand, b.demand], axis=1)
    x = np.concatenate([a.x, b.x], axis=1)
    history = build_supervised_data(d, x, a.categories, np.arange(12, d.shape[1]))
    old = history.features.reshape(d.shape[1] - 12, len(a.item_ids), -1)
    features = np.concatenate([old[:, :, :17], old[:, :, 57:58], x[:, 12:, :].transpose(1, 0, 2)], axis=2)
    names = history.feature_names[:17] + ['category_code'] + [f'x{i}' for i in range(1, 21)]
    np.savez_compressed(OUT / 'data.npz', demand=d, features=features.astype('float32'), item_ids=a.item_ids, feature_names=np.array(names))
    manifest = dict(seed=42, train_periods=60, validation_split=42, test_periods=28, sku_count=len(a.item_ids), features=names, timing='current x known; current demand revealed AFTER order; immediate delivery', service='per-SKU fraction of periods with end inventory >1e-8, target .95', capacity_multiplier=1.05, initial_inventory_cover=1.25, source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, demand_min=float(d.min()), demand_max=float(d.max()), nonfinite_count=int((~np.isfinite(d)).sum()), note='History features for t use only demand before t. Test history updates after sales/demand observation; no refitting on test.')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f'Frozen {d.shape} demand, {features.shape} features', flush=True)

def lagged():
    OUT = ROOT / 'results/generated/online_learning'
    SOURCE = ROOT / 'results/generated/ready_immediate/data.npz'
    raw = np.load(SOURCE, allow_pickle=False)
    d, f, ids = (raw['demand'], raw['features'], raw['item_ids'])
    assert d.shape == (6000, 88) and f.shape == (76, 6000, 38)
    assert np.isfinite(d).all() and np.isfinite(f).all() and (d >= 0).all()
    assert len(np.unique(ids)) == len(ids)
    safe = f[1:].copy()
    safe[:, :, -20:] = f[:-1, :, -20:]
    names = raw['feature_names'].copy()
    names[-20:] = np.array([f'x{i}_lag_1' for i in range(1, 21)])
    target = np.arange(13, 88)
    assert np.array_equal(safe[:, :, 0], d[:, target - 1].T.astype(np.float32))
    folder = OUT / '数据/processed'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'panel.npz'
    np.savez_compressed(path, demand=d, features=safe, item_ids=ids, target_periods=target, feature_names=names)
    audit = dict(source=str(SOURCE.relative_to(ROOT)), source_sha256=digest(SOURCE), frozen_input=str(path.relative_to(OUT)), frozen_sha256=digest(path), source_shape=list(f.shape), frozen_shape=list(safe.shape), nonfinite_before=0, nonfinite_after=0, duplicate_skus=0, transformations=['shift x one day', 'skip unsupported first feature row'], leakage_control='features for target t use d and x through t-1; no scalers fit here')
    (OUT / '图表').mkdir(exist_ok=True)
    (OUT / '图表/data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print('Frozen safe panel', safe.shape, flush=True)

def mpc():
    WORK = ROOT / 'results/generated/adaptive_mpc'
    source = ROOT / 'results/generated/ready_immediate/data.npz'
    raw = np.load(source, allow_pickle=False)
    d, f, ids = (raw['demand'], raw['features'], raw['item_ids'])
    assert d.shape == (6000, 88) and f.shape == (76, 6000, 38)
    assert len(np.unique(ids)) == len(ids)
    assert np.isfinite(d).all() and np.isfinite(f).all() and (d >= 0).all()
    names = list(raw['feature_names'])
    assert names[18:] == [f'x{i}' for i in range(1, 21)]
    x = f[:, :, 18:].transpose(1, 0, 2).astype('float64')
    category = f[0, :, 17].astype('int64')
    assert np.array_equal(f[:, :, 17], np.broadcast_to(category, f[:, :, 17].shape))
    target = WORK / '数据/processed/panel.npz'
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, demand=d, x_from_period_13=x, item_ids=ids, categories=category, train_periods=60)
    audit = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), frozen_input=str(target), frozen_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), shape=list(d.shape), duplicate_skus=0, nonfinite=0, negative_demand=0, before_missing=0, after_missing=0, removed_skus=0, steps=['extract raw demand, raw x and category; discard old fitted transforms', 'retain all values, including train/test demand reset'], leakage_control='features rebuilt from observations through origin; train-only fits', limitations='raw x before period 13 unavailable in cache; use origins >=13')
    (WORK / '图表').mkdir(exist_ok=True)
    (WORK / '图表/data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print('DATA AUDIT PASS', d.shape, flush=True)
    return audit

def service() -> None:
    OUT = ROOT / 'results/generated/service_policy'
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_panel(ROOT / 'data/replenishment_train.xlsx')
    test = load_panel(ROOT / 'data/replenishment_test.xlsx')
    if not np.array_equal(train.item_ids, test.item_ids):
        raise ValueError('Train/test SKU alignment differs.')
    demand = np.concatenate([train.demand, test.demand], axis=1)
    x = np.concatenate([train.x, test.x], axis=1)
    periods = np.arange(12, demand.shape[1] + 1)
    supervised = build_supervised_data(demand, x, train.categories, periods, include_target=False)
    features = supervised.features.reshape(len(periods), len(train.item_ids), -1)
    reference = np.stack([np.maximum(demand[:, t - 12:t].mean(axis=1), 1.0) for t in periods])
    names = supervised.feature_names
    scale_names = [name for name in names if name.startswith('demand_') or name == 'recent_slope']
    scale_cols = [names.index(name) for name in scale_names]
    features[:, :, scale_cols] /= reference[:, :, None]
    fit = features[:30].reshape(-1, len(names))
    mean = fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(fit.std(axis=0, dtype=np.float64), 1e-06).astype(np.float32)
    normalized = np.clip((features - mean) / std, -15, 15).astype(np.float32)
    on_hand = train.initial_inventory.copy()
    arrival = np.zeros(len(train.item_ids))
    for t in range(12):
        available = on_hand + arrival
        on_hand = np.maximum(available - train.demand[:, t], 0)
        history = train.demand[:, max(0, t - 11):t + 1]
        target = history.mean(axis=1) + history.std(axis=1)
        arrival = np.maximum(target - on_hand, 0)
    path = OUT / 'prepared_data.npz'
    np.savez_compressed(path, item_ids=train.item_ids, categories=train.categories, demand=demand.astype(np.float32), features=normalized, reference=reference.astype(np.float32), warm_inventory=on_hand.astype(np.float32), test_initial_inventory=test.initial_inventory.astype(np.float32), scaler_mean=mean, scaler_std=std, feature_names=np.asarray(names), n_train=train.demand.shape[1])
    manifest = {'sources': {name: hashlib.sha256((ROOT / 'data' / name).read_bytes()).hexdigest() for name in ['replenishment_train.xlsx', 'replenishment_test.xlsx']}, 'prepared_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'n_skus': len(train.item_ids), 'train_periods': 60, 'test_periods': 28, 'fit_periods': '13-42', 'validation_periods': '43-60', 'feature_scaler_fit': '13-42 only', 'feature_names': names, 'feature_timing': 'target t uses demand and x through t-1 only', 'missing_values': 0, 'test_usage': 'Never used for optimization, checkpoint, multiplier or parameter selection'}
    (OUT / 'data_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({'prepared': str(path), 'shape': normalized.shape}, ensure_ascii=False), flush=True)

def turnover():
    OUT = ROOT / 'results/generated/turnover_dfl'
    src = ROOT / 'results/generated/online_learning/数据/processed/panel.npz'
    a = np.load(src, allow_pickle=False)
    assert a['demand'].shape == (6000, 88) and a['features'].shape == (75, 6000, 38)
    assert np.isfinite(a['features']).all() and np.isfinite(a['demand']).all()
    assert (a['demand'] >= 0).all() and len(np.unique(a['item_ids'])) == 6000
    np.testing.assert_array_equal(a['features'][:, :, 0], a['demand'][:, a['target_periods'] - 1].T.astype(np.float32))
    dst = OUT / '数据/processed/panel.npz'
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **{k: a[k] for k in a.files})
    (OUT / '图表').mkdir(exist_ok=True)
    audit = dict(source=str(src.relative_to(ROOT)), source_sha256=digest(src), frozen_input=str(dst.relative_to(OUT)), frozen_sha256=digest(dst), source_shape=list(a['features'].shape), frozen_shape=list(a['features'].shape), nonfinite_before=0, nonfinite_after=0, steps=['inherit audited lagged-x input', 'validate identities and historical demand', 'freeze canonical panel'], leakage_control='prediction scalers fitted on prefix only; known business parameters use full training60')
    (OUT / '图表/data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print('Canonical input frozen', flush=True)
