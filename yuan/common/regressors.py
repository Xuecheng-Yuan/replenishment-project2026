"""common.regressors: actual model/processing implementation, not a legacy runner alias.

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
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelSpec:
    name: str
    candidates: tuple[dict, ...]
    tune_sample_limit: int | None = None
    final_sample_limit: int | None = None

def model_specs() -> list[ModelSpec]:
    return [ModelSpec('linear_regression', ({},)), ModelSpec('ridge', tuple(({'alpha': value} for value in (0.1, 1.0, 10.0, 100.0)))), ModelSpec('lasso', tuple(({'alpha': value} for value in (0.01, 0.1, 1.0, 10.0)))), ModelSpec('elastic_net', tuple(({'alpha': alpha, 'l1_ratio': ratio} for alpha in (0.01, 0.1, 1.0) for ratio in (0.2, 0.5, 0.8)))), ModelSpec('svr', ({'C': 1.0, 'epsilon': 0.05}, {'C': 10.0, 'epsilon': 0.05}, {'C': 10.0, 'epsilon': 0.1}, {'C': 50.0, 'epsilon': 0.1}), tune_sample_limit=15000, final_sample_limit=20000), ModelSpec('random_forest', ({'n_estimators': 180, 'max_depth': 18, 'min_samples_leaf': 1}, {'n_estimators': 180, 'max_depth': 24, 'min_samples_leaf': 2}, {'n_estimators': 180, 'max_depth': None, 'min_samples_leaf': 3}, {'n_estimators': 240, 'max_depth': None, 'min_samples_leaf': 1}), tune_sample_limit=120000), ModelSpec('xgboost', ({'n_estimators': 240, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.9}, {'n_estimators': 320, 'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.9}, {'n_estimators': 220, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 0.9}, {'n_estimators': 320, 'max_depth': 10, 'learning_rate': 0.05, 'subsample': 0.8})), ModelSpec('lightgbm', ({'n_estimators': 300, 'num_leaves': 31, 'learning_rate': 0.05, 'min_child_samples': 20}, {'n_estimators': 400, 'num_leaves': 63, 'learning_rate': 0.05, 'min_child_samples': 20}, {'n_estimators': 300, 'num_leaves': 63, 'learning_rate': 0.08, 'min_child_samples': 50}, {'n_estimators': 450, 'num_leaves': 127, 'learning_rate': 0.04, 'min_child_samples': 50})), ModelSpec('catboost', ({'iterations': 300, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 3.0}, {'iterations': 400, 'depth': 8, 'learning_rate': 0.05, 'l2_leaf_reg': 5.0}, {'iterations': 300, 'depth': 8, 'learning_rate': 0.08, 'l2_leaf_reg': 8.0}, {'iterations': 450, 'depth': 10, 'learning_rate': 0.04, 'l2_leaf_reg': 8.0}))]

def build_regressor(name,parameters,n_jobs=8):
    from importlib import import_module
    return import_module('models.'+name).build(parameters,n_jobs)
