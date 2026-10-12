"""models.lightgbm: actual model/processing implementation, not a legacy runner alias.

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
def build(parameters, n_jobs=8):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(**parameters, objective='regression_l1', subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, n_jobs=n_jobs, random_state=42, verbosity=-1)

def fit(features,target,parameters,n_jobs=2):
    """Fit this estimator/pipeline; chronological split is caller-controlled."""
    model=build(parameters,n_jobs)
    model.fit(features,target)
    return model

if __name__=='__main__':
    from common.cli import start
    start('lightgbm', globals(), default_protocol='legacy')
