"""models.svr: actual model/processing implementation, not a legacy runner alias.

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
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def build(parameters, n_jobs=8):
    regressor = make_pipeline(StandardScaler(), PCA(n_components=0.95, svd_solver='full'), SVR(kernel='rbf', gamma='scale', cache_size=2048, **parameters))
    return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())

def fit(features,target,parameters,n_jobs=2):
    """Fit this estimator/pipeline; chronological split is caller-controlled."""
    model=build(parameters,n_jobs)
    model.fit(features,target)
    return model

if __name__=='__main__':
    from common.cli import start
    start('svr', globals(), default_protocol='legacy')
