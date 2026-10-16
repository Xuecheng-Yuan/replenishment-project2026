"""Shared training-risk waterlines; CLI runs the current 95%-95.5% model.

Archived static-reserve experiment: service95_revision/history/conservative_99_percent.
"""
import sys
import numpy as np
from extend_with_ro import ROOT, LABELS, box
from constraint_ablation import FOLDS, digest
HOME=ROOT/'experiments/service95_revision'
sys.path.insert(0,str(ROOT/'experiments/rolling_dro_v3'))
import model_v2 as model
import run_v3 as pipeline
from run_v2 import ss_quantiles


def levels(residual, labels, hot, recipe, cache=None):
    method=recipe['method'];cache={} if cache is None else cache
    if method=='RO':return np.full(len(labels),box(residual,recipe['gamma'])[1]),None
    if method=='sS':
        def quantile(p):
            key=('ss',p)
            if key not in cache:cache[key]=ss_quantiles(residual,labels,p)
            return cache[key]*recipe['scale']
        return quantile(recipe['p_s']),quantile(recipe['p_S'])
    key=(method,recipe['beta'])
    if key not in cache:
        groups=labels if method=='SAA' else pipeline.variant_labels(labels,hot)[method]
        cache[key]=model.risk_levels(residual,groups,recipe['beta'],0.)[0]
    return (cache[key]+recipe['radius'])*recipe['scale'],None


if __name__=='__main__':
    from service_band_revision import main
    main()
