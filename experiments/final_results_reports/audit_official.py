"""Replay every frozen official policy; no model refit or parameter selection."""
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/constraint_mechanism'))
from analyze_results import audit_published

if __name__=='__main__':
    (HERE/'evidence').mkdir(exist_ok=True)
    audit_published(HERE/'evidence',np.load(ROOT/'experiments/constraint_mechanism/outputs/shared_arrays.npz'))
