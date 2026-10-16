"""lasso：模型构造；统一训练和评价见training/forecast.py。"""
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def build(parameters, n_jobs=8):
    return make_pipeline(
        StandardScaler(),
        Lasso(**parameters, max_iter=2000, tol=0.001, selection='random', random_state=42)
    )
