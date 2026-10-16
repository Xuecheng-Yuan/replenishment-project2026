"""linear_regression：模型构造；统一训练和评价见training/forecast.py。"""
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def build(parameters, n_jobs=8):
    return make_pipeline(StandardScaler(), LinearRegression(n_jobs=n_jobs))
