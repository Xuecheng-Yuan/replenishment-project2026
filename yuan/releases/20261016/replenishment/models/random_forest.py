"""random_forest：模型构造；统一训练和评价见training/forecast.py。"""
from sklearn.ensemble import RandomForestRegressor

def build(parameters, n_jobs=8):
    return RandomForestRegressor(**parameters, n_jobs=n_jobs, max_features=0.8, random_state=42)
