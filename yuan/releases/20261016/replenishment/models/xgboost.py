"""xgboost：模型构造；统一训练和评价见training/forecast.py。"""

def build(parameters, n_jobs=8):
    from xgboost import XGBRegressor
    return XGBRegressor(
        **parameters,
        objective='reg:squarederror',
        tree_method='hist',
        max_bin=256,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=n_jobs,
        random_state=42
    )
