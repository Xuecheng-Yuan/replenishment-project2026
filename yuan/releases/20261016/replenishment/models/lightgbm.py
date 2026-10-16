"""lightgbm：模型构造；统一训练和评价见training/forecast.py。"""

def build(parameters, n_jobs=8):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        **parameters,
        objective='regression_l1',
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=n_jobs,
        random_state=42,
        verbosity=-1
    )
