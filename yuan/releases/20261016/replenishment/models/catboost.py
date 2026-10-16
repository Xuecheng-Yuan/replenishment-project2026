"""catboost：模型构造；统一训练和评价见training/forecast.py。"""

def build(parameters, n_jobs=8):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        **parameters,
        loss_function='RMSE',
        random_seed=42,
        thread_count=n_jobs,
        verbose=False,
        allow_writing_files=False
    )
