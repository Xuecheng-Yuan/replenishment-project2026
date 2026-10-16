"""模型唯一清单：名称、分组和入口统一，不再使用历史别名。"""

STAT_MODELS = ['baseline', 'naive', 'moving_average', 'exponential_smoothing', 'arima']
ML_MODELS = ['linear_regression', 'ridge', 'lasso', 'elastic_net', 'svr',
             'random_forest', 'xgboost', 'lightgbm', 'catboost']
FORECAST_MODELS = STAT_MODELS + ML_MODELS
DECISION_MODELS = ['spo_linear', 'xgboost_spo', 'lightgbm_spo',
                   'direct_xgboost', 'direct_lightgbm']
NEURAL_MODELS = ['nn_scenario', 'dfl_improved', 'service_policy',
                'online_learning', 'turnover_dfl']
ENSEMBLE_MODELS = ['equal_mean', 'global_stacking', 'sku_stacking', 'sku_router']
MODELS = FORECAST_MODELS + DECISION_MODELS + NEURAL_MODELS + ['adaptive_mpc'] + ENSEMBLE_MODELS

TITLES = dict(zip(MODELS, [
    '历史均值Baseline', '上一期需求', '移动平均', '阻尼指数平滑', 'ARIMA(p,1,0)',
    '线性回归', 'Ridge', 'Lasso', '弹性网络', 'SVR', '随机森林', 'XGBoost', 'LightGBM', 'CatBoost',
    '线性SPO+', 'XGBoost-SPO', 'LightGBM-SPO', '直接决策XGBoost', '直接决策LightGBM',
    '概率网络与内层优化', '改进DFL（4期）', '直接策略网络', '在线概率网络与内层优化',
    '库存比DFL（7期）', '自适应情景MPC', '等权融合', '全局Stacking', '逐SKU收缩Stacking', '逐SKU硬路由'
]))


def group(name):
    """只负责分组，不隐藏模型实现。"""
    for title, names in [('历史统计', STAT_MODELS), ('机器学习', ML_MODELS),
                         ('决策学习', DECISION_MODELS), ('神经网络与DFL', NEURAL_MODELS),
                         ('融合', ENSEMBLE_MODELS)]:
        if name in names:
            return title
    return '滚动优化'
