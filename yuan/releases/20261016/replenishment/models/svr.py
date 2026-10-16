"""svr：模型构造；统一训练和评价见training/forecast.py。"""
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def build(parameters, n_jobs=8):
    regressor = make_pipeline(
        StandardScaler(),
        PCA(n_components=0.95, svd_solver='full'),
        SVR(kernel='rbf', gamma='scale', cache_size=2048, **parameters)
    )
    return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())
