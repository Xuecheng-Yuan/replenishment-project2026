"""有限动作决策学习的实际损失和学习器，无旧版训练入口。"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

def hybrid_grad_hess(
    true_costs: np.ndarray,
    predicted_costs: np.ndarray,
    *,
    alpha: float,
    temperature: float,
    hessian_floor: float=1e-06
) -> tuple[np.ndarray, np.ndarray]:
    """MSE和平滑SPO+的解析梯度及对角Hessian。"""
    actual = np.asarray(true_costs, dtype=float)
    prediction = np.asarray(predicted_costs, dtype=float)
    if actual.shape != prediction.shape or actual.ndim != 2:
        raise ValueError('true_costs 和 predicted_costs 必须是同形二维数组。')
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('alpha 必须在 [0, 1] 内。')
    if temperature <= 0 or hessian_floor <= 0:
        raise ValueError('temperature 和 hessian_floor 必须为正。')
    n_actions = actual.shape[1]
    logits = (actual - 2.0 * prediction) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    oracle = np.argmin(actual, axis=1)
    oracle_one_hot = np.zeros_like(actual)
    oracle_one_hot[np.arange(len(actual)), oracle] = 1.0
    gradient = (1.0 - alpha) * (prediction - actual) / n_actions + 2.0 * alpha * (oracle_one_hot - probabilities)
    hessian = (1.0 - alpha) / n_actions + 4.0 * alpha * probabilities * (1.0 - probabilities) / temperature + hessian_floor
    return (gradient.astype(np.float32), hessian.astype(np.float32))

@dataclass
class XGBSPOHybrid:
    """XGBoost有限动作成本学习器，可组合MSE与平滑SPO+。"""
    alpha: float
    temperature: float = 0.5
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    n_jobs: int = 8
    random_seed: int = 42
    model_: object | None = field(init=False, default=None, repr=False)
    cost_scale_: float | None = field(init=False, default=None)

    def fit(self, features: np.ndarray, costs: np.ndarray) -> 'XGBSPOHybrid':
        """归一化动作成本，以自定义梯度和对角Hessian训练多输出XGBoost。"""
        from xgboost import XGBRegressor
        X = np.asarray(features, dtype=np.float32)
        C = np.asarray(costs, dtype=np.float32)
        if X.ndim != 2 or C.ndim != 2 or len(X) != len(C) or (C.shape[1] < 2):
            raise ValueError('features/costs 必须是行数相同的二维数组，且至少有两个动作。')
        if not np.isfinite(X).all() or not np.isfinite(C).all():
            raise ValueError('训练数据包含非有限值。')
        positive = C[C > 0]
        self.cost_scale_ = float(np.median(positive)) if positive.size else 1.0
        if not np.isfinite(self.cost_scale_) or self.cost_scale_ <= 0:
            self.cost_scale_ = 1.0
        normalized = C / self.cost_scale_

        def objective(y_true: np.ndarray, y_pred: np.ndarray):
            """内层代理目标：库存需求比，加服务水平95%—96%的双侧罚项。"""
            return hybrid_grad_hess(y_true, y_pred, alpha=self.alpha, temperature=self.temperature)
        self.model_ = XGBRegressor(
            objective=objective,
            multi_strategy='one_output_per_tree',
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            tree_method='hist',
            max_bin=256,
            n_jobs=self.n_jobs,
            random_state=self.random_seed
        )
        self.model_.fit(X, normalized, verbose=False)
        return self

    def predict_costs(self, features: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.cost_scale_ is None:
            raise ValueError('模型尚未拟合。')
        prediction = np.asarray(self.model_.predict(np.asarray(features, dtype=np.float32)), dtype=float)
        return prediction * self.cost_scale_

    def predict_actions(self, features: np.ndarray) -> np.ndarray:
        return np.argmin(self.predict_costs(features), axis=1)

    def save_model(self, path: str | Path) -> None:
        if self.model_ is None:
            raise ValueError('模型尚未拟合。')
        self.model_.save_model(Path(path))
