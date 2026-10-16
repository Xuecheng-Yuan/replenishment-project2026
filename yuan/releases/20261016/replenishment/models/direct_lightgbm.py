"""有限动作决策学习的实际损失和学习器，无旧版训练入口。"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import numpy as np

def direct_spo_rank_grad_hess(
    true_costs: np.ndarray,
    predicted_costs: np.ndarray,
    *,
    spo_weight: float,
    rank_weight: float,
    spo_temperature: float=0.5,
    rank_temperature: float=1.0,
    hessian_floor: float=1e-06
) -> tuple[np.ndarray, np.ndarray]:
    """混合损失的解析梯度；Hessian使用对角近似和数值下界。"""
    actual = np.asarray(true_costs, dtype=float)
    prediction = np.asarray(predicted_costs, dtype=float)
    if actual.shape != prediction.shape or actual.ndim != 2:
        raise ValueError('true_costs 和 predicted_costs 必须是同形二维数组。')
    if spo_weight < 0 or rank_weight < 0 or spo_weight + rank_weight > 1:
        raise ValueError('损失权重必须非负且 spo_weight + rank_weight <= 1。')
    if spo_temperature <= 0 or rank_temperature <= 0 or hessian_floor <= 0:
        raise ValueError('温度和Hessian下界必须为正。')
    n_samples, n_actions = actual.shape
    mse_weight = 1.0 - spo_weight - rank_weight
    gradient = mse_weight * (prediction - actual) / n_actions
    hessian = np.full_like(actual, mse_weight / n_actions)
    oracle = np.argmin(actual, axis=1)
    rows = np.arange(n_samples)
    if spo_weight > 0:
        logits = (actual - 2.0 * prediction) / spo_temperature
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        oracle_one_hot = np.zeros_like(actual)
        oracle_one_hot[rows, oracle] = 1.0
        gradient += 2.0 * spo_weight * (oracle_one_hot - probabilities)
        hessian += 4.0 * spo_weight * probabilities * (1.0 - probabilities) / spo_temperature
    if rank_weight > 0:
        oracle_cost = actual[rows, oracle][:, None]
        regret = np.maximum(actual - oracle_cost, 0.0)
        regret_sum = regret.sum(axis=1, keepdims=True)
        pair_weight = np.divide(regret, regret_sum, out=np.zeros_like(regret), where=regret_sum > 0)
        oracle_prediction = prediction[rows, oracle][:, None]
        pair_logits = np.clip((oracle_prediction - prediction) / rank_temperature, -40.0, 40.0)
        pair_probability = 1.0 / (1.0 + np.exp(-pair_logits))
        pair_gradient = pair_weight * pair_probability / rank_temperature
        pair_curvature = pair_weight * pair_probability * (1.0 - pair_probability) / rank_temperature ** 2
        gradient -= rank_weight * pair_gradient
        hessian += rank_weight * pair_curvature
        np.add.at(gradient, (rows, oracle), rank_weight * pair_gradient.sum(axis=1))
        np.add.at(hessian, (rows, oracle), rank_weight * pair_curvature.sum(axis=1))
    hessian += hessian_floor
    return (gradient.astype(np.float32), hessian.astype(np.float32))

@dataclass
class LGBMDirectSPOPolicy:
    """直接动作评分：平滑SPO+、遗憾加权排序和MSE。"""
    spo_weight: float
    rank_weight: float
    spo_temperature: float = 0.5
    rank_temperature: float = 1.0
    n_estimators: int = 40
    num_leaves: int = 63
    learning_rate: float = 0.05
    min_child_samples: int = 100
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    n_jobs: int = 8
    random_seed: int = 42
    model_: object | None = field(init=False, default=None, repr=False)
    cost_scale_: float | None = field(init=False, default=None)
    actions_: np.ndarray | None = field(init=False, default=None, repr=False)
    uses_action_targets: bool = field(init=False, default=True, repr=False)

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        costs: np.ndarray,
        actions: np.ndarray
    ) -> 'LGBMDirectSPOPolicy':
        """展开库存动作，用MSE、平滑SPO+和遗憾加权排序的混合目标训练。"""
        from lightgbm import LGBMRegressor
        X = np.asarray(features, dtype=np.float32)
        B = np.asarray(targets, dtype=np.float32)
        C = np.asarray(costs, dtype=np.float32)
        A = np.asarray(actions, dtype=np.float32)
        if X.ndim != 2 or B.ndim != 2 or C.shape != B.shape or (len(X) != len(B)):
            raise ValueError('features、targets和costs的维度不一致。')
        if B.shape[1] != len(A) or len(A) < 2:
            raise ValueError('动作数量与targets列数不一致。')
        if not np.isfinite(X).all() or not np.isfinite(B).all() or (not np.isfinite(C).all()):
            raise ValueError('训练数据包含非有限值。')
        positive = C[C > 0]
        self.cost_scale_ = float(np.median(positive)) if positive.size else 1.0
        if not np.isfinite(self.cost_scale_) or self.cost_scale_ <= 0:
            self.cost_scale_ = 1.0
        self.actions_ = A.copy()
        expanded = self._expand_features(X, B, A)
        labels = (C / self.cost_scale_).reshape(-1)
        n_actions = len(A)

        def objective(y_true: np.ndarray, y_pred: np.ndarray):
            """内层代理目标：库存需求比，加服务水平95%—96%的双侧罚项。"""
            gradient, hessian = direct_spo_rank_grad_hess(
                np.asarray(y_true).reshape(-1, n_actions),
                np.asarray(y_pred).reshape(-1, n_actions),
                spo_weight=self.spo_weight,
                rank_weight=self.rank_weight,
                spo_temperature=self.spo_temperature,
                rank_temperature=self.rank_temperature
            )
            return (gradient.reshape(-1), hessian.reshape(-1))
        self.model_ = LGBMRegressor(
            objective=objective,
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            n_jobs=self.n_jobs,
            random_state=self.random_seed,
            verbosity=-1
        )
        self.model_.fit(expanded, labels)
        return self

    def predict_costs(self, features: np.ndarray, targets: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.cost_scale_ is None or self.actions_ is None:
            raise ValueError('模型尚未拟合。')
        X = np.asarray(features, dtype=np.float32)
        B = np.asarray(targets, dtype=np.float32)
        if B.shape != (len(X), len(self.actions_)):
            raise ValueError('预测targets维度与模型动作数不一致。')
        expanded = self._expand_features(X, B, self.actions_)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='X does not have valid feature names')
            prediction = np.asarray(self.model_.predict(expanded), dtype=float)
        return prediction.reshape(len(X), len(self.actions_)) * self.cost_scale_

    def predict_actions(self, features: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return np.argmin(self.predict_costs(features, targets), axis=1)

    def save_model(self, path: str | Path) -> None:
        if self.model_ is None:
            raise ValueError('模型尚未拟合。')
        self.model_.booster_.save_model(str(Path(path)))

    @staticmethod
    def _expand_features(features: np.ndarray, targets: np.ndarray, actions: np.ndarray) -> np.ndarray:
        n_samples = len(features)
        n_actions = len(actions)
        return np.column_stack([
            np.repeat(features, n_actions, axis=0),
            np.tile(actions, n_samples),
            targets.reshape(-1)
        ]).astype(np.float32)
