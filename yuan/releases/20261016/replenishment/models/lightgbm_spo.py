"""有限动作决策学习的实际损失和学习器，无旧版训练入口。"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from .xgboost_spo import hybrid_grad_hess

@dataclass
class LGBMSPOHybrid:
    """LightGBM展开动作特征，学习MSE和平滑SPO+混合损失。"""
    alpha: float
    temperature: float = 0.5
    n_estimators: int = 100
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

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        costs: np.ndarray,
        actions: np.ndarray
    ) -> 'LGBMSPOHybrid':
        """每个样本展开全部动作，自定义目标按整组动作计算耦合梯度。"""
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
            gradient, hessian = hybrid_grad_hess(
                np.asarray(y_true).reshape(-1, n_actions),
                np.asarray(y_pred).reshape(-1, n_actions),
                alpha=self.alpha,
                temperature=self.temperature
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
