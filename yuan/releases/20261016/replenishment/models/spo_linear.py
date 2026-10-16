"""有限动作决策学习的实际损失和学习器，无旧版训练入口。"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class SPOPlusLinear:
    """有限动作成本向量的线性SPO+学习器。"""
    epochs: int = 12
    batch_size: int = 2048
    learning_rate: float = 0.05
    ridge: float = 0.0001
    random_seed: int = 42
    coefficients_: np.ndarray | None = field(init=False, default=None, repr=False)
    feature_mean_: np.ndarray | None = field(init=False, default=None, repr=False)
    feature_scale_: np.ndarray | None = field(init=False, default=None, repr=False)
    cost_scale_: float | None = field(init=False, default=None)
    history_: list[dict] = field(init=False, default_factory=list)

    def fit(self, features: np.ndarray, costs: np.ndarray) -> 'SPOPlusLinear':
        """标准化特征与成本，按mini-batch平均次梯度最小化线性SPO+。"""
        X = np.asarray(features, dtype=np.float32)
        C = np.asarray(costs, dtype=np.float32)
        if X.ndim != 2 or C.ndim != 2 or len(X) != len(C):
            raise ValueError('features/costs 必须是行数相同的二维数组。')
        if len(X) == 0 or C.shape[1] < 2 or (not np.isfinite(X).all()) or (not np.isfinite(C).all()):
            raise ValueError('训练数据为空、动作不足或包含非有限值。')
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0 or (self.ridge < 0):
            raise ValueError('训练超参数不合法。')
        self.feature_mean_ = X.mean(axis=0, dtype=np.float64).astype(np.float32)
        raw_scale = X.std(axis=0, dtype=np.float64).astype(np.float32)
        self.feature_scale_ = np.where(raw_scale > 1e-06, raw_scale, 1.0).astype(np.float32)
        Xn = np.clip((X - self.feature_mean_) / self.feature_scale_, -10.0, 10.0)
        Xn = np.column_stack([np.ones(len(Xn), dtype=np.float32), Xn]).astype(np.float32)
        positive = C[C > 0]
        self.cost_scale_ = float(np.median(positive)) if positive.size else 1.0
        if not np.isfinite(self.cost_scale_) or self.cost_scale_ <= 0:
            self.cost_scale_ = 1.0
        Cn = C / self.cost_scale_
        n_samples, n_features = Xn.shape
        n_actions = Cn.shape[1]
        weights = np.zeros((n_features, n_actions), dtype=np.float32)
        averaged = np.zeros_like(weights)
        averaging_weight = 0.0
        rng = np.random.default_rng(self.random_seed)
        step = 0
        self.history_ = []
        for epoch in range(self.epochs):
            order = rng.permutation(n_samples)
            mismatch_count = 0
            for start in range(0, n_samples, self.batch_size):
                indices = order[start:start + self.batch_size]
                xb = Xn[indices]
                cb = Cn[indices]
                predicted = xb @ weights
                true_best = np.argmin(cb, axis=1)
                adversarial_best = np.argmin(2.0 * predicted - cb, axis=1)
                mismatch_count += int(np.count_nonzero(true_best != adversarial_best))
                direction = np.zeros((len(indices), n_actions), dtype=np.float32)
                rows = np.arange(len(indices))
                direction[rows, true_best] += 2.0
                direction[rows, adversarial_best] -= 2.0
                gradient = xb.T @ direction / len(indices)
                gradient[1:] += self.ridge * weights[1:]
                step += 1
                rate = self.learning_rate / np.sqrt(step)
                weights -= rate * gradient
                averaged += rate * weights
                averaging_weight += rate
            current = averaged / averaging_weight
            predictions = Xn @ current
            chosen = np.argmin(predictions, axis=1)
            oracle = np.argmin(Cn, axis=1)
            spo_loss = float(np.mean(Cn[np.arange(n_samples), chosen] - Cn[np.arange(n_samples), oracle]))
            self.history_.append({
                'epoch': epoch + 1,
                'spo_loss_normalized': spo_loss,
                'oracle_mismatch_share': mismatch_count / n_samples
            })
        self.coefficients_ = averaged / averaging_weight
        return self

    def predict_costs(self, features: np.ndarray) -> np.ndarray:
        """Predict an action-cost vector for each row."""
        if self.coefficients_ is None or self.feature_mean_ is None or self.feature_scale_ is None:
            raise ValueError('模型尚未拟合。')
        X = np.asarray(features, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != len(self.feature_mean_):
            raise ValueError('预测特征维度与训练时不一致。')
        Xn = np.clip((X - self.feature_mean_) / self.feature_scale_, -10.0, 10.0)
        Xn = np.column_stack([np.ones(len(Xn), dtype=np.float32), Xn]).astype(np.float32)
        return Xn @ self.coefficients_ * float(self.cost_scale_)

    def predict_actions(self, features: np.ndarray) -> np.ndarray:
        """Return the optimization oracle decision for every row."""
        return np.argmin(self.predict_costs(features), axis=1)

    def state_dict(self) -> dict[str, np.ndarray | float | list[dict]]:
        if self.coefficients_ is None or self.feature_mean_ is None or self.feature_scale_ is None:
            raise ValueError('模型尚未拟合。')
        return {
            'coefficients': self.coefficients_,
            'feature_mean': self.feature_mean_,
            'feature_scale': self.feature_scale_,
            'cost_scale': float(self.cost_scale_),
            'history': self.history_
        }
