"""兼容入口；统一使用problem_based_methods评价器。"""

from .problem_evaluation import EvaluationResult, InventoryEvaluator

__all__ = ["EvaluationResult", "InventoryEvaluator"]
