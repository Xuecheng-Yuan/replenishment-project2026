"""兼容模型入口的训练边界测试。"""

import inspect

import pandas as pd

from replenishment.models import BaselineConfig, MeanVarianceBaseline


def test_baseline_alias_uses_current_train_split_and_refit() -> None:
    """兼容名称仍执行当前problem-based方法。"""

    train = pd.DataFrame(
        {
            "item_id": [1, 2],
            **{
                f"period_{period}": [10 + period % 2, 20 + period % 3]
                for period in range(1, 11)
            },
        }
    )
    model = MeanVarianceBaseline(
        BaselineConfig(validation_periods=4, z_min=0, z_max=1, z_step=0.5)
    ).fit(train)
    artifacts = model.get_artifacts()

    assert artifacts.fit_periods == 6
    assert artifacts.validation_periods == 4
    assert len(artifacts.parameters) == 2
    assert (artifacts.parameters["s"] == artifacts.parameters["S"]).all()
    assert artifacts.parameters["mean_demand"].tolist() == [10.5, 21.0]


def test_fit_interface_does_not_accept_test_data() -> None:
    """拟合接口只有train需求，防止test参与参数选择。"""

    parameters = list(inspect.signature(MeanVarianceBaseline.fit).parameters)
    assert parameters == ["self", "train_demand"]
