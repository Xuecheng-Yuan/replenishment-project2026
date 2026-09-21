"""均值–方差 baseline 的训练逻辑测试。"""

import pandas as pd

from replenishment.models import BaselineConfig, MeanVarianceBaseline


def test_baseline_uses_train_internal_time_split() -> None:
    """确认前段估计、后段校准，并在最终阶段使用完整 train。"""

    train = pd.DataFrame(
        {
            "item_id": [1, 2],
            "period_1": [10, 20],
            "period_2": [10, 20],
            "period_3": [12, 18],
            "period_4": [11, 19],
            "period_5": [10, 20],
            "period_6": [11, 21],
        }
    )
    model = MeanVarianceBaseline(
        BaselineConfig(validation_periods=2, z_min=0, z_max=2, z_step=0.5)
    ).fit(train)
    artifacts = model.get_artifacts()

    assert artifacts.fit_periods == 4
    assert artifacts.validation_periods == 2
    assert len(artifacts.parameters) == 2
    assert (artifacts.parameters["s"] == artifacts.parameters["S"]).all()
    assert artifacts.train_statistics["mean_demand"].tolist() == [10.666666666666666, 19.666666666666668]


def test_fit_interface_does_not_accept_test_data() -> None:
    """拟合接口只有 train_demand，防止把 test 指标用于参数选择。"""

    import inspect

    parameters = list(inspect.signature(MeanVarianceBaseline.fit).parameters)
    assert parameters == ["self", "train_demand"]
