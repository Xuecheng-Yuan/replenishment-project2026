# Experiments

每个正式实验使用独立脚本，并采用 `run_<method>.py` 命名。脚本只负责读取配置、调用 `src/replenishment/` 中的可复用逻辑并保存结果，不应重复实现核心算法。

- `run_baseline.py`：运行静态均值–方差 baseline。
- `run_time_series.py`：运行时间序列滚动验证、冻结最终策略并评价 test。
