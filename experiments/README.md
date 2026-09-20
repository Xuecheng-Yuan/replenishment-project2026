# Experiments

每个正式实验使用独立脚本，并采用 `run_<method>.py` 命名。脚本只负责读取配置、调用 `src/replenishment/` 中的可复用逻辑并保存结果，不应重复实现核心算法。

