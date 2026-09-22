# Experiments

正式实验使用 `run_<method>.py` 命名。脚本只负责参数解析、调用核心模块和打印结果，不重复实现模型。

- `run_problem_based_methods.py`：运行正式problem-based baseline。
- `run_baseline.py`：兼容旧命令，转发到同一个problem-based实验。

运行命令：

```bash
python -m experiments.run_problem_based_methods
```
