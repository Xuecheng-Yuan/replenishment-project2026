# 自备授权数据

放replenishment_train.xlsx（60期）、replenishment_test.xlsx（28期）到本目录。每个文件包含连续Period_1、Period_2……sheet，每期6000行SKU；所有sheet和train/test的item_id身份/顺序一致，不能重复。

第k张sheet：item_id、category、initial_inventory、demand_k、x1–x20。需求和初始库存非负，需求/x有限且无缺失。测试sheet周期编号重新从1开始。后期缓存仅有第13期以来x，严格滞后模型共同从目标第14期开始，不补造缺失早期x。

本包不包含Excel、二进制模型或原始SKU轨迹。数据读取只为复现用户自备合法数据；没有合适数据时仅阅读默认历史成绩。若真实业务只见销量，不能当作未截断的完整需求，此问题另行处理。变更数据必须新克隆运行。
