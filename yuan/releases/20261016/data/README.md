# 数据放置

原始Excel不随代码提交，不自动从GitHub下载。在本机运行时将`replenishment_train.xlsx`、`replenishment_test.xlsx`放在此目录，或在prepare命令中传入你的实际路径。

训练60期、测试28期、6000SKU、20个匿名特征；需求非负，SKU必须对齐。目标期的x不能提前使用。程序会核验输入并冻结哈希。

`artifacts/panel.npz`仅保存网站和订单核验所需需求、初始库存及SKU信息，不含训练所需完整20特征，因此不能拿它替代训练原始输入。
