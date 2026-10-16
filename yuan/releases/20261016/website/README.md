# 离线网站

`offline/index.html`是便携单文件版本，包含页面代码、公式和完整冻结数据；不需要公网服务器。也可在本包根目录运行`python3 run.py serve --port 8793`，打开`http://127.0.0.1:8793/`。本机服务器仅绑定127.0.0.1。

`app/`为保留共享运行时的网站可编辑工程。只在`src/content/`修改内容，保留`AGENTS.md`要求；不要手动修改受保护的运行时或构建清单。

数据来自本包`configs/model_catalog.json`、`results/benchmark/`和`artifacts/`。更新文档/轨迹后先运行`tools/build_docs.py`、`tools/build_website_data.py --complete`，再按`app/AGENTS.md`使用Data插件构建和export-offline。插件是网站重新构建的工具依赖，不是查看离线网站的依赖。

负责人筛选区分Yuan、Kristina和全部；“算法原理”解释输入、训练、公式及下单步骤，可下载对应MD；“逐期实验”按SKU、模型、周期查看真实库存与订单；“模型对比”支持三指标坐标选择与经验前沿。

完整离线HTML含完整研究数据，默认不入Git。请先确认共享权限，不要把“离线”误认为“匿名化”或“可无条件公开”。
