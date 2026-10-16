# 新旧两版定稿实验结果报告（2026-10-16）

两份报告锁定此前正式交付的六方法结果，补齐模型解读、60期训练使用方式、参数、整体与爆品/长尾表现、四期预测误差和复现说明。与 `../constraint_mechanism` 的原因探索报告分开保存；诊断实验不覆盖正式结果。

- [旧版：需求满足率≥95%](reports/旧版_需求满足率_最终实验结果报告.pdf)：分类DRO，J=0.168340、F=96.5615%。
- [新版：正库存比例95%–95.5%](reports/新版_正库存比例_最终实验结果报告.pdf)：(s,S)，J=0.345958、P=95.2810%。
- 同目录 `.md` 可直接在GitHub阅读；`.json` 为PDF排版源；`figures` 含六方法整体、爆品/长尾、28期动态共6张图。
- `evidence` 为重新回放正式政策得到的摘要、类别及逐期结果、原文件哈希与历史配对区间。所有数值均有正式明细来源。两版终稿各7页。

整体服务达标不等于每个商品或类别均达标。新版(s,S)相对分类DRO只是点估计较低，配对区间跨0。J是累计期末库存/需求、越低越好，与常见销售/平均库存的周转率不同。

在项目根目录运行（Python，依赖见 `experiments/rolling_dro_v3/requirements_v3.txt`）：

```powershell
# 完整回放核验新旧各6个冻结策略；不重选参数
python experiments/final_results_reports/audit_official.py
# 生成报告文字、表格和6张图（需matplotlib）
python experiments/final_results_reports/build_reports.py
# PDF生成额外安装 reportlab>=4,<5、pypdf>=5,<7；需要中文字体
python experiments/constraint_mechanism/render_reports.py --reports experiments/final_results_reports/reports
# 校验定稿哈希、正式结果数值与报告表格（不重建PDF）
python experiments/final_results_reports/verify_reports.py
```

完整审计读取约201.6万条正式轨迹；`verify_reports.py`适用于CI快速核验。重建PDF可能因字体或库版本使文件字节变化，先保留定稿副本；SHA256SUMS.json记录的是本次已交付版本。代码无需Git元数据。最终拟合包含训练文件全部60期，测试使用独立测试文件28期、周期重新起算；此前已多次查看测试结果，因此不声称全新盲测。
