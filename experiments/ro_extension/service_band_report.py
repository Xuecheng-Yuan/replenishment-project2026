"""Conclusion-first report of the actual narrow-service-band experiment."""
import json
import pandas as pd
from service95_revision import ROOT,HOME,LABELS


def write_report():
    out=HOME/'outputs_v4';cur=pd.read_csv(out/'summary.csv').query("scope=='all_28_periods'").set_index('method')
    design=json.loads((out/'frozen_parameters.json').read_text(encoding='utf-8'));audit=json.loads((out/'test_audit.json').read_text(encoding='utf-8'))
    best=cur.sort_values('inventory_demand_ratio').iloc[0]
    lines=['# 服务水平95%附近：重跑与交付报告','',
        f'六种方法完整28期正库存比例为{cur.positive_stock_ratio.min():.4%}–{cur.positive_stock_ratio.max():.4%}，全部达到95%–95.5%。在本次候选与反馈控制器中，{LABELS[best.name]}库存需求比最低，为{best.inventory_demand_ratio:.6f}。',
        '', '新版继续用训练文件全部60期最终拟合，测试使用独立replenishment_test.xlsx的28期。测试从Period_1重新开始，使用原始初始库存、提前期1、零初始在途，缺货按丢失处理。测试计分包含首期、全部6000商品。',
        '', '## 本次做了什么', '',
        '1. 恢复95%≤P≤95.5%的双边服务约束，撤下服务超过99%的高库存方案。高库存方案的冻结参数、结果和代码保存在history/conservative_99_percent，便于查阅。',
        '2. 删除静态训练平均需求储备下限，保留训练60期及四个训练切片风险水位的保护包络作为各方法的基础库存结构。',
        '3. 增加因果服务预算反馈：本期需求发生后，统计截至本期已有正库存商品天，按完整28期目标95.25%计算剩余时期所需服务比例，结合训练残差与已经发生的预测误差，选择下一期补货水位。',
        '4. 在训练内部的28期周期重启留出上，从108个反馈候选中按95%–95.5%筛选，再最小化库存需求比。原三轮向前/重启验证保留为12期前缀诊断。选参用途、计分范围和时间关系在下文分开说明。',
        '5. 重跑六方法并更新Excel、逐商品明细、版本对照、全部6000/爆品480/长尾5520动画、公式说明、代码、本地文件包和GitHub。',
        '', '## 完整28期测试结果', '',
        '|方法|正库存商品天/168000|正库存比例P|需求满足率F|库存需求比J↓|95%–95.5%|',
        '|---|---:|---:|---:|---:|---|']
    for m,r in cur.iterrows():lines.append(f'|{LABELS[m]}|{int(r.positive_stock_sku_periods):,}|{r.positive_stock_ratio:.4%}|{r.fill_rate:.4%}|{r.inventory_demand_ratio:.6f}|{"达到" if .95<=r.positive_stock_ratio<=.955 else "未达"}|')
    delta=(cur.loc['sS','inventory_demand_ratio']-cur.loc['DRO_category','inventory_demand_ratio'])/cur.loc['sS','inventory_demand_ratio']
    lines+=['',f'三种DRO中，爆品/长尾DRO的库存需求比最低，为{cur.loc["DRO_category","inventory_demand_ratio"]:.6f}，比传统(s,S)高{-delta:.2%}。本次没有获得DRO优于传统基线的库存结果。RO与不分组DRO非常接近：两者采用统一残差池，公共反馈会抵消大量整体水位倍率差异。',
        '', 'P按每个商品期末库存严格大于0计数，再除以6000×28。恰好售罄记为0；F按销量/需求计算。双边约束作用于全部商品完整28期，不要求每一天或每个商品各自都在该区间。J是累计期末库存/累计需求，越低越好，与通常的销量/平均库存周转次数方向相反。',
        '', '## 训练60期、选参及独立测试如何使用', '',
        '|用途|拟合数据|检验数据|说明|', '|---|---|---|---|',
        '|预测超参数与前缀诊断1|训练1–24|训练25–36|向前与周期重启两情景|',
        '|预测超参数与前缀诊断2|训练1–36|训练37–48|向前与周期重启两情景|',
        '|预测超参数与前缀诊断3|训练1–48|训练49–60|向前与周期重启两情景|',
        '|反馈策略选参|训练29–60|训练1–28|分块回溯重启留出，拟合与检验不重叠|',
        '|最终拟合|训练全部1–60|独立测试文件1–28|参数先冻结，再测试，逐期仅吸收已发生需求|',
        '', '28期重启留出用于校准“周期重新起算”的业务情景。拟合段位于检验段之后，因此它属于训练内部回溯迁移检验，不是向前预测性能估计。原三轮12期诊断仍只用更早数据拟合预测；前缀未满28期，不用其12期整体P判断完整28期双边约束是否满足。每个情景均从原始库存、零在途开始。',
        '预测超参数沿用同一三轮训练验证选出的blend=1、window=3、prior=0。基础风险参数来自仅使用训练数据选出的六方法方案；本次进一步训练公共反馈的窗口、先验权重和预测偏置。每种方法18个反馈候选，共108个，按训练1–28留出P落在[95%,95.5%]筛选，再按J和候选号排序。',
        f'最终拟合保存60×6000＝{design["final_fit_observations"]:,}个需求观测及其SHA256：`{design["training_y_sha256"]}`。前期数据参与最终预测、聚类和历史残差计算；最早18期作为残差预测的暖启动，19–60期产生扩展窗预测残差。',
        '', '## 数学模型与解读', '',
        r'目标：$\min J=\sum_{i=1}^{N}\sum_{t=1}^{28}I_{it}/\sum_{i,t}d_{it}$，约束：$0.95\le P=\sum_{i,t}\mathbf1\{I_{it}>0\}/(28N)\le0.955$，$N=6000$。需求满足率$F=\sum s_{it}/\sum d_{it}$作为补充指标。',
        r'库存流转：$A_{it}=I_{i,t-1}+q_{i,t-1}$，$s_{it}=\min(A_{it},d_{it})$，$I_{it}=\max(A_{it}-d_{it},0)$，$\ell_{it}=d_{it}-s_{it}$，$q_{it}\in\mathbb Z_+$，初始在途为0。',
        r'本期t需求发生后，用当时已有信息预测后四期$m_{it,h}$，仅执行最近一期订单。DRO基础水位$z_{ig}^{(b)}=\kappa(r_{(\lceil\beta M\rceil)}+\varepsilon)$，样本$r=d/m$来自拟合切片b中的扩展窗预测残差，分布集合$\mathcal P_b=\{Q:W_\infty(Q,\widehat P_b)\le\varepsilon\}$。',
        '特征聚类DRO、不分组DRO、爆品/长尾DRO分别使用特征池、统一池和两品类池。各切片按商品对齐后取最大基础水位z*；零半径经验模型取ε=0。传统RO用各训练切片μ+Γσ上界的最大值。传统(s,S)使用逐商品与聚类经验分位数加权的双水位。',
        r'累计服务预算：$C_t=\sum_{\tau=1}^{t}\sum_i\mathbf1\{I_{i\tau}>0\}$。内部目标$p^*=0.9525$，下一期所需比例$u_t=\mathrm{clip}((28Np^*-C_t)/(N\max(28-t,1))+b,0,1)$。b为训练选择的预测偏置。',
        r'反馈后的目标库存为$S_{it,h}(\lambda)=\lfloor\lambda m_{it,h}z_i^*\rfloor+1$。整数订单$q_{it,h}=\max(0,\lceil S_{it,h}(\lambda)-\widehat I_{it,h-1}\rceil)$，(s,S)另按同一λ缩放触发水位s；未来计划库存按预测需求递推。选择最小非负λ，使估计的下一期总体正库存比例不低于$u_t$。搜索上限128，16次二分。',
        r'服务响应估计：每个商品池保留训练残差CDF与最近w期已发生的单步预测误差CDF，混合权重为$a/(a+n)$，其中a为训练先验权重、n为已有近期误差期数。CDF用128个中点分位数近似。估计服务率为$N^{-1}\sum_i\widehat F_{g,t}((I_{it}+q_{it})/m_{it,1})$，严格不等式按左侧CDF计数。',
        '当前期末库存、已发生需求、训练残差和历史单步预测误差均已知；下一期实际需求不参与λ选择。首期没有本业务段中提前形成的单步预测，因而不把首期需求除以事后修正预测伪装成单步误差。',
        '动态λ可以小于1，因此基础DRO机会水位的固定β保护不能直接解释为执行策略的服务保证。本次交付的是DRO/RO基础水位加公共服务预算控制器，达标证据为完整测试实际计数；在当前候选控制器中比较J，不宣称无约束策略空间全局最优。',
        '', '## 为什么这次能靠近95%', '',
        '上一轮静态风险水位在训练和测试重新起算之间存在偏移；直接增加储备又把P推到99%以上。本次将库存水平与累计服务预算联系：截至本期已高于目标时，降低后续补货水位；落后时，提高后续水位。保留风险分组差异的同时，对总体服务水平实施反馈，避免固定储备长期积压。',
        '本次比高库存中间方案显著降低J，但与PR7未达标方案相比，多数方法仍需要更多库存来满足服务要求。Excel和version_comparison.csv保留历史需求满足率版、PR7版、本次95%带宽版三组结果；高库存中间方案单独保留，不作为最终交付结果。',
        '', '## 特征聚类的依据', '',
        '五个需求描述为平均需求、变异系数、零需求比例、95%分位数/均值、后8期/前8期需求比。原始20特征按拟合切片取时间均值、标准化并PCA，取前三个主成分，以0.5权重与标准化需求描述合并。6类K-means固定随机种子、8次初始化，选最小平方误差解。',
        '每轮仅用拟合切片聚类，最终聚类用全部60期；不使用测试未来特征。最终反馈分组沿用相应方法的最终聚类、统一池或原始爆品/长尾类别。',
        '', '## 验证记录、适用范围与复现', '',
        f'测试前冻结参数SHA256：`{audit["frozen_parameters_sha256"]}`。完整新版1,008,000条逐商品轨迹与原始需求对账，首期{audit["first_positive"]}个商品正库存；即使首期不可控，28期理论上限为{audit["max_possible_service"]:.6%}。实际P统计没有排除首期或挑选商品。',
        '同一测试文件在前面模型修订中已多次评估，本轮属于重复测试；参数数值按声明的训练规则选择，在线反馈只读取当时已发生的测试需求。对任意未见需求不保证一定落入窄区间，泛化需要新的未使用数据检验。',
        '当前结果：outputs_v4；逐期缩放与预算：service_feedback.csv；完整六方法决策：detail_*.csv.gz和decision_comparison_full.csv.gz；训练候选108条、已选七情景42条。Excel有三版对照、全部6000商品最近一期决策及12商品完整28期。动画覆盖全部6000、480爆品、5520长尾及10个固定SKU。',
        '', '```powershell',
        'python -m pip install -r experiments/rolling_dro_v3/requirements_v3.txt',
        'python experiments/ro_extension/service_band_revision.py',
        'python experiments/ro_extension/service95_artifacts.py',
        'node experiments/ro_extension/build_workbooks.mjs --service95',
        'python scripts/build_animation.py',
        'python experiments/ro_extension/verify_service95.py',
        'python scripts/verify_animation.py', '```',
        'Python数值模型可独立安装requirements运行；Excel重建需Node.js及可用的@oai/artifact-tool，本机自动使用Codex自带环境。已交付Excel可直接打开，动画离线运行。',
        '', '方法参考：[RSOME多阶段库存示例](https://xiongpengnus.github.io/rsome/example_dro_inv)。该示例为零提前期和积压成本；本项目按提前期1、失销、四期滚动与正库存比例改写。']
    segment=pd.read_csv(out/'summary_by_segment.csv')
    split=['## 爆品与长尾品分别表现','', '|方法|类别|正库存比例P|需求满足率F|库存需求比J|','|---|---|---:|---:|---:|']
    for r in segment.itertuples():split.append(f'|{LABELS[r.method]}|{r.segment}|{r.positive_stock_ratio:.4%}|{r.fill_rate:.4%}|{r.inventory_demand_ratio:.6f}|')
    split+=['','总体P=(480×爆品P+5520×长尾P)/6000。各类别不承担独立95%下限，本次爆品P低于总体，表中如实展示；若要求每类都达到95%，需要另设分类约束并重新求解。','']
    where=lines.index('## 训练60期、选参及独立测试如何使用');lines[where:where]=split
    (HOME/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (HOME/'README.md').write_text('# 当前：服务水平95%–95.5%\n\n[任务报告与数学模型](REPORT.md) · [六方法Excel与三版对照](outputs_v4/replenishment_comparison.xlsx) · [全量动画](../../汇报成果/2026-10-15/动画/index.html)\n\n用训练文件全部60期最终拟合，独立测试文件28期。六种方法全部落在95%–95.5%。服务预算反馈只使用已发生需求。历史PR7版保留在../full60_revision；高库存中间方案保留在history/conservative_99_percent。\n',encoding='utf-8')
    (HOME/'DELIVERY_README.md').write_text('# 李佳蔚：服务95%附近重跑文件包\n\n先读任务完成报告.md；当前Excel为新版_全60期_六方法.xlsx，含三版对照。动画/index.html支持全部6000、爆品480、长尾5520和10个SKU。项目目录保留可运行代码、原始Excel、完整轨迹和历史实验。\n\n当前约束95%≤P≤95.5%，六种方法全部达到。P按全部商品完整28期末库存>0的商品天计数。当前结论以本次报告为准，旧报告与单因素实验均标明历史。\n',encoding='utf-8')


if __name__=='__main__':write_report()
