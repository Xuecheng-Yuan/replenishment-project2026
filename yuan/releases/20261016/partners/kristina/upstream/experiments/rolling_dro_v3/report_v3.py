"""Pyramid-structured Chinese report matching run_v3.py and its actual outputs."""
from pathlib import Path
import pandas as pd

LABELS={'DRO_cluster':'特征聚类DRO','DRO_global':'不分组DRO','DRO_category':'爆品/长尾DRO',
        'sS':'传统(s,S)','SAA':'零半径经验模型'}
ORDER=list(LABELS)


def stock_presence(out:Path):
    """Count positive period-ending stock, including exact-demand stockouts."""
    totals=[]; by_sku=[]
    for method in ORDER:
        path=out/f'detail_{method}.csv'
        frame=pd.read_csv(path if path.exists() else path.with_suffix('.csv.gz'),
                          usecols=['period','item_id','ending'])
        frame['positive']=frame.ending>0
        for scope,part in [('all_28_periods',frame),
                           ('controllable_periods_2_to_28',frame[frame.period>=2])]:
            totals.append(dict(method=method,scope=scope,
                positive_stock_sku_periods=int(part.positive.sum()),
                total_sku_periods=len(part),positive_stock_ratio=float(part.positive.mean())))
        grouped=frame.groupby('item_id').positive.agg(['sum','count']).reset_index()
        grouped.columns=['item_id','positive_stock_periods','total_periods']
        grouped.insert(1,'method',method)
        grouped['positive_stock_ratio']=grouped.positive_stock_periods/grouped.total_periods
        by_sku.append(grouped)
    overall=pd.DataFrame(totals)
    overall.to_csv(out/'positive_stock_service.csv',index=False)
    pd.concat(by_sku,ignore_index=True).to_csv(out/'positive_stock_by_sku.csv',index=False)
    return overall


def build_report(out:Path,runs,settings,forecast_scores,seconds):
    presence=stock_presence(out)
    summary=pd.DataFrame([dict(method=m,**runs[m]['summary']) for m in ORDER]).set_index('method')
    feasible=summary[summary.positive_stock_ratio.between(.95,.955)]
    winner=feasible.inventory_demand_ratio.idxmin() if len(feasible) else None
    baseline=summary.loc['sS']
    lines=['# 多周期补货实验报告：正库存服务约束修订版','','## 一、结论先行','']
    if winner:
        r=summary.loc[winner]
        lines += [f'完整测试集6000个商品×28天上，{LABELS[winner]}的正库存服务水平为**{r.positive_stock_ratio:.2%}**，落在95%–95.5%范围内，库存需求比为**{r.inventory_demand_ratio:.6f}**，在达标策略中最低。']
    else:
        closest=(summary.positive_stock_ratio-.95).abs().idxmin()
        lines += [f'完整测试集上，**没有策略落在95%–95.5%的正库存服务范围内**。距离95%最近的是{LABELS[closest]}，实际为{summary.loc[closest,"positive_stock_ratio"]:.2%}。参数在训练集内部确定后冻结，未使用测试结果再次选参。']
    lines += ['', '本版目标是最小化库存需求比；服务约束已从销量/需求改为期末库存大于0的商品－天占比。95%–95.5%是用户确认的服务范围。销量/需求仅作为补充指标，不能用它判定本版是否达标。', '',
      '## 二、五种策略完整重跑结果','',
      '| 方法 | 正库存服务水平 | 95%–95.5%判定 | 库存需求比（越低越好） | 相对(s,S)变化 | 需求满足率（补充） | 平均库存/商品 |',
      '|---|---:|:---:|---:|---:|---:|---:|']
    for m in ORDER:
        r=summary.loc[m]; status='范围内' if .95<=r.positive_stock_ratio<=.955 else ('低于95%' if r.positive_stock_ratio<.95 else '高于95.5%')
        lines.append(f'| {LABELS[m]} | {r.positive_stock_ratio:.2%} | {status} | {r.inventory_demand_ratio:.6f} | {r.inventory_demand_ratio/baseline.inventory_demand_ratio-1:+.2%} | {r.fill_rate:.2%} | {r.avg_inventory_per_sku:.2f} |')
    lines += ['', '### 服务水平的计数依据','',
      '| 方法 | 正库存商品－天数 | 全部商品－天数 | 完整28天 | 第2–28天 |','|---|---:|---:|---:|---:|']
    for m in ORDER:
        g=presence[presence.method==m].set_index('scope'); a=g.loc['all_28_periods']; b=g.loc['controllable_periods_2_to_28']
        lines.append(f'| {LABELS[m]} | {int(a.positive_stock_sku_periods):,} | {int(a.total_sku_periods):,} | {a.positive_stock_ratio:.2%} | {b.positive_stock_ratio:.2%} |')
    lines += ['', '分母为6000×28＝168,000。每个商品都评价28天，所以总体比例也等于6000个商品各自比例的平均值。严格使用期末库存>0；期末恰好为0，即使当天需求全部满足，也记为0。原始数据标为Period，本版按题图每周期一天解释，不能从期末记录推断一天内库存是否始终为正。全部商品各策略计数见`positive_stock_by_sku.csv`，总体计数见`positive_stock_service.csv`。', '',
      '### 训练选择与测试判定','',
      '| 方法 | β或p_s | ε或p_S | 额外倍率κ | 重启验证正库存服务 | 前向验证正库存服务（诊断） | 重启验证库存需求比 |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for m in ORDER:
        r=settings['selected'][m]
        lines.append(f'| {LABELS[m]} | {r.get("beta",r.get("p_s")):.4f} | {r.get("radius",r.get("p_S")):.4f} | {r.get("scale",1.):.3f} | {r["reset_service"]:.2%} | {r["forward_service"]:.2%} | {r["reset_inventory_ratio"]:.6f} |')
    lines += ['', '预测参数在训练集内部两个折的平均WAPE最小处确定。前向折拟合第1–36天、评价第37–60天；重启折拟合第29–60天、评价第1–28天。由于测试Period_1重新起算，决策参数以重启验证的95%–95.5%正库存比例为可行条件，在可行候选中最小化库存需求比。前向折作为时间外推诊断单独报告，不再强加不相容的第二个服务范围。', '',
      'SAA与传统(s,S)加入训练选择的共同安全倍率κ≥1，避免经验尾部样本过短使其没有可行候选。SAA的模糊集半径仍为0；κ是外层政策校准，不能解释为新的分布鲁棒半径。三种DRO不使用κ。各方法均只用训练集选参，候选记录在`policy_validation.csv`。', '',
      '## 三、数学模型与解读','',r'''### 1. 库存、订货与丢失销售

设$I_{it}$为商品$i$在第$t$天需求后的期末库存，$q_{it}$为当天需求实现后下单量，提前期$L=1$。期初可用库存为$A_{it}=I_{i,t-1}+q_{i,t-1}$，销量$Y_{it}=\min(A_{it},D_{it})$，损失$L_{it}=(D_{it}-A_{it})_+$，期末库存$I_{it}=(A_{it}-D_{it})_+$。所有订单为非负整数；未满足需求丢失，不结转。第1天只有给定初始库存、初始在途为0；第28天下单在第29天到货，单列终期在途。

### 2. 优化目标及服务约束

$$J_H=\frac{\sum_{t=1}^{H}\sum_i I_{it}}{\sum_{t=1}^{H}\sum_i D_{it}},\qquad S_H=\frac1{NH}\sum_{t=1}^{H}\sum_i\mathbf1(I_{it}>0).$$

$$\min J_H\qquad\mathrm{s.t.}\quad0.95\le S_H\le0.955.$$

分子累计每一天的期末库存，分母包括丢失需求。该指标为库存需求比，越低越好。服务比例以商品－天为等权统计单元，不能用销量/需求或无缺货天数替换。上述是完整路径的业务目标；未来需求未知时，采用下面的分布鲁棒机会约束生成水位，再通过训练验证校准实际路径服务范围。训练验证不能保证未见测试路径一定落在该范围。

### 3. 特征聚类、预测与模糊集

使用5项训练需求形态：对数平均需求、变异系数、零需求比例、上尾/均值、前后需求趋势。20项原始特征取训练均值，标准化后压缩为3个主成分，权重0.5，与需求形态共同输入K-means，K=6、随机种子固定。K和权重为预设。另两种DRO分别共用1组，或按原始爆品/长尾标签分为2组。三种方法只改变残差分组，各自独立选参。

线性趋势与滚动近期误差修正生成未来4天预测$m_{i,t+h|t}$。选中的点预测采用商品自身趋势，特征聚类用于共享相对预测误差。未来特征是否可提前获得没有给定，因此不使用未来特征或未来真实需求。训练期扩展窗口的一步样本外预测生成$r_j=D_j/\hat D_j$，按组共享经验分布。样本邻域为$[\max(0,r_j-\varepsilon),r_j+\varepsilon]$；所有样本等权且保留尾部，构成一维$W_\infty$模糊集。

### 4. 正库存的分布鲁棒机会约束

对每组选择整数目标水位$B_{i,t+h}$，要求

$$\inf_{Q\in\mathcal P_g(\varepsilon)}Q(m_{i,t+h}R<B_{i,t+h})\ge\beta_g.$$

需求小于水位才会留下正库存，因此必须使用严格不等号。令$u_j=r_j+\varepsilon$，最坏分布将每个样本放到上端点。排序$u_{(1)}\le\cdots\le u_{(n)}$，令$k=\lceil\beta n\rceil$，可取

$$z_g=u_{(k)},\qquad B_{i,t+h}=\lfloor m_{i,t+h}z_g\rfloor+1.$$

该整数水位严格超过至少k个上端点，最坏正库存概率至少为k/n≥β。固定β和ε后，库存目标随水位单调不减，取这一最小整数水位即可。该解针对本版正库存机会约束，不使用上一版控制缺货量比例的stop-loss LP。β为内部条件概率下界，完整评价包括不可控制的第1天，因此β与完整28天的实际服务比例不必相等；上界95.5%由训练路径选参控制，不声称对所有未来分布都保证上界。

### 5. 四天滚动执行与传统(s,S)

每天需求实现后更新$I_{it}$并预测未来4天，订单$q_{it}=(B_{i,t+1}-I_{it})_+$。代码沿四天预测均值路径生成4笔计划，只执行第1笔；下一天需求实现后重新预测和优化。没有共享产能、预算、固定订货费等跨期耦合，最低水位可按天分解，远期计划不等于提前执行未来订单。

传统(s,S)在同一预测上用商品与聚类经验残差的收缩分位形成两档水位：$s=\lfloor\kappa m Q_i(p_s)\rfloor+1$、$S=\lfloor\kappa m Q_i(p_S)\rfloor+1$。期末库存低于s时订至S，否则不下单。SAA使用同样的正库存分位逻辑但ε=0，并以κ校准有限样本的水位。它们都在相同训练验证服务范围内选参。
''', '## 四、验证与本地复现','',
      '五策略均完整重跑，输出每策略168,000行库存、销量、丢失需求、四天预测和计划订单。核验库存守恒、提前期1、丢失销售、订单非负整数、第一笔实际执行、正库存比例与库存需求比。期末库存恰好为0的边界另有单元测试。', '',
      '```powershell\npython -m pip install -r experiments/rolling_dro_v3/requirements_v3.txt\npython experiments/rolling_dro_v3/run_v3.py\npython experiments/rolling_dro_v3/build_visualizations.py\npython -m unittest discover -s experiments/rolling_dro_v3/tests -p test_v2.py -v\n```','',
      '默认读取仓库data/raw两份Excel，也可用--data-dir指定目录。Excel由build_workbook_v3.mjs刷新；模型和图表可由普通Python环境复现，无需商业求解器。完整gzip轨迹、SHA256清单、冻结参数、逐期与商品级结果均在outputs_v4。', '',
      '测试集曾用于早期版本偏差诊断，本次属于修订后的重复评估。按新服务定义重跑，未用测试服务水平重新选择参数。总体达标不代表每个商品或每天都达标。', '',
      '方法参考：[RSOME多阶段库存示例](https://xiongpengnus.github.io/rsome/example_dro_inv)。该示例采用零提前期及可负库存成本模型；本项目按提前期1和丢失销售改写，并采用样本邻域的正库存机会约束，未直接套用示例的目标函数或成本。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
