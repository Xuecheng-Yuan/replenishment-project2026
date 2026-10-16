"""Build the two final result reports from audited official outputs, never diagnostics."""
from pathlib import Path
import hashlib,json,shutil,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
METHODS=['DRO_cluster','DRO_global','DRO_category','sS','SAA','RO']
NAMES=dict(zip(METHODS,['特征聚类DRO','不分组DRO','爆品/长尾DRO','传统(s,S)','SAA（零半径）','传统RO']))
H=lambda t:dict(kind='heading',text=t)
P=lambda t:dict(kind='paragraph',text=t)
def T(h,r):return dict(kind='table',headers=h,rows=[[str(x) for x in a] for a in r])
def I(p,c):return dict(kind='figure',path=p,caption=c)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pct(x):return f'{x:.4%}'
def markdown(blocks):
    lines=[]
    for b in blocks:
        if b['kind']=='heading':lines+=['## '+b['text'],'']
        elif b['kind']=='paragraph':lines += [b['text'],'']
        elif b['kind']=='table':
            lines+=['|'+'|'.join(b['headers'])+'|','|'+'|'.join(['---']*len(b['headers']))+'|']
            lines+=['|'+'|'.join(r)+'|' for r in b['rows']];lines+=['']
        else:lines += [f"![{b['caption']}]({b['path']})",'']
    return '\n'.join(lines).rstrip()+'\n'

def main():
    evidence=HERE/'evidence';reports=HERE/'reports';figs=reports/'figures'
    figs.mkdir(parents=True,exist_ok=True)
    pub=pd.read_csv(evidence/'published_summary.csv');seg=pd.read_csv(evidence/'published_segments.csv')
    audit=json.loads((evidence/'published_audit.json').read_text())
    assert audit['all_frozen_policy_replays_passed']
    pair_source=ROOT/'experiments/constraint_mechanism/outputs/paired_contrasts.csv'
    pair=pd.read_csv(pair_source).query("family=='published'")
    pair.to_csv(evidence/'official_paired_intervals.csv',index=False)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':9,'figure.dpi':150})
    colors=['#2563eb','#0891b2','#16a34a','#f97316','#9333ea','#64748b']
    sources=[]
    for version,folder,title in [('old','demand_fill_v3/with_ro','旧版_需求满足率_最终实验结果报告'),('current','service95_revision','新版_正库存比例_最终实验结果报告')]:
        old=version=='old';home=ROOT/'experiments'/folder/'outputs_v4'
        frozen=json.loads((home/'frozen_parameters.json').read_text(encoding='utf-8'))
        s=pub[pub.version.eq(version)].set_index('method').loc[METHODS]
        g=seg[seg.version.eq(version)&seg.category.ne('all')]
        period=pd.read_csv(home/'results_by_period.csv')
        forecast_path=ROOT/'experiments/demand_fill_v3/with_ro/outputs_v4/forecast_test_metrics.csv'
        forecast=pd.read_csv(forecast_path)
        for p in (home/'summary.csv',home/'results_by_period.csv',home/'frozen_parameters.json',forecast_path):
            sources.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
        period.to_csv(evidence/f'{version}_official_periods.csv',index=False)
        forecast.to_csv(evidence/f'{version}_forecast_metrics.csv',index=False)
        feasible=s.fill_rate.ge(.95) if old else s.positive_stock_ratio.between(.95,.955)
        assert feasible.all() and len(s)==6
        best=s.loc[feasible].sort_values('inventory_demand_ratio').iloc[0];bestid=best.name
        comparison=pair[pair.envelope.eq(version)].iloc[0]
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        axes[0].bar(range(6),s.inventory_demand_ratio,color=colors)
        for i,v in enumerate(s.inventory_demand_ratio):axes[0].text(i,v,f'{v:.4f}',ha='center',va='bottom',fontsize=8)
        axes[0].set_ylabel('库存需求比 J（越低越好）');axes[0].set_ylim(0,s.inventory_demand_ratio.max()*1.15)
        service=s.fill_rate if old else s.positive_stock_ratio
        axes[1].bar(range(6),service*100,color=colors)
        axes[1].axhline(95,color='#dc2626',ls='--');axes[1].set_ylabel(('需求满足率 F' if old else '正库存比例 P')+'（%）')
        axes[1].set_ylim(94.5,100 if old else 96)
        if not old:axes[1].axhline(95.5,color='#dc2626',ls='--')
        for ax in axes:ax.set_xticks(range(6),METHODS,rotation=30,ha='right');ax.grid(axis='y',alpha=.2)
        fig.savefig(figs/f'{version}_overall.png');plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
        for ax,col,label in zip(axes,['inventory_demand_ratio','fill_rate','positive_stock_ratio'],['J（越低越好）','F（%）','P（%）']):
            for j,(cat,c) in enumerate([('hot','#f97316'),('tail','#2563eb')]):
                z=g[g.category.eq(cat)].set_index('method').loc[METHODS]
                ax.bar(np.arange(6)+(j-.5)*.36,z[col]*(1 if col=='inventory_demand_ratio' else 100),width=.36,color=c,label='爆品' if cat=='hot' else '长尾品')
            ax.set_xticks(range(6),METHODS,rotation=40,ha='right');ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
        axes[0].legend();fig.savefig(figs/f'{version}_segments.png');plt.close(fig)
        fig,axes=plt.subplots(2,1,figsize=(10,5.8),layout='constrained')
        for m,c in zip(METHODS,colors):
            z=period[period.method.eq(m)]
            axes[0].plot(z.period,z.positive_stock_ratio*100,label=NAMES[m],color=c)
            axes[1].plot(z.period,z.inventory_demand_ratio,color=c)
        axes[0].set_ylabel('当期正库存比例（%）');axes[1].set_ylabel('当期库存需求比');axes[1].set_xlabel('测试期（1–28重新起算）')
        axes[0].legend(ncol=3,fontsize=8)
        for ax in axes:ax.grid(alpha=.2)
        fig.savefig(figs/f'{version}_periods.png');plt.close(fig)
        b=[H(title.replace('_','：',1)),P('定稿日期：2026年10月16日。对象：6000种商品、训练文件60期、测试文件28期。本文锁定正式交付的六方法结果；本次重新回放核验，不重新选择参数。'),
           H('1. 结论先行'),
           P(f'在本版本服务约束下，六种正式方法全部达标。测试集库存需求比最低的是{NAMES[bestid]}：J={best.inventory_demand_ratio:.6f}，需求满足率F={pct(best.fill_rate)}，正库存比例P={pct(best.positive_stock_ratio)}。这里的“最低”仅指已冻结的六种方案在本测试集上的数值比较，不表示求得所有可能策略的全局最优。'),
           P('旧版按销量/需求保证整体需求满足率不低于95%；库存需求比目标越小越好。最优的分类DRO相对(s,S)降低库存需求比约2.18%。' if old else '新版按所有商品的正库存商品天比例约束95%–95.5%，不是按销量/需求约束。最优的(s,S)为95.2810%，满足“达到95%且不要过高”的要求；分类DRO为95.2821%，J比(s,S)高约0.274%。'),
           P(f'焦点比较ΔJ=分类DRO−(s,S)={comparison.delta_J:+.6f}，商品分层配对重抽样95%区间为[{comparison.ci_low:+.6f}, {comparison.ci_high:+.6f}]。'+('该区间在0以下，支持本样本中分类DRO的库存优势。' if old else '该区间跨0，因此只能报告(s,S)的点估计更低，不能认定其稳定优于分类DRO。')),
           I(f'figures/{version}_overall.png','图1：六方法整体正式测试结果；服务线仅用于本版本的汇总约束。'),
           H('2. 数据、业务时序与评价口径'),
           T(['项目','本次冻结设置'],[['训练文件','replenishment_train.xlsx，60期×6000商品=360000条需求观测；每条记录有20个给定特征。'],['测试文件','replenishment_test.xlsx，28期×6000商品=168000个商品天；Period_1作为新业务段起点。'],['商品组成','爆品480种，长尾品5520种；分别占商品数8%和92%。'],['需求权重','爆品占测试需求79.9954%，长尾占20.0046%。标签采用数据给定分类。'],['库存时序','沿用输入期初库存，期初在途为0；本期需求结束后订货，下期需求之前到货。'],['滚动规则','每次预测未来4期、生成4期计划，只执行最近一期订货；提前期1期，无另加盘点周期。'],['缺货','允许缺货，缺口即丢失销售，不结转为欠单；库存不得为负。']]),
           P('记d_it为商品i第t期需求，A_it为需求前可用库存，I_it为需求后的期末库存，q_it为第t期需求结束后订货量，初始I_i0由文件给出，q_i0=0。每日决策一次是离散模型时间步，不额外叠加盘点等待期。'),
           P('A_it = I_i,t−1 + q_i,t−1；x_it = min(A_it,d_it)；I_it = max(A_it−d_it,0)；l_it = d_it−x_it。q_it为非负整数，只依赖训练信息和截至t期已经发生的测试需求。'),
           P('J = Σ_i,t I_it / Σ_i,t d_it（累计期末库存/累计需求，越低越好）；F = Σ_i,t x_it / Σ_i,t d_it（需求满足率）；P = Σ_i,t 1{I_it>0}/(6000×28)（正库存比例）。恰好售罄、期末I=0的商品天不计入P，即使没有丢失销量。无缺货比例C=Σ1{l_it=0}/168000另列，不能与P混用。'),
           P('本项目曾口头把J称为“周转率”，本文统一写“库存需求比”。常见销售量/平均库存定义的库存周转率通常越高越好，与这里越低越好的J不是同一指标。报告中的平均期末库存为ΣI/168000。'),
           H('3. 数学目标与六种决策方法'),
           P('政策选择形式：在声明的候选策略集合Π内，min_π J(π)，其中π满足上述库存平衡与非预知条件，并满足'+('F(π)≥0.95。' if old else '0.95≤P(π)≤0.955。')+'需求总量对同一段评估固定，因此比较J等价于比较累计期末库存。候选筛选在训练文件内部进行，测试只报告冻结方案的表现。'),
           P('实际求解不是一次求解6000商品×28期的联合随机整数规划，而是以预测和鲁棒风险水位构造可执行策略，再在训练回放中选择参数。没有输入仓容、采购预算或订货固定成本，不添加这些约束；四期后续计划按预测需求推进虚拟库存，下一日重新计算。'),
           P('设m_i,t,h为未来第h期预测，r=d/m为非负相对预测误差。经验样本分布记为P̂_g，g为商品所属组。DRO集合P_g={Q:W∞(Q,P̂_g)≤ε，Q支持于非负数}。其含义是每个历史相对误差样本可在半径ε内移动，而不假设只有一个确定需求分布。ε无需求单位。'),
           P('旧版风险水位z_g=min{z≥0:sup_Q∈P_g E_Q[(R−z)⁺−(1−β)R]≤0}。这是组内鲁棒期望缺货约束，β是内部目标；目标β不等于最终测试集F。每个样本r的最坏值取区间[max(0,r−ε),r+ε]两端中的较大值，再对样本求平均；用单调二分找最小可行z。实现使用该分段线性结构，不依赖大型LP。' if old else '新版基础水位z_g为W∞集合下的鲁棒β分位数：z_g=经验β分位数(R_g)+ε，对应sup_Q Q(R>z_g)≤1−β的基础机会约束。部署水位还取全60期和四个训练拟合切片水位的逐商品最大值。随后反馈乘数λ可以小于1，所以最终执行策略不继承基础β的逐商品机会保证；实际P靠训练校准并在测试如实验证。'),
           T(['方法','分组与风险处理'],[['特征聚类DRO','使用训练需求形态与给定特征分成6组，每组有独立误差分布及鲁棒水位。'],['不分组DRO','6000商品共用同一相对误差池及鲁棒水位。'],['爆品/长尾DRO','使用文件中爆品/长尾标签分2组，每组独立鲁棒水位。'],['SAA','特征组经验分布，鲁棒半径ε=0，是同构的非鲁棒参照。'],['传统(s,S)','库存低于s才订至S；两水位由商品自身与特征组误差分位数加权估计。'],['传统RO','相对需求不确定集合U=[max(0,μ−Γσ),μ+Γσ]，只优化最坏需求，不优化分布。'+('旧版z=β(μ+Γσ)。' if old else '新版基础z=μ+Γσ，再经包络和共同反馈。')]]),
           P('旧版执行：非(s,S)方法S_i,t,h=m_i,t,h z_i，q=ceil(max(S−I,0))；(s,S)在I<s时补至S。新版执行：S_i,t,h=floor(λ_t m_i,t,h z_i)+1；(s,S)的触发点也按λ缩放并加1，其余方法补至S。+1用于严格期末正库存计数。四期计划使用逐期预测更新虚拟库存，只实施h=1。两版的水位和反馈实现不同，不能将历史排序变化全部归因于单独更换服务公式。'),
           H('4. 特征聚类依据与60期训练如何使用'),
           P('聚类的5个需求形态变量来自拟合段：log(1+均值)、log(1+变异系数)、零需求比例、log(1+95%分位数/均值)、log(1+末8期均值/前8期均值)。这5列标准化。20个给定特征先在拟合时间段内取均值、标准化，再取PCA前三主成分，主成分标准化后权重为0.5，与需求形态拼接。K-means固定6组、随机种子20260929、8次初始化、最多60次迭代，保留组内平方和最小解。'),
           P('上述分组依据是需求规模、波动、间歇性、尾部和变化趋势，以及给定特征所反映的相似性；不是把测试效果最好的商品事后分在一起，也不是凭商品名称分组。因20个特征的业务含义未另给出，本文只解释数值处理，不虚构字段含义。爆品/长尾标签与6组特征聚类是两套分类方案。'),
           P('两版最终模型均重新拟合训练文件第1–60期，全部360000条需求用于最终需求拟合和训练特征汇总。验证时必须留出观测来评价候选，不能把同一批已用于拟合的60期拟合优度当作独立验证成绩。先在训练内部选参数，再全60期重拟合，是完整使用训练数据；不是把前28期弃用。'),
           P('共同冻结预测参数blend=1、window=3、prior=0：商品线性趋势加最近3个已观测周期的误差校正，预测截断下限为1。每期观测结束后更新，只用过去和当前需求预测未来4期。相对误差库采用训练内部扩展窗口，前18期为热身，此后每6期扩展一次，形成第19–60期共42期的误差；前18期用于拟合热身，并没有从最终模型丢弃。误差生成在相应外层拟合的分组内进行，并非每次重新嵌套聚类。'),
           P('旧版选参：训练1–36期拟合、37–60期向前验证；另用29–60期拟合、1–28期作为重启业务段回放。后者是训练内不相交的回溯验证，不是时间上向前的预测检验。筛选reset F≥95%及可控段forward F≥95.8057628%的候选，按reset J、forward J排序。RO亦只用训练文件校准不确定集合宽度。' if old else '新版选参：基础设计来自此前训练内候选冻结设置；反馈参数用训练29–60期拟合，1–28期作为28期重启回放。每方法窗口{1,3,7}×先验{0,3}×偏移{−0.01,0,+0.01}共18个候选，六方法共108个。筛选训练P∈[95%,95.5%]后最小J。另有1–24→25–36、1–36→37–48、1–48→49–60的12期向前/重启前缀诊断；这些前缀不是28期服务约束的替代验证。'),
           H('5. 最终冻结参数与测试协议')]
        if old:
            rows=[]
            for m in METHODS:
                r=frozen['selected'][m]
                params=f"β={r['beta']:.9f}, ε={r['radius']}" if m.startswith('DRO') or m=='SAA' else f"p_s={r['p_s']}, p_S={r['p_S']}" if m=='sS' else f"Γ={r['gamma']}, β={r['beta']:.9f}"
                rows.append([NAMES[m],params,pct(r['reset_fill']),f"{r['reset_inventory_ratio']:.6f}"])
            b += [T(['方法','冻结参数','训练reset F','训练reset J'],rows),P('RO的全训练误差μ=1.008739591，σ=0.168145381，相对需求区间为[0.824115962,1.193363220]。此区间只定义训练拟合的不确定集合，并非声称测试需求必然全部落在其中。')]
        else:
            rows=[]
            for m in METHODS:
                r=frozen['selected'][m]
                p=f"β={r['beta']}, ε={r['radius']}" if 'beta' in r else f"p_s={r['p_s']}, p_S={r['p_S']}" if m=='sS' else f"Γ={r['gamma']}"
                rows.append([NAMES[m],p,str(r['candidate']),pct(r['train_service']),f"{r['train_inventory_ratio']:.6f}"])
            b += [T(['方法','基础风险参数','候选ID','训练P','训练J'],rows),P('六方法所选反馈参数相同：response_window=3、response_prior=3、response_bias=+0.01；内部累计目标为95.25%。基础水位取全60期与训练切片(1–24、1–36、1–48、29–60)包络；不再添加此前99%高服务试验的静态储备。'),
                  P('反馈解释：截至t期累计正库存商品天C_t；剩余期数28−t。下一期请求比例b_t=clip((0.9525×168000−C_t)/(6000×max(28−t,1))+0.01,0,1)。按训练误差CDF和最近已实现的一步误差CDF估算下一期正库存比例，寻找最小λ≥0使估计值达到b_t。CDF使用128个中点分位数近似，先验权重为a/(a+n)，a=3。首期需求没有对应的测试前一期预测，不作为一步误差；后续只加入已经发生的误差。'),
                  P('λ搜索从上界1倍增至最多128，随后二分16次。该过程是基于经验CDF的因果反馈，并非对未知测试分布提供硬性概率保证。末期订货仍记录，但28期预算是固定评估长度；迁移到其他长度必须修改预算并重新验证。')]
        b += [P('最终60期模型、分组和风险水位确定后，才对测试文件滚动运行。测试第1期需求不由测试前订货控制，六方法首期相同：丢失763368件，5357/6000个商品期末库存为正（89.2833%）。所有整体指标均包含这一期，不剔除不利样本。'),
              P('本项目历经多次依据用户新要求修改，测试文件此前已被反复查看。因此这是修订后同一测试集上的正式结果，不能称为全新未见过的盲测；也没有用本次测试结果重新搜索并替换冻结参数。'),
              H('6. 六方法正式实验结果'),
              T(['方法','F：销量/需求','P：正库存比例','J：库存/需求','无缺货比例C'],[[NAMES[m],pct(r.fill_rate),pct(r.positive_stock_ratio),f'{r.inventory_demand_ratio:.6f}',pct(r.cycle_service)] for m,r in s.iterrows()]),
              T(['方法','正库存商品天','累计期末库存','丢失销量','平均期末库存/商品天'],[[NAMES[m],f'{int(r.positive_stock_sku_periods):,}/168000',f'{int(r.ending_inventory_units):,}',f'{int(r.lost_units):,}',f'{r.avg_ending_inventory_per_sku:.4f}'] for m,r in s.iterrows()]),
              P(f'最低J方案的可手算核验：J={int(best.ending_inventory_units):,}/{int(best.demand_units):,}={best.inventory_demand_ratio:.9f}；F={int(best.sales_units):,}/{int(best.demand_units):,}={pct(best.fill_rate)}；P={int(best.positive_stock_sku_periods):,}/168000={pct(best.positive_stock_ratio)}。六方法面对同一总需求90465437件，J的分母完全一致。'),
              T(['方法','第28期期末库存','第28期订货/期末在途'],[[NAMES[m],f'{int(r.final_stock):,}',f'{int(r.final_pipeline):,}'] for m,r in s.iterrows()]),
              P('第28期所下订单在第29期到货，故不增加前28期库存指标。报告仍单列期末在途，避免把末期订货当作已消耗或已到库。旧版无反馈而持续订货、与新版固定预算反馈末期订货很小，是两个已部署策略的差别，不能仅凭28期J判断更长周期的成本优劣。'),
              H('7. 爆品、长尾品与全部商品'),
              T(['方法','类别','F','P','J'],[[NAMES[r.method],'爆品' if r.category=='hot' else '长尾品',pct(r.fill_rate),pct(r.positive_stock_ratio),f'{r.inventory_demand_ratio:.6f}'] for r in g.itertuples()]),
              I(f'figures/{version}_segments.png','图2：六方法均展示爆品和长尾品，不只展示爆品或最优方案。'),
              P('整体F按需求量加权，整体P按商品天等权：480种爆品贡献约80%的需求，却仅占8%的P权重。'+('因此整体F达95%并不要求整体P达95%，旧版P约74%–81%并不违反本版约束。' if old else '因此整体P达到95%并不意味着每个类别或每种商品都达到95%。例如(s,S)的爆品P低于95%，长尾P高于95%，汇总仍达到95.2810%；本版没有设置逐类别95%的额外约束。')),
              H('8. 滚动表现与四期预测质量'),
              P('下图展示六种方案的逐期变化，随后分别列出四个预测步长的误差；按完整28期汇总评价，不用某一天的好坏替代整体表现。'),
              I(f'figures/{version}_periods.png','图3：28期逐期P与J。整体服务目标约束整个评估段，不要求每一天都落在95%–95.5%。'),
              T(['预测步长h','可核对单元数','WAPE','RMSE（件）'],[[int(r.horizon),int(r.n_cells),pct(r.wape),f'{r.rmse:.4f}'] for r in forecast.itertuples()]),
              P('预测误差仅对测试段内能观察到目标期的预测计算，h=1至4分别有27、26、25、24个预测起点×6000商品。最后一期预测未来4期仍可形成计划，但超出测试段的需求未知，不计入误差；测试首期需求也没有对应本段之前的一步预测。WAPE=Σ|预测−需求|/Σ需求，RMSE=√均方误差。'),
              P('六方法使用同一套四期需求预测，差异主要来自风险分组、水位和决策。核验全部新旧正式轨迹，最大预测差异仅3.64×10⁻¹²，来自旧版RO浮点舍入；焦点分类DRO和(s,S)新旧逐值相同。预测步长越长误差越大，说明保留四期计划同时只执行第一步具有实际意义，但并不构成全局最优证明。'),
              H('9. 证据核验、比较边界与复现'),
              P('本次重新读取12份正式压缩轨迹（新旧各6份），共2016000条商品天记录。逐条核验需求、到货、销量、期末库存、订货和4期计划，并用冻结政策回放逐值比对；摘要表和分组表由轨迹重新计算。证据位于evidence/published_audit.json、published_summary.csv和published_segments.csv，记录轨迹及参数SHA256。'),
              P('配对区间来自已完成机制实验中的正式历史对比：按爆品480、长尾5520分层，抽样单位为商品完整28期路径，共1000次，随机种子20261016。该区间以训练模型和策略固定为条件，不包含重新训练或时间分布变化的不确定性。保留其来源文件哈希；本报告没有把诊断候选最优结果替换成正式结果。'),
              P('两版服务约束不同，且旧版使用鲁棒期望缺货水位，新版使用基础分位数、包络和共同反馈。因此旧版J较低并不能直接证明策略更好；它只保证另一种服务要求。原因探索及受控消融另见experiments/constraint_mechanism/reports/机制探索报告.pdf与完整实验报告.pdf，定稿结果与解释实验分开保存。'),
              T(['用途','项目根目录中的位置或命令'],[['本版正式Excel',f'experiments/{folder}/outputs_v4/replenishment_comparison.xlsx'],['本版逐期完整明细',f'experiments/{folder}/outputs_v4/detail_<method>.csv.gz'],['本版冻结参数',f'experiments/{folder}/outputs_v4/frozen_parameters.json'],['重新核验两版正式政策','python experiments/final_results_reports/audit_official.py'],['重建两份报告文字及图','python experiments/final_results_reports/build_reports.py'],['PDF渲染','python experiments/constraint_mechanism/render_reports.py --reports experiments/final_results_reports/reports'],['本版模型实现','experiments/demand_fill_v3/model_v2.py、run_v3.py；RO见experiments/ro_extension/extend_with_ro.py' if old else 'experiments/ro_extension/service_band_revision.py；验证器verify_service95.py']]),
              P('本地文件包根目录为C:/26OR资料/大作业/李佳蔚；报告独立放在“新旧两版定稿实验结果报告”，完整代码、数据和正式Excel在“项目”目录。复现需Python及项目requirements_v3.txt依赖；生成图片还需matplotlib，渲染PDF需reportlab、pypdf及中文字体。无需Git元数据即可生成报告。报告重建读取冻结正式结果，不重新调参；重新训练请另建输出目录以保留定稿证据。'),
              P('来源：训练和测试原始文件SHA256见evidence/source_manifest.json；输入全面流式核验记录见experiments/constraint_mechanism/outputs/input_audit.json。方法参考RSOME官方多阶段DRO库存示例：https://xiongpengnus.github.io/rsome/example_dro_inv 。本项目使用专用水位计算和滚动政策回放，没有将RSOME示例求解器结果冒充为本项目全局最优解。')]
        # SimHei lacks Unicode superscript minus/digits and combining accents.
        # Use explicit ASCII math operators to keep all model formulas readable.
        def printable(value):
            if isinstance(value,str):
                return value.replace('(R−z)⁺','max(R-z,0)').replace('3.64×10⁻¹²','3.64e-12').replace('P̂','P_emp').replace('−','-')
            if isinstance(value,list):return [printable(v) for v in value]
            if isinstance(value,dict):return {k:printable(v) for k,v in value.items()}
            return value
        b=printable(b)
        content=dict(title=title,version=version,official_results_only=True,blocks=b)
        (reports/f'{title}.json').write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (reports/f'{title}.md').write_text(markdown(b),encoding='utf-8')
    sources.append(dict(path=pair_source.relative_to(ROOT).as_posix(),sha256=sha(pair_source)))
    raw=ROOT/'experiments/constraint_mechanism/outputs/input_audit.json'
    (evidence/'source_manifest.json').write_text(json.dumps(dict(official_sources=sources,input_audit=json.loads(raw.read_text()),paired_interval_source=pair_source.relative_to(ROOT).as_posix()),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Built two final reports and six official-result figures.')

if __name__=='__main__':main()
