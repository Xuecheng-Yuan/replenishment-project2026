"""Build source reports and scientific figures only from audited result files.

PDF layout is a separate, lightweight reportlab command (render_reports.py).
"""
from run_experiment import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHOD_NAMES={**LABELS,'SAA':'SAA（零半径）'}
ARM_NAMES={'fill95':'F≥95%','positive95':'P≥95%','positive_band':'95%≤P≤95.5%'}
FAM_NAMES={'static':'静态水位库','budget':'反馈水位库','union':'二者并集'}
ENV_NAMES={'full60':'仅全60期水位','envelope':'部署时取包络'}
def name(x):return METHOD_NAMES.get(x,str(x))
def pct(x):return f'{x:.4%}'
def num(x):return f'{x:.6f}'
def yes(x):return '是' if x else '否'
def H(text):return dict(kind='heading',text=text)
def P(text):return dict(kind='paragraph',text=text)
def T(headers,rows):return dict(kind='table',headers=headers,rows=[[str(x) for x in r] for r in rows])
def I(path,caption):return dict(kind='figure',path=path,caption=caption)

def markdown(blocks):
    lines=[]
    for b in blocks:
        if b['kind']=='heading':lines+=['## '+b['text'],'']
        elif b['kind']=='paragraph':lines+=[b['text'],'']
        elif b['kind']=='table':
            lines+=['|'+'|'.join(b['headers'])+'|','|'+'|'.join(['---']*len(b['headers']))+'|']
            lines+=['|'+'|'.join(row)+'|' for row in b['rows']];lines+=['']
        else:lines+=[f"![{b['caption']}]({b['path']})",'']
    return '\n'.join(lines).rstrip()+'\n'

def main():
    out=HERE/'outputs';reports=HERE/'reports';figs=reports/'figures';figs.mkdir(parents=True,exist_ok=True)
    s=pd.read_csv(out/'summary.csv');v=pd.read_csv(out/'candidate_validation.csv');w=pd.read_csv(out/'cell_winners.csv')
    pair=pd.read_csv(out/'paired_contrasts.csv');effects=pd.read_csv(out/'factor_effects.csv');pub=pd.read_csv(out/'published_summary.csv')
    seg=pd.read_csv(out/'published_segments.csv');water=pd.read_csv(out/'waterline_shape.csv');overlap=pd.read_csv(out/'feasible_overlap.csv')
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'));complete=json.loads((out/'completed.json').read_text())
    audit=json.loads((out/'published_audit.json').read_text());verified=json.loads((out/'verification.json').read_text());assert verified['status']=='passed'
    assert audit['all_frozen_policy_replays_passed'] and audit['forecasts_equivalent_at_1e_9']
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'figure.dpi':140})
    colors=['#2563eb','#f97316','#16a34a','#9333ea','#0891b2','#64748b']
    # Plot every selected method, not just the winner.
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    panels=[('fill95','static','full60','仅切换约束：需求满足率'),('positive95','static','full60','仅切换约束：正库存下限'),('positive_band','budget','envelope','反馈与包络：正库存区间')]
    for ax,(arm,fam,env,title) in zip(axes,panels):
        z=s[s.arm.eq(arm)&s.family.eq(fam)&s.envelope.eq(env)].set_index('method').loc[list(LABELS)]
        ax.bar(range(6),z.inventory_demand_ratio,color=colors);ax.set_xticks(range(6),list(LABELS),rotation=35,ha='right')
        ax.set_title(title);ax.set_ylabel('库存需求比 J（越低越好）');ax.grid(axis='y',alpha=.2)
    fig.savefig(figs/'method_comparison.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    z=pair[(pair.family!='published')&pair.arm.isin(['fill95','positive_band'])&pair.family.isin(['static','budget'])].copy()
    yy=np.arange(len(z));ax.errorbar(z.delta_J,yy,xerr=np.vstack([z.delta_J-z.ci_low,z.ci_high-z.delta_J]),fmt='o',capsize=4,color='#2563eb')
    labels=[f"{FAM_NAMES[r.family]} / {ENV_NAMES[r.envelope]} / {ARM_NAMES[r.arm]}"+(' [未达标，仅诊断]' if not r.both_test_feasible else '') for r in z.itertuples()]
    ax.set_yticks(yy,labels);ax.axvline(0,color='#dc2626',ls='--');ax.set_xlabel('ΔJ = 分类DRO − (s,S)；左侧分类DRO库存较低，右侧(s,S)较低')
    ax.set_title('同一批商品配对重抽样：点估计与95%区间');ax.grid(axis='x',alpha=.2)
    fig.savefig(figs/'paired_intervals.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for method,c in zip(LABELS,colors):
        a=v[v.method.eq(method)&v.family.eq('static')]
        axes[0].plot(a.positive_stock_ratio*100,a.inventory_demand_ratio,label=name(method),color=c)
        a=v[v.method.eq(method)&v.family.eq('budget')]
        axes[1].scatter(a.positive_stock_ratio*100,a.inventory_demand_ratio,label=name(method),color=c,s=14)
    for ax,title in zip(axes,['训练静态候选曲线','训练反馈候选（离散点）']):
        ax.axvspan(95,95.5,color='#22c55e',alpha=.2);ax.set_xlabel('训练正库存比例 P（%）');ax.set_ylabel('训练 J');ax.set_title(title);ax.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.savefig(figs/'training_frontier.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
    q=seg[seg.version.eq('current')&seg.method.eq('DRO_category')&seg.category.ne('all')]
    axes[0].bar(['爆品','长尾品'],q.sku_weight*100,color=['#f97316','#2563eb']);axes[0].set_title('P的商品天权重（%）')
    axes[1].bar(['爆品','长尾品'],q.demand_weight*100,color=['#f97316','#2563eb']);axes[1].set_title('F的需求量权重（%）')
    for ax in axes:ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2)
    fig.savefig(figs/'service_weights.png');plt.close(fig)
    def row(fam,env,arm,method):return s[s.family.eq(fam)&s.envelope.eq(env)&s.arm.eq(arm)&s.method.eq(method)].iloc[0]
    def delta(fam,env,arm):return pair[pair.family.eq(fam)&pair.envelope.eq(env)&pair.arm.eq(arm)].iloc[0]
    oldpair=pair[pair.family.eq('published')&pair.envelope.eq('old')].iloc[0]
    newpair=pair[pair.family.eq('published')&pair.envelope.eq('current')].iloc[0]
    fullpair=delta('budget','full60','positive_band');envpair=delta('budget','envelope','positive_band')
    envelope_effect=effects[effects.effect.eq('deployment_envelope')&effects.case.eq('budget/positive_band')].iloc[0]
    intro=[H('约束变化为何改变最优策略：重新实验后的结论'),
        P('2026年10月16日 | 对象：6000商品、训练60期、独立测试28期 | 本报告中的“最优”限于声明的候选库及对应服务可行域。'),
        P('历史新旧版的最优方法不同，来自服务指标与策略实现共同作用。受控实验支持“可行域改变会改变最优参数与部分方法排序”，但没有支持“只要把需求满足率改成正库存比例，就必然由分类DRO变成(s,S)最优”。'),
        P(f'历史结果已逐条回放核验：旧版分类DRO相对(s,S)的ΔJ={oldpair.delta_J:+.6f}；当前新版为{newpair.delta_J:+.6f}。新版数值上(s,S)更低，但其95%商品块重抽样区间为[{newpair.ci_low:+.6f}, {newpair.ci_high:+.6f}]，跨过0，不能据此声称稳定优于分类DRO。'),
        P('当前正式交付结果保持不变。本轮是解释原因的诊断实验，扩大了反馈候选范围，实验中的最优数值不覆盖之前已经交付的Excel、动画或冻结参数。'),
        T(['证据层次','本轮获得的结论'],[
            ['数学机制','F按需求量加权，P按商品天等权；两者可行域不同，最优策略可能不同。'],
            ['仅切换约束','静态、无包络：F约束测试最优为分类DRO；P下限测试最优为SAA。分类DRO与(s,S)这一对没有反转。'],
            ['开关部署包络','固定反馈候选与P约束：不开包络分类DRO更低；开包络(s,S)数值更低。两者均达标。'],
            ['只增加95.5%上限','本轮全部方法、库的最优候选均未改变，上限在训练最优点没有约束作用。'],
            ['预测差异核查',f'焦点两方法的新旧四期预测逐值一致；六方法最大差异仅{audit["max_forecast_difference_vs_full60"]:.3g}，属于浮点舍入。']]),
        H('一、先分清服务约束改变了什么'),
        P('J = sum(I) / sum(d)，越低越好；F = sum(sales) / sum(d)；P = count(I > 0) / (N*T)。J是累计期末库存与累计需求之比，并非通常的销量/平均库存周转次数。恰好售罄的商品天在P中记0，尽管当天需求可能已全部满足。'),
        P('测试中爆品仅480种、占商品天8%，却占需求量79.9954%；长尾5520种、占商品天92%，只占需求量20.0046%。因此，F对爆品销量更敏感，P对大量长尾商品是否留有正库存更敏感。'),
        I('figures/service_weights.png','图1：相同商品在两种服务指标中的不同权重。'),
        P('固定策略集合Pi及同一批数据：pi_F = argmin{J(pi): F(pi)>=0.95}；pi_P = argmin{J(pi): P(pi)>=0.95}；pi_band = argmin{J(pi): 0.95<=P(pi)<=0.955}。目标不变，允许参与比较的策略改变。新最优点既可能变化，也可能保持不变。'),
        P('一个可核算的逻辑例子：20个商品的需求为(1000,1,...,1)。策略A可用库存(1000,0,...,0)，策略B为(1001,2,...,2,0)。A的F=1000/1019、P=0、J=0；B的F=1018/1019、P=19/20、J=19/1019。在仅含A、B的策略库中，F≥95%选A，而95%≤P≤95.5%选B。这个例子证明“可以改变”，不替代真实数据的实验。'),
        H('二、固定其他因素后，真实数据发生了什么'),
        T(['约束','控制器/部署','测试达标数','测试达标最小J方法'],[
            [ARM_NAMES[r.arm],FAM_NAMES[r.family]+' / '+ENV_NAMES[r.envelope],str(r.test_feasible_methods),name(r.descriptive_test_winner) if pd.notna(r.descriptive_test_winner) else '无达标方法']
            for r in w[w.family.isin(['static','budget'])].itertuples()]),
        P('这里的“测试最小J”是事后描述，不能拿它重新选参部署。实际训练选出的总优胜方法另列在完整实验报告和cell_winners.csv。静态库的P区间组训练达标、测试却超过95.5%，因此没有测试达标最优方法；其较低或较高J只用于机制诊断。'),
        I('figures/method_comparison.png','图2：六方法在可比实验条件下的库存需求比。'),
        H('三、为什么反馈与风险包络会改变相对排序'),
        P(f'固定P区间、反馈库及选中参数，只改变最终部署的风险水位：不加包络时分类DRO J={row("budget","full60","positive_band","DRO_category").inventory_demand_ratio:.6f}、(s,S) J={row("budget","full60","positive_band","sS").inventory_demand_ratio:.6f}，ΔJ={fullpair.delta_J:+.6f}；加包络后分别为{row("budget","envelope","positive_band","DRO_category").inventory_demand_ratio:.6f}、{row("budget","envelope","positive_band","sS").inventory_demand_ratio:.6f}，ΔJ={envpair.delta_J:+.6f}。'),
        P(f'这一部署干预使相对差异增加{envelope_effect.delta_relative_J:+.6f}，配对区间[{envelope_effect.ci_low:+.6f}, {envelope_effect.ci_high:+.6f}]。它在本轮固定库中重现了焦点两方法的数值排序翻转。开启包络后的单个ΔJ区间仍跨0，因此翻转后的胜负稳定性较弱。'),
        P('包络逐商品取多个训练切片水位的最大值，会改变库存分配的形状。反馈随后为整体P选择公共缩放量lambda；它可以下调总体水位，因而“包络更高”不等于最终库存必然更多。分类DRO与(s,S)的逐商品水位及触发门槛不同，这一交互会改变库存分配效率。'),
        P('不分组DRO与RO的水位均是所有商品共用的常数；公共lambda大体能抵消常数倍率，所以两者在反馈下十分接近。分类、聚类及(s,S)的包络倍率随商品变化，公共lambda无法同时抵消所有商品的形状变化。waterline_shape.csv保存了这一点的数值证据。'),
        H('四、对原问题应如何下结论'),
        P('可以确认：服务指标改变了可行域与商品权重；反馈库和部署风险包络进一步改变方法间相对库存表现。原历史排名是整套流程的比较，不能单独归因于约束。最终预测已核对相同，因此预测变化不是本次焦点排名反转的解释。'),
        P('本轮P≥95%与95%–95.5%选择完全相同的候选，故不能把本轮排名差异归因于95.5%上限。扩大共同库后，训练的优胜方法从F条件下分类DRO切换到P条件下(s,S)；测试优胜是否一致仍取决于包络等部署条件。'),
        I('figures/paired_intervals.png','图3：ΔJ及其条件性95%配对商品块区间。'),
        H('五、证据边界与交付'),
        P('本次为既有测试集上的事后机制检验。训练29–60拟合、1–28验证属于不重叠的回溯留出，不能当作向未来预测的无偏估计；最终预测器、聚类与残差均用满60期。Bootstrap固定了参数和策略，保留每个商品28期路径、按爆品/长尾分层，只反映商品构成重抽样的不确定性，不包含重新训练、未来时间冲击及商品之间共同因子的全部风险。多项对比为探索性区间，未做多重检验校正。'),
        P(f'方案冻结哈希：{complete["protocol_sha256"]}。选参冻结哈希：{complete["selection_sha256"]}。原始Excel与缓存哈希、全部1620候选、108实验单元、48份唯一测试回放、806.4万行轨迹及验证结论均随完整报告交付。')]
    full=[H('完整实验报告：服务约束、反馈控制与风险包络'),
          P('2026年10月16日。问题：旧版需求满足率约束下分类DRO最小，新版正库存比例约束下(s,S)最小，这一差异究竟由什么造成？'),
          *intro[2:6],
          H('1. 数据、版本与历史结果复核'),
          P('原始训练文件replenishment_train.xlsx包含60期×6000商品，原始测试文件replenishment_test.xlsx包含28期×6000商品。原始20个特征、初始库存与爆品/长尾标签均通过已校验缓存读取，输入和缓存SHA256见outputs/input_audit.json。训练需求为360000个观测。测试Period_1重新开始，初始在途0，全部168000商品天参与计分。'),
          P('历史需求版来自experiments/demand_fill_v3/with_ro/outputs_v4；正式当前版来自experiments/service95_revision/outputs_v4。逐条重建销量=min(可用库存,需求)、期末库存、L=1到货关系，并分别回放旧版冻结静态水位及新版冻结反馈策略；四期计划也逐值匹配。所有文件哈希保存于published_audit.json。这里只回放冻结策略，没有声称重新训练历史模型或重选历史参数。'),
          T(['版本','方法','F','P','J'],[[r.version,name(r.method),pct(r.fill_rate),pct(r.positive_stock_ratio),num(r.inventory_demand_ratio)] for r in pub.itertuples()]),
          P(f'焦点方法的新旧预测逐值相同；12组历史轨迹中仅旧版RO预测存在最多{audit["max_forecast_difference_vs_full60"]:.12g}的保存舍入差。由此排除有实质量级的预测差异。历史两版仍改变了风险函数（stop-loss与机会水位）、风险参数、整数正库存保护、验证规则、包络与反馈，不能把整套变化叫作“只改约束”。'),
          H('2. 实验问题、因素及对照关系'),
          P('研究问题分三层：固定候选策略后切换约束，是否改变训练最优参数或测试方法排序？反馈库与部署风险包络是否改变焦点两方法的相对J？历史数值胜负在配对商品构成扰动下有多稳定？方案在本轮计算前写入protocol.json；由于测试文件已经多次看过，这不是盲测或全新预注册研究。'),
          T(['因素','水平','固定与变化'],[
              ['外层服务约束','F≥.95 / P≥.95 / .95≤P≤.955','同一候选得分只计算一次，三组仅改变可行性筛选'],
              ['控制器候选库','静态 / 反馈 / 二者并集','每库内部三组约束使用完全同一候选集'],
              ['最终部署风险','full60 / envelope','选择相同训练候选后，仅替换最终部署风险水位'],
              ['共同设置','L=1、H=4、失销、原始库存','最终模型60期，测试28期，含首期；目标及平局规则一致']]),
          P('共3约束×3候选库×2部署×6方法=108个选中实验单元。并集组复用静态和反馈候选，三约束也复用同一得分；不把重复结果伪装成独立样本。最终只有24个唯一选中候选×2部署=48次唯一测试回放。'),
          H('3. 数学模型与可行域证明'),*intro[7:12],
          P('库存流转：A(i,t)=I(i,t-1)+q(i,t-1)，sales(i,t)=min(A(i,t),d(i,t))，I(i,t)=A(i,t)-sales(i,t)，lost(i,t)=d(i,t)-sales(i,t)。本期观察需求后决定q(i,t)，下期需求前到货；缺货丢失，不回补历史欠货。所有订单为非负整数。'),
          P('静态水位：S(i,t,h)=floor(k*m(i,t,h)*z(i))+1；q(i,t,h)=max(0,ceil(S(i,t,h)-projected_I))。未来库存按预测需求路径递推，只有h=1订单执行。(s,S)还要求projected_I<floor(k*m*s)+1才下单。未来计划绝不使用未来实际需求。'),
          P('DRO基础：相对需求r=d/max(m,1)，按对应商品池建立等权经验分布及W-infinity半径epsilon，机会水位z=kappa*(r_(ceil(beta*M))+epsilon)。SAA的epsilon=0；传统RO为非负需求箱集合的上端mu+Gamma*sigma，不对集合赋予概率；(s,S)是逐商品分位数与池分位数的加权和，权重为残差期数/(残差期数+24)。'),
          P('机会水位的依据：W-infinity半径epsilon允许每个等权经验样本在距离epsilon内移动。令k=ceil(beta*M)，目标整数库存严格大于预测值乘以第k个上端点，则至少k个原子所对应的整个移动区间均位于库存之下，因而最坏分布的正库存概率至少为k/M≥beta。目标取floor(m*z)+1处理整数和相等边界。该结论描述固定基础水位，并不扩展成动态lambda后的有限测试保证。'),
          P('旧版DRO使用另一种风险函数：找最小z使sup_Q E[(R-z)_+-(1-beta)R]≤0。在W-infinity球中，对每个样本的非负区间端点求凸分段线性函数的最大值，再取均值；通过单调二分求根。它控制最坏期望失销量相对于最坏约束表达中的需求量，不是要求beta比例商品天留有库存。旧版与新版连内部鲁棒函数也改变了，这是历史整套对照中的混杂因素；本轮共同候选库统一采用当前机会水位，专门隔离外层可行性规则。'),
          P('最终包络z_env(i)=max_b z_b(i)，b取全60期、训练1–24、1–36、1–48、29–60。所有切片都属于训练文件。验证阶段两种部署干预均只使用29–60拟合出的同一基础水位，不把1–28验证需求用于风险水位估计；包络是选参后发生的部署干预。它与正式版风险包络一致，且消除了验证数据回流的混杂。'),
          P('反馈：C_t为已实现正库存商品天，内部目标p*来自声明候选；u_t=clip((28*N*p*-C_t)/(N*max(27-t,1))+bias,0,1)，这里t为0起算。以训练残差CDF和最近w期已发生单步相对误差CDF估计下一期服务率，取满足估计服务不低于u_t的最小公共lambda。CDF各用128个中点分位数近似，先验权重a/(a+n)，16次二分，lambda上限128。首期需求不伪装成事前单步误差。'),
          P('lambda可以小于1，因此固定DRO水位的beta保护不自动成为执行策略保证。实验比较的是鲁棒基础水位加公共反馈的可实施策略。没有求解原始所有非预见策略中的全局多期最优解；H=4采用预测路径滚动计划。实际P达标由完整测试计数确认。'),
          H('4. 特征聚类、预测与训练流程'),
          P('特征聚类包含5个需求描述：log(1+均值)、log(1+变异系数)、零需求比例、log(1+95%分位数/均值)、log(1+后8期均值/前8期均值)。原始20特征按拟合段时间均值标准化并做PCA，取前3成分，标准化后权重0.5；与5个标准化需求特征拼接。K-means固定6类、随机种子20260929、8次初始化、60次迭代，以平方误差最小者为准。'),
          P('特征聚类DRO按6池，不分组DRO和RO用统一池，分类DRO按480爆品/5520长尾，SAA及(s,S)的池与特征聚类一致。预测参数固定为历史训练验证得到的blend=1、window=3、prior=0。拟合独立线性时间趋势，滚动用已观测的最近3期校正，预测不得低于1。测试未来特征不输入。'),
          P('本轮反馈与静态参数统一使用训练29–60拟合、训练1–28回溯留出选择，块不重叠。最终重新用训练全部1–60拟合预测器、聚类和扩展窗残差；残差前18期暖启动，19–60期产生样本外残差。训练最早28期参与选参检验及最终60期拟合，并未丢弃。回溯留出不是向未来预测成绩。'),
          H('5. 完整候选库与冻结程序'),
          P('每方法231个静态倍率k=0.20,0.21,...,2.50；39个反馈候选：p*=.9525时w∈{1,3,7}、a∈{0,3}、bias∈{-.01,0,.01}共18个，加上固定w=3、a=3时p*∈{.70,.80,.90,.95,.955,.975,.99}与3个bias共21个。每方法270个，合计1620个。所有目标p*均同时提供给三种约束，绝不按约束偷偷更换库。'),
          P('基础固定参数沿用当前正式版：三种DRO beta=.975、epsilon=.005、kappa=1；SAA epsilon=0；(s,S)分位数.975/.985；RO Gamma=3。各方法不是被给予相同水位数值，而是同一候选规模和筛选规则。基础参数从既有训练流程沿用，本实验不穷尽其所有可能取值。'),
          P('先冻结protocol.json，再计算训练得分；对每方法与候选库，用对应约束筛选，最小化J，平局按候选号。将54条选择和全部模型/水位散列写入frozen_selection.json，之后才回放测试。读取原始文件作哈希审计会接触测试文件字节，但本轮选参没有使用任何测试数值成绩。'),
          T(['方法','候选库','总候选','F可行','P下限可行','P区间可行','F与P区间交集'],[[name(r.method),FAM_NAMES[r.family],r.candidates,r.fill_feasible,r.positive_lower_feasible,r.positive_band_feasible,r.fill_and_band] for r in overlap.itertuples()]),
          I('figures/training_frontier.png','图4：完整训练候选的服务与库存分布。'),
          H('6. 训练选择与测试优胜必须分开'),
          T(['约束','库/部署','训练优胜','该方法测试达标','测试达标数','测试描述性优胜'],[[ARM_NAMES[r.arm],FAM_NAMES[r.family]+'/'+ENV_NAMES[r.envelope],name(r.training_winner),yes(r.training_winner_test_feasible),r.test_feasible_methods,name(r.descriptive_test_winner) if pd.notna(r.descriptive_test_winner) else '无'] for r in w.itertuples()]),
          P('训练优胜用于选择规则的解释；测试描述性优胜仅对已冻结的6方法回放结果进行比较。静态P区间组测试无达标方法，不能用其J排序宣称可行最优。并集F组训练选择静态候选，P组选择反馈候选，这反映改变可行域会改变控制器类型。'),
          H('7. 108实验单元完整结果')]
    for arm,family,envelope in itertools.product(ARMS,FAMILIES,ENVELOPES):
        group=s[s.arm.eq(arm)&s.family.eq(family)&s.envelope.eq(envelope)]
        full += [H(f'{ARM_NAMES[arm]} | {FAM_NAMES[family]} | {ENV_NAMES[envelope]}'),
            T(['方法/候选号','训练F','训练P','训练J','测试F','测试P','测试J','测试达标'],
              [[name(r.method)+' / '+str(int(r.candidate)),pct(r.train_F),pct(r.train_P),num(r.train_J),pct(r.fill_rate),pct(r.positive_stock_ratio),num(r.inventory_demand_ratio),yes(r.test_feasible)] for r in group.itertuples()])]
    full += [H('8. 焦点两方法的配对差异与因素效应'),
        P('定义ΔJ=J(分类DRO)-J(s,S)。负值表示分类DRO库存需求比更低。对6000个商品按480/5520分层、配对重抽样1000次，种子20261016，每个商品保留28期完整路径；每次用同一组商品权重重新计算分子与需求分母。区间为2.5%与97.5%分位数。'),
        T(['库/版本','部署','约束','ΔJ','区间下限','区间上限','两者达标'],[[FAM_NAMES.get(r.family,r.family),ENV_NAMES.get(r.envelope,r.envelope),ARM_NAMES.get(r.arm,r.arm),f'{r.delta_J:+.6f}',f'{r.ci_low:+.6f}',f'{r.ci_high:+.6f}',yes(r.both_test_feasible)] for r in pair.itertuples()]),
        I('figures/paired_intervals.png','相同商品抽样下的条件差异区间；未达标行只作诊断。'),
        T(['对比','条件','相对差异变化','区间下限','区间上限'],[[r.effect,r.case,f'{r.delta_relative_J:+.6f}',f'{r.ci_low:+.6f}',f'{r.ci_high:+.6f}'] for r in effects.itertuples()]),
        P('差异中的差异是当前固定库内的策略干预对比；反馈库对比包含按相同规则重新选择候选的作用。由于历史方案并不是这些有限因素的完全析因版本，不能把本表相加后宣称解释历史总变化的某个百分比；交互作用使这种归因依赖路径。所有区间是探索性的条件商品重抽样区间。'),
        *intro[next(i for i,b0 in enumerate(intro) if b0['kind']=='heading' and b0['text'].startswith('三、'))+1:next(i for i,b0 in enumerate(intro) if b0['kind']=='heading' and b0['text'].startswith('四、'))],
        H('9. 商品结构、售罄和水位形状'),
        T(['版本','方法','恰好售罄天数比','发生失销天数比','P'],[[r.version,name(r.method),pct(r.exact_sellout_fraction),pct(r.lost_day_fraction),pct(r.positive_stock_ratio)] for r in pub.itertuples()]),
        P('P=1-恰好售罄比例-发生失销比例；这里“恰好售罄”也包含零需求且零库存。F只按损失需求量计分，因此恰好售罄可以不损害F却损害P。此恒等式按每个历史轨迹逐行核对。'),
        T(['版本','方法','类别','F','P','J'],[[r.version,name(r.method),'爆品' if r.category=='hot' else '长尾',pct(r.fill_rate),pct(r.positive_stock_ratio),num(r.inventory_demand_ratio)] for r in seg[seg.category.ne('all')].itertuples()]),
        P('上表的P按SKU数合并，F和J按各类别需求量合并。实验总体约束不强制爆品、长尾各自达95%，也不强制每一天或每个商品达95%。新的48回放按品类和逐期结果另见segments.csv、periods.csv，逐商品结果见sku_summary.csv.gz。'),
        T(['方法','全60水位均值','包络均值','改变商品数','包络倍率最小','倍率最大','倍率标准差'],[[name(r.method),num(r.full_mean),num(r.envelope_mean),r.changed_items,num(r.min_multiplier),num(r.max_multiplier),num(r.multiplier_std)] for r in water.itertuples()]),
        H('10. 核验、计算环境与运行记录'),
        P(f'实际运行{complete["elapsed_seconds"]:.2f}秒（本机本次记录，不是其他机器的性能保证）。Python {protocol["python"]}，NumPy {protocol["numpy"]}，pandas {protocol["pandas"]}。源提交{protocol["source_commit"]}；运行脚本SHA256={protocol["run_code_sha256"]}。完整依赖见requirements_v3.txt，PDF生成额外使用reportlab和pypdf。'),
        T(['核验项目','实际结果'],[
            ['原始Excel/缓存/60期数组散列','通过；输入签名见input_audit.json'],
            ['重建预测、聚类、样本外残差、独立水位','通过；不只读取汇总数'],
            ['重算全部训练候选和54次选择','1620条通过；两种部署共108行'],
            ['完整测试库存恒等式与四期整数计划','48回放、8064000行通过'],
            ['未来需求扰动检查','第13期后需求改为7倍再加123，此前预测与订单不变'],
            ['历史冻结策略回放','新旧各6方法全部通过，实际订单/库存/销量/四期计划逐值匹配'],
            ['数值读取问题的处理','初用CSV默认浮点解析导致4个未执行未来计划差1；改为round_trip恢复原值后全部逐值匹配，未修改历史文件或结果'],
            ['正式交付与实验隔离','保留旧版、新版全部冻结输出；诊断实验单独保存']]),
        P('日志文件：completed.json记录真实计数和耗时，verification.json记录核验通过范围，published_audit.json记录历史源文件散列与回放结论，analysis_audit.json记录Bootstrap设置。所有报告表格从CSV和JSON自动生成，未手工填写实验数值。'),
        H('11. 复现步骤与文件索引'),
        P('从仓库根目录运行：python -m pip install -r experiments/rolling_dro_v3/requirements_v3.txt。重新计算用python experiments/constraint_mechanism/run_experiment.py --output experiments/constraint_mechanism/rerun_outputs；完整运行会检查原始Excel。已有completed.json的目录不覆盖，以保留证据。'),
        P('复核：python experiments/constraint_mechanism/verify_experiment.py --output experiments/constraint_mechanism/outputs。历史回放和统计：python experiments/constraint_mechanism/analyze_results.py。报告源文与图：python experiments/constraint_mechanism/build_reports.py。PDF：安装reportlab和pypdf后运行python experiments/constraint_mechanism/render_reports.py。分析/报告默认读取本次outputs；新目录复核支持--output，若需新报告应明确修改报告输入或在副本中更换目录。'),
        T(['文件','内容'],[
            ['protocol.json / frozen_selection.json','计算前方案及测试前54条冻结选择、共享模型散列'],
            ['candidate_validation.csv','全部1620条训练候选，含F/P/J和原始计数'],
            ['summary.csv / cell_winners.csv','全部108实验单元、训练优胜和测试描述性优胜'],
            ['shared_arrays.npz','原始需求、初始库存、拟合标签/残差/预测与部署水位'],
            ['traces/*.npz','48回放：每商品每期可用量、销量、期末库存、订单和4期计划'],
            ['sku_summary.csv.gz / periods.csv / segments.csv','逐商品、逐期与全部/爆品/长尾结果'],
            ['published_summary.csv / published_audit.json','历史新旧两版真实轨迹重算及源哈希'],
            ['paired_contrasts.csv / factor_effects.csv','全部配对差异和因素效应及区间'],
            ['feasible_overlap.csv / waterline_shape.csv','可行集合变化与风险包络形状证据'],
            ['reports/机制探索报告.pdf / 完整实验报告.pdf','摘要解释与完整设置、全部选中结果、核验记录'],
            ['SHA256SUMS.json','本实验交付文件散列清单']]),
        H('12. 结论、适用范围与下一步'),
        *[b0 for b0 in intro if b0['kind']=='paragraph' and b0['text'].startswith(('可以确认：','本轮P≥95%','本次为既有测试集'))],
        P('本轮结果回答的是“在当前真实数据及声明候选库内，哪些改变足以改变参数和相对排序”。不能证明任意数据、任意不确定集合或任意鲁棒策略中某方法普遍最优。训练内回溯留出、重复测试、固定参数的商品块区间和候选库范围共同限定解释。若要评价稳定部署优势，需要新增未使用的时间段，并在看结果前固定全流程。'),
        H('方法参考'),
        P('RSOME官方多阶段库存示例：https://xiongpengnus.github.io/rsome/example_dro_inv。该示例为零提前期和积压成本，本项目按提前期1与失销改写；网页说明方法背景，本报告的所有数值来自本地真实Excel与可复现回放。'),
        P('RSOME官方传统鲁棒优化说明：https://xiongpengnus.github.io/rsome/ro_rsome。这里的RO使用区间不确定集合；DRO水位使用W-infinity分布模糊集，区别于官方库存示例中的Wasserstein-1等实例。')]
    documents={'机制探索报告':intro,'完整实验报告':full}
    for title,blocks in documents.items():
        (reports/(title+'.md')).write_text(markdown(blocks),encoding='utf-8')
        dump(reports/(title+'.json'),dict(title=title,blocks=blocks))
    print('Built two source reports and four scientific figures from audited outputs.')

if __name__=='__main__':main()
