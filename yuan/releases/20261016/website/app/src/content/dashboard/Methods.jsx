import React,{useMemo} from 'react';
import {DataComponent,RichNarrative,Section,useDataApp} from '../../data-app-public.jsx';
import {MetricStrip} from './Tradeoff.jsx';
import {routes,pct,saveFile} from './lab-data.js';
import formulas from '../assets/formulas.json';

export function Methods({rows,current,onSelect,onReplay}){
 const {snapshot}=useDataApp(),docRow=snapshot.queries.methods.rows.find(r=>r.model===current.model);
 const dossier=useMemo(()=>docRow?JSON.parse(docRow.payload):null,[docRow]);
 const route=current.model==='service_policy'?['历史＋库存状态','直接策略网络','服务与库存损失','产生下一期订单']:
  current.model==='nn_scenario'?['概率需求网络','8个需求情景','有限步内层优化','执行下一期订单']:
  current.model==='dfl_improved'?['概率需求网络','8个需求情景','有限步内层优化','4期决策损失反馈']:routes[current.family];
 return <div className="lab-method-layout"><aside className="lab-method-list" aria-label="模型目录">{[...new Set(rows.map(r=>r.family))].map(f=><section key={f}><h3>{f}</h3>{rows.filter(r=>r.family===f).map(r=><button aria-pressed={current.model===r.model} key={r.model} onClick={()=>onSelect(r.model)}><span>{r.name}</span><small>{r.owner==='kristina'?'K':'Y'}</small></button>)}</section>)}</aside>
  <div className="lab-method-main">
   <DataComponent id="method-summary" queryId="models" title={current.name} sourceRows={[current]} displayRows={[current]} kind="custom" variant="card"><div className="lab-method-heading"><span className="lab-badge">{current.family}</span><span>{current.owner}</span><button onClick={()=>onReplay(current.model)}>查看库存回放 ↗</button></div><MetricStrip row={current}/><p className="lab-caption">版本：{current.version} · {current.test_status}。测试集6000 SKU、完整28期。{current.protocol}。</p></DataComponent>
   <Section id="method-route-head" title="模型如何从输入走向决策" spacing="section"><div className="lab-process">{route.map((step,i)=><div key={step}><span>0{i+1}</span><strong>{step}</strong>{i<3&&<b aria-hidden="true">→</b>}</div>)}</div>{current.family==='概率网络 · DFL'&&<div className="lab-feedback">↖ 决策损失反馈到需求网络 · 未DFL微调模型不执行此反馈训练</div>}</Section>
   <DataComponent id="inventory-math" queryId="models" title="统一的一期到货库存动态" kind="custom" sourceRows={[current]} displayRows={[]} variant="card">
    <div className="lab-math-row"><math display="block"><mrow><msub><mi>A</mi><mi>t</mi></msub><mo>=</mo><msub><mi>I</mi><mrow><mi>t</mi><mo>−</mo><mn>1</mn></mrow></msub><mo>+</mo><msub><mi>q</mi><mrow><mi>t</mi><mo>−</mo><mn>1</mn></mrow></msub></mrow></math><span>上期期末库存 ＋ 上期订单到货</span></div>
    <div className="lab-math-row"><math display="block"><mrow><msub><mi>s</mi><mi>t</mi></msub><mo>=</mo><mi>min</mi><mo>(</mo><msub><mi>A</mi><mi>t</mi></msub><mo>,</mo><msub><mi>d</mi><mi>t</mi></msub><mo>)</mo><mo>,</mo><mspace width="1em"/><msub><mi>I</mi><mi>t</mi></msub><mo>=</mo><msub><mi>A</mi><mi>t</mi></msub><mo>−</mo><msub><mi>s</mi><mi>t</mi></msub></mrow></math><span>销量不超过可用库存，未满足需求流失</span></div>
    <p className="lab-caption">需求发生后，看到当期需求和期末库存，再下单qₜ，下一期到货。首期到货为0，末期订单仍是在途，不计入期末库存。</p>
   </DataComponent>
   {dossier?<DataComponent id="method-dossier" queryId="methods" title="模型输入、训练与数学细节" kind="custom" sourceRows={[docRow]} displayRows={[docRow]} variant="card">
    <div className="release-method-actions"><code>{dossier.code}</code>{dossier.markdown&&<button onClick={()=>saveFile(current.model+'.md',dossier.markdown,'text/markdown;charset=utf-8')}>下载本模型中文说明 MD</button>}</div>
    {dossier.decision_steps&&<details className="lab-disclosure" open><summary>这一个模型到底怎样决定补多少货？</summary><ol className="release-decision-steps">{dossier.decision_steps.map((step,i)=><li key={i}>{step}</li>)}</ol></details>}
    <details className="lab-disclosure" open><summary>使用了哪些数据，怎样训练？</summary><RichNarrative id={'method-data-'+current.model} value={dossier.data+'\n\n'+dossier.training}/></details>
    <details className="lab-disclosure" open={['turnover_dfl','dfl_improved','nn_scenario'].includes(current.model)}><summary>完整公式与推导</summary>{dossier.blocks.map((b,i)=>b.type==='text'?<RichNarrative key={i} id={'derivation-'+current.model+'-'+i} value={b.value}/>:<figure className="lab-formula" key={i}>{b.mathml?<div role="math" dangerouslySetInnerHTML={{__html:b.mathml}}/>:formulas[b.tex]?<img src={formulas[b.tex]} alt={b.explanation||'模型数学公式'}/>:<p>当前公式未保存渲染版本，请展开查看LaTeX源码。</p>}<figcaption>{b.explanation}</figcaption><details><summary>复制LaTeX源码</summary><pre>{b.tex}</pre></details></figure>)}</details>
    <details className="lab-disclosure"><summary>决策步骤、参数与方法限制</summary><RichNarrative id={'method-limits-'+current.model} value={dossier.decision+'\n\n'+dossier.limits}/><pre>{JSON.stringify(dossier.parameters,null,2)}</pre></details>
   </DataComponent>:<DataComponent id="method-additional" queryId="models" title="方案说明" kind="custom" sourceRows={[current]} displayRows={[current]} variant="card"><RichNarrative id={'extra-method-'+current.model} value={current.owner==='kristina'?'Kristina的冻结服务约束版本。特征聚类、不分组、爆品/长尾分组分别改变经验残差的分组方式；传统(s,S)使用上下水位，零半径经验模型不增加DRO半径。服务约束使用严格大于零的整数水位。\n\n拟合使用29—60期，1—28期重启校准，不是严格前向验证；与个人模型相同的是测试需求、初始库存和评价公式，而不是训练协议。准确参数见下方。':'先由时间顺序OOF生成候选预测，再学习全局或逐SKU融合权重，最终通过统一补货规则产生订单。四种融合方式不等于把多种订货量直接相加。预测误差更小也不保证库存决策更好。'}/><pre>{JSON.stringify(JSON.parse(current.parameters),null,2)}</pre></DataComponent>}
   {current.family==='SPO · 决策学习'&&<p className="lab-method-warning">SPO系列采用有限动作近似；直接决策模型的动作与成本参数预测模型不同，不能一概称为直接求解原非线性模型的标准SPO+。</p>}
   {current.family==='概率网络 · DFL'&&<Experiment/>}
  </div></div>;
}
function Experiment(){
 const {snapshot}=useDataApp(),rows=snapshot.queries.experiments.rows;
 return <DataComponent id="latest-fill-experiment" queryId="experiments" title="缺货感知DFL：已测试但未采纳" kind="custom" sourceRows={rows} displayRows={rows} variant="card"><div className="lab-experiment-results">{rows.map(r=><div key={r.model}><strong>{r.model}</strong><span>服务水平 {pct(r.service)}</span><span>满足率 {pct(r.fill)}</span><span>库存需求比 {r.J.toFixed(4)}</span></div>)}</div><p className="lab-method-warning">满足率上升，但库存需求比增加24.91%，不符合用户要求。未替换原7期DFL，也未把失败候选加入主模型排行榜。</p></DataComponent>;
}
