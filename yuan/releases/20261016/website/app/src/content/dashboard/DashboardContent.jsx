import React,{useState} from 'react';
import {DataComponent,Dropdown,EvidenceChart,RichNarrative,Section,useDataApp,useDashboardTabs} from '../../data-app-public.jsx';
import {Tradeoff} from './Tradeoff.jsx';
import {Replay} from './Replay.jsx';
import {Methods} from './Methods.jsx';
import {num,pct,routes} from './lab-data.js';
import './lab.css';

const tabs=[{id:'overview',label:'研究概览'},{id:'methods',label:'算法原理'},{id:'replay',label:'逐期实验'},{id:'comparison',label:'模型对比'},{id:'data',label:'数据探索'},{id:'code',label:'模型与代码'}].map(t=>({...t,focusFields:['model']}));

export function DashboardContent(){
 const shell=useDataApp(),{snapshot}=shell;
 const {activeTabId}=useDashboardTabs(tabs);
 const tab=tabs.some(t=>t.id===activeTabId)?activeTabId:'overview';
 const [owner,setOwner]=useState('yuan xuecheng');
 const allRows=snapshot.queries.models.rows;
 const rows=allRows.filter(r=>owner==='all'||r.owner===owner);
 const [selected,setSelected]=useState('turnover_dfl');
 const [sku,setSku]=useState(snapshot.queries.skus.rows[0].sku);
 const requested=shell.viewFocus?.model||selected;
 const modelId=rows.some(r=>r.model===requested)?requested:rows[0].model;
 const current=rows.find(r=>r.model===modelId)||rows[0];
 const select=model=>{setSelected(model);shell.setDashboardFocus({model});};
 const choose=model=>{setSelected(model);shell.exploreDashboard('methods',{focus:{model}});};
 const replay=(model=modelId)=>{setSelected(model);shell.exploreDashboard('replay',{focus:{model}});};
 return <article className="lab-page">
  <div className="lab-page-header"><div><span className="lab-kicker">2026.10.16 · 个人29方案 / Kristina最新6方案</span><h1>{tabs.find(t=>t.id===tab)?.label}</h1></div><Dropdown label="负责人范围" showLabel value={owner} choices={['yuan xuecheng','kristina','all']} choiceLabels={{'yuan xuecheng':'Yuan · 29个个人方案',kristina:'Kristina · 最新6个方案',all:'全部35个方案'}} onChange={setOwner}/><div className="lab-scope"><span>6,000 <small>SKU</small></span><i/><span>28 <small>测试期</small></span><span className="lab-badge">一期到货</span></div></div>
  {tab==='overview'&&<div className="lab-overview-layout">
   <div className="lab-overview-main"><Tradeoff rows={rows} selected={modelId} onSelect={select} onChoose={choose} compact/></div>
   <aside className="lab-overview-aside">
    <DataComponent id="research-map" queryId="models" title="从预测到库存决策" kind="custom" sourceRows={rows} displayRows={rows} variant="card">
     <div className="lab-research-map">{['Baseline','时间序列','ML · PTO','SPO · 决策学习','概率网络 · DFL','Kristina · 鲁棒优化'].filter(f=>rows.some(r=>r.family===f)).map((family,i)=><button key={family} onClick={()=>choose(rows.find(r=>r.family===family).model)}><span className="lab-route-index">0{i+1}</span><div><strong>{family}</strong><small>{routes[family][1]} → {routes[family][2]}</small></div><b>↗</b></button>)}</div>
     <p className="lab-caption">SPO与可微优化式DFL同属广义决策聚焦学习；这里按实验流水线区分，而非互斥理论分类。</p>
    </DataComponent>
    {rows.some(r=>r.model==='turnover_dfl')&&<DataComponent id="dfl-reference" queryId="models" title="DFL研究主线" kind="custom" sourceRows={rows.filter(r=>['nn_scenario','dfl_improved','turnover_dfl'].includes(r.model))} displayRows={rows.filter(r=>['nn_scenario','dfl_improved','turnover_dfl'].includes(r.model))} variant="card">
     <div className="lab-reference-chain">{rows.filter(r=>['nn_scenario','dfl_improved','turnover_dfl'].includes(r.model)).map(r=><button key={r.model} onClick={()=>choose(r.model)}><span>{r.name}</span><strong>{r.J.toFixed(4)}<small> J</small></strong><span>{pct(r.service)} <small>服务水平</small></span></button>)}</div>
     <p className="lab-caption">7期DFL降低库存，但满足率低于未微调内层模型，不作“全面更优”的判断。</p><button className="lab-primary" onClick={()=>replay('turnover_dfl')}>打开逐期库存实验 →</button>
    </DataComponent>}
    <DataComponent id="research-contract" queryId="models" kind="custom" title="共同评价规则" sourceRows={rows} displayRows={[]} variant="card"><div className="lab-contract"><div><span>服务水平</span><p>期末库存 &gt; 0 的商品—周期比例</p></div><div><span>满足率 · Fill Rate</span><p>实际销量 / 需求总量</p></div><div><span>库存需求比 J ↓</span><p>期末库存累计 / 需求总量</p></div></div></DataComponent>
   </aside>
  </div>}
  {tab==='methods'&&<Methods rows={rows} current={current} onSelect={select} onReplay={replay}/>}
  {tab==='replay'&&<Replay rows={rows} initialModel={modelId} selectedSku={sku} onSku={setSku}/>}
  {tab==='comparison'&&<Tradeoff rows={rows} selected={modelId} onSelect={select} onChoose={choose}/>}
  {tab==='data'&&<Explore selectedSku={sku} onSku={setSku} onReplay={()=>replay()}/>}
  {tab==='code'&&<CodeIndex rows={rows} onChoose={choose}/>}
  <footer className="lab-footer"><span>离线审阅版 · 2026.10.16 · 未上传GitHub</span><span>Yuan：预测与决策学习 · Kristina：鲁棒优化＋服务反馈（aec11c8）</span></footer>
 </article>;
}

function CodeIndex({rows,onChoose}){
 return <Section id="code-index" title="从模型找到代码与补货决策" spacing="none">
  <p className="lab-caption">训练共用数据、特征和评价器，模型文件只保留算法差异；当前网站是冻结测试轨迹的重放，不会在浏览器重新训练。桌面的两个ML项目未纳入。</p>
  <div className="release-code-list">{rows.map(r=><DataComponent key={r.model} id={'code-'+r.model} queryId="models" kind="custom" title={r.name} sourceRows={[r]} displayRows={[r]} variant="card">
   <div className="lab-selection-head"><span>{r.owner}</span><span className="lab-badge">{r.test_status}</span></div>
   <p><code>{r.model}</code> · {r.version}</p><pre>{r.code}</pre><p className="lab-caption">{r.protocol}</p>
   <button className="lab-primary" onClick={()=>onChoose(r.model)}>查看数据、公式与逐步决策 →</button>
  </DataComponent>)}</div>
 </Section>;
}

function Explore({selectedSku,onSku,onReplay}){
 const {snapshot}=useDataApp(),all=snapshot.queries.skus.rows;
 const [category,setCategory]=useState('all'),[measure,setMeasure]=useState('train_cv'),[search,setSearch]=useState('');
 const scoped=all.filter(r=>category==='all'||r.category===category);
 const active=scoped.find(r=>r.sku===selectedSku)||scoped[0];
 const matches=scoped.filter(r=>r.sku.toLowerCase().includes(search.toLowerCase())).slice(0,8);
 const vals=scoped.map(r=>r[measure]),max=Math.max(...vals),step=Math.max(max/16,1e-6);
 const bins=Array.from({length:16},(_,i)=>({区间:Number((i*step).toFixed(3)),商品数:vals.filter(v=>Math.min(15,Math.floor(v/step))===i).length}));
 const p=JSON.parse(active.payload),history=[...p.history,...p.demand].map((d,i)=>({周期:i+1,'训练需求':i<60?d:null,'测试需求':i>=60?d:null}));
 return <>
  <Section id="explore-head" title="需求规模与波动" spacing="none" filters={<><Dropdown label="商品类别" showLabel value={category} choices={['all',...new Set(all.map(r=>r.category))]} choiceLabels={{all:'全部商品'}} onChange={setCategory}/><Dropdown label="分布指标" showLabel value={measure} choices={['train_cv','train_mean','zero_fraction']} choiceLabels={{train_cv:'需求变异系数 CV',train_mean:'平均需求',zero_fraction:'零需求比例'}} onChange={setMeasure}/></>}>
   <EvidenceChart id="demand-histogram" queryId="skus" title="训练期需求特征分布" variant="card" sourceRows={scoped} rows={bins} height={310} spec={{type:'bar',x:'区间',y:'商品数',colors:{商品数:'var(--chart-1)'},distribution:true,showLegend:false,startAtZero:true}}/>
  </Section>
  <Section id="sku-explore-head" title="单商品需求历史" spacing="section">
   <div className="lab-sku-search"><label>查找商品<input value={search} onChange={e=>setSearch(e.target.value)} placeholder="输入SKU编号"/></label><div>{matches.map(r=><button key={r.sku} aria-pressed={r.sku===active.sku} onClick={()=>onSku(r.sku)}>{r.sku}</button>)}</div></div>
   <DataComponent id="sku-profile" queryId="skus" title={active.sku+' · '+active.category} kind="metrics" sourceRows={[active]} displayRows={[active]} variant="card"><div className="lab-profile-stats"><span>训练期平均需求 <strong>{num(active.train_mean)}</strong></span><span>变异系数 CV <strong>{active.train_cv.toFixed(2)}</strong></span><span>零需求比例 <strong>{pct(active.zero_fraction*100)}</strong></span><button className="lab-primary" onClick={()=>{onSku(active.sku);onReplay();}}>查看该商品补货 →</button></div></DataComponent>
   <EvidenceChart id="sku-history" queryId="skus" title="60期训练需求与28期测试需求" variant="card" sourceRows={[active]} rows={history} height={330} spec={{type:'line',x:'周期',y:'训练需求',fields:['训练需求','测试需求'],colors:{训练需求:'var(--chart-2)',测试需求:'var(--accent)'},stackable:false}}/>
  </Section>
  <details className="lab-disclosure"><summary>特征使用与数据范围</summary><RichNarrative id="feature-rules" value="原数据有20个匿名特征。不假设下一期x提前已知：预测使用已观察历史及滞后特征。各模型具体输入维数不同，见算法原理。网页内嵌真实需求历史和冻结库存轨迹，未内嵌原始20特征矩阵，不展示伪造特征值。\n\n测试期1—28在需求历史图中对应第61—88期。这里只浏览，不重新训练模型。"/></details>
 </>;
}
