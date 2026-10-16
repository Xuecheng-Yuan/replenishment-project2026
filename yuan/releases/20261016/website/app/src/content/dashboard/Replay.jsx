import React,{useEffect,useMemo,useRef,useState} from 'react';
import {DataComponent,DataTable,Dropdown,EvidenceChart,RichNarrative,Section,SegmentedControl,Slider,useDataApp} from '../../data-app-public.jsx';
import {MetricStrip} from './Tradeoff.jsx';
import {cumulative,decodePayload,exportSVG,num,palette,saveCSV,saveFile,skuTrajectory,teaching} from './lab-data.js';

const measures={ending:'期末库存',order:'本期订货量',sales:'实际销量',forecast:'需求预测'};
export function Replay({rows,initialModel,selectedSku,onSku}){
 const {snapshot}=useDataApp(),skus=snapshot.queries.skus.rows,allPeriod=snapshot.queries.periods.rows;
 const [ids,setIds]=useState([...new Set([initialModel,'nn_scenario','lightgbm'])].slice(0,3));
 const [mode,setMode]=useState('research'),[scope,setScope]=useState('sku'),[measure,setMeasure]=useState('ending');
 const [period,setPeriod]=useState(1),[playing,setPlaying]=useState(false),[speed,setSpeed]=useState(900);
 const [search,setSearch]=useState(''),[loaded,setLoaded]=useState(null),[error,setError]=useState('');
 const [coefficient,setCoefficient]=useState(1.65),[initial,setInitial]=useState(null),[demandScale,setDemandScale]=useState(1);
 const host=useRef(null);
 const sku=skus.find(r=>r.sku===selectedSku)||skus[0],current=rows.find(r=>r.model===ids[0]);
 const found=skus.filter(r=>r.sku.toLowerCase().includes(search.toLowerCase())).slice(0,7);
 useEffect(()=>{
  let live=true;setLoaded(null);setError('');
  Promise.all(ids.map(async model=>[model,await decodePayload(snapshot.queries.trajectories.rows.find(r=>r.model===model))]))
   .then(pairs=>{if(live)setLoaded(Object.fromEntries(pairs));}).catch(e=>{if(live)setError(e.message);});
  return ()=>{live=false;};
 },[ids.join('|'),snapshot]);
 useEffect(()=>{if(!playing)return;const timer=setInterval(()=>setPeriod(p=>{if(p>=28){setPlaying(false);return 28;}return p+1;}),speed);return ()=>clearInterval(timer);},[playing,speed]);
 const simulated=useMemo(()=>teaching(sku,coefficient,initial??sku.initial,demandScale),[sku,coefficient,initial,demandScale]);
 const sources=snapshot.queries.trajectories.rows.filter(r=>ids.includes(r.model));
 const scopedSources=mode==='demo'?[sku]:scope==='warehouse'?allPeriod.filter(r=>ids.includes(r.model)&&r.period<=period):sources;
 const traces=useMemo(()=>{
  if(mode==='demo')return {teaching:simulated};
  if(scope==='warehouse')return Object.fromEntries(ids.map(id=>[id,allPeriod.filter(r=>r.model===id).map(r=>({period:r.period,demand:r.demand,sales:r.sales,ending:r.ending,order:r.orders,arrival:r.arrivals,opening:r.period===1?skus.reduce((s,r)=>s+r.initial,0):allPeriod.find(p=>p.model===id&&p.period===r.period-1).ending,forecast:null}))]));
  return loaded?Object.fromEntries(ids.map(id=>[id,skuTrajectory(sku,id,loaded[id])])):{};
 },[mode,scope,ids.join('|'),loaded,sku,simulated]);
 const activeIds=mode==='demo'?['teaching']:ids;
 const labels=Object.fromEntries(activeIds.map(id=>[id,id==='teaching'?'教学推演':rows.find(r=>r.model===id).name]));
 const active=traces[activeIds[0]];
 const state=active?.[period-1];
 const stats=mode==='research'&&scope==='warehouse'?allPeriod.find(r=>r.model===ids[0]&&r.period===period):active?cumulative(active,period):null;
 const forecastIds=activeIds.filter(id=>measure!=='forecast'||traces[id]?.some(r=>r.forecast!==null));
 const chartRows=Array.from({length:28},(_,t)=>{
  const r={周期:t+1,'真实需求':t<period?(active?.[t].demand??null):null};
  forecastIds.forEach(id=>{r[labels[id]]=t<period?(traces[id]?.[t][measure]??null):null;});
  return r;
 });
 const fields=['真实需求',...forecastIds.map(id=>labels[id])];
 const colors={'真实需求':'var(--secondary)',...Object.fromEntries(forecastIds.map((id,i)=>[labels[id],palette[i]]))};
 const chooseModel=id=>{setPlaying(false);setIds(prev=>[id,...prev.filter(v=>v!==id)].slice(0,4));};
 const toggle=id=>setIds(prev=>prev.includes(id)?prev.length===1?prev:prev.filter(v=>v!==id):prev.length<4?[...prev,id]:prev);
 const exportRows=()=>saveCSV(activeIds.flatMap(id=>(traces[id]||[]).filter(r=>r.period<=period).map(r=>({sku:scope==='warehouse'?'全部6000 SKU':sku.sku,model:labels[id],...r}))),mode==='demo'?'教学推演_非真实模型结果.csv':'冻结测试轨迹.csv');
 return <>
  <div className="lab-workbench-controls"><SegmentedControl value={mode} onChange={v=>{setMode(v);setPlaying(false);}} options={[{value:'research',label:'科研实验 · 真实结果'},{value:'demo',label:'教学推演 · 非模型成绩'}]}/>{mode==='research'&&<SegmentedControl value={scope} onChange={setScope} options={[{value:'sku',label:'单商品'},{value:'warehouse',label:'全部6000 SKU'}]}/>}</div>
  {(scope==='sku'||mode==='demo')&&<div className="lab-sku-search"><label>查找商品<input value={search} onChange={e=>setSearch(e.target.value)} placeholder={'当前 '+sku.sku}/></label><div>{found.map(r=><button key={r.sku} aria-pressed={r.sku===sku.sku} onClick={()=>{onSku(r.sku);setInitial(null);setPlaying(false);}}>{r.sku}</button>)}</div><span className="lab-badge">{sku.category}</span></div>}
  {mode==='research'?<div className="lab-model-compare-controls"><Dropdown label="主模型" showLabel value={ids[0]} choices={rows.map(r=>r.model)} choiceLabels={Object.fromEntries(rows.map(r=>[r.model,r.name]))} onChange={chooseModel}/><details className="lab-model-chooser"><summary>叠加对比 · {ids.length} / 4</summary><div>{rows.map(r=><label key={r.model}><input type="checkbox" checked={ids.includes(r.model)} disabled={!ids.includes(r.model)&&ids.length>=4} onChange={()=>toggle(r.model)}/>{r.name}</label>)}</div></details><span className="lab-caption">模型参数已冻结，此处仅回放。</span></div>:<DataComponent id="teaching-assumptions" queryId="skus" kind="custom" title="教学推演参数 · 不计入真实模型对比" sourceRows={[sku]} displayRows={[sku]} variant="card"><div className="lab-demo-sliders"><Slider label="目标库存波动系数" min={0} max={4} step={.05} value={coefficient} onChange={setCoefficient}/><Slider label="初始库存" min={0} max={Math.max(100,Math.ceil(sku.train_mean*10))} step={1} value={initial??sku.initial} onChange={setInitial}/><Slider label="需求缩放倍数" min={.5} max={2} step={.05} value={demandScale} onChange={setDemandScale}/></div><p className="lab-caption">使用该SKU的实际需求路径作反事实推演。目标库存＝最近6期已观察需求均值＋系数×标准差；不是SPO或DFL，也不重新训练任何模型。</p></DataComponent>}
  {error&&<p role="alert" className="lab-method-warning">{error}</p>}
  {!active&&!error&&<div className="lab-loading" role="status">正在解码所选模型的真实库存轨迹…</div>}
  {active&&<>
   <DataComponent id="replay-cumulative-metrics" queryId={mode==='demo'?'skus':scope==='warehouse'?'periods':'trajectories'} title={(mode==='demo'?'教学推演':current.name)+' · 累计至测试第'+period+'期'} kind="metrics" variant="card" sourceRows={mode==='demo'?[sku]:scope==='warehouse'?allPeriod.filter(r=>r.model===ids[0]&&r.period<=period):sources} displayRows={[stats]}><MetricStrip row={stats}/></DataComponent>
   <div ref={host} className="lab-replay-chart">
    <EvidenceChart id="inventory-replay-plot" queryId={mode==='demo'?'skus':scope==='warehouse'?'periods':'trajectories'} title={measures[measure]+'与需求 · '+(scope==='warehouse'&&mode==='research'?'全仓汇总':sku.sku)} sourceRows={mode==='demo'?[sku]:scope==='warehouse'?allPeriod.filter(r=>ids.includes(r.model)):sources} rows={chartRows} variant="card" height={350} headerControls={<Dropdown label="曲线指标" showLabel value={measure} choices={Object.keys(measures)} choiceLabels={measures} onChange={setMeasure}/>}
     spec={{type:'line',x:'周期',y:'真实需求',fields,colors,stackable:false,valueDecimals:0}}>
     {measure==='forecast'&&activeIds.filter(id=>!forecastIds.includes(id)).length>0&&<p className="lab-caption">{activeIds.filter(id=>!forecastIds.includes(id)).map(id=>labels[id]).join('、')}：未接入需求预测输出。真实库存、销量和订货轨迹仍可查看。</p>}
     <div className="lab-timeline"><button className="lab-play" onClick={()=>{if(period===28)setPeriod(1);setPlaying(p=>!p);}} aria-label={playing?'暂停回放':'播放回放'}>{playing?'Ⅱ':'▶'}</button><button disabled={period===1} onClick={()=>{setPlaying(false);setPeriod(p=>p-1);}} aria-label="上一期">‹</button><div className="lab-timeline-track"><Slider label="测试周期" labelPlacement="inline" min={1} max={28} step={1} value={period} onChange={v=>{setPeriod(v);setPlaying(false);}} formatValue={v=>'第 '+v+' / 28 期'}/></div><button disabled={period===28} onClick={()=>{setPlaying(false);setPeriod(p=>p+1);}} aria-label="下一期">›</button><select aria-label="播放速度" value={speed} onChange={e=>setSpeed(Number(e.target.value))}><option value={1500}>0.5×</option><option value={900}>1×</option><option value={350}>2×</option></select></div>
     <div className="lab-export-row"><span>订单下一期到货 · 指标随周期累计重算</span><button onClick={()=>exportSVG(host.current,'库存轨迹')}>导出 SVG</button><button onClick={exportRows}>导出当前轨迹 CSV</button><button onClick={()=>saveFile('补货实验配置.json',JSON.stringify({mode,scope,models:ids,sku:sku.sku,period,measure,coefficient,initial:initial??sku.initial,demandScale},null,2))}>保存实验配置</button></div>
    </EvidenceChart>
   </div>
   <DataComponent id="period-inventory-flow" queryId={mode==='demo'?'skus':scope==='warehouse'?'periods':'trajectories'} title={'第'+period+'期库存流 · '+(mode==='demo'?'教学推演':current.name)} kind="custom" sourceRows={scopedSources} displayRows={[state]} variant="card">
    <div className="lab-inventory-flow">{[['上期期末库存',state.opening,''],['本期到货',state.arrival,'＋'],['实际销量',state.sales,'−'],['本期期末库存',state.ending,'＝']].map(([title,value,sign],i)=><React.Fragment key={title}>{i>0&&<b>{sign}</b>}<div><span>{title}</span><strong>{num(value)}</strong></div></React.Fragment>)}</div>
    <div className="lab-order-note"><span>本期需求 <strong>{num(state.demand)}</strong> · 未满足需求 <strong>{num(state.lost??state.demand-state.sales)}</strong></span><span>本期下单 <strong>{num(state.order)}</strong> → <b>{period===28?'测试期外到货':'第'+(period+1)+'期到货'}</b></span></div>
   </DataComponent>
   <details className="lab-disclosure"><summary>查看各模型截至当前周期的准确指标</summary><DataComponent id="replay-model-values" queryId={mode==='demo'?'skus':scope==='warehouse'?'periods':'trajectories'} title="当前比较模型" kind="table" sourceRows={scopedSources} displayRows={activeIds.map(id=>({model:labels[id],...(scope==='warehouse'&&mode==='research'?allPeriod.find(r=>r.model===id&&r.period===period):cumulative(traces[id],period))}))}><DataTable rowKey="model" rows={activeIds.map(id=>({model:labels[id],...(scope==='warehouse'&&mode==='research'?allPeriod.find(r=>r.model===id&&r.period===period):cumulative(traces[id],period))}))} columns={[{field:'model',label:'模型'},{field:'service',label:'服务水平',renderCell:v=>v.toFixed(2)+'%'},{field:'fill',label:'满足率',renderCell:v=>v===null?'无需求':v.toFixed(2)+'%'},{field:'J',label:'库存需求比',renderCell:v=>v===null?'无需求':v.toFixed(4)}]}/></DataComponent></details>
  </>}
 </>;
}
