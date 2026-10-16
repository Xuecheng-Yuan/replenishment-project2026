import React,{useMemo,useState} from 'react';
import {EvidenceChart,DataComponent,DataTable,Dropdown,Section,RichNarrative} from '../../data-app-public.jsx';
import ZoomableScatter from './ZoomableScatter.jsx';
import {front,metricNames,fmt,pct} from './lab-data.js';

export function MetricStrip({row,label}){
 return <div className="lab-metrics" data-reviewed-rows aria-label={label||row.name}>
  {[['service','服务水平','按商品—周期计数'],['J','库存需求比','总期末库存 / 总需求 ↓'],['fill','满足率 · Fill Rate','按需求数量计量']].map(([k,title,hint],i)=><div className="lab-metric" key={k} data-tone={i}><span>{title}</span><strong>{fmt(row[k],k)}</strong><small>{hint}</small></div>)}
 </div>;
}

export function Tradeoff({rows,onChoose,selected,onSelect,compact=false}){
 const [family,setFamily]=useState('all'),[x,setX]=useState('J'),[y,setY]=useState('service'),[goal,setGoal]=useState('higher');
 const scope=rows.filter(r=>family==='all'||r.family===family);
 const current=scope.find(r=>r.model===selected)||scope.find(r=>r.model==='turnover_dfl')||scope[0];
 const aliases={J:'inventory_demand_ratio',service:'ready_percent',fill:'fill_percent'};
 const original=scope.map(r=>({...r,inventory_demand_ratio:r.J,ready_percent:r.service,fill_percent:r.fill}));
 const pareto=new Set(front(scope,x,y,goal).map(r=>r.model));
 const ranked=(row,key)=>1+scope.filter(other=>key==='J'?other[key]<row[key]:key==='service'&&goal==='target'?Math.abs(other[key]-95)<Math.abs(row[key]-95):other[key]>row[key]).length;
 const plotted=original.map(r=>({...r,'横轴位置':r[x],'纵轴位置':r[y],
  '横轴原值':fmt(r[x],x),'纵轴原值':fmt(r[y],y),'横轴名次':ranked(r,x),'纵轴名次':ranked(r,y),
  '标记':r.model===current.model?'当前模型':'其他方案'}));
 return <div className="lab-tradeoff">
  <Section id={compact?'overview-comparison-head':'comparison-head'} title={compact?'模型位置与库存—服务权衡':'全模型指标比较'} spacing="none" filters={<Dropdown label="方案方向" showLabel value={family} choices={['all',...new Set(rows.map(r=>r.family))]} choiceLabels={{all:'全部方向'}} onChange={setFamily}/> }>
   <div className="lab-axis-controls">
    <Dropdown label="横坐标" showLabel value={x} choices={Object.keys(metricNames).filter(k=>k!==y)} choiceLabels={metricNames} onChange={setX}/>
    <button className="lab-icon-button" aria-label="交换横纵坐标" onClick={()=>{setX(y);setY(x);}}>⇄</button>
    <Dropdown label="纵坐标" showLabel value={y} choices={Object.keys(metricNames).filter(k=>k!==x)} choiceLabels={metricNames} onChange={setY}/>
    <Dropdown label="服务优化方向" showLabel value={goal} choices={['higher','target']} choiceLabels={{higher:'越高越好',target:'越接近95%越好'}} onChange={setGoal}/>
   </div>
   <EvidenceChart id={compact?'overview-scatter':'all-scatter'} queryId="models" title={metricNames[x]+' × '+metricNames[y]} rows={plotted} sourceRows={scope} height={430} variant="card"
    renderPlot={(plot,chart)=><ZoomableScatter id={compact?'overview-scatter':'all-scatter'} rows={plotted} selected={current.model} xKey={aliases[x]} yKey={aliases[y]} scale="raw" chart={chart} fallbackPlot={plot} onHighlight={onSelect} onChoose={onChoose} paretoIds={pareto}/>}
    spec={{type:'scatter',x:'横轴位置',y:'纵轴位置',series:'标记',xLabel:metricNames[x],yLabel:metricNames[y],showLegend:true,colors:{'当前模型':'var(--accent)','其他方案':'var(--chart-2)'}}}/>
  </Section>
  <DataComponent id={compact?'overview-selection':'comparison-selection'} queryId="models" title={current.name} sourceRows={[current]} displayRows={[current]} variant="card" kind="metrics">
   <div className="lab-selection-head"><span className="lab-badge">{current.family}</span><small>{current.owner}</small></div>
   <MetricStrip row={current}/>
   <div className="lab-inline-actions"><button onClick={()=>onChoose(current.model)}>查看模型原理 ↗</button><span>{pareto.has(current.model)?'位于当前两指标的经验前沿':'未位于当前两指标的经验前沿'}</span></div>
  </DataComponent>
  <p className="lab-caption">前沿由当前筛选范围的真实测试点计算。平滑曲线仅为经验插值，不代表理论最优；缩放不改变计算范围。相同评价口径不等于相同训练协议。</p>
  {!compact&&<>
   <Section id="service-fill-head" title="服务水平与满足率：相同百分比，不同含义" spacing="section">
    <RatePairs rows={scope} onChoose={onChoose}/>
   </Section>
   <details className="lab-disclosure"><summary>查看准确数值与来源</summary><DataComponent id="all-exact-values" queryId="models" title="测试集准确指标" kind="table" sourceRows={scope} displayRows={scope}>
    <DataTable rows={scope} rowKey="model" pageSize={12} onRowSelect={r=>onChoose(r.model)} columns={[
     {field:'name',label:'模型'},{field:'owner',label:'负责人'},{field:'service',label:'服务水平',renderCell:pct},
     {field:'fill',label:'满足率',renderCell:pct},{field:'J',label:'库存需求比 ↓',renderCell:v=>v.toFixed(6)}]}/>
   </DataComponent></details>
  </>}
 </div>;
}

function RatePairs({rows,onChoose}){
 const [range,setRange]=useState('focus');
 const low=range==='focus'?Math.max(0,Math.floor(Math.min(...rows.flatMap(r=>[r.service,r.fill])))-1):0;
 const pos=v=>Math.max(0,Math.min(100,(v-low)/(100-low)*100));
 return <DataComponent id="paired-rates" queryId="models" kind="custom" title="服务水平与满足率对照" variant="card" sourceRows={rows} displayRows={rows}
  headerControls={<Dropdown label="坐标范围" showLabel value={range} choices={['focus','full']} choiceLabels={{focus:'聚焦差异',full:'完整0%—100%'}} onChange={setRange}/>}>
  <div className="lab-pair-key"><span><i/>服务水平</span><span><i/>满足率</span><small>局部横轴 {low}%—100%</small></div>
  <div className="lab-pair-axis"><span>{low}%</span><span>{((low+100)/2).toFixed(0)}%</span><span>100%</span></div>
  <div className="lab-pairs" data-reviewed-rows>{rows.map(r=><button className="lab-pair" key={r.model} onClick={()=>onChoose(r.model)} title={r.name+'：服务水平'+pct(r.service)+'，满足率'+pct(r.fill)}>
   <span>{r.name}</span><div className="lab-pair-track"><b style={{left:pos(Math.min(r.service,r.fill))+'%',width:Math.abs(pos(r.service)-pos(r.fill))+'%'}}/><i style={{left:pos(r.service)+'%'}}/><i className="fill" style={{left:pos(r.fill)+'%'}}/></div><span className="lab-pair-numbers">{pct(r.service)} <em>/</em> {pct(r.fill)}</span>
  </button>)}</div>
 </DataComponent>;
}
