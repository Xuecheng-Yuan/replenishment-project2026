import React,{useEffect,useRef,useState} from "react";
import {ResponsiveContainer,ScatterChart,Scatter,CartesianGrid,XAxis,YAxis,ReferenceLine} from "recharts";
import {ChartMark,ChartTooltip,RichNarrative} from "../../data-app-public.jsx";
import {frontierCurve} from "./frontier-curve.js";

const pointX="横轴位置",pointY="纵轴位置";
const span=a=>a[1]-a[0];
function fitted(values){const lo=Math.min(...values),hi=Math.max(...values),pad=Math.max((hi-lo)*.08,.02);return [Math.max(0,lo-pad),hi+pad];}
function clampDomain(a,limits){let s=Math.min(span(a),span(limits));let lo=Math.max(limits[0],Math.min(a[0],limits[1]-s));return [lo,lo+s];}
export default function ZoomableScatter({rows,selected,reference,onHighlight,onReference,xKey,yKey,scale,chart,fallbackPlot,onChoose,id,paretoIds}){
 const ref=useRef(null),drag=useRef(null);
 const rate=key=>["ready_percent","fill_percent"].includes(key);
 const fit=(key,values)=>{const domain=fitted(values);return rate(key)&&scale==="raw"?[Math.max(0,domain[0]),Math.min(100,domain[1])]:domain;};
 const all=()=>({x:fit(xKey,rows.map(r=>r[pointX])),y:fit(yKey,rows.map(r=>r[pointY]))});
 const near=()=>{
  if(yKey==="ready_percent"&&scale!=="rank"){
   const close=rows.filter(r=>Math.abs(r.ready_percent-95)<=3);
   if(!close.length)return all();
   return {x:fit(xKey,close.map(r=>r[pointX])),y:scale==="log"?[Math.log10(93),Math.log10(99)]:[92,98]};
  }
  if(xKey==="ready_percent"&&scale!=="rank"){
   const close=rows.filter(r=>Math.abs(r.ready_percent-95)<=3);
   if(!close.length)return all();
   return {x:scale==="log"?[Math.log10(93),Math.log10(99)]:[92,98],y:fit(yKey,close.map(r=>r[pointY]))};
  }
  return all();
 };
 const [view,setView]=useState(near),[moving,setMoving]=useState(false),[curveMode,setCurveMode]=useState("points");
 const populationKey=rows.map(r=>r.model).sort().join("|");
 useEffect(()=>{setView(near());},[xKey,yKey,scale,populationKey]);
 const transformed95=scale==="log"?Math.log10(96):95;
 const targetAxis=scale!=="rank"?(yKey==="ready_percent"?"y":xKey==="ready_percent"?"x":null):null;
 const targetInView=targetAxis&&transformed95>=view[targetAxis][0]&&transformed95<=view[targetAxis][1];
 const inView=r=>r[pointX]>=view.x[0]&&r[pointX]<=view.x[1]&&r[pointY]>=view.y[0]&&r[pointY]<=view.y[1];
 const visibleRows=rows.filter(inView),current=rows.find(r=>r.model===selected);
 const limits=()=>{const full=all();return {x:rate(xKey)&&scale==="raw"?[0,100]:[0,Math.max(full.x[1]*1.2,view.x[1])],y:rate(yKey)&&scale==="raw"?[0,100]:[0,Math.max(full.y[1]*1.2,view.y[1])]};};
 const front=paretoIds?rows.filter(r=>paretoIds.has(r.model)).sort((a,b)=>a[pointX]-b[pointX]||a[pointY]-b[pointY]):[];
 const curve=frontierCurve(front,pointX,pointY);
 const smooth=curveMode==="smooth"&&curve.points.length>0;
 const tick=(value,key)=>Number(value.toFixed(key==="inventory_demand_ratio"?3:2))+(rate(key)&&scale==="raw"?"%":"");
 const move=(dx,dy)=>{let bounds=limits();setView(v=>({x:clampDomain(v.x.map(n=>n+dx*span(v.x)),bounds.x),y:clampDomain(v.y.map(n=>n+dy*span(v.y)),bounds.y)}));};
 const zoom=factor=>{const bounds=limits();setView(v=>{const resize=(a,l)=>{const center=(a[0]+a[1])/2,s=Math.max(.00001,span(a)*factor);return clampDomain([center-s/2,center+s/2],l);};return {x:resize(v.x,bounds.x),y:resize(v.y,bounds.y)};});};
 const locate=()=>setView(v=>({x:clampDomain([current[pointX]-span(v.x)/2,current[pointX]+span(v.x)/2],limits().x),y:clampDomain([current[pointY]-span(v.y)/2,current[pointY]+span(v.y)/2],limits().y)}));
 const start=e=>{if(e.button!==0||e.target.closest?.('button'))return;drag.current={x:e.clientX,y:e.clientY,view};ref.current.setPointerCapture(e.pointerId);setMoving(true);};
 const pan=e=>{if(!drag.current)return;const b=ref.current.getBoundingClientRect(),dx=(e.clientX-drag.current.x)/Math.max(60,b.width-94),dy=(e.clientY-drag.current.y)/330,bounds=limits();setView({x:clampDomain(drag.current.view.x.map(n=>n-dx*span(drag.current.view.x)),bounds.x),y:clampDomain(drag.current.view.y.map(n=>n+dy*span(drag.current.view.y)),bounds.y)});};
 const stop=()=>{drag.current=null;setMoving(false);};
 const mark=({cx,cy,payload})=>{
  if(!Number.isFinite(cx)||!Number.isFinite(cy)||!inView(payload))return null;
  const active=payload.model===selected,paired=payload.model===reference,onFront=paretoIds?.has(payload.model);
  const shortName=payload.name.length>17?payload.name.slice(0,16)+"…":payload.name;
  const labelOffset=payload[pointY]>view.y[0]+.8*span(view.y)?22:payload[pointY]<view.y[0]+.2*span(view.y)?-15:paired?21:-15;
  const actions=[...(onHighlight&&!active?[{label:"突出显示"+payload.name,onSelect:()=>onHighlight(payload.model)}]:[]),...(onReference&&!active&&!paired?[{label:"将"+payload.name+"设为对照",onSelect:()=>onReference(payload.model)}]:[]),...(onChoose?[{label:"查看"+payload.name+"详解",onSelect:()=>onChoose(payload.model)}]:[])];
  return <g>
   {(active||paired)&&<text x={cx+14} y={cy+labelOffset} textAnchor={payload[pointX]>(view.x[0]+view.x[1])/2?"end":"start"} transform={payload[pointX]>(view.x[0]+view.x[1])/2?"translate(-28 0)":undefined} fill={active?"var(--accent)":"var(--chart-2)"} fontSize={12} fontWeight={600} stroke="var(--surface)" strokeWidth={4} paintOrder="stroke" pointerEvents="none">{active?"● ":"■ "}{shortName}</text>}
   <foreignObject x={cx-10} y={cy-10} width={20} height={20} overflow="visible">
   <ChartMark className={"zoom-point"+(active?" current":"")+(paired?" reference":"")+(onFront?" pareto-point":"")} style={{background:onFront&&!active&&!paired?"var(--surface)":chart.colors?.[payload[chart.series]]??(active?"var(--accent)":"var(--secondary)"),...(onFront?{outline:"2px solid var(--chart-4)",outlineOffset:2}:{})}}
    aria-label={payload.name+"，"+payload["横轴原值"]+"，"+payload["纵轴原值"]}
    tooltip={<ChartTooltip active label={payload.name} details={[{label:chart.xLabel,value:payload["横轴原值"]},{label:chart.yLabel,value:payload["纵轴原值"]},{label:"横轴名次",value:payload["横轴名次"]+" / "+rows.length},{label:"纵轴名次",value:payload["纵轴名次"]+" / "+rows.length},...(paretoIds?[{label:"帕累托前沿",value:onFront?"是":"否"}]:[])]}/>}
    context={{kind:"chart",chartType:"scatter",label:payload.name,value:payload[pointY],row:payload,actions}}/>
  </foreignObject></g>;
 };
 // Respect saved chart edits. Other families/bindings use the shared renderer.
 if(chart.type!=="scatter"||chart.x!==pointX||chart.y!==pointY||chart.series!=="标记")return fallbackPlot;
 return <div className="zoom-scatter">
  <div className="zoom-tools" aria-label="散点图缩放和平移">
   <button disabled={!targetAxis} onClick={()=>setView(near())}>聚焦95%附近</button><button onClick={()=>setView(all())}>显示全部{rows.length}方案</button>
   <button aria-label="放大图表" onClick={()=>zoom(.7)}>＋ 放大</button><button aria-label="缩小图表" onClick={()=>zoom(1/.7)}>− 缩小</button>
   <button onClick={locate}>定位当前模型</button>
   {reference&&<button onClick={()=>{const pair=rows.filter(r=>[selected,reference].includes(r.model));setView({x:fit(xKey,[...pair.map(r=>r[pointX]),...(targetAxis==="x"?[transformed95]:[])]),y:fit(yKey,[...pair.map(r=>r[pointY]),...(targetAxis==="y"?[transformed95]:[])])});}}>聚焦两个模型</button>}
   <div className="zoom-arrows"><button aria-label="向左查看" onClick={()=>move(-.4,0)}>←</button><button aria-label="向右查看" onClick={()=>move(.4,0)}>→</button><button aria-label="向上查看" onClick={()=>move(0,.4)}>↑</button><button aria-label="向下查看" onClick={()=>move(0,-.4)}>↓</button></div>
   {paretoIds&&<label className="frontier-mode">前沿显示 <select aria-label="前沿显示" value={curveMode} onChange={e=>setCurveMode(e.target.value)}><option value="points">实际点与折线</option><option value="smooth">经验平滑曲线（非理论最优）</option></select></label>}
  </div>
  <div className="zoom-window-status" data-reviewed-rows aria-live="polite">当前视野 {visibleRows.length} / {rows.length} 个方案 · 横轴 {tick(view.x[0],xKey)}—{tick(view.x[1],xKey)} · 纵轴 {tick(view.y[0],yKey)}—{tick(view.y[1],yKey)}{paretoIds?" · 前沿 "+front.length+" 个方案":""}{!inView(current)?" · 当前模型在视野外，可点击“定位当前模型”":""}{reference&&!inView(rows.find(r=>r.model===reference))?" · 对照模型在视野外，可点击“聚焦两个模型”":""}{targetAxis&&!targetInView?" · 95%参考线在视野外":""}</div>
  <div ref={ref} className={"zoom-plot"+(moving?" moving":"")} data-chart-interaction-root data-zoom-x={view.x.join(",")} data-zoom-y={view.y.join(",")}
   onPointerDown={start} onPointerMove={pan} onPointerUp={stop} onPointerCancel={stop}>
   <ResponsiveContainer width="100%" height={430}>
    <ScatterChart margin={{top:30,right:20,bottom:36,left:8}} accessibilityLayer>
     <CartesianGrid stroke="var(--border-strong)" strokeDasharray="3 5"/>
     <XAxis type="number" dataKey={pointX} domain={view.x} allowDataOverflow tickCount={5} tickFormatter={n=>tick(n,xKey)} stroke="var(--secondary)" tickLine={false} height={42}/>
     <YAxis type="number" dataKey={pointY} domain={view.y} allowDataOverflow tickCount={5} tickFormatter={n=>tick(n,yKey)} stroke="var(--secondary)" tickLine={false} width={66}/>
     {targetInView&&<ReferenceLine {...{[targetAxis]:transformed95}} ifOverflow="hidden" stroke="var(--text)" strokeWidth={1.5} strokeDasharray="7 5" label={{value:"服务水平95%",position:"insideTopRight",fill:"var(--text)",fontSize:13}}/>}
     {!smooth&&front.slice(1).map((row,i)=><ReferenceLine key={front[i].model+"-"+row.model} className="pareto-frontier-segment" segment={[{x:front[i][pointX],y:front[i][pointY]},{x:row[pointX],y:row[pointY]}]} ifOverflow="hidden" stroke="var(--chart-4)" strokeWidth={1.5} strokeDasharray="4 3"/>)}
     {smooth&&<Scatter className="empirical-frontier-curve" data={curve.points.map(p=>({[pointX]:p.x,[pointY]:p.y}))} shape={()=>null} line={{stroke:"var(--chart-4)",strokeWidth:1.5,strokeDasharray:"6 4"}} lineType="joint" fill="var(--chart-4)" isAnimationActive={false}/>}
     <Scatter name={rows.length+"个方案"} data={rows} shape={mark} isAnimationActive={false}/>
    </ScatterChart>
   </ResponsiveContainer>
  </div>
  <div className="zoom-axis-labels"><span>横轴：{chart.xLabel}</span><span>纵轴：{chart.yLabel}</span></div>
  {paretoIds&&curveMode==="smooth"&&<p className="version-dot-note" data-reviewed-rows>{curve.reason}。曲线不是新实验点或置信区间，不表示插值位置有可执行策略，也没有证明理论最优。</p>}
  {paretoIds&&<details className="reading-disclosure compact-disclosure"><summary>平滑曲线与理论前沿有什么区别？</summary><RichNarrative id={id+":curve-method"} value="这里可选的是PCHIP保形三次插值：连接当前有限方案的经验前沿点，保持单调，避免普通高阶多项式产生虚假的波峰，不外推。\n\n理论前沿需要定义完整策略集合、固定服务目标并逐点求解（或证明）最小库存；现有不同模型的一组测试成绩不足以证明这条边界。真正验证连续权衡应在训练/验证集扫描服务目标或风险参数，冻结方案后再测试，不能拿最终测试集反复调参。"/><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html" target="_blank" rel="noreferrer">PCHIP方法说明（SciPy官方文档）</a></details>}
  {chart.showLegend!==false&&<div className="zoom-legend"><span><i/>其他方案</span><span><i className="current"/>{current.name}</span>{reference&&<span><i className="reference"/>{rows.find(r=>r.model===reference)?.name}</span>}{paretoIds&&<span><i className="pareto"/>帕累托前沿</span>}{targetAxis&&<span><i className="target"/>95%参考线</span>}</div>}
  <details className="reading-disclosure compact-disclosure"><summary>缩放、平移与95%参考线的说明</summary><RichNarrative id={id+":zoom-guide"} value={"拖动图内空白处可以平移，也可用箭头按钮；“＋ / −”调整视野，“显示全部"+rows.length+"方案”恢复全景。有邻近点时默认聚焦95%上下3个百分点，否则显示全范围；**展示窗口不是达标容差**。坐标轴裁剪范围在上方明示；隐藏在视野外的方案仍保留在数据中。"}/></details>
 </div>;
}
