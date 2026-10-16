// 真实结果与教学推演分开；浏览器不训练模型，不改变冻结测试结果。
export const palette=['#19605c','#3266b0','#b48232','#92619a','#63827b'];
export const metricNames={service:'服务水平',fill:'满足率 · Fill Rate',J:'库存需求比 J'};
export const pct=n=>Number.isFinite(n)?n.toFixed(2)+'%':'未保存';
export const num=n=>Number.isFinite(n)?n.toLocaleString('zh-CN',{maximumFractionDigits:0}):'—';
export const fmt=(n,key)=>!Number.isFinite(n)?'—':key==='J'?n.toFixed(4):pct(n);
export function front(rows,x,y,goal='higher'){
 const score=(r,k)=>k==='J'?r[k]:k==='service'&&goal==='target'?Math.abs(r[k]-95):-r[k];
 return rows.filter(a=>!rows.some(b=>score(b,x)<=score(a,x)&&score(b,y)<=score(a,y)&&(score(b,x)<score(a,x)||score(b,y)<score(a,y))));
}
const decoded=new Map();
export async function decodePayload(row){
 if(decoded.has(row.model))return decoded.get(row.model);
 const inflate=async value=>{
  if(!value)return null;
  if(!globalThis.DecompressionStream)throw new Error('此浏览器不支持解压，请使用新版Chrome或内置浏览器。');
  const binary=Uint8Array.from(atob(value),c=>c.charCodeAt(0));
  const buffer=await new Response(new Blob([binary]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer();
  return new Float64Array(buffer);
 };
 const result={data:await inflate(row.payload),forecast:await inflate(row.forecast_payload)};
 decoded.set(row.model,result);
 return result;
}
export function skuTrajectory(sku,model,decoded){
 const {demand}=JSON.parse(sku.payload),data=decoded.data,n=6000*28,offset=sku.index*28;
 return demand.map((d,t)=>{
  const ending=data[offset+t],order=data[n+offset+t];
  const opening=t?data[offset+t-1]:sku.initial;
  const arrival=t?data[n+offset+t-1]:0;
  const prediction=decoded.forecast?.[offset+t];
  return {model,period:t+1,demand:d,opening,arrival,available:opening+arrival,ending,order,
    sales:Math.min(d,opening+arrival),lost:Math.max(d-opening-arrival,0),forecast:Number.isFinite(prediction)?prediction:null};
 });
}
export function cumulative(rows,through=28){
 const a=rows.filter(r=>r.period<=through);
 const demand=a.reduce((s,r)=>s+r.demand,0),sales=a.reduce((s,r)=>s+r.sales,0),stock=a.reduce((s,r)=>s+r.ending,0);
 return {service:100*a.filter(r=>r.ending>0).length/a.length,fill:demand?100*sales/demand:null,J:demand?stock/demand:null};
}
export function teaching(sku,k=1.65,initial=sku.initial,multiplier=1){
 const p=JSON.parse(sku.payload),history=[...p.history],out=[];
 let inv=initial,pending=0;
 for(let t=0;t<28;t++){
  const demand=p.demand[t]*multiplier,opening=inv,arrival=pending;
  const available=opening+arrival,sales=Math.min(demand,available);
  inv=available-sales;history.push(demand);
  const h=history.slice(-6),mu=h.reduce((s,x)=>s+x,0)/h.length;
  const sd=Math.sqrt(h.reduce((s,x)=>s+(x-mu)**2,0)/h.length);
  pending=Math.ceil(Math.max(mu+k*sd-inv,0));
  out.push({model:'teaching',period:t+1,demand,opening,arrival,available,sales,ending:inv,order:pending,lost:demand-sales,forecast:mu});
 }
 return out;
}
export function saveFile(name,content,type='application/json'){
 const url=URL.createObjectURL(new Blob([content],{type}));
 const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);
}
export function saveCSV(rows,name='补货轨迹.csv'){
 if(!rows.length)return;
 const keys=Object.keys(rows[0]);
 const escape=v=>'"'+String(v??'').replaceAll('"','""')+'"';
 saveFile(name,'\ufeff'+[keys,...rows.map(r=>keys.map(k=>r[k]))].map(r=>r.map(escape).join(',')).join('\r\n'),'text/csv;charset=utf-8');
}
export function exportSVG(root,name){
 const source=root?.querySelector('svg');if(!source)return;
 const clone=source.cloneNode(true),a=[source,...source.querySelectorAll('*')],b=[clone,...clone.querySelectorAll('*')];
 a.forEach((node,i)=>{
  const computed=getComputedStyle(node);
  for(const key of ['fill','stroke','font-family','font-size','font-weight','color'])b[i].style.setProperty(key,computed.getPropertyValue(key));
 });
 clone.setAttribute('xmlns','http://www.w3.org/2000/svg');
 saveFile(name+'.svg',new XMLSerializer().serializeToString(clone),'image/svg+xml');
}
export const routes={
 'Baseline':['逐SKU历史需求','均值与波动','校准目标库存','扣减库存下单'],
 '时间序列':['历史需求','时序预测','补货规则','库存回放'],
 'ML · PTO':['历史＋滞后特征','预测误差训练','校准目标库存','库存回放'],
 'SPO · 决策学习':['历史＋库存状态','有限动作成本预测','SPO风格决策损失','选择补货动作'],
 '概率网络 · DFL':['概率需求网络','8个需求情景','有限步内层优化','7期决策损失反馈'],
 '融合与选择':['时间顺序OOF预测','学习融合权重','统一补货校准','冻结测试'],
 '鲁棒优化与经验策略':['历史需求与特征','分组与残差分布','鲁棒水位或经验分位','服务约束补货']
 ,'Kristina · 鲁棒优化':['训练需求与聚类','基础风险水位','累计服务预算反馈','执行最近一期订单']
 ,'扩展 · 在线与MPC':['历史需求与库存','滚动预测或在线更新','多期优化与校准','执行最近一期订单']
 ,'扩展 · 融合与选择':['时间顺序OOF预测','学习融合权重','统一补货校准','冻结测试']
};
