// PCHIP, Fritsch–Butland weighted harmonic slopes. No extrapolation or new observed rows.
// Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html
export function frontierCurve(rows,xKey,yKey){
 const nodes=[...rows].map(r=>({x:r[xKey],y:r[yKey]})).filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)).sort((a,b)=>a.x-b.x);
 if(nodes.length<3)return {points:[],reason:'至少需要3个不同横坐标的前沿点；当前保留原始点或折线'};
 if(nodes.some((p,i)=>i&&p.x<=nodes[i-1].x))return {points:[],reason:'前沿存在相同横坐标、不同纵坐标，不能拟合单值曲线'};
 const h=nodes.slice(1).map((p,i)=>p.x-nodes[i].x),d=h.map((v,i)=>(nodes[i+1].y-nodes[i].y)/v);
 if(d.some(v=>v>0)&&d.some(v=>v<0))return {points:[],reason:'当前前沿在原始坐标中不单调（例如服务目标两侧）；不绘制容易误导的单调曲线'};
 const edge=(h0,h1,d0,d1)=>{let m=((2*h0+h1)*d0-h0*d1)/(h0+h1);if(Math.sign(m)!==Math.sign(d0))return 0;if(Math.sign(d0)!==Math.sign(d1)&&Math.abs(m)>3*Math.abs(d0))return 3*d0;return m;};
 const slopes=[edge(h[0],h[1],d[0],d[1])];
 for(let i=1;i<nodes.length-1;i++){const a=d[i-1],b=d[i],w1=2*h[i]+h[i-1],w2=h[i]+2*h[i-1];slopes.push(a*b<=0?0:(w1+w2)/(w1/a+w2/b));}
 slopes.push(edge(h.at(-1),h.at(-2),d.at(-1),d.at(-2)));
 const points=[];
 for(let i=0;i<h.length;i++)for(let j=0;j<24;j++){
  const t=j/24,t2=t*t,t3=t2*t;
  points.push({x:nodes[i].x+t*h[i],y:(2*t3-3*t2+1)*nodes[i].y+(t3-2*t2+t)*h[i]*slopes[i]+(-2*t3+3*t2)*nodes[i+1].y+(t3-t2)*h[i]*slopes[i+1]});
 }
 points.push(nodes.at(-1));return {points,reason:'PCHIP保形插值；只在已观测前沿点之间，不向区间外推'};
}
