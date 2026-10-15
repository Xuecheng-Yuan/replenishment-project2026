// One builder for both six-policy deliveries. Use the bundled artifact-tool.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {loadArtifactTool} from './artifact_runtime.mjs';
const {Workbook,SpreadsheetFile}=await loadArtifactTool();
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const labels={DRO_cluster:'特征聚类DRO',DRO_global:'不分组DRO',DRO_category:'爆品/长尾DRO',sS:'传统(s,S)',SAA:'零半径经验模型',RO:'传统区间RO'};
const read=async(out,name)=>{
 const lines=(await fs.readFile(path.join(out,name),'utf8')).replace(/^\uFEFF/,'').trimEnd().split(/\r?\n/);
 const cols=lines.shift().split(',');
 return lines.map(l=>Object.fromEntries(l.split(',').map((v,i)=>[cols[i],v!==''&&Number.isFinite(Number(v))?Number(v):v])));
};
const service95=process.argv.includes('--service95');
const full60=process.argv.includes('--full60')||service95;
for(const version of (full60?['rolling_dro_v3']:['demand_fill_v3','rolling_dro_v3'])){
 const out=service95?path.join(root,'experiments/service95_revision/outputs_v4'):full60?path.join(root,'experiments/full60_revision/new_positive/outputs_v4'):path.join(root,'experiments',version,'with_ro/outputs_v4');
 const old=version==='demand_fill_v3';
 const data=(await read(out,'summary.csv')).filter(r=>r.scope==='all_28_periods');
 const wb=Workbook.create();
 function sheet(name,title,headers,rows){
  const sh=wb.worksheets.add(name);sh.showGridLines=false;
  sh.getRange('A2').values=[[title]];sh.getRange('A2').format.font={name:'Microsoft YaHei',size:15,bold:true,color:'#17324D'};
  const n=headers.length, count=rows.length;
  sh.getRangeByIndexes(3,0,1,n).values=[headers];
  sh.getRangeByIndexes(3,0,1,n).format={fill:'#17324D',font:{name:'Microsoft YaHei',size:10,bold:true,color:'#FFFFFF'},rowHeight:36,wrapText:true,verticalAlignment:'center'};
  for(let k=0;k<count;k+=300){const part=rows.slice(k,k+300);sh.getRangeByIndexes(k+4,0,part.length,n).values=part;}
  sh.getRangeByIndexes(4,0,count,n).format={font:{name:'Microsoft YaHei',size:10,color:'#17324D'},rowHeight:21,verticalAlignment:'center'};
  sh.getRangeByIndexes(4,0,count,n).setNumberFormat('#,##0.00');
  sh.getRangeByIndexes(0,0,count+5,n).format.columnWidth=18;
  sh.getRange('A:A').format.columnWidth=24;
  sh.freezePanes.freezeRows(4);
  return sh;
 }
 const rows=data.map(r=>[labels[r.method],r.sales_units,r.demand_units,null,r.positive_stock_sku_periods,r.total_sku_periods,null,r.ending_inventory_units,null,null]);
 const sh=sheet('结论',old?'需求满足率版：六种补货方法':'正库存天数版：六种补货方法',
  ['方法','销量','需求','需求满足率','正库存商品天','全部商品天','正库存天数比例','累计期末库存','库存需求比↓','服务要求'],rows);
 for(let i=0;i<6;i++){
  const r=i+5;
  sh.getRange(`D${r}`).formulas=[[`=B${r}/C${r}`]];
  sh.getRange(`G${r}`).formulas=[[`=E${r}/F${r}`]];
  sh.getRange(`I${r}`).formulas=[[`=H${r}/C${r}`]];
  sh.getRange(`J${r}`).formulas=[[old?`=IF(D${r}>=95%,"达到95%","低于95%")`:`=IF(AND(G${r}>=95%,G${r}<=95.5%),"范围内","范围外")`]];
 }
 sh.getRange('D5:D10').setNumberFormat('0.0000%');sh.getRange('G5:G10').setNumberFormat('0.0000%');sh.getRange('I5:I10').setNumberFormat('0.000000');sh.getRange('B5:C10').setNumberFormat('#,##0');sh.getRange('E5:F10').setNumberFormat('#,##0');sh.getRange('H5:H10').setNumberFormat('#,##0');
 sh.getRange('J5:J10').format.horizontalAlignment='center';
 sh.getRange('A12').values=[['服务定义：'+(old?'销量÷需求≥95%':'期末库存>0的商品－天数÷全部商品－天数，在95%–95.5%之间')]];
 sh.getRange('A14').values=[['库存需求比＝累计期末库存÷累计需求，越低越好。6000商品，28天；两项服务指标同时报告。']];
 sh.getRange('A16').values=[['来源：同目录summary.csv、frozen_parameters.json及detail_*.csv.gz；完整数学模型见上一级REPORT.md。']];
 if(full60)sh.getRange('A18').values=[[service95?'训练29–60拟合、1–28重启留出选反馈；三轮前缀诊断；全部60期最终拟合，独立测试28期。':'三轮1–24→25–36、1–36→37–48、1–48→49–60合并选参，最终60期拟合，独立测试文件28期。']];
 const segment=await read(out,'summary_by_segment.csv');
 const seg=sheet('爆品长尾结果','全部爆品和全部长尾品的独立测试表现',
  ['方法','类别','SKU数','销量','需求','需求满足率','正库存商品天','全部商品天','正库存比例','累计期末库存','库存需求比'],
  segment.map(r=>[labels[r.method],r.segment,r.skus,r.sales_units,r.demand_units,null,r.positive_stock_sku_periods,r.total_sku_periods,null,r.ending_inventory_units,null]));
 for(let i=0;i<segment.length;i++){
  const r=i+5;seg.getRange(`F${r}`).formulas=[[`=D${r}/E${r}`]];seg.getRange(`I${r}`).formulas=[[`=G${r}/H${r}`]];seg.getRange(`K${r}`).formulas=[[`=J${r}/E${r}`]];
 }
 seg.getRange('F5:F16').setNumberFormat('0.0000%');seg.getRange('I5:I16').setNumberFormat('0.0000%');seg.getRange('K5:K16').setNumberFormat('0.000000');
 seg.getRange('C5:E16').setNumberFormat('#,##0');seg.getRange('G5:H16').setNumberFormat('#,##0');seg.getRange('J5:J16').setNumberFormat('#,##0');
 const period=await read(out,'results_by_period.csv');
 const ps=sheet('逐期结果','逐期服务和库存', ['方法','周期','需求满足率','正库存天数比例','库存需求比','损失需求','订单量'],period.map(r=>[labels[r.method],r.period,r.fill_rate,r.positive_stock_ratio,r.inventory_demand_ratio,r.lost_units,r.order_units]));
 ps.getRange('C5:D172').setNumberFormat('0.0000%');ps.getRange('E5:E172').setNumberFormat('0.000000');
 ps.getRange('B5:B172').setNumberFormat('0');ps.getRange('B5:B172').format.horizontalAlignment='center';ps.getRange('F5:G172').setNumberFormat('#,##0');
 const methods=Object.keys(labels);
 const headers=['周期','item_id','类别','聚类','需求','预测H1','预测H2','预测H3','预测H4',...methods.map(m=>labels[m]+'订单'),...methods.map(m=>labels[m]+'损失'),...methods.map(m=>labels[m]+'期末库存')];
 const toRows=rs=>rs.map(r=>[r.period,String(r.item_id),r.category,r.cluster,r.actual_demand,r.forecast_h1,r.forecast_h2,r.forecast_h3,r.forecast_h4,...methods.map(m=>r['order_'+m]),...methods.map(m=>r['lost_'+m]),...methods.map(m=>r['ending_'+m])]);
 for(const [name,title,file] of [['最近一期全部SKU','第28期全部商品：预测与六种决策','latest_decisions.csv'],['完整周期样本','前12个商品完整28期：预测与六种决策','sample_decisions.csv']]){
  const records=await read(out,file);const detail=sheet(name,title,headers,toRows(records));const last=records.length+4;
  detail.getRange(`A5:A${last}`).setNumberFormat('0');detail.getRange(`A5:B${last}`).format.horizontalAlignment='center';detail.getRange(`B5:B${last}`).setNumberFormat('@');
  detail.getRange(`D5:E${last}`).setNumberFormat('#,##0');detail.getRange(`J5:AA${last}`).setNumberFormat('#,##0');
 }
 const val=service95?await read(out,'candidate_validation.csv'):full60?await read(path.join(root,'experiments/full60_revision/outputs'),'candidate_validation.csv'):await read(out,'ro_validation.csv');
 if(service95){
  const chosen=JSON.parse(await fs.readFile(path.join(out,'frozen_parameters.json'),'utf8')).selected;
  const foldScores=await read(out,'candidate_validation_by_fold.csv');
  const rs=foldScores.filter(r=>r.candidate===chosen[r.method].candidate);
  const fd=sheet('训练验证','已选策略：28期重启留出用于选参，三轮12期前缀用于诊断',
   ['方法','轮次','情景','拟合时期','验证时期','计分时期','正库存比例','需求满足率','库存需求比'],
   rs.map(r=>[labels[r.method],r.fold,r.case==='reset'?'周期重启':'日历延续',r.fit_periods,r.validation_periods,r.score_periods,r.positive_stock_ratio,r.fill_rate,r.inventory_demand_ratio]));
  fd.getRange('B5:B46').setNumberFormat('0');fd.getRange('G5:H46').setNumberFormat('0.0000%');fd.getRange('I5:I46').setNumberFormat('0.000000');
  const lib=sheet('策略候选','108个反馈候选：训练28期重启P在95%–95.5%内最小化J',
   ['方法','候选号','正库存比例','需求满足率','库存需求比','近期窗口','训练先验','响应偏置'],
   val.map(r=>[labels[r.method],r.candidate,r.positive_stock_ratio,r.fill_rate,r.inventory_demand_ratio,r.response_window,r.response_prior,r.response_bias]));
  lib.getRange(`B5:B${val.length+4}`).setNumberFormat('0');lib.getRange(`C5:D${val.length+4}`).setNumberFormat('0.0000%');lib.getRange(`E5:E${val.length+4}`).setNumberFormat('0.000000');lib.getRange(`F5:G${val.length+4}`).setNumberFormat('0');lib.getRange(`H5:H${val.length+4}`).setNumberFormat('0.00');
  const feedback=await read(out,'service_feedback.csv');
  const fb=sheet('服务预算','逐期仅用已发生信息调整下一期水位；末期订单到达第29期',
   ['方法','周期','累计正库存商品天','累计正库存比例','下一期预算比例','估计下一期服务','水位倍率λ','预算目标'],
   feedback.map(r=>[labels[r.method],r.period,r.cumulative_positive,r.cumulative_service,r.requested_next_service,r.estimated_next_service,r.scale,r.feedback_target]));
  fb.getRange('B5:C172').setNumberFormat('#,##0');fb.getRange('D5:F172').setNumberFormat('0.0000%');fb.getRange('G5:G172').setNumberFormat('0.000000');fb.getRange('H5:H172').setNumberFormat('0.0000%');
  const comparison=await read(out,'version_comparison.csv');
  const comp=sheet('版本对比','历史需求版、上一轮正库存版与本次服务95%版：训练方案不同',
   ['版本','方法','销量','需求','需求满足率','正库存商品天','全部商品天','正库存比例','累计期末库存','库存需求比','本版服务达标'],
   comparison.map(r=>[r.version,labels[r.method],r.sales_units,r.demand_units,null,r.positive_stock_sku_periods,r.total_sku_periods,null,r.ending_inventory_units,null,null]));
  comp.getRange('A:A').format.columnWidth=32;comp.getRange('B:B').format.columnWidth=24;
  comp.getRange('K5:K22').format.horizontalAlignment='center';
  for(let i=0;i<comparison.length;i++){const r=i+5;comp.getRange(`E${r}`).formulas=[[`=C${r}/D${r}`]];comp.getRange(`H${r}`).formulas=[[`=F${r}/G${r}`]];comp.getRange(`J${r}`).formulas=[[`=I${r}/D${r}`]];comp.getRange(`K${r}`).formulas=[[i<6?`=IF(E${r}>=95%,"达到95%","低于95%")`:`=IF(AND(H${r}>=95%,H${r}<=95.5%),"范围内","范围外")`]];}
  comp.getRange('C5:D22').setNumberFormat('#,##0');comp.getRange('F5:G22').setNumberFormat('#,##0');comp.getRange('I5:I22').setNumberFormat('#,##0');comp.getRange('E5:E22').setNumberFormat('0.0000%');comp.getRange('H5:H22').setNumberFormat('0.0000%');comp.getRange('J5:J22').setNumberFormat('0.000000');
 }else if(full60){
  const ro=sheet('RO训练选参','三轮验证合并36期：RO宽度仅从训练集选择', ['区间宽度Γ','需求满足率','正库存比例','库存需求比'],val.filter(r=>r.method==='RO').map(r=>[r.gamma,r.fill_rate,r.positive_stock_ratio,r.inventory_demand_ratio]));
  ro.getRange(`B5:C${val.filter(r=>r.method==='RO').length+4}`).setNumberFormat('0.0000%');
  ro.getRange(`D5:D${val.filter(r=>r.method==='RO').length+4}`).setNumberFormat('0.000000');
  const chosen=JSON.parse(await fs.readFile(path.join(out,'frozen_parameters.json'),'utf8')).selected;
  const foldScores=await read(path.join(root,'experiments/full60_revision/outputs'),'candidate_validation_by_fold.csv');
  const rows=foldScores.filter(r=>r.candidate===chosen[r.method].candidate);
  const fold=sheet('三轮验证','正库存组已选候选：逐轮展示，按三轮合并值选参',
   ['方法','轮次','拟合时期','验证时期','需求满足率','正库存比例','库存需求比'],
   rows.map(r=>[labels[r.method],r.fold,r.fit_periods,r.validation_periods,r.fill_rate,r.positive_stock_ratio,r.inventory_demand_ratio]));
  fold.getRange(`E5:F${rows.length+4}`).setNumberFormat('0.0000%');fold.getRange(`G5:G${rows.length+4}`).setNumberFormat('0.000000');
  fold.getRange(`B5:B${rows.length+4}`).setNumberFormat('0');fold.getRange(`B5:D${rows.length+4}`).format.horizontalAlignment='center';
 }
 else sheet('RO训练选参','RO宽度搜索：仅使用训练集', ['区间宽度Γ','前推需求满足率','重置需求满足率','前推正库存比例','重置正库存比例','前推库存需求比','重置库存需求比'],val.map(r=>[r.gamma,r.forward_fill,r.reset_fill,r.forward_service,r.reset_service,r.forward_inventory_ratio,r.reset_inventory_ratio]));
 wb.recalculate();
 console.log((await wb.inspect({kind:'table',range:'结论!A4:J10',include:'values,formulas',tableMaxRows:7,tableMaxCols:10,maxChars:1200})).ndjson);
 console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:10},maxChars:1000})).ndjson);
 const png=await wb.render({sheetName:'结论',range:full60?'A1:J19':'A1:J17',scale:1.3,format:'png'});
 await fs.writeFile(path.join(out,'workbook_preview.png'),new Uint8Array(await png.arrayBuffer()));
 if(full60){
  const qa=process.env.REPLENISHMENT_QA_DIR??path.join(root,'../_delivery_tools/rolling_cv_qa');await fs.mkdir(qa,{recursive:true});
  const views=[['爆品长尾结果','A1:K16'],['逐期结果','A1:G12'],['最近一期全部SKU','A1:O12'],['完整周期样本','A1:O12'],...(service95?[['训练验证','A1:I12'],['策略候选','A1:H12'],['服务预算','A1:H12'],['版本对比','A1:K22']]:[['RO训练选参','A1:D12'],['三轮验证','A1:G15']])];
  for(const [name,range] of views){
   const image=await wb.render({sheetName:name,range,scale:1.15,format:'png'});
   await fs.writeFile(path.join(qa,`新版_${name}.png`),new Uint8Array(await image.arrayBuffer()));
  }
 }
 await(await SpreadsheetFile.exportXlsx(wb)).save(path.join(out,'replenishment_comparison.xlsx'));
 console.log('saved',version);
}
