// One builder for both six-policy deliveries. Use the bundled artifact-tool.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const labels={DRO_cluster:'特征聚类DRO',DRO_global:'不分组DRO',DRO_category:'爆品/长尾DRO',sS:'传统(s,S)',SAA:'零半径经验模型',RO:'传统区间RO'};
const read=async(out,name)=>{
 const lines=(await fs.readFile(path.join(out,name),'utf8')).replace(/^\uFEFF/,'').trimEnd().split(/\r?\n/);
 const cols=lines.shift().split(',');
 return lines.map(l=>Object.fromEntries(l.split(',').map((v,i)=>[cols[i],v!==''&&Number.isFinite(Number(v))?Number(v):v])));
};
const full60=process.argv.includes('--full60');
for(const version of (full60?['rolling_dro_v3']:['demand_fill_v3','rolling_dro_v3'])){
 const out=full60?path.join(root,'experiments/full60_revision/new_positive/outputs_v4'):path.join(root,'experiments',version,'with_ro/outputs_v4');
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
 sh.getRange('A12').values=[['服务定义：'+(old?'销量÷需求≥95%':'期末库存>0的商品－天数÷全部商品－天数，在95%–95.5%之间')]];
 sh.getRange('A14').values=[['库存需求比＝累计期末库存÷累计需求，越低越好。6000商品，28天；两项服务指标同时报告。']];
 sh.getRange('A16').values=[['来源：同目录summary.csv、frozen_parameters.json及detail_*.csv.gz；完整数学模型见上一级REPORT.md。']];
 const period=await read(out,'results_by_period.csv');
 const ps=sheet('逐期结果','逐期服务和库存', ['方法','周期','需求满足率','正库存天数比例','库存需求比','损失需求','订单量'],period.map(r=>[labels[r.method],r.period,r.fill_rate,r.positive_stock_ratio,r.inventory_demand_ratio,r.lost_units,r.order_units]));
 ps.getRange('C5:D172').setNumberFormat('0.0000%');ps.getRange('E5:E172').setNumberFormat('0.000000');
 const methods=Object.keys(labels);
 const headers=['周期','item_id','类别','聚类','需求','预测H1','预测H2','预测H3','预测H4',...methods.map(m=>labels[m]+'订单'),...methods.map(m=>labels[m]+'损失'),...methods.map(m=>labels[m]+'期末库存')];
 const toRows=rs=>rs.map(r=>[r.period,String(r.item_id),r.category,r.cluster,r.actual_demand,r.forecast_h1,r.forecast_h2,r.forecast_h3,r.forecast_h4,...methods.map(m=>r['order_'+m]),...methods.map(m=>r['lost_'+m]),...methods.map(m=>r['ending_'+m])]);
 sheet('最近一期全部SKU','第28期全部商品：预测与六种决策',headers,toRows(await read(out,'latest_decisions.csv')));
 sheet('完整周期样本','前12个商品完整28期：预测与六种决策',headers,toRows(await read(out,'sample_decisions.csv')));
 const val=full60?await read(path.join(root,'experiments/full60_revision/outputs'),'candidate_validation.csv'):await read(out,'ro_validation.csv');
 if(full60) sheet('RO训练选参','1–32期拟合，33–60期向前验证', ['区间宽度Γ','需求满足率','正库存比例','库存需求比'],val.filter(r=>r.method==='RO').map(r=>[r.gamma,r.fill_rate,r.positive_stock_ratio,r.inventory_demand_ratio]));
 else sheet('RO训练选参','RO宽度搜索：仅使用训练集', ['区间宽度Γ','前推需求满足率','重置需求满足率','前推正库存比例','重置正库存比例','前推库存需求比','重置库存需求比'],val.map(r=>[r.gamma,r.forward_fill,r.reset_fill,r.forward_service,r.reset_service,r.forward_inventory_ratio,r.reset_inventory_ratio]));
 wb.recalculate();
 console.log((await wb.inspect({kind:'table',range:'结论!A4:J10',include:'values,formulas',tableMaxRows:7,tableMaxCols:10,maxChars:1200})).ndjson);
 console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:10},maxChars:1000})).ndjson);
 const png=await wb.render({sheetName:'结论',range:'A1:J17',scale:1.3,format:'png'});
 await fs.writeFile(path.join(out,'workbook_preview.png'),new Uint8Array(await png.arrayBuffer()));
 await(await SpreadsheetFile.exportXlsx(wb)).save(path.join(out,'replenishment_comparison.xlsx'));
 console.log('saved',version);
}
