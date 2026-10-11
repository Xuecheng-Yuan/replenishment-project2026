// Rebuild the presentation workbook from run_v3.py CSV outputs.
// Requires @oai/artifact-tool from Codex's bundled workspace dependencies.
import fs from 'node:fs/promises';
import path from 'node:path';
import readline from 'node:readline';
import { createReadStream, existsSync } from 'node:fs';
import { createGunzip } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const here=path.dirname(fileURLToPath(import.meta.url));
const out=path.join(here,'outputs_v4');
const req=(filename)=>path.join(out,filename);
const readCsv=async filename=>{
  const txt=(await fs.readFile(req(filename),'utf8')).replace(/^\uFEFF/,'').trimEnd();
  const [header,...lines]=txt.split(/\r?\n/);
  return {columns:header.split(','),rows:lines.map(line=>line.split(','))};
};
const asNum=v=>v===''?null:Number(v);
const n=(row,cols,key)=>Number(row[cols.indexOf(key)]);
const s=(row,cols,key)=>row[cols.indexOf(key)];
const wb=Workbook.create();
const navy='#17324D', teal='#0F6B77', pale='#E9F3F4', orange='#C6781D', textColor='#17324D';
function styleSheet(sheet,color=teal){sheet.showGridLines=false;sheet.tabColor=color;}
function title(sheet,value,end){
  sheet.mergeCells(`A1:${end}1`);
  const cell=sheet.getRange('A1');cell.values=[[value]];
  sheet.getRange(`A1:${end}1`).format={fill:navy,font:{name:'Microsoft YaHei',size:16,bold:true,color:'#FFFFFF'},rowHeight:34,verticalAlignment:'center'};
}
function header(sheet,address){sheet.getRange(address).format={fill:teal,font:{name:'Microsoft YaHei',size:10,bold:true,color:'#FFFFFF'},rowHeight:29,verticalAlignment:'center'};}
function setSize(sheet,lastCol){sheet.getRange(`A:A`).format.columnWidth=16;sheet.getRange(`B:${lastCol}`).format.columnWidth=13;}
function putRows(sheet,start,rows,size=500){
  for(let k=0;k<rows.length;k+=size){
    const block=rows.slice(k,k+size);
    sheet.getRangeByIndexes(start-1+k,0,block.length,block[0].length).values=block;
  }
}
const summary=await readCsv('summary.csv');
const selected=JSON.parse(await fs.readFile(req('frozen_parameters.json'),'utf8'));
const methods=['DRO_cluster','DRO_global','DRO_category','sS','SAA'];
const labels={'DRO_cluster':'特征聚类DRO','DRO_global':'不分组DRO','DRO_category':'爆品/长尾DRO','sS':'传统(s,S)','SAA':'零半径经验模型'};
const all=new Map(summary.rows.filter(r=>s(r,summary.columns,'scope')==='all_28_periods').map(r=>[s(r,summary.columns,'method'),r]));

// One visible answer: service, stock, turnover, and training-selected settings.
{
  const sh=wb.worksheets.add('结论');styleSheet(sh);title(sh,'补货策略对比：完整28期测试','J');
  sh.getRange('A3:J4').merge();
  sh.getRange('A3').values=[['需求满足率＝总销量/总需求；28期库存周转率（数量口径，次）＝总销量/全仓平均在手库存，越高越好。缺货按丢失处理，分子用实际销量。']];
  sh.getRange('A3:J4').format={fill:pale,font:{name:'Microsoft YaHei',size:11,color:textColor},wrapText:true,rowHeight:25};
  sh.getRange('A6:J6').values=[['方法','需求满足率','平均库存/品种','28期库存周转率（次）','损失需求','周转率相对(s,S)','95%目标','风险组数','内部目标β','鲁棒半径ε']];
  header(sh,'A6:J6');
  const count={'DRO_cluster':6,'DRO_global':1,'DRO_category':2,'sS':null,'SAA':6};
  const rows=methods.map(m=>{
    const r=all.get(m),p=selected.selected[m];
    return [labels[m],n(r,summary.columns,'fill_rate'),n(r,summary.columns,'avg_inventory_per_sku'),
      n(r,summary.columns,'turnover'),n(r,summary.columns,'lost_units'),null,null,count[m],
      p.beta??null,p.radius??null];
  });
  sh.getRange('A7:J11').values=rows;
  for(let excelRow=7;excelRow<=11;excelRow++){
    sh.getRange(`F${excelRow}`).formulas=[[`=D${excelRow}/$D$10-1`]];
    sh.getRange(`G${excelRow}`).formulas=[[`=IF(B${excelRow}>=95%,"达到","未达")`]];
  }
  sh.getRange('B7:B11').setNumberFormat('0.00%');
  sh.getRange('F7:F11').setNumberFormat('+0.00%;-0.00%;0.00%');
  sh.getRange('G7:G11').format.horizontalAlignment='center';
  sh.getRange('C7:D11').setNumberFormat('#,##0.00');
  sh.getRange('E7:E11').setNumberFormat('#,##0');
  sh.getRange('I7:J11').setNumberFormat('0.0000');
  sh.getRange('A9:J9').format.fill='#E8F4EA';
  sh.getRange('A13:J14').merge();
  sh.getRange('A13').values=[['全部SKU、全部28期、五种决策的逐格结果保存在 decision_comparison_full.csv；本簿展示全量SKU汇总、最近一期全部SKU决策及可阅读的完整周期样本。']];
  sh.getRange('A13:J14').format={fill:'#FFF4E6',font:{name:'Microsoft YaHei',size:10,color:textColor},wrapText:true,rowHeight:25};
  sh.getRange('A16:J17').merge();
  sh.getRange('A16').values=[['下单时点：本期需求实现后；提前期1，下一期需求前到货；未满足需求直接丢失。所有风险参数只用训练集内部验证选择。']];
  sh.getRange('A16:J17').format={font:{name:'Microsoft YaHei',size:10,color:textColor},wrapText:true,rowHeight:25};
  setSize(sh,'J');sh.getRange('A:A').format.columnWidth=23;
  sh.getRange('C:C').format.columnWidth=17;sh.getRange('D:D').format.columnWidth=25;
  sh.getRange('E:E').format.columnWidth=17;sh.getRange('F:F').format.columnWidth=21;
  sh.getRange('G:G').format.columnWidth=15;
  sh.freezePanes.freezeRows(6);
}

// Every period x every method, with no sample truncation.
{
  const data=await readCsv('results_by_period.csv');
  const sh=wb.worksheets.add('逐期比较');styleSheet(sh,'#2E6D85');title(sh,'逐期对比：五种策略','J');
  sh.getRange('A3:J3').values=[['周期','方法','需求满足率','平均库存/品种','当期销量','当期损失','当期下单','当期需求','当期库存周转率（次）','终期在途']];header(sh,'A3:J3');
  const rows=data.rows.map(r=>[n(r,data.columns,'period'),labels[s(r,data.columns,'method')],n(r,data.columns,'fill_rate'),
    n(r,data.columns,'avg_inventory_per_sku'),n(r,data.columns,'sales_units'),n(r,data.columns,'lost_units'),
    n(r,data.columns,'order_units'),n(r,data.columns,'demand_units'),n(r,data.columns,'turnover'),
    n(r,data.columns,'final_pipeline')]);
  putRows(sh,4,rows);sh.getRange(`C4:C${rows.length+3}`).setNumberFormat('0.00%');
  sh.getRange(`D4:J${rows.length+3}`).setNumberFormat('#,##0.00');
  setSize(sh,'J');sh.getRange('B:B').format.columnWidth=22;
  sh.getRange('I:I').format.columnWidth=25;sh.freezePanes.freezeRows(3);
}

// Full 6000-SKU aggregation; source CSV has every SKU-period-method observation.
let selectedIds=new Set(), skuSummary=[],sampleRows=[],latestRows=[],comparisonColumns=null;
{
  const assignment=await readCsv('cluster_assignment.csv');
  const hot=assignment.rows.filter(r=>r[assignment.columns.indexOf('category')]==='爆品').slice(0,6);
  const tail=assignment.rows.filter(r=>r[assignment.columns.indexOf('category')]==='长尾品').slice(0,6);
  selectedIds=new Set([...hot,...tail].map(r=>r[0]));
  const bySku=new Map();
  const fullPath=req('decision_comparison_full.csv');
  const input=existsSync(fullPath)?createReadStream(fullPath):createReadStream(fullPath+'.gz').pipe(createGunzip());
  input.setEncoding('utf8');
  const stream=readline.createInterface({input,crlfDelay:Infinity});
  let columns=null;
  for await(const line of stream){
    if(!columns){columns=line.replace(/^\uFEFF/,'').split(',');comparisonColumns=columns;continue;}
    const r=line.split(',');const id=s(r,columns,'item_id');
    if(selectedIds.has(id)) sampleRows.push(r);
    if(n(r,columns,'period')===28) latestRows.push(r);
    let a=bySku.get(id);
    if(!a){a={id:Number(id),category:s(r,columns,'category'),cluster:n(r,columns,'cluster'),demand:0,
      orders:Object.fromEntries(methods.map(m=>[m,0])),lost:Object.fromEntries(methods.map(m=>[m,0]))};bySku.set(id,a);}
    a.demand+=n(r,columns,'actual_demand');
    for(const m of methods){a.orders[m]+=n(r,columns,`order_${m}`);a.lost[m]+=n(r,columns,`lost_${m}`);}
  }
  skuSummary=[...bySku.values()].sort((a,b)=>a.id-b.id);
  const sh=wb.worksheets.add('SKU汇总');styleSheet(sh,'#43836D');title(sh,'全部6000个SKU：28期订单与损失','N');
  sh.getRange('A3:N3').values=[['item_id','类别','聚类','实际需求','聚类DRO下单','不分组DRO下单','类别DRO下单','(s,S)下单','SAA下单',
    '聚类DRO损失','不分组DRO损失','类别DRO损失','(s,S)损失','SAA损失']];header(sh,'A3:N3');
  const rows=skuSummary.map(a=>[a.id,a.category,a.cluster,a.demand,...methods.map(m=>a.orders[m]),...methods.map(m=>a.lost[m])]);
  putRows(sh,4,rows);setSize(sh,'N');sh.getRange('B:B').format.columnWidth=13;
  sh.getRange('D:N').format.columnWidth=17;sh.getRange('D4:N6003').setNumberFormat('#,##0');
  sh.freezePanes.freezeRows(3);
}

// Twelve SKUs, all 28 periods and all 4 horizons. The full 168000-row table
// remains in decision_comparison_full.csv next to the workbook.
{
  const sh=wb.worksheets.add('决策明细示例');styleSheet(sh,orange);title(sh,'逐SKU逐期对比示例（爆品、长尾品各6个）','U');
  const headers=['周期','item_id','类别','聚类','实际需求','预测H1','预测H2','预测H3','预测H4',
    ...methods.map(m=>`${labels[m]}订单`),...methods.map(m=>`${labels[m]}损失`),
    '聚类DRO期末库存','类别DRO期末库存'];
  sh.getRangeByIndexes(2,0,1,headers.length).values=[headers];header(sh,'A3:U3');
  const rows=sampleRows.map(r=>[
    n(r,comparisonColumns,'period'),n(r,comparisonColumns,'item_id'),s(r,comparisonColumns,'category'),n(r,comparisonColumns,'cluster'),
    n(r,comparisonColumns,'actual_demand'),...Array.from({length:4},(_,i)=>n(r,comparisonColumns,`forecast_h${i+1}`)),
    ...methods.map(m=>n(r,comparisonColumns,`order_${m}`)),...methods.map(m=>n(r,comparisonColumns,`lost_${m}`)),
    n(r,comparisonColumns,'ending_DRO_cluster'),n(r,comparisonColumns,'ending_DRO_category')]);
  putRows(sh,4,rows,200);setSize(sh,'U');sh.getRange('J:U').format.columnWidth=17;
  sh.getRange(`E4:U${rows.length+3}`).setNumberFormat('#,##0.00');sh.freezePanes.freezeRows(3);
}
// All 6000 SKUs at the most recent decision point, with each forecast horizon
// and policy action side by side. Full history remains in the adjacent CSV.
{
  const sh=wb.worksheets.add('最近一期全部SKU');styleSheet(sh,'#315B8A');title(sh,'第28期：全部SKU的四期预测与五种决策','U');
  const headers=['周期','item_id','类别','聚类','实际需求','预测H1','预测H2','预测H3','预测H4',
    ...methods.map(m=>`${labels[m]}订单`),...methods.map(m=>`${labels[m]}损失`),
    '聚类DRO期末库存','类别DRO期末库存'];
  sh.getRangeByIndexes(2,0,1,headers.length).values=[headers];header(sh,'A3:U3');
  const rows=latestRows.map(r=>[
    n(r,comparisonColumns,'period'),n(r,comparisonColumns,'item_id'),s(r,comparisonColumns,'category'),n(r,comparisonColumns,'cluster'),
    n(r,comparisonColumns,'actual_demand'),...Array.from({length:4},(_,i)=>n(r,comparisonColumns,`forecast_h${i+1}`)),
    ...methods.map(m=>n(r,comparisonColumns,`order_${m}`)),...methods.map(m=>n(r,comparisonColumns,`lost_${m}`)),
    n(r,comparisonColumns,'ending_DRO_cluster'),n(r,comparisonColumns,'ending_DRO_category')]);
  putRows(sh,4,rows,300);setSize(sh,'U');sh.getRange('J:U').format.columnWidth=17;
  sh.getRange(`E4:U${rows.length+3}`).setNumberFormat('#,##0.00');sh.freezePanes.freezeRows(3);
}

wb.recalculate();
const preview=await wb.render({sheetName:'结论',range:'A1:J17',scale:1.6,format:'png'});
await fs.writeFile(req('workbook_preview.png'),new Uint8Array(await preview.arrayBuffer()));
const inspected=await wb.inspect({kind:'table',range:'结论!A6:J11',include:'values,formulas',tableMaxRows:7,tableMaxCols:10,maxChars:4000});
console.log(inspected.ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:50},summary:'final error scan'});
console.log(errors.ndjson);
const output=await SpreadsheetFile.exportXlsx(wb);
await output.save(req('replenishment_comparison.xlsx'));
console.log('saved',req('replenishment_comparison.xlsx'));
