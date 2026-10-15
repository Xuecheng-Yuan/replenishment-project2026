import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {loadArtifactTool} from './artifact_runtime.mjs';
const {Workbook,SpreadsheetFile}=await loadArtifactTool();
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const out=path.join(root,'experiments/full60_revision/outputs');
const data=JSON.parse(await fs.readFile(path.join(out,'workbook_data.json'),'utf8'));
const wb=Workbook.create();const labels=data.labels;
function table(name,title,headers,rows){
 const s=wb.worksheets.add(name);s.showGridLines=false;s.getRange('A2').values=[[title]];
 s.getRange('A2').format.font={name:'Microsoft YaHei',size:15,bold:true,color:'#17324D'};
 const n=headers.length;s.getRangeByIndexes(3,0,1,n).values=[headers];
 s.getRangeByIndexes(3,0,1,n).format={fill:'#17324D',font:{name:'Microsoft YaHei',size:10,color:'#FFFFFF',bold:true},rowHeight:38,wrapText:true,verticalAlignment:'center'};
 s.getRangeByIndexes(4,0,rows.length,n).values=rows;
 s.getRangeByIndexes(4,0,rows.length,n).format={font:{name:'Microsoft YaHei',size:10,color:'#17324D'},rowHeight:23};
 s.getRangeByIndexes(0,0,rows.length+4,n).format.columnWidth=18;
 s.getRange('A:A').format.columnWidth=30;s.getRange('B:B').format.columnWidth=24;
 s.getRangeByIndexes(4,2,rows.length,n-2).setNumberFormat('#,##0.000000');
 s.freezePanes.freezeRows(4);return s;
}
const headers=['版本或约束','方法','销量','需求','需求满足率','正库存商品天','全部商品天','正库存比例','累计期末库存','库存需求比↓','测试服务达标'];
for(const [name,title,records] of [
 ['单因素对照','三轮滚动验证选参，全部60期最终拟合，仅改变服务约束',data.controlled],
 ['历史新旧两版','历史结果：旧需求版、32期版与60期一次划分版',data.historical]]){
 const rows=records.map(r=>[r.version??r.arm,labels[r.method],r.sales_units,r.demand_units,null,r.positive_stock_sku_periods,r.total_sku_periods,null,r.ending_inventory_units,null,null]);
 const s=table(name,title,headers,rows);
 for(let i=0;i<records.length;i++){
  const r=i+5;const fill=records[i].arm==='fill'||records[i].version?.startsWith('旧版');
  s.getRange(`E${r}`).formulas=[[`=C${r}/D${r}`]];s.getRange(`H${r}`).formulas=[[`=F${r}/G${r}`]];s.getRange(`J${r}`).formulas=[[`=I${r}/D${r}`]];
  s.getRange(`K${r}`).formulas=[[fill?`=IF(E${r}>=95%,"达标","未达标")`:`=IF(AND(H${r}>=95%,H${r}<=95.5%),"达标","未达标")`]];
 }
 const last=records.length+4;
 s.getRange(`C5:D${last}`).setNumberFormat('#,##0');s.getRange(`F5:G${last}`).setNumberFormat('#,##0');s.getRange(`I5:I${last}`).setNumberFormat('#,##0');
 s.getRange(`E5:E${last}`).setNumberFormat('0.0000%');s.getRange(`H5:H${last}`).setNumberFormat('0.0000%');
 s.getRange(`K5:K${last}`).format.horizontalAlignment='center';
 if(name==='单因素对照'){
  s.getRange('A18').values=[['选参：1–24→25–36、1–36→37–48、1–48→49–60三轮合并36期；最终全部60期拟合。']];
  s.getRange('A20').values=[['测试未达标的策略如实标注。训练参数冻结后，不用测试集回调库存水位。']];
  s.getRange('A22').values=[['95%–95.5%作用于三轮合并正库存比例；各轮表现见逐轮验证，独立测试见本页。']];
 }
}
const cat=table('爆品长尾证据','按商品类别核对服务与库存', ['约束','方法','类别','SKU数','需求满足率','正库存比例','累计期末库存','累计需求','库存需求比'],data.category.map(r=>[r.arm,labels[r.method],r.category,r.skus,r.fill_rate,r.positive_stock_ratio,r.ending_units,r.demand_units,r.inventory_demand_ratio]));
cat.getRange('E5:F28').setNumberFormat('0.0000%');
cat.getRange('D5:D28').setNumberFormat('#,##0');cat.getRange('G5:H28').setNumberFormat('#,##0');
const bootstrap=table('配对区间','相对(s,S)库存需求比差异，负值表示库存更少', ['约束','方法','库存比差异ΔJ','95%下界','95%上界','抽样次数'],data.bootstrap.map(r=>[r.arm,labels[r.method],r.delta_vs_sS,r.ci95_lower,r.ci95_upper,r.replicates]));
bootstrap.getRange('F5:F16').setNumberFormat('#,##0');
const select=table('冻结选参','同一候选库，两种服务约束分别选参', ['约束','方法','候选编号','训练需求满足率','训练正库存比例','训练库存需求比','β或分位数s','半径或分位数S','安全倍率','RO宽度Γ'],data.selected.map(r=>[r.arm,labels[r.method],r.candidate,r.train_fill,r.train_positive,r.train_inventory_ratio,r.recipe.beta??r.recipe.p_s??null,r.recipe.radius??r.recipe.p_S??null,r.recipe.scale??null,r.recipe.gamma??null]));
select.getRange('D5:E16').setNumberFormat('0.0000%');
select.getRange('C5:C16').setNumberFormat('#,##0');
select.getRange('A18').values=[['训练指标为三轮验证原始总量合并后的36期指标，非最终60期模型的训练内回放成绩。']];
const folds=table('逐轮验证','已选候选逐轮成绩，选参约束作用于36期合并值',
 ['约束','方法','轮次','拟合时期','验证时期','销量','需求','需求满足率','正库存商品天','全部商品天','正库存比例','累计期末库存','库存需求比'],
 data.selected_folds.map(r=>[r.arm,labels[r.method],r.fold,r.fit_periods,r.validation_periods,r.sales_units,r.demand_units,null,r.positive_stock_sku_periods,r.total_sku_periods,null,r.ending_inventory_units,null]));
for(let i=0;i<data.selected_folds.length;i++){
 const r=i+5;
 folds.getRange(`H${r}`).formulas=[[`=F${r}/G${r}`]];
 folds.getRange(`K${r}`).formulas=[[`=I${r}/J${r}`]];
 folds.getRange(`M${r}`).formulas=[[`=L${r}/G${r}`]];
}
const foldLast=data.selected_folds.length+4;
folds.getRange(`H5:H${foldLast}`).setNumberFormat('0.0000%');folds.getRange(`K5:K${foldLast}`).setNumberFormat('0.0000%');
folds.getRange(`F5:G${foldLast}`).setNumberFormat('#,##0');folds.getRange(`I5:J${foldLast}`).setNumberFormat('#,##0');folds.getRange(`L5:L${foldLast}`).setNumberFormat('#,##0');folds.getRange(`C5:C${foldLast}`).setNumberFormat('0');
folds.getRange(`C5:E${foldLast}`).format.horizontalAlignment='center';
const forecast=table('预测参数验证','三轮合计绝对误差 / 三轮合计需求选择预测参数',
 ['趋势混合blend','窗口window','先验prior','合计绝对误差','合计需求','36期WAPE'],
 data.forecast_validation.map(r=>[r.blend,r.window,r.prior,r.absolute_error,r.demand_units,null]));
for(let i=0;i<data.forecast_validation.length;i++)forecast.getRange(`F${i+5}`).formulas=[[`=D${i+5}/E${i+5}`]];
forecast.getRange(`D5:E${data.forecast_validation.length+4}`).setNumberFormat('#,##0');forecast.getRange(`F5:F${data.forecast_validation.length+4}`).setNumberFormat('0.0000%');
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'逐轮验证!A4:M10',include:'values,formulas',tableMaxRows:7,tableMaxCols:13,maxChars:1500})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:10},maxChars:1000})).ndjson);
const png=await wb.render({sheetName:'单因素对照',range:'A1:K23',scale:1.2,format:'png'});
await fs.writeFile(path.join(out,'comparison_workbook_preview.png'),new Uint8Array(await png.arrayBuffer()));
const foldPng=await wb.render({sheetName:'逐轮验证',range:'A1:M14',scale:1.15,format:'png'});
await fs.writeFile(path.join(out,'rolling_validation_workbook_preview.png'),new Uint8Array(await foldPng.arrayBuffer()));
const qa=process.env.REPLENISHMENT_QA_DIR??path.join(root,'../_delivery_tools/rolling_cv_qa');await fs.mkdir(qa,{recursive:true});
for(const [name,range] of [['历史新旧两版','A1:K12'],['爆品长尾证据','A1:I12'],['配对区间','A1:F12'],['冻结选参','A1:J18'],['预测参数验证','A1:F13']]){
 const image=await wb.render({sheetName:name,range,scale:1.15,format:'png'});
 await fs.writeFile(path.join(qa,`总对照_${name}.png`),new Uint8Array(await image.arrayBuffer()));
}
await(await SpreadsheetFile.exportXlsx(wb)).save(path.join(out,'新旧及单因素总对照.xlsx'));
console.log('saved ablation comparison workbook');
