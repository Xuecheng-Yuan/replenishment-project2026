import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
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
 s.getRange('A:A').format.columnWidth=24;s.getRange('B:B').format.columnWidth=24;
 s.getRangeByIndexes(4,2,rows.length,n-2).setNumberFormat('#,##0.000000');
 s.freezePanes.freezeRows(4);return s;
}
const headers=['版本或约束','方法','销量','需求','需求满足率','正库存商品天','全部商品天','正库存比例','累计期末库存','库存需求比↓','测试服务达标'];
for(const [name,title,records] of [
 ['历史新旧两版','历史交付：两版各六种方法',data.historical],
 ['单因素对照','固定模型和候选库，仅改变服务约束',data.controlled]]){
 const rows=records.map(r=>[r.version??r.arm,labels[r.method],r.sales_units,r.demand_units,null,r.positive_stock_sku_periods,r.total_sku_periods,null,r.ending_inventory_units,null,null]);
 const s=table(name,title,headers,rows);
 for(let i=0;i<records.length;i++){
  const r=i+5;const fill=records[i].arm==='fill'||records[i].version?.startsWith('旧版');
  s.getRange(`E${r}`).formulas=[[`=C${r}/D${r}`]];s.getRange(`H${r}`).formulas=[[`=F${r}/G${r}`]];s.getRange(`J${r}`).formulas=[[`=I${r}/D${r}`]];
  s.getRange(`K${r}`).formulas=[[fill?`=IF(E${r}>=95%,"达标","未达标")`:`=IF(AND(H${r}>=95%,H${r}<=95.5%),"达标","未达标")`]];
 }
 s.getRange('C5:D16').setNumberFormat('#,##0');s.getRange('F5:G16').setNumberFormat('#,##0');s.getRange('I5:I16').setNumberFormat('#,##0');
 s.getRange('E5:E16').setNumberFormat('0.0000%');s.getRange('H5:H16').setNumberFormat('0.0000%');
 if(name==='单因素对照'){
  s.getRange('A18').values=[['本次最终模型使用全部60期；选参采用1–32期拟合、33–60期按时间向前验证。']];
  s.getRange('A20').values=[['测试未达标的策略如实标注。训练参数冻结后，不用测试集回调库存水位。']];
  s.getRange('A22').values=[['历史两版仅供追溯，最新结果和结论见单因素对照与新版60期报告。']];
 }
}
const cat=table('爆品长尾证据','按商品类别核对服务与库存', ['约束','方法','类别','SKU数','需求满足率','正库存比例','累计期末库存','累计需求','库存需求比'],data.category.map(r=>[r.arm,labels[r.method],r.category,r.skus,r.fill_rate,r.positive_stock_ratio,r.ending_units,r.demand_units,r.inventory_demand_ratio]));
cat.getRange('E5:F28').setNumberFormat('0.0000%');
table('配对区间','相对(s,S)库存需求比差异，负值表示库存更少', ['约束','方法','库存比差异ΔJ','95%下界','95%上界','抽样次数'],data.bootstrap.map(r=>[r.arm,labels[r.method],r.delta_vs_sS,r.ci95_lower,r.ci95_upper,r.replicates]));
const select=table('冻结选参','同一候选库，两种服务约束分别选参', ['约束','方法','候选编号','训练需求满足率','训练正库存比例','训练库存需求比','β或分位数s','半径或分位数S','安全倍率','RO宽度Γ'],data.selected.map(r=>[r.arm,labels[r.method],r.candidate,r.train_fill,r.train_positive,r.train_inventory_ratio,r.recipe.beta??r.recipe.p_s??null,r.recipe.radius??r.recipe.p_S??null,r.recipe.scale??null,r.recipe.gamma??null]));
select.getRange('D5:E16').setNumberFormat('0.0000%');
wb.recalculate();
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:10},maxChars:1000})).ndjson);
const png=await wb.render({sheetName:'单因素对照',range:'A1:K23',scale:1.2,format:'png'});
await fs.writeFile(path.join(out,'comparison_workbook_preview.png'),new Uint8Array(await png.arrayBuffer()));
await(await SpreadsheetFile.exportXlsx(wb)).save(path.join(out,'新旧及单因素总对照.xlsx'));
console.log('saved ablation comparison workbook');
