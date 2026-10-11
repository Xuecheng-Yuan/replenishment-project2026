"""Check the delivered Excel against every numerical source used in its views."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from openpyxl import load_workbook
root=Path(__file__).resolve().parent
out=root/'outputs_v4'
book=load_workbook(out/'replenishment_comparison.xlsx',read_only=False,data_only=True)
names=[s.title for s in book.worksheets]
assert len(names)==5,names
summary=pd.read_csv(out/'summary.csv').query("scope == 'all_28_periods'").set_index('method')
methods=('DRO_cluster','DRO_global','DRO_category','sS','SAA')
cover=book.worksheets[0]
for row,method in enumerate(methods,start=7):
    source=summary.loc[method]
    for letter,column in [('B','fill_rate'),('C','avg_inventory_per_sku'),('D','turnover'),('E','lost_units')]:
        assert np.isclose(cover[f'{letter}{row}'].value,source[column]),(method,column)
    assert cover[f'G{row}'].value=='达到'
period=pd.read_csv(out/'results_by_period.csv')
period_sheet=book.worksheets[1]
assert period_sheet.max_row==143
for idx,record in enumerate(period.itertuples(index=False),start=4):
    assert period_sheet[f'A{idx}'].value==record.period
    assert np.isclose(period_sheet[f'C{idx}'].value,record.fill_rate)
sku=book.worksheets[2]; latest=book.worksheets[4]; sample=book.worksheets[3]
assert sku.max_row==6003 and latest.max_row==6003 and sample.max_row==339
full_path=out/'decision_comparison_full.csv'
full=pd.read_csv(full_path if full_path.exists() else full_path.with_suffix('.csv.gz'))
aggregated=full.groupby('item_id',sort=True).agg({'actual_demand':'sum',**{f'order_{m}':'sum' for m in methods},**{f'lost_{m}':'sum' for m in methods}})
for excel_row in range(4,6004):
    item_id=int(sku[f'A{excel_row}'].value)
    line=aggregated.loc[item_id]
    assert np.isclose(sku[f'D{excel_row}'].value,line.actual_demand)
    for index,method in enumerate(methods,start=5):
        assert np.isclose(sku.cell(excel_row,index).value,line[f'order_{method}'])
        assert np.isclose(sku.cell(excel_row,index+5).value,line[f'lost_{method}'])
    assert latest[f'A{excel_row}'].value==28
    recent=full[(full.item_id==item_id)&(full.period==28)].iloc[0]
    assert np.isclose(latest[f'E{excel_row}'].value,recent.actual_demand)
    assert np.isclose(latest[f'F{excel_row}'].value,recent.forecast_h1)
    for index,method in enumerate(methods,start=10):
        assert np.isclose(latest.cell(excel_row,index).value,recent[f'order_{method}'])
book.close()
check=dict(workbook='replenishment_comparison.xlsx',sheets=names,summary_methods=5,
           compared_period_rows=140,sku_aggregate_rows=6000,latest_decision_rows=6000,
           illustrative_rows=336,source_rows=len(full),status='pass')
(out/'workbook_checks.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(check,ensure_ascii=False,indent=2))
