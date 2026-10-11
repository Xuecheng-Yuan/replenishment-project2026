"""Rebuild the narrative report from saved numerical results without rerunning models."""
from pathlib import Path
import json
import pandas as pd
from report_v3 import build_report
out=Path(__file__).resolve().parent/'outputs_v4'
data=pd.read_csv(out/'summary.csv')
runs={}
for method in data.method.unique():
    group=data[data.method==method].set_index('scope')
    runs[method]=dict(summary=group.loc['all_28_periods'].to_dict(),
                      controlled=group.loc['controllable_periods_2_to_28'].to_dict())
parameters=json.loads((out/'frozen_parameters.json').read_text(encoding='utf-8'))
forecasts=pd.read_csv(out/'forecast_test_metrics.csv').to_dict('records')
build_report(out,runs,parameters,forecasts,0.)
print(out/'REPORT.md')
