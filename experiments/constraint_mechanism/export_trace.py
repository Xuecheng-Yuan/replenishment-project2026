"""Export a frozen replay without requiring an Excel or custom runtime."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def main():
    home=Path(__file__).resolve().parent/'outputs'
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=int,required=True);p.add_argument('--envelope',choices=('full60','envelope'),required=True)
    p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    if a.destination.exists():raise FileExistsError('Choose a new CSV destination to preserve existing evidence.')
    b=np.load(home/'shared_arrays.npz');t=np.load(home/'traces'/f'c{a.candidate}_{a.envelope}.npz');y=b['test_y']
    table=dict(period=np.repeat(np.arange(1,29),6000),item_id=np.tile(b['item_id'],28),demand=y.ravel(),
               available=t['commencing'].ravel(),sales=t['sold'].ravel(),ending=t['ending'].ravel(),order=t['orders'].ravel(),lost=(y-t['sold']).ravel())
    for h in range(4):table[f'forecast_h{h+1}']=b['final_forecast'][:,h].ravel();table[f'planned_order_h{h+1}']=t['plans'][:,h].ravel()
    pd.DataFrame(table).to_csv(a.destination,index=False,encoding='utf-8-sig',compression='infer')
    print('Exported',y.size,'rows to',a.destination)

if __name__=='__main__':main()
