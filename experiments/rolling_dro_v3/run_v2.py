"""Shared (s,S) and diagnostics helpers; legacy entry forwards to the current experiment."""
import numpy as np
import pandas as pd

def savecsv(frame,path):
    frame.to_csv(path,index=False,encoding='utf-8-sig')

def ss_quantiles(residual,labels,p):
    individual=np.quantile(residual,p,axis=0)
    pooled=np.empty(len(labels))
    for g in np.unique(labels): pooled[labels==g]=np.quantile(residual[:,labels==g],p)
    weight=len(residual)/(len(residual)+24)
    return weight*individual+(1-weight)*pooled

def feature_diagnostics(y,x,out):
    # Causal lagged x only: future x values were not stated to be available.
    target=y[1:]/np.maximum(y[:-1],1)
    target=np.clip(target,0,10).ravel()
    rows=[]
    for j in range(20):
        lag=x[:-1,:,j].ravel()
        rows.append(dict(feature=f'x{j+1}',lag1_demand_ratio_correlation=float(np.corrcoef(lag,target)[0,1]),
                         used_in_clustering=True,used_as_future_covariate=False))
    savecsv(pd.DataFrame(rows),out/'feature_diagnostics.csv')


if __name__ == "__main__":
    from run_v3 import main
    main()
