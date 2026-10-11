"""Causal clustered forecasting and exact scalar W-infinity fill-rate DRO.

L=1: after observing period t, q_t arrives before demand t+1.
There are no order/setup/capacity costs in the supplied data. Four-stage
conditional service constraints consequently admit just-in-time base stocks.
The scalar robust stop-loss root is solved exactly (to numerical tolerance),
not by an affine inventory relaxation that permits artificial lost sales.
"""
from dataclasses import dataclass
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import linprog
from scipy.spatial.distance import cdist


def feature_matrix(y, x):
    mu = np.maximum(y.mean(0), 1.)
    cv = y.std(0)/mu
    demand = np.column_stack((np.log1p(mu), np.log1p(cv), (y==0).mean(0),
                             np.log1p(np.quantile(y,.95,axis=0)/mu),
                             np.log1p(y[-8:].mean(0)/np.maximum(y[:8].mean(0),1))))
    xm = x.mean(0)
    # Twenty supplied features enter via PCA; block weight avoids swamping
    # five demand descriptors with twenty nearly redundant columns.
    z = (xm-xm.mean(0))/np.maximum(xm.std(0),1e-8)
    _,_,vt = np.linalg.svd(z,full_matrices=False)
    pcs = z@vt[:3].T
    a = (demand-demand.mean(0))/np.maximum(demand.std(0),1e-8)
    b = (pcs-pcs.mean(0))/np.maximum(pcs.std(0),1e-8)
    return np.column_stack((a,b*.5)), demand, xm


def cluster(y,x,k=6):
    f, demand, xm = feature_matrix(y,x)
    best = None
    rng = np.random.default_rng(20260929)
    for _ in range(8):
        centers, labels = kmeans2(f, f[rng.choice(len(f),k,replace=False)], iter=60, minit='matrix')
        loss = ((f-centers[labels])**2).sum()
        if best is None or loss<best[0]: best=(loss,labels,centers)
    return best[1], demand, xm


@dataclass
class Forecaster:
    y: np.ndarray
    group: np.ndarray
    blend: float = .5
    window: int = 6
    prior: float = 3.
    time_start: int = 0

    def __post_init__(self):
        self.y = np.asarray(self.y,float)
        t=np.arange(self.time_start,self.time_start+len(self.y),dtype=float)
        centered=t-t.mean()
        self.slope=(centered[:,None]*self.y).sum(0)/(centered@centered)
        self.intercept=self.y.mean(0)-t.mean()*self.slope
        self.shape=np.zeros_like(self.y)
        self.group_coeff={}
        for g in np.unique(self.group):
            m=self.group==g
            gmean=self.y[:,m].mean(1)
            coeff=np.polyfit(t,gmean,1)
            self.group_coeff[int(g)]=coeff
            self.shape[:,m]=np.maximum(np.polyval(coeff,t),1)[:,None]
        self.level=(self.y/np.maximum(self.shape,1)).mean(0)

    def base(self,times):
        times=np.asarray(times)
        gshape=np.zeros((len(times),self.y.shape[1]))
        for g,coeff in self.group_coeff.items():
            gshape[:,self.group==g]=np.maximum(np.polyval(coeff,times),1)[:,None]
        pooled=gshape*self.level
        individual=np.maximum(self.intercept+times[:,None]*self.slope,1)
        return np.maximum(self.blend*individual+(1-self.blend)*pooled,1)

    def forecast(self,observed,start,next_time,horizon=4):
        """Only already observed demands enter. start is its absolute period."""
        pred=self.base(np.arange(next_time,next_time+horizon))
        if len(observed):
            k=min(len(observed),self.window)
            times=np.arange(start+len(observed)-k,start+len(observed))
            # Additive intercept correction stays well-defined when a fitted
            # trend extrapolates near zero at a new episode's first period.
            correction=(observed[-k:]-self.base(times)).mean(0)
            pred+=correction*k/(self.prior+k)
        return np.maximum(pred,1.)


def rolling_predictions(model,y,start,history=None):
    hist=np.empty((0,y.shape[1])) if history is None else np.asarray(history)
    hist_start=start-len(hist)
    ans=[]
    for t in range(len(y)):
        ans.append(model.forecast(hist,hist_start,start+t,4))
        hist=np.concatenate((hist,y[t:t+1]))
    return np.asarray(ans)


def residual_oof(y,group,blend,window,prior,first=18,time_start=0):
    """Expanding-window out-of-fit predictions; never use fitted residuals."""
    ys=[]; ps=[]
    for cut in range(first,len(y),6):
        end=min(cut+6,len(y))
        model=Forecaster(y[:cut],group,blend,window,prior,time_start=time_start)
        p=rolling_predictions(model,y[cut:end],time_start+cut,history=y[:cut])[:,0]
        ys.append(y[cut:end]); ps.append(p)
    return np.concatenate(ys)/np.maximum(np.concatenate(ps),1),np.concatenate(ps)


def robust_fill_level(samples,beta,radius,tol=1e-10):
    """Minimum z s.t. sup_Q E[(R-z)+ - (1-beta)R] <= 0.

    Q is a W-infinity ball, with base metric |.|, around equal-weight
    nonnegative samples. Each atom can move inside [max(0,r-eps),r+eps].
    A convex piecewise affine integrand attains its maximum at an endpoint.
    This controls the worst-case ratio of expected sales to expected demand.
    """
    r=np.asarray(samples,float)
    lo=np.maximum(0,r-radius); hi=r+radius
    a=1-beta
    left=0.; right=float(hi.max())
    for _ in range(60):
        mid=(left+right)/2
        worst=np.maximum(np.maximum(lo-mid,0)-a*lo,np.maximum(hi-mid,0)-a*hi).mean()
        if worst>0: left=mid
        else: right=mid
        if right-left<tol: break
    return right


def robust_fill_lp_check(samples,beta,radius):
    """Independent endpoint epigraph LP for numerical verification only."""
    r=np.asarray(samples,float); n=len(r); a=1-beta
    lo=np.maximum(0,r-radius); hi=r+radius
    A=[]; b=[]
    for j in range(n):
        for v in (lo[j],hi[j]):
            row=np.zeros(n+1); row[0]=-1; row[j+1]=-1
            A.append(row); b.append(-beta*v)
            row=np.zeros(n+1); row[j+1]=-1
            A.append(row); b.append(a*v)
    row=np.r_[0,np.ones(n)/n]; A.append(row); b.append(0.)
    result=linprog(np.r_[1,np.zeros(n)],A_ub=np.asarray(A),b_ub=np.asarray(b),
                   bounds=[(0,None)]+[(None,None)]*n,method='highs')
    if not result.success: raise RuntimeError(result.message)
    return result.x[0]


def risk_levels(residual,labels,beta,radius):
    z=np.zeros(len(labels)); raw={}
    for g in np.unique(labels):
        samples=residual[:,labels==g].ravel()
        # Keep all tails: no trimming or synthetic circular scenario shifts.
        level=robust_fill_level(samples,beta,radius)
        z[labels==g]=level
        raw[int(g)]=dict(level=float(level),n=len(samples),mean=float(samples.mean()),
                         q95=float(np.quantile(samples,.95)))
    return z,raw


def summarize(demand,sold,ending,commencing,orders):
    T,n=demand.shape
    mean_stock=((commencing+ending)/2).sum()/T
    end_stock=ending.sum()/T
    sales=sold.sum(); total=demand.sum()
    return dict(fill_rate=float(sales/total),cycle_service=float(np.mean(sold>=demand-1e-9)),
                avg_inventory_per_sku=float(mean_stock/n),avg_ending_inventory_per_sku=float(end_stock/n),
                turnover=float(sales/mean_stock) if mean_stock else float('inf'),
                turnover_ending=float(sales/end_stock) if end_stock else float('inf'),
                demand_units=float(total),sales_units=float(sales),lost_units=float(total-sales),
                order_units=float(orders.sum()),final_stock=float(ending[-1].sum()),
                final_pipeline=float(orders[-1].sum()),periods=T,skus=n)


def simulate(y,initial,forecast,low,high=None,keep=False):
    """Forecast[t,h] is formed AFTER seeing y[t], for t+1+h.

    Plan all four periods using mean-path stock, execute only q[0]. Future
    recourse is q_h(I)=max(S_h-I,0), not an order fixed with future demand.
    All methods share the initial zero pipeline and integer order rounding.
    """
    T,n=y.shape
    inventory=np.array(initial,float); pipe=np.zeros(n)
    commencing=[]; ending=[]; sold=[]; orders=[]; plans=[]; targets=[]
    for t in range(T):
        available=inventory+pipe
        sale=np.minimum(available,y[t]); inventory=available-sale
        highlevel=forecast[t]*(low if high is None else high)[None,:]
        lowlevel=forecast[t]*low[None,:]
        projected=inventory.copy(); plan=[]
        for h in range(4):
            q=np.maximum(0,np.ceil(highlevel[h]-projected))
            if high is not None: q=np.where(projected<lowlevel[h],q,0)
            plan.append(q)
            projected=np.maximum(0,projected+q-forecast[t,h])
        pipe=plan[0]
        commencing.append(available); ending.append(inventory.copy()); sold.append(sale); orders.append(pipe)
        if keep: plans.append(np.array(plan)); targets.append(highlevel)
    c,e,s,q=map(np.asarray,(commencing,ending,sold,orders))
    result={'summary':summarize(y,s,e,c,q),'controlled':summarize(y[1:],s[1:],e[1:],c[1:],q[1:])}
    if keep: result.update(commencing=c,ending=e,sold=s,orders=q,plans=np.asarray(plans),targets=np.asarray(targets))
    return result


def make_forecasts(model,y,start=0,history=None):
    hist=np.empty((0,y.shape[1])) if history is None else np.asarray(history)
    hist_start=start-len(hist); result=[]
    for t in range(len(y)):
        hist=np.concatenate((hist,y[t:t+1]))
        result.append(model.forecast(hist,hist_start,start+t+1,4))
    return np.asarray(result)
