"""Training-only tuning, four-period receding-horizon DRO, locked test replay.

Run: python run_v2.py
Numerical outputs are CSV/NPZ model artifacts, not changes to source Excel.
"""
from pathlib import Path
import argparse, hashlib, json, platform, time
import numpy as np
import pandas as pd
from model_v2 import (cluster, Forecaster, rolling_predictions, residual_oof,
                      risk_levels, simulate, make_forecasts, summarize)
from config import Config
from data_io import load_period_data

ROOT=Path(__file__).resolve().parent


def savecsv(frame,path):
    frame.to_csv(path,index=False,encoding='utf-8-sig')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs_v3')
    parser.add_argument('--data-dir',type=Path,default=ROOT.parent,
                        help='directory containing replenishment_train.xlsx and replenishment_test.xlsx')
    parser.add_argument('--refresh-cache',action='store_true',help='force a fresh read of both Excel workbooks')
    args=parser.parse_args(); out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    started=time.time()
    source_dir=args.data_dir.resolve(); cache_dir=ROOT/'.cache'; cache_file=cache_dir/'period_data.npz'
    audit_path=cache_dir/'source_manifest_v2.json'
    signatures=[]
    for name in ('replenishment_train.xlsx','replenishment_test.xlsx'):
        path=source_dir/name
        if not path.is_file(): raise FileNotFoundError(path)
        signatures.append(dict(file=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    current=None
    if audit_path.is_file(): current=json.loads(audit_path.read_text(encoding='utf-8'))
    verified=(current is not None and current.get('inputs')==signatures and cache_file.is_file()
              and current.get('cache_sha256')==hashlib.sha256(cache_file.read_bytes()).hexdigest())
    if args.refresh_cache or not verified:
        print('[data] source/cache changed or absent; rebuilding cache from Excel',flush=True)
        load_period_data(Config(data_dir=source_dir,cache_dir=cache_dir,output_dir=out),refresh_cache=True)
        current=dict(inputs=signatures,cache_sha256=hashlib.sha256(cache_file.read_bytes()).hexdigest(),
                     validation='streamed every period, checked item_id/category/initial inventory and finite nonnegative demands')
        audit_path.write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'input_audit.json').write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
    # Held-out test outcomes enter only after training has frozen parameters.
    data=np.load(cache_file)
    y=data['train_y'].astype(float); x=data['train_x'].astype(float)
    ids=data['item_id']; initial=data['initial_inventory'].astype(float)
    split=36; target=.95; test_length=28
    labels,_,_=cluster(y[:split],x[:split])
    reset_labels,_,_=cluster(y[28:],x[28:])
    print('Training selection: forward fold 1:36 -> 37:60; reset fold 29:60 -> 1:28.',flush=True)
    scores=[]
    for blend in (0.,.5,1.):
        for window in (3,6,12):
            for prior in (0.,.5,1.,3.):
                model=Forecaster(y[:split],labels,blend,window,prior)
                pred=rolling_predictions(model,y[split:],split,history=y[:split])[:,0]
                score=abs(pred-y[split:]).sum()/y[split:].sum()
                reset_model=Forecaster(y[28:],reset_labels,blend,window,prior,time_start=28)
                reset_pred=rolling_predictions(reset_model,y[:28],0)[:,0]
                reset_score=abs(reset_pred-y[:28]).sum()/y[:28].sum()
                scores.append(dict(blend=blend,window=window,prior=prior,validation_wape=float((score+reset_score)/2),
                                   forward_wape=float(score),reset_wape=float(reset_score)))
    best=min(scores,key=lambda z:z['validation_wape'])
    savecsv(pd.DataFrame(scores),out/'forecast_validation.csv')
    settings={k:best[k] for k in ('blend','window','prior')}
    print('Forecast selected:',best,flush=True)
    model=Forecaster(y[:split],labels,**settings)
    val_fc=make_forecasts(model,y[split:],split,history=y[:split])
    residual,_=residual_oof(y[:split],labels,**settings)
    reset_model=Forecaster(y[28:],reset_labels,**settings,time_start=28)
    reset_fc=make_forecasts(reset_model,y[:28],0)
    reset_residual,_=residual_oof(y[28:],reset_labels,**settings,time_start=28)
    # Startup losses cannot be repaired by later orders. Reserve their budget
    # using TRAINING first-28-period volumes, never held-out test outcomes.
    train_total=y[:test_length].sum()
    startup_loss=np.maximum(y[0]-initial,0).sum()
    controllable_total=y[1:test_length].sum()
    control_target=1-((1-target)*train_total-startup_loss)/controllable_total
    control_target=float(np.clip(control_target,target,.999))
    print('Operational validation fill target including training startup reserve:',control_target,flush=True)
    validation=[]; dro_candidates=[]; ss_candidates=[]
    for beta in (control_target,min(control_target+.005,.999),min(control_target+.01,.999)):
        for radius in (0.,.025,.05,.10,.20,.35,.5,.75,1.):
            low,_=risk_levels(residual,labels,beta,radius)
            r=simulate(y[split:],initial,val_fc,low)['controlled']
            reset_low,_=risk_levels(reset_residual,reset_labels,beta,radius)
            reset_r=simulate(y[:28],initial,reset_fc,reset_low)['summary']
            row=dict(method='DRO' if radius>0 else 'SAA',beta=beta,radius=radius,**r,
                     reset_fill=reset_r['fill_rate'],reset_inventory=reset_r['avg_inventory_per_sku'])
            validation.append(row)
            if radius>0: dro_candidates.append(row)
    # Strong (s,S) benchmark, shared forecasts; empirically estimated per-SKU
    # quantiles are regularised with the cluster pool due to 18 OOF periods.
    for p in (.50,.60,.65,.75,.80,.85,.90,.925,.95,.975,.99,.995,.999):
        for gap in (.0,.025,.05,.10):
            pS=min(p+gap,.999)
            lower=ss_quantiles(residual,labels,p)
            upper=ss_quantiles(residual,labels,pS)
            r=simulate(y[split:],initial,val_fc,lower,upper)['controlled']
            reset_r=simulate(y[:28],initial,reset_fc,ss_quantiles(reset_residual,reset_labels,p),
                             ss_quantiles(reset_residual,reset_labels,pS))['summary']
            row=dict(method='sS',p_s=p,p_S=pS,**r,reset_fill=reset_r['fill_rate'],reset_inventory=reset_r['avg_inventory_per_sku'])
            validation.append(row); ss_candidates.append(row)
    def choose(rows):
        feasible=[r for r in rows if r['fill_rate']>=control_target and r['reset_fill']>=target]
        if not feasible:
            savecsv(pd.DataFrame(validation),out/'policy_validation.csv')
            raise RuntimeError('No validation-feasible setting; expand TRAINING grid.')
        return min(feasible,key=lambda r:r['reset_inventory'])
    selected_dro=choose(dro_candidates); selected_ss=choose(ss_candidates)
    saa_rows=[r for r in validation if r['method']=='SAA']
    # SAA is an ablation: keep a failed training service constraint visible.
    saa_feasible=[r for r in saa_rows if r['fill_rate']>=control_target and r['reset_fill']>=target]
    selected_saa=min(saa_feasible,key=lambda r:r['reset_inventory']) if saa_feasible else max(saa_rows,key=lambda r:r['reset_fill'])
    savecsv(pd.DataFrame(validation),out/'policy_validation.csv')
    print('Selected DRO',selected_dro,flush=True)
    print('Selected sS',selected_ss,flush=True)
    # Final refit on all TRAINING data after fixing hyperparameters.
    labels,descriptors,xmeans=cluster(y,x)
    final=Forecaster(y,labels,**settings)
    residual,pred_oof=residual_oof(y,labels,**settings)
    low,profile=risk_levels(residual,labels,selected_dro['beta'],selected_dro['radius'])
    saa,_=risk_levels(residual,labels,selected_saa['beta'],0.)
    sslo=ss_quantiles(residual,labels,selected_ss['p_s'])
    sshi=ss_quantiles(residual,labels,selected_ss['p_S'])
    frozen=dict(target_fill=target,validation_control_target=control_target,lead_time=1,horizon=4,
                order_timing='after demand t, arrival before demand t+1',test_periods_restart=True,
                forecast=settings,DRO=selected_dro,sS=selected_ss,SAA=selected_saa,
                validation_split=split,training_periods=len(y),skus=len(ids),seed=20260929,
                test_used_for_parameter_selection=False,test_used_for_model_diagnosis=True,
                reset_validation='fit periods 29:60, replay held-out 1:28; training-only blocked validation',
                python=platform.python_version(),
                uncertainty='rectangular W-infinity, absolute distance on nonnegative relative demand')
    payload=json.dumps(frozen,ensure_ascii=False,indent=2)
    (out/'frozen_parameters.json').write_text(payload,encoding='utf-8')
    np.savez_compressed(out/'fitted_model.npz',train_y=y,labels=labels,risk_multiplier=low,
                        ss_lower=sslo,ss_upper=sshi,relative_oof=residual,item_id=ids)
    profile_rows=[]
    for g,v in profile.items():
        m=labels==g
        profile_rows.append(dict(cluster=g,skus=int(m.sum()),train_mean=float(y[:,m].mean()),
                                 train_cv=float((y[:,m].std(0)/y[:,m].mean(0)).mean()),**v))
    savecsv(pd.DataFrame(profile_rows),out/'cluster_profile.csv')
    assignment=pd.DataFrame(dict(item_id=ids,cluster=labels))
    for j in range(20): assignment[f'train_mean_x{j+1}']=xmeans[:,j]
    savecsv(assignment,out/'cluster_assignment.csv')
    feature_diagnostics(y,x,out)
    print('Parameters frozen. Starting one held-out TEST evaluation.',flush=True)
    test=data['test_y'].astype(float)
    assert test.shape==(test_length,len(ids))
    fc=make_forecasts(final,test,start=0)
    runs={
        'DRO':simulate(test,initial,fc,low,keep=True),
        'sS':simulate(test,initial,fc,sslo,sshi,keep=True),
        'SAA':simulate(test,initial,fc,saa,keep=True),
    }
    summaries=[]; period_rows=[]; segment_rows=[]
    for name,r in runs.items():
        summaries.append(dict(method=name,scope='all_28_periods',**r['summary']))
        summaries.append(dict(method=name,scope='controllable_periods_2_to_28',**r['controlled']))
        for t in range(len(test)):
            m=summarize(test[t:t+1],r['sold'][t:t+1],r['ending'][t:t+1],r['commencing'][t:t+1],r['orders'][t:t+1])
            period_rows.append(dict(method=name,period=t+1,**m))
        for g in range(6):
            m=labels==g
            stats=summarize(test[:,m],r['sold'][:,m],r['ending'][:,m],r['commencing'][:,m],r['orders'][:,m])
            segment_rows.append(dict(method=name,cluster=g,**stats))
        detail=pd.DataFrame(dict(period=np.repeat(np.arange(1,len(test)+1),len(ids)),
                                 item_id=np.tile(ids,len(test)),cluster=np.tile(labels,len(test)),
                                 demand=test.ravel(),available=r['commencing'].ravel(),
                                 sales=r['sold'].ravel(),lost=(test-r['sold']).ravel(),
                                 ending=r['ending'].ravel(),order=r['orders'].ravel(),
                                 arrival_period=np.repeat(np.arange(2,len(test)+2),len(ids))))
        for h in range(4):
            detail[f'forecast_h{h+1}']=fc[:,h,:].ravel()
            detail[f'target_h{h+1}']=r['targets'][:,h,:].ravel()
            detail[f'planned_order_h{h+1}']=r['plans'][:,h,:].ravel()
        savecsv(detail,out/f'detail_{name}.csv')
        matrix=pd.DataFrame(r['orders'].T,columns=[f'order_after_period_{t+1}' for t in range(len(test))])
        matrix.insert(0,'item_id',ids); savecsv(matrix,out/f'orders_{name}.csv')
        print(name,r['summary'],flush=True)
    frozen['test_first_period_lost']=float((test[0]-runs['DRO']['sold'][0]).sum())
    savecsv(pd.DataFrame(summaries),out/'summary.csv')
    savecsv(pd.DataFrame(period_rows),out/'results_by_period.csv')
    savecsv(pd.DataFrame(segment_rows),out/'summary_by_cluster.csv')
    # Horizon-specific forecasting accuracy scored only at available targets.
    forecast_scores=[]
    for h in range(4):
        truth=test[h+1:]; pred=fc[:len(test)-h-1,h]
        forecast_scores.append(dict(horizon=h+1,n_cells=truth.size,
                                    wape=float(abs(truth-pred).sum()/truth.sum()),
                                    rmse=float(np.sqrt(np.mean((truth-pred)**2)))))
    savecsv(pd.DataFrame(forecast_scores),out/'forecast_test_metrics.csv')
    # Paired SKU bootstrap quantifies precision conditional on this time path.
    rng=np.random.default_rng(20260929); bootstrap=[]
    for _ in range(1000):
        idx=rng.integers(0,len(ids),len(ids)); pair=[]
        for name in ('DRO','sS'):
            r=runs[name]
            sale=r['sold'][:,idx].sum(); inv=((r['commencing'][:,idx]+r['ending'][:,idx])/2).sum()/len(test)
            pair.append((sale/test[:,idx].sum(),sale/inv))
        bootstrap.append([pair[0][0]-pair[1][0],pair[0][1]/pair[1][1]-1])
    intervals=np.quantile(bootstrap,[.025,.5,.975],axis=0)
    savecsv(pd.DataFrame(intervals,index=['p025','p50','p975'],columns=['fill_difference','turnover_relative_change']).reset_index(),out/'paired_bootstrap.csv')
    write_report(out,runs,frozen,forecast_scores,intervals,time.time()-started)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'model_v2.py',ROOT/'run_v2.py',out/'frozen_parameters.json')}
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Completed in',round(time.time()-started,1),'seconds:',out,flush=True)


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


def write_report(out,runs,settings,fc,ci,seconds):
    a=runs['DRO']['summary']; b=runs['sS']['summary']
    change=a['turnover']/b['turnover']-1
    status='达到' if a['fill_rate']>=.95 else '未达到'
    body=f'''# 聚类与分布鲁棒滚动补货：修订模型实测报告

训练集 60 期，测试集 28 期，每期 6000 个 SKU。DRO 在完整测试集的需求满足率为 **{a['fill_rate']:.2%}**，{status} 95% 目标。数量周转次数相对 (s,S) 变化 **{change:+.2%}**。

## 测试结果

| 方法 | 需求满足率 | 无缺货 SKU-周期占比 | 平均库存/品种 | 28期数量周转次数 | 损失需求 |
|---|---:|---:|---:|---:|---:|
'''
    for name in ('DRO','sS','SAA'):
        r=runs[name]['summary']
        body+=f"| {name} | {r['fill_rate']:.2%} | {r['cycle_service']:.2%} | {r['avg_inventory_per_sku']:.2f} | {r['turnover']:.3f} | {r['lost_units']:,.0f} |\n"
    body+=f'''
需求满足率 = 总销量/总需求。平均库存采用各期 `(到货后需求前库存 + 期末库存)/2` 的时间平均。周转次数 = 28 期总销量/全仓平均库存，越高越好；没有单位成本，不能计算货币口径库存周转率，也没有按年折算。另提供期末库存口径周转次数作敏感性对照。

初始在途设为零，第一期只能使用文件给定初始库存，第一期损失为 {settings['test_first_period_lost']:,.0f} 件。第二期起订单开始到货。所有主表指标包含第一期。第28期末订单在第29期到货，单列 final_pipeline，不充当测试期销量。

## 训练、验证和测试隔离

训练内部使用两个验证折。前向折使用第1–36期拟合，回放第37–60期；重启折使用第29–60期拟合，回放独立留出的第1–28期。重启折是离线分块验证，拟合数据的周期编号可以晚于验证区间，用于检验跨文件从早期编号重新开始的外推。每折的聚类、残差风险分布均只从该折拟合区间获得。拟合区间内部采用扩展窗口生成样本外残差，不用该折验证需求估计风险分布。

模型修订前已经查看过一轮测试结果（满足率93.36%），据此发现重启场景的验证不足。本版是重复测试评估，不是从未查看过测试集的独立最终检验。预测参数、半径和(s,S)阈值的数值仍全部由上述训练内部验证选择。选择完成后使用完整60期训练数据重拟合，写入冻结参数文件，再执行本版测试回放。第一轮结果保存在 outputs_v2。

选定预测参数：{settings['forecast']}。前向折可控期间的目标按训练集首28期估计启动缺货预算后提高为 {settings['validation_control_target']:.2%}。候选还须在完整重启折达到95%，然后选重启折平均库存最低的一项。训练期调参结果见 policy_validation.csv。SAA 消融若无验证可行点，保留验证满足率最高的候选并报告其不足。

测试 Period_1 重新起算。每期需求实现后，使用当期及以前的真实需求更新水平，并预测接下来四期。需求真实值用于模拟丢失销售和更新模型；这要求历史 demand 字段反映真实需求，而非被缺货截断的销量。

## 聚类和预测

六类 K-means 输入为训练需求的水平、波动、零需求比例、上尾和增长特征，并加入 x1–x20 训练均值标准化后的前三个主成分。点预测候选包含簇级趋势与SKU线性趋势的混合，验证最终选择SKU趋势（blend=1）。最近3期已观测需求用于加性偏差修正，避免起始预测接近零时乘法修正失效。聚类用于共享需求预测误差分布和鲁棒水位，因此仍直接影响决策。

每个时点只使用已观察到的特征；未来 x1–x20 的可得性未提供，因此不使用测试未来特征。现有特征全部进入聚类，滞后特征与需求变化的线性相关诊断另存。相关较低不等价于证明没有非线性预测价值。

## 分布鲁棒模型

以预测均值 m 为尺度，将样本外需求转成相对需求 R。对每个簇，经验分布的每个样本 r 可以移动到 `[max(0,r-epsilon), r+epsilon]`，构成以绝对距离为底层距离的 W-infinity 球。各期采用矩形化条件模糊集。

对目标满足率 beta，求最小库存水位 z，使所有 Q 满足 `E_Q[(R-z)+] <= (1-beta) E_Q[R]`。这约束的是期望销量与期望需求的比值，而非无缺货概率。

令 `a=1-beta, l=max(0,r-epsilon), u=r+epsilon`，等价约束为：

`mean(max((l-z)+ - a*l, (u-z)+ - a*u)) <= 0`。

该式来自凸分段线性函数在区间上的最大值位于端点。采用单调二分求最小 z；用独立的线性规划端点表述核对数值解。最终采用 beta={settings['DRO']['beta']:.6f}、epsilon={settings['DRO']['radius']:.3f}。SAA 是同一约束的 epsilon=0 消融。鲁棒约束对指定模糊集成立，测试实际服务水平仍需实测验证。

每期形成四个水位 `S_h=m_h*z`，场景条件补货规则为 `q_h(I)=max(0,ceil(S_h-I))`。四期计划按预测均值路径展示，实际只执行第一笔，下一期观测需求后重算。各期共享一步预测残差的边际模糊集，这是对未来路径的矩形近似；本次执行的一步决策使用对应一步残差，远期计划在执行前会重新预测。库存和损失通过正部函数精确更新，不允许同时虚增库存和损失。

数据只提供周期汇总需求，无法还原期内连续盘点触发时刻。本模型采用本期需求实现后立即订货、下期需求前到货的离散近似；不额外添加盘点周期或将保护期改成两期。

在没有订货固定费用、采购量限制、批量限制或采购价格变化的设定下，满足逐期条件服务约束的最小水位策略具有逐路径库存支配性：多订当前库存不会降低后续满足水位所需的库存。因此四期问题可分解，远期预测不会人为抬高当前订单。这里不宣称四期前瞻本身带来优于单期的收益。若加入真实容量或批量约束，需要重新求解耦合的多期问题。

## (s,S) 对照与解释

(s,S) 采用相同簇、预测、信息可见范围和仿真时序，库存头寸低于 s 才补到 S。阈值基于SKU与簇残差分位数的收缩估计，训练内部选择 p_s={settings['sS']['p_s']:.3f}、p_S={settings['sS']['p_S']:.3f}。候选包含 s=S 特例，以免人为削弱对照。未给订货固定费用，因此未擅自加入 EOQ 或经济成本参数。

DRO 与 (s,S) 的实测满足率略有差异，这里比较的是两者均超过95%的验证选定策略，而非严格相同服务水平的配对最优解。周转提升较小，不构成所有数据或所有(s,S)参数下的支配结论。结果也不能归因于“LP求解器效果不好”：同一分布鲁棒约束可通过等价LP求解，分布假设与服务约束才是区别。

按 SKU 成对重抽样1000次，周转变化的95%区间为 [{ci[0,1]:+.2%}, {ci[2,1]:+.2%}]，满足率差的区间为 [{ci[0,0]:+.2%}, {ci[2,0]:+.2%}]。该区间以观测到的28期时间路径为条件，不涵盖未来时间变化或供应中断。

## 复现与输出

在项目目录运行 `python run_v2.py`。新流程写入 outputs_v3，保留原流程和旧结果用于核对。运行耗时约 {seconds:.1f} 秒，不含源Excel核验。运行 `python -m unittest discover -s tests -p test_v2.py -v` 可核验鲁棒根与独立LP一致性、提前期、库存守恒、未来信息隔离及四期分解。

- summary.csv：完整测试期和可控期间的对照指标。
- detail_DRO.csv / detail_sS.csv：逐SKU逐期的预测、四期计划、实际执行订单、到货、销量、缺货和库存。
- orders_DRO.csv：每期需求实现后下达的订单矩阵。
- frozen_parameters.json / fitted_model.npz：冻结的参数及模型。
- input_audit.json：原始Excel SHA256和逐单元格缓存核验。
- forecast_test_metrics.csv：四个预测跨度各自的测试误差。

方法参考：[RSOME 多阶段库存示例](https://xiongpengnus.github.io/rsome/example_dro_inv)、[RSOME 分布鲁棒建模文档](https://xiongpengnus.github.io/rsome/dro_rsome)。本实现借鉴样本邻域模糊集思想，改为非负丢失销售及满足率约束，并用解析端点等价式求解，没有直接复制示例中的积压库存目标。
'''
    (out/'REPORT.md').write_text(body,encoding='utf-8')


if __name__=='__main__': main()
