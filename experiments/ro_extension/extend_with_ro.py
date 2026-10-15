"""Add a box-uncertainty RO comparator without changing the five published policies.

Run from a clone: python experiments/ro_extension/extend_with_ro.py --version all
Requires the existing requirements_v3.txt. No commercial solver is needed: the
monotone box robust counterpart has an exact endpoint solution.
"""
from pathlib import Path
import argparse, gzip, hashlib, json, shutil, subprocess, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = ('demand_fill_v3', 'rolling_dro_v3')
LABELS = {'DRO_cluster':'特征聚类DRO','DRO_global':'不分组DRO','DRO_category':'爆品/长尾DRO','sS':'传统(s,S)','SAA':'零半径经验模型','RO':'传统区间RO'}

def box(samples, gamma):
    """Relative-demand interval; gamma is width, not a coupled budget."""
    r = np.asarray(samples, float)
    mu, sigma = float(r.mean()), float(r.std())
    return max(0., mu-gamma*sigma), mu+gamma*sigma

def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def main(version):
    home = ROOT/'experiments'/version
    sys.path.insert(0, str(home))
    import run_v3 as pipeline
    import model_v2 as model
    old = version == VERSIONS[0]
    base = home/'outputs_v4'
    out = home/'with_ro'/'outputs_v4'
    out.mkdir(parents=True, exist_ok=True)
    cache = home/'.cache'
    if not cache.exists() and (ROOT/'experiments/rolling_dro_v3/.cache').exists():
        shutil.copytree(ROOT/'experiments/rolling_dro_v3/.cache', cache)
    data = pipeline.verified_data(ROOT/'data/raw', out, False)
    y, x = data['train_y'].astype(float), data['train_x'].astype(float)
    initial = data['initial_inventory'].astype(float)
    params = json.loads((base/'frozen_parameters.json').read_text(encoding='utf-8'))
    settings = params['forecast']
    gf = model.cluster(y[:36], x[:36])[0]
    gr = model.cluster(y[28:], x[28:])[0]
    ff = model.make_forecasts(model.Forecaster(y[:36], gf, **settings), y[36:], 36, history=y[:36])
    fr = model.make_forecasts(model.Forecaster(y[28:], gr, **settings, time_start=28), y[:28], 0)
    rf = model.residual_oof(y[:36], gf, **settings)[0]
    rr = model.residual_oof(y[28:], gr, **settings, time_start=28)[0]
    beta = params.get('forward_control_target', 1.) if old else 1.
    def simulate(demand, forecasts, samples, gamma, keep=False):
        level = np.full(demand.shape[1], beta*box(samples, gamma)[1])
        kw = {} if old else {'positive_guard':True}
        return model.simulate(demand, initial, forecasts, level, keep=keep, **kw)
    rows = []
    def evaluate(gamma):
        f = simulate(y[36:], ff, rf, gamma)['controlled']
        r = simulate(y[:28], fr, rr, gamma)['summary']
        rows.append(dict(method='RO', gamma=float(gamma), beta=beta,
            forward_fill=f['fill_rate'], reset_fill=r['fill_rate'],
            forward_service=f.get('positive_stock_ratio',np.nan), reset_service=r.get('positive_stock_ratio',np.nan),
            forward_inventory_ratio=f['inventory_demand_ratio'], reset_inventory_ratio=r['inventory_demand_ratio']))
    def choose():
        return pipeline.selected(rows, params.get('forward_control_target'))
    print(f'[{version}] Calibrating interval width on training folds only', flush=True)
    for g in np.arange(0, 8.0001, .025): evaluate(round(g, 6))
    best = choose()
    if best is None: raise RuntimeError('No training-feasible RO width in [0,8]; inspect validation before changing search.')
    for g in np.arange(max(0,best['gamma']-.025), best['gamma']+.0251, .001):
        if not any(abs(g-r['gamma'])<1e-8 for r in rows): evaluate(round(g,6))
    best = choose()
    pipeline.csv(pd.DataFrame(rows), out/'ro_validation.csv')
    bank = np.load(base/'fitted_model.npz', allow_pickle=False)
    lower, upper = box(bank['relative_residual'], best['gamma'])
    params['selected']['RO'] = dict(best, lower_relative=lower, upper_relative=upper,
        uncertainty_set='nonnegative relative-demand box [max(0,mu-gamma*sigma),mu+gamma*sigma]',
        mu=float(bank['relative_residual'].mean()), sigma=float(bank['relative_residual'].std()))
    params['ro_calibration'] = 'training only; width grid 0:0.025:8 then local 0.001 refinement; original selection rules'
    write_json(out/'frozen_parameters.json', params)  # Freeze before accessing test demands.
    test = data['test_y'].astype(float)
    runs = {}
    forecasts = None
    # Reuse exact original traces: no re-estimation of the other five policies.
    for method in pipeline.METHODS:
        frame = pd.read_csv(base/f'detail_{method}.csv.gz')
        shape = test.shape
        arrays = {dest:frame[src].to_numpy().reshape(shape) for dest,src in
                  [('commencing','available'),('ending','ending'),('sold','sales'),('orders','order')]}
        arrays['plans'] = np.stack([frame[f'planned_order_h{h}'].to_numpy().reshape(shape) for h in range(1,5)],axis=1)
        arrays['targets'] = np.stack([frame[f'target_h{h}'].to_numpy().reshape(shape) for h in range(1,5)],axis=1)
        if forecasts is None:
            forecasts = np.stack([frame[f'forecast_h{h}'].to_numpy().reshape(shape) for h in range(1,5)],axis=1)
        runs[method] = arrays
    runs['RO'] = simulate(test, forecasts, bank['relative_residual'], best['gamma'], keep=True)
    def summarize(demand,sold,ending,commencing,orders):
        result = model.summarize(demand,sold,ending,commencing,orders)
        result.update(positive_stock_ratio=float(np.mean(ending>0)),
            positive_stock_sku_periods=int(np.count_nonzero(ending>0)),total_sku_periods=int(ending.size))
        return result
    for r in runs.values():
        r['summary'] = summarize(test,r['sold'],r['ending'],r['commencing'],r['orders'])
        r['controlled'] = summarize(test[1:],r['sold'][1:],r['ending'][1:],r['commencing'][1:],r['orders'][1:])
    pipeline.summarize = summarize
    pipeline.METHODS = (*pipeline.METHODS,'RO')
    pipeline.output_results(out,runs,test,forecasts,data['item_id'],bank['labels'],data['is_hot'])
    for method in pipeline.METHODS[:-1]:
        # Preserve original gzip bytes and avoid duplicating large Git blobs.
        shutil.copy2(base/f'detail_{method}.csv.gz',out/f'detail_{method}.csv.gz')
        (out/f'detail_{method}.csv').unlink()
    for name in ('feature_diagnostics.csv','forecast_validation.csv','forecast_test_metrics.csv','cluster_assignment.csv','risk_profiles.csv'):
        if (base/name).exists(): shutil.copy2(base/name,out/name)
    next_forecast = forecasts[:-1,0]
    write_json(out/'ro_test_coverage.json',dict(relative_lower=lower,relative_upper=upper,
        realized_next_period_interval_coverage=float(np.mean((test[1:]>=next_forecast*lower)&(test[1:]<=next_forecast*upper))),
        note='Diagnostic only; not used to select gamma; observed demands can fall outside the uncertainty set.'))
    report(version, out, params)
    print(f'[{version}] Selected gamma={best["gamma"]}; test RO={runs["RO"]["summary"]}',flush=True)

def report(version, out, params):
    old = version == VERSIONS[0]
    summary = pd.read_csv(out/'summary.csv').query("scope == 'all_28_periods'")
    metric = 'fill_rate' if old else 'positive_stock_ratio'
    feasible = summary[summary[metric]>=.95]
    if not old: feasible = feasible[feasible[metric]<=.955]
    winner = feasible.sort_values('inventory_demand_ratio').iloc[0] if len(feasible) else None
    lines = ['# 六种方法补货实验报告：'+('需求满足率版' if old else '正库存天数版'),'',
        ('满足测试服务要求的方法中，库存需求比最低的是'+LABELS[winner['method']]+f'（{winner["inventory_demand_ratio"]:.6f}）。') if winner is not None else '测试中没有方法满足指定服务范围；结果如实保留。',
        '本次增加传统区间RO，原有五种策略逐商品、逐期的订单、销量及库存保持原结果。RO参数仅在训练集内部选择，未使用测试集调参。','',
        '|方法|需求满足率|正库存天数比例|库存需求比↓|损失需求|', '|---|---:|---:|---:|---:|']
    for _,r in summary.iterrows(): lines.append(f'|{LABELS[r["method"]]}|{r.fill_rate:.4%}|{r.positive_stock_ratio:.4%}|{r.inventory_demand_ratio:.6f}|{r.lost_units:,.0f}|')
    lines += ['', '## 目标、服务口径与库存流转','',
      r'最小化 $J=\sum_{i,t}I_{it}/\sum_{i,t}d_{it}$。这是累计期末库存与需求之比，越低越好；并非通常定义的销售/平均库存周转次数。',
      r'需求满足率 $F=\sum s_{it}/\sum d_{it}$；正库存天数比例 $P=\sum\mathbf1\{I_{it}>0\}/(NT)$。',
      ('本版服务约束为 F≥95%，P为补充指标。' if old else '本版服务要求为95%≤P≤95.5%，F为补充指标。'),
      r'$a_{it}=I_{i,t-1}+q_{i,t-1}$，$s_{it}=\min(a_{it},d_{it})$，$I_{it}=(a_{it}-d_{it})_+$，$\ell_{it}=d_{it}-s_{it}$。',
      '提前期为1；期末观察需求后下单，下一期需求前到货。缺货直接丢失，不递延。每次形成未来四期预测与补货计划，仅执行第一期订单，其余在下次滚动重算。初始库存与零初始在途量沿用原实验。',
      '', '## 新增传统RO的数学模型及解读','',
      r'设 $m_{i,t+h}$ 是只使用已观测历史的预测，训练期样本外相对需求为 $r=d/\max(m,1)$。由训练残差计算全局均值μ与标准差σ。',
      r'需求不确定集合为 $\mathcal U_{it}(\Gamma)=\{\mathbf d: m_{i,t+h}\max(0,\mu-\Gamma\sigma)\le d_{i,t+h}\le m_{i,t+h}(\mu+\Gamma\sigma),\ h=1,\ldots,4\}$。',
      'Γ是区间宽度系数，不是跨期共享的不确定性预算；本RO不做聚类，因此可以与不分组DRO直接对照。μ、σ描述集合边界，并未在集合上赋予概率分布。',
      r'令 $U_h=m_h(\mu+\Gamma\sigma)$，$B_h$ 为计划需求发生前的库存。',
      (r'旧版鲁棒约束为 $(d_h-B_h)_+\le(1-\beta)d_h,\ \forall d_h\in\mathcal U$，等价于 $B_h\ge\beta U_h$。β取原版训练前推控制目标。' if old else r'新版鲁棒约束为 $B_h>d_h,\ \forall d_h\in\mathcal U$。整数库存的最小目标为 $S_h=\lfloor U_h\rfloor+1$，确保集合内需求实现后库存严格大于0。'),
      r'在这些约束下最小化非负持有库存；因目标对目标库存单调，最优目标取约束的最小值。旧版 $S_h=\beta U_h$，新版如上。令 $\widetilde I_0=I_t$，$q_h=\max(0,\lceil S_h-\widetilde I_{h-1}\rceil)$，$\widetilde I_h=\max(\widetilde I_{h-1}+q_h-m_h,0)$。仅执行$q_1$。',
      '这是箱式集合下可分离、无固定订货成本的库存目标策略的解析鲁棒对应式，不声称求得原始含服务指标的全局多期随机最优解。四期计划采用预测路径，后续决策通过滚动反馈实现。',
      '传统RO保护集合内的每个需求点；DRO保护分布模糊集内的最坏期望/概率。RO在集合内的逐期保护不等于测试总体服务保证：测试需求可能超出集合，首期也不受本轮订单控制。集合覆盖率见ro_test_coverage.json。',
      '', '## 训练选参与可复现性','',
      'Γ先搜索0到8、步长0.025，再在所选点附近以0.001细化。选参表为ro_validation.csv；冻结文件先写入磁盘，再评估测试需求。',
      ('旧版：前推训练1–36/验证37–60及重置训练29–60/验证1–28均满足原服务门槛，选择重置库存需求比最小者；最终残差沿用原版全60期拟合。' if old else '新版：训练29–60、验证1–28的正库存比例须在95%–95.5%，选库存需求比最小者；前推折只作诊断。最终残差沿用原版重置模型，不重新拟合。'),
      '重置折是训练数据内部的场景验证，包含较晚时期拟合、较早时期验证；它不是纯前向预测。测试Period_1重新起算，四期预测逐期吸收已经实现的测试需求。',
      f'冻结RO参数：Γ={params["selected"]["RO"]["gamma"]}；μ={params["selected"]["RO"]["mu"]:.6f}；σ={params["selected"]["RO"]["sigma"]:.6f}。',
      '原有五种方法的完整数学模型见[原数学报告](../outputs_v4/REPORT.md)。聚类依据见'+('[特征聚类说明](../docs/CLUSTERING.md)。' if old else '[特征聚类说明](../../../docs/rolling_dro_v3/CLUSTERING.md)。')+'本报告补充RO并列出六种方法完整测试结果。',
      '', '## 文件与运行','',
      '从仓库根目录安装原版本requirements_v3.txt，然后运行：', '```sh',
      f'python experiments/ro_extension/extend_with_ro.py --version {version}',
      'python experiments/ro_extension/verify_ro.py', '```',
      'summary.csv为六种方法汇总；results_by_period.csv为逐期结果；orders_*.csv为全部商品订单；detail_*.csv.gz为完整轨迹；decision_comparison_full.csv.gz为168,000行并列决策结果。Excel展示汇总及逐商品订单，HTML展示对比。工作簿重建命令见ro_extension/README.md。',
      '', '参考：[RSOME传统鲁棒优化：不确定集合与forall约束](https://xiongpengnus.github.io/rsome/ro_rsome)。']
    (out.parent/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Small CSVs for an Excel presentation; full trace stays in compressed CSV.
    pipeline_csv = lambda df,name: df.to_csv(out/name,index=False,encoding='utf-8-sig')
    pipeline_csv(summary,'excel_summary.csv')
    wide = pd.read_csv(out/'decision_comparison_full.csv')
    pipeline_csv(wide[wide.period==wide.period.max()], 'latest_decisions.csv')
    pipeline_csv(wide[wide.item_id.isin(wide.item_id.unique()[:12])], 'sample_decisions.csv')
    (out.parent/'.gitignore').write_text('outputs_v4/detail_*.csv\noutputs_v4/decision_comparison_full.csv\noutputs_v4/*.inspect.ndjson\n',encoding='utf-8')

def publish():
    for version in VERSIONS:
        out=ROOT/'experiments'/version/'with_ro/outputs_v4'
        for name in ('detail_RO.csv','decision_comparison_full.csv'):
            source=out/name
            if source.exists() and (not (out/(name+'.gz')).exists() or source.stat().st_mtime>(out/(name+'.gz')).stat().st_mtime):
                with source.open('rb') as src, (out/(name+'.gz')).open('wb') as dst:
                    with gzip.GzipFile(filename='',fileobj=dst,mode='wb',mtime=0) as z: shutil.copyfileobj(src,z)
        # Compute coverage in demand units, including forecasts below one.
        frame=pd.read_csv(out/'detail_RO.csv.gz',usecols=['demand','forecast_h1'])
        demand=frame.demand.to_numpy().reshape(28,6000)[1:]
        forecast=frame.forecast_h1.to_numpy().reshape(28,6000)[:-1]
        coverage=json.loads((out/'ro_test_coverage.json').read_text(encoding='utf-8'))
        coverage['realized_next_period_interval_coverage']=float(np.mean(
            (demand>=forecast*coverage['relative_lower'])&(demand<=forecast*coverage['relative_upper'])))
        write_json(out/'ro_test_coverage.json',coverage)
        files = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()
                 if p.is_file() and p.name!='SHA256.json' and p.suffix not in ('.ndjson',) and p.name not in ('detail_RO.csv','decision_comparison_full.csv')}
        write_json(out/'SHA256.json',files)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--version',choices=(*VERSIONS,'all'),default='all'); ap.add_argument('--publish',action='store_true'); a=ap.parse_args()
    if a.publish: publish()
    elif a.version=='all':
        for version in VERSIONS: subprocess.run([sys.executable,__file__,'--version',version],check=True)
    else: main(a.version)
