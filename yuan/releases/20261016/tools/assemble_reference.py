"""把已有正式轨迹整理成唯一结果目录；只复算，不训练，不访问桌面ML工程。"""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import shutil
import sys
import zlib

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from replenishment.evaluation import metrics, validate
from replenishment.registry import MODELS, TITLES, FORECAST_MODELS, DECISION_MODELS


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                              default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def family(name):
    if name == 'baseline': return 'Baseline'
    if name in FORECAST_MODELS[:5]: return '时间序列'
    if name in FORECAST_MODELS: return 'ML · PTO'
    if name in DECISION_MODELS: return 'SPO · 决策学习'
    if name in ['online_learning', 'adaptive_mpc']: return '扩展 · 在线与MPC'
    if name in ['equal_mean', 'global_stacking', 'sku_stacking', 'sku_router']: return '扩展 · 融合与选择'
    return '概率网络 · DFL'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-source', type=Path, required=True,
                        help='只在首次整理历史结果时使用；训练程序不依赖此目录。')
    args = parser.parse_args()
    project = args.project_source.resolve()
    old = project/'outputs/teacher_retrained_20261013'
    refinement = project/'outputs/spo_action_refinement_20261015'
    site = ROOT/'website/app'
    snapshot = read(project/'reports/replenishment_lab_20261015/src/data.json')
    prior_rows = {r['model']: r for r in snapshot['queries']['models']['rows']}
    prior_docs = {r['model']: json.loads(r['payload']) for r in snapshot['queries']['methods']['rows']}
    raw = dict(np.load(old/'数据/processed/input.npz', allow_pickle=False))
    panel = {k: raw[k] for k in ['train_demand','test_demand','initial','item_ids','categories']}
    np.savez_compressed(ROOT/'artifacts/panel.npz', **panel)
    selected_spo = {r['model']: r for r in read(refinement/'图表/全部结果.json')['results']
                    if r['validation_selected']}
    records, catalog, source_files = [], [], []

    def register(name, title, owner, group, arrays, prediction, selected, version,
                 protocol, code, files, dossier):
        a,b,s,q = arrays
        validate(raw['test_demand'], raw['initial'], a,b,s,q)
        m = metrics(raw['test_demand'], a,b,s,q)
        row = dict(model=name, name=title, owner=owner, family=group, version=version,
                   protocol=protocol, code=code, **m)
        row['test_status'] = '目标带内' if .95<=m['ready_rate']<=.96 else '低于95%' if m['ready_rate']<.95 else '高于96%'
        row['selection'] = selected
        records.append(row)
        np.savez_compressed(ROOT/'artifacts/trajectories'/f'{name}.npz', ending=b, orders=q,
                            forecast=prediction if prediction is not None else np.empty((0,)))
        save(ROOT/'configs/models'/f'{name}.json', dict(model=name,version=version,parameters=selected))
        save(ROOT/'artifacts/methods'/f'{name}.json', dossier)
        catalog.append(dict(model=name, name=title, owner=owner, family=group, version=version,
                            protocol=protocol, code=code, document=f'docs/models/{name}.md',
                            parameters=f'configs/models/{name}.json',
                            trajectory=f'artifacts/trajectories/{name}.npz'))
        for file in files:
            source_files.append(dict(source=str(file.relative_to(project)) if file.is_relative_to(project) else str(file.relative_to(ROOT)), sha256=sha(file)))
        print('VERIFIED', name, round(m['ready_rate'],6), round(m['inventory_demand_ratio'],6), flush=True)

    for name in MODELS:
        parent = project/'outputs/sku_stacking_20261013' if family(name)=='扩展 · 融合与选择' else old
        version = 'service_retrained_20261013'
        path = parent/'results'/name/'trajectory.npz'
        selected = read(parent/'models'/name/'selection.json') if (parent/'models'/name/'selection.json').exists() else prior_rows[name].get('parameters', '{}')
        if isinstance(selected,str): selected=json.loads(selected)
        if name in selected_spo:
            chosen=selected_spo[name]
            version='spo_refinement_20261015/'+chosen['version']
            path=refinement/'results'/name/chosen['version']/'trajectory.npz'
            selected=dict(chosen['parameters'], validation=chosen['validation'],
                          validation_selected=True, version=chosen['version'])
        z=np.load(path,allow_pickle=False)
        np.testing.assert_array_equal(z['demand'], raw['test_demand'])
        arrays=[z[k] for k in ['available','ending','sales','orders']]
        prediction=None
        if name in FORECAST_MODELS:
            archive=np.load(parent/'models'/name/'policy.npz',allow_pickle=False)
            prediction=archive['forecast'][:,:28]
        code=('replenishment/training/forecast.py' if name in FORECAST_MODELS else
              'replenishment/training/decision.py + replenishment/training/action_grid.py' if name in DECISION_MODELS else
              'replenishment/training/ensemble.py' if family(name)=='扩展 · 融合与选择' else
              'replenishment/training/mpc.py' if name=='adaptive_mpc' else 'replenishment/training/neural.py')
        dossier=prior_docs.get(name,dict(data='时间顺序OOF专家预测。',training='只在训练资料内学习融合权重和补货参数。',
            blocks=[],decision='先融合预测，再按目标库存规则扣减当前库存下单。',limits='融合不保证优于单模型。',parameters={},source='历史融合实验'))
        dossier['parameters']=selected
        dossier['release_version']=version
        dossier['code']=code
        if name in selected_spo:
            dossier['training']+='\n\n当前版本按验证结果选择：'+selected_spo[name]['version']+'。细化范围只由拟合段oracle成本分布确定，保留0和4端点。训练/验证成绩不从旧版本挪用。'
            dossier['decision']+='\n\n本版本实际动作网格：'+str(selected['actions'])+'；风险偏移z='+str(selected['z'])+'。动作编号不是固定的需求倍数。当前库存参与动作特征及目标库存扣减。'
            if name=='direct_lightgbm': dossier['limits']+='\n\n当前验证选中版本在测试服务水平低于95%，必须保留失败标记，不把原版测试成绩替换过来。'
        protocol='前42期拟合；第13—60期连续库存预热，第43—60期验证评分；最终60期重拟合'
        if family(name)=='扩展 · 融合与选择': protocol='时间顺序OOF训练融合权重，训练内部验证校准，最终专家用60期拟合'
        title=prior_rows[name]['name']
        if name=='baseline': title='历史均值 Baseline'
        if name in selected_spo and selected_spo[name]['version']=='B_refine_only': title+=' · 动作细化'
        register(name,title,'yuan xuecheng',family(name),arrays,prediction,selected,version,protocol,code,
                 [path, path.with_name('summary.json')],dossier)

    upstream=ROOT/'partners/kristina/upstream'
    partner=upstream/'experiments/service95_revision/outputs_v4'
    frozen=read(partner/'frozen_parameters.json')
    summary=pd.read_csv(partner/'summary.csv').query("scope=='all_28_periods'").set_index('method')
    names={'DRO_cluster':'特征聚类 DRO','DRO_global':'不分组 DRO','DRO_category':'爆品/长尾 DRO',
           'sS':'传统 (s,S)','SAA':'零半径经验模型','RO':'传统区间 RO'}
    ids=[str(i) for i in raw['item_ids']]
    index=pd.MultiIndex.from_product([ids,range(1,29)],names=['item_id','period'])
    for method,title in names.items():
        path=partner/f'detail_{method}.csv.gz'
        frame=pd.read_csv(path, dtype={'item_id':str})
        assert not frame.duplicated(['item_id','period']).any() and len(frame)==6000*28
        frame=frame.set_index(['item_id','period']).reindex(index)
        assert not frame.isna().any().any()
        np.testing.assert_array_equal(frame.demand.to_numpy().reshape(6000,28),raw['test_demand'])
        arrays=[frame[k].to_numpy().reshape(6000,28) for k in ['available','ending','sales','order']]
        prediction=np.full((6000,28),np.nan)
        prediction[:,1:]=frame.forecast_h1.to_numpy().reshape(6000,28)[:,:27]
        m=metrics(raw['test_demand'],*arrays)
        for key in ['fill_rate','inventory_demand_ratio']:
            np.testing.assert_allclose(m[key],summary.loc[method,key],rtol=1e-12)
        np.testing.assert_allclose(m['ready_rate'],summary.loc[method,'positive_stock_ratio'],rtol=1e-12)
        name='kristina_'+method
        protocol='最终60期拟合；三轮前向诊断＋训练29—60拟合/1—28回溯反馈校准；测试只吸收已发生需求'
        dossier=dict(data='训练60期需求与历史x1—x20；x用于聚类，不提前使用测试未来x。',
          training='采用固定Git提交aec11c8的正式服务预算反馈版。训练内部回溯重启留出不属于严格前向验证。',
          blocks=[dict(type='text',value='基础方法决定逐商品风险水位；公共反馈根据已经发生的服务水平和需求误差，修正下一期总体服务预算。机制探索实验是诊断，不替换正式六方案。')],
          decision='本期需求揭示后：累计服务计数→计算剩余服务预算→训练残差与已发生误差估计服务响应→二分下一期水位倍率→扣减期末库存→整数订单；仅执行四期计划的第一期。传统(s,S)另保留触发水位。',
          limits='执行策略是基础DRO/RO/经验水位加公共反馈，不能把固定DRO机会约束保证直接套到动态缩放后策略。目标95%—95.5%，并非逐SKU95%。',
          parameters=frozen,code='partners/kristina/upstream/experiments/ro_extension/service_band_revision.py:run_budget',
          release_version='aec11c8/service95_revision',source='Kristina固定GitHub提交的正式逐SKU轨迹')
        register(name,title+' · 服务反馈','kristina','Kristina · 鲁棒优化',arrays,prediction,frozen,
                 'aec11c8/service95_revision',protocol,dossier['code'],[path,partner/'summary.csv'],dossier)

    save(ROOT/'configs/model_catalog.json',catalog)
    save(ROOT/'results/benchmark/metrics.json',records)
    fields=['model','name','owner','family','version','ready_rate','fill_rate','inventory_demand_ratio',
            'average_ending_total_inventory','test_status','protocol','code']
    pd.DataFrame(records)[fields].to_csv(ROOT/'results/benchmark/metrics.csv',index=False,encoding='utf-8-sig')
    save(ROOT/'results/benchmark/provenance.json',dict(sources=source_files,
        note='既有正式实验轨迹的独立重算，不代表本次整理后全量重训。今日两个桌面ML项目及其混合对照分支未纳入。'))
    for src,dest in [(refinement/'计算结果.md','SPO动作细化_实验报告.md'),
                     (project/'outputs/dfl_fill_guarded_20261015/计算结果.md','DFL满足率改进_未采纳.md')]:
        shutil.copy2(src,ROOT/'results/research'/dest)
    save(ROOT/'results/research/spo_variants.json',read(refinement/'图表/全部结果.json')['results'])
    save(ROOT/'artifacts/sku_rows.json',snapshot['queries']['skus']['rows'])
    save(ROOT/'artifacts/experiments.json',snapshot['queries']['experiments']['rows'])
    print('COMPLETE',len(records),'models; copied metrics are not new training scores')


if __name__=='__main__':
    main()
