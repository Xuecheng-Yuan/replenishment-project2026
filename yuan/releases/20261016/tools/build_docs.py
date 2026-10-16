"""从唯一模型目录和已核验指标生成中文索引及每个模型的决策说明。"""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text())


def clean_copy(value):
    """展示文字使用统一研究术语，不改变任何公式、数据或上游原文。"""
    if isinstance(value,str):
        return value.replace('新老师目标','服务水平目标').replace('老师真实指标','真实评价指标').replace('老师原分式模型','原始分式模型')
    if isinstance(value,list): return [clean_copy(v) for v in value]
    if isinstance(value,dict): return {k:clean_copy(v) for k,v in value.items()}
    return value


def steps(model):
    if model.startswith('kristina_'):
        return ['只用训练资料拟合预测、分组与基础风险水位。',
                '观察本期真实需求，更新当前期末库存及累计服务计数。',
                '按剩余周期的服务预算，用训练残差和已发生误差估计下一期服务响应。',
                '二分选择水位倍率；传统(s,S)还检查当前库存是否低于触发水位。',
                '将目标库存减去当前库存，向上取整下单；只执行四期计划的第一期订单。']
    if model in ['spo_linear','xgboost_spo','lightgbm_spo','direct_xgboost','direct_lightgbm']:
        return ['截至本期期末，使用历史需求、滞后x和当前真实库存构造状态。',
                '线性/直接模型的需求参考来自MA6；树SPO的需求中心来自前缀拟合的基础ML。',
                '模型预测各有限动作的成本，选择一个动作；它不是直接预测下一期真实需求。',
                'SPO构造中心＋动作×误差尺度，直接决策构造中心×动作；再加验证集已选风险偏移。',
                '目标库存扣掉当前期末库存，非负部分向上取整下单，下一期到货。']
    if model=='service_policy':
        return ['历史特征、SKU嵌入、当前库存和历史参考量进入直接策略网络。',
                '网络输出目标库存，加验证集选择的风险修正。',
                '减去当前库存，非负取整后形成下一期订单；测试不反向训练。']
    if model in ['nn_scenario','dfl_improved','turnover_dfl','online_learning']:
        return ['使用38维历史特征和SKU嵌入，输出对数需求的位置与尺度。',
                '依据拟合前缀的相关结构生成8个需求情景，当前库存进入内层。',
                '执行有限步内层优化，得到下一期订单；并非精确全局最优或商业求解器结果。',
                'nn_scenario只训练概率预测；DFL用库存与服务代理损失微调，4期/7期指训练回放窗口。',
                '测试执行整数订单，重新计算真实服务水平；仅online_learning在需求揭示后更新模型。']
    if model=='adaptive_mpc':
        return ['从历史特征滚动预测四期需求与不确定性。',
                '结合训练期PCA/Kmeans分组、当前库存和情景优化四期订单。',
                '只执行第一期，观察新需求后重新规划；验证集选风险参数，测试不调参。']
    if model in ['equal_mean','global_stacking','sku_stacking','sku_router']:
        return ['用时间顺序OOF预测学习融合/选择规则，不按测试误差挑专家。',
                '将14个历史统计与ML专家的下一期预测按已冻结权重组合，硬路由只选一个专家。',
                '使用验证集校准的风险尺度和系数形成目标库存。',
                '扣掉当前库存、非负取整，形成下一期订单；不是把专家订单直接相加。']
    return ['只使用截至本期已观测的需求；ML还使用滞后一期及更早的x。',
            '历史统计或ML预测下一期需求；普通ML训练不直接优化服务约束。',
            '加上验证集校准的风险缓冲，得到下一期目标库存。',
            '扣除当前期末库存，非负部分向上取整下单；提前期一期。']


def main():
    catalog=read(ROOT/'configs/model_catalog.json');metrics={r['model']:r for r in read(ROOT/'results/benchmark/metrics.json')}
    index=['# 模型索引与代码入口','','个人29个模型按Baseline→时间序列→ML→SPO→DFL组织，扩展单列；Kristina最新6个模型独立署名。',
           '网站、表格、逐期回放和此索引共同读取configs/model_catalog.json。不是一模型复制一套评价器。','',
           '| 模型 | 负责人 | 当前版本 | 实现 | 中文文档 |','|---|---|---|---|---|']
    decisions=['# 每个模型怎样决定补多少货','','共同时间顺序：上期订单到货→本期需求发生→销量与期末库存→决定下期订单。',
               '同一服务指标不等于同一训练协议；普通ML是先预测再决策，SPO学习有限动作成本，DFL通过决策损失微调概率网络。','']
    all_results=['# 当前发布模型测试结果','','既有正式轨迹的独立重算；不是这次整理后全量重训。全部6000SKU、28期，包含首期。',
        '服务水平是期末库存>0的SKU-期比例；Fill Rate是销量/需求；J是总期末库存/总需求，越低越好。','',
        '| 模型 | 负责人 | 库存需求比J | 服务水平 | Fill Rate | 状态 |','|---|---|---:|---:|---:|---|']
    for row in catalog:
        name=row['model'];m=metrics[name];path=ROOT/row['document']
        if name.startswith('kristina_'):
            differences={
                'kristina_DRO_cluster':'按训练特征聚类，组内汇集残差，构造带正半径的分布鲁棒风险水位。',
                'kristina_DRO_global':'不按商品分组，使用全体训练残差构造共同风险刻画。',
                'kristina_DRO_category':'按爆品/长尾组汇集残差，组别改变风险水位估计。',
                'kristina_sS':'使用触发水位s和补至水位S，库存低于s才补至S。',
                'kristina_SAA':'使用经验残差分布，DRO半径为零，不额外增加分布鲁棒保护。',
                'kristina_RO':'用传统残差不确定区间构造风险水位，不等同于概率分布集合DRO。'}
            text=f'# {row["name"]}（{name}）\n\n负责人：Kristina。原始算法源码不改写、不并入个人方向成果。\n\n## 1. 输入和训练\n\n训练60期需求、历史匿名特征与共同初始库存；测试28期按时点揭示需求。特征用于分组，不能提前读未来x。正式版本包含训练内部回溯重启校准，不等同于严格前向验证。\n\n## 2. 本模型与其他方法的区别\n\n'+differences[name]+'六方法均叠加共同服务预算反馈，不能只按基础DRO/RO名称解释实际执行结果。\n\n## 3. 服务预算怎样修正订单\n\n令C_t为截至t期累计库存>0的商品—期次数，N为商品数，T=28。目标p=0.9525，b为训练内部选择的偏移。剩余服务请求为\n\n$$\np_{t+1}^{req}=\\operatorname{clip}\\left(\\frac{pNT-C_t}{N\\max(T-t,1)}+b,0,1\\right).\n$$\n\n只用训练残差和已经观察到的测试误差估计服务响应，再二分寻找水位倍率a。基础预测为f、风险倍数为h，则目标库存及订单为\n\n$$\nS_{i,t+1}=\\lfloor a_t f_{i,t+1}h_i\\rfloor+1,\\qquad q_{it}=\\max\\{0,\\lceil S_{i,t+1}-I_{it}\\rceil\\}.\n$$\n\n传统(s,S)还要求当前库存低于缩放后的触发水位才订货。+1用于严格正库存的整数水位；经验服务响应不是未来真实服务水平的保证。末期订单仍留作在途。\n\n## 4. 推导与限制\n\n若剩余M个商品—期，想达到总体p，需要再实现约pNT-C_t次正库存，因此以所需次数除以M构造请求；请求截断在0与1。已实现库存不能撤回，也可能使目标不可达。动态缩放后不直接继承固定DRO水位的原始保证。完整基础风险模型见[原始数学报告](../../partners/kristina/upstream/experiments/service95_revision/REPORT.md)。'
        elif path.exists():
            text=path.read_text().split('## 5.')[0].rstrip()
            text=text.replace('方向：决策学习。','方向：SPO/有限动作决策学习。')
        else:
            text=f'# {row["name"]}（{name}）\n\n负责人：Kristina。原始算法与报告保留在partners/kristina/upstream，未改动其算法源码。'
        sequence=steps(name)
        result=f'库存需求比 **{m["inventory_demand_ratio"]:.6f}**；服务水平 **{m["ready_rate"]:.4%}**；Fill Rate **{m["fill_rate"]:.4%}**。状态：**{m["test_status"]}**。'
        text+='\n\n## 5. 当前发布版本与真实测试结果\n\n版本：`'+row['version']+'`。\n\n'+result+'\n\n'+row['protocol']+'。\n\n以上为既有正式实验轨迹复算，不是新代码全量重训。细化版没有保存的样本内训练指标不从旧粗网格版本移用。\n'
        text+='\n## 6. 补货决策逐步说明\n\n'+'\n'.join(f'{i+1}. {s}' for i,s in enumerate(sequence))+'\n'
        text+='\n## 7. 对应代码、参数和回放\n\n实现：`'+row['code']+'`。\n\n冻结参数：[`'+row['parameters']+'`](../../'+row['parameters']+')。\n\n'
        if not name.startswith('kristina_'):
            text+='```bash\npython run.py --workdir runs/main train --models '+name+'\npython run.py --workdir runs/main evaluate --models '+name+'\n```\n\n'
        else:
            text+='Kristina代码作为独立上游项目运行，避免与个人方向的同名Python包混用。先看[独立运行说明](../../partners/kristina/README.md)。\n\n'
        text+='查看已保存订单，不需要重新训练：\n\n```bash\npython run.py replay --model '+name+' --sku 1 --through 28 --output runs/replay/'+name+'_sku1.csv\n```\n\n训练与评价不读取历史指标作预测。测试不达标与失败候选如实保留；当前实现为有限计算预算，不声称理论全局最优。\n'
        if name in ['spo_linear','xgboost_spo','lightgbm_spo','direct_xgboost','direct_lightgbm']:
            p=m['selection']
            text+='\n### 当前实际执行的动作\n\n`A = '+str(p['actions'])+'`，`z = '+str(p['z'])+'`，`λ = '+str(p['penalty'])+'`。\n\n新训练入口比较粗网格与训练成本分布细化网格，仅在验证选择，不再硬编码所有模型执行0、0.5、…、4。执行对齐C/D为历史诊断，详见results/research，不冒充本入口又完整重跑了四版本实验。\n'
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
        dossier_path=ROOT/'artifacts/methods'/f'{name}.json';dossier=clean_copy(read(dossier_path))
        if name in ['spo_linear','xgboost_spo','lightgbm_spo','direct_xgboost','direct_lightgbm']:
            actions=m['selection']['actions']
            dossier['training']=dossier['training'].replace('9个动作a=0,.5,…,4', '冻结动作集合A='+str(actions))
        if name.startswith('kristina_'):
            dossier['blocks']=[dict(type='text',value=differences[name]+' 六方法另有公共服务预算反馈。'),
                dict(type='formula',tex=r'S_{i,t+1}=\lfloor a_t f_{i,t+1}h_i\rfloor+1',
                     mathml='<math display="block"><mrow><mi>S</mi><mo>=</mo><mo>⌊</mo><mi>a</mi><mi>f</mi><mi>h</mi><mo>⌋</mo><mo>+</mo><mn>1</mn></mrow></math>',
                     explanation='a是仅用已发生需求校准的公共倍率；f是下一期预测，h是该方法的基础风险水位。'),
                dict(type='text',value='截至t期累计服务次数为C，剩余次数需求=p×N×28−C。将其除以剩余商品—期数，加训练内部选出的偏移，截断到[0,1]，得到下一期服务请求。用训练残差与已发生误差估计响应，二分选择倍率；(s,S)还检查触发水位。详细公式可下载本模型MD，完整基础模型见上游REPORT.md。')]
        dossier['decision_steps']=sequence;dossier['markdown']=text
        dossier_path.write_text(json.dumps(dossier,ensure_ascii=False,indent=2))
        index.append(f'| {row["name"]} | {row["owner"]} | {row["version"]} | `{row["code"]}` | [{name}](models/{name}.md) |')
        decisions+=['## '+row['name'],'','实现：`'+row['code']+'`。','']+[f'{i+1}. {s}' for i,s in enumerate(sequence)]+['',result,'']
        all_results.append(f'| {row["name"]} | {row["owner"]} | {m["inventory_demand_ratio"]:.6f} | {m["ready_rate"]:.4%} | {m["fill_rate"]:.4%} | {m["test_status"]} |')
    (ROOT/'docs/模型索引.md').write_text('\n'.join(index))
    (ROOT/'docs/补货决策总览.md').write_text('\n'.join(decisions))
    (ROOT/'results/benchmark/README.md').write_text('\n'.join(all_results))
    print('DOCS',len(catalog),'model documents with decision steps and measured metrics')


if __name__=='__main__':main()
