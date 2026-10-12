"""One consistent CLI; each model file owns its estimator/loss/decision code."""
import argparse
import json
from pathlib import Path
import sys
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[1]
ML={'linear_regression','ridge','lasso','elastic_net','svr','random_forest','xgboost','lightgbm','catboost'}
TS={'naive','moving_average','exponential_smoothing','arima'}

@contextmanager
def arguments(values):
    previous=sys.argv
    sys.argv=[previous[0],*values]
    try:yield
    finally:sys.argv=previous

def call(fn,values=()):
    with arguments(values):return fn()

def historical(name):
    rows=json.loads((ROOT/'results/variants.json').read_text(encoding='utf-8'))
    selected=[r for r in rows if r['model_file']=='models/'+name+'.py']
    for r in selected:
        print(r['group'],r['id'],json.dumps(r['metrics'],ensure_ascii=False))
    return selected

def start(name,namespace,default_protocol='legacy'):
    parser=argparse.ArgumentParser(description=name+'：实际模型实现与历史结果。')
    parser.add_argument('--train',action='store_true',help='训练与评价；默认只读历史成绩。')
    parser.add_argument('--protocol',choices=['legacy','ready_immediate','lead_one'],default=default_protocol)
    parser.add_argument('--window',type=int,choices=[3,6,12],default=12,help='移动平均窗口。')
    parser.add_argument('--round',choices=['all','1','2','3','4'],default='all',help='MPC开发轮次。')
    parser.add_argument('--n-jobs',type=int,default=2)
    args=parser.parse_args()
    if not args.train:historical(name);return
    if args.n_jobs<1:parser.error('--n-jobs须为正。')
    missing=[p.name for p in [ROOT/'data/replenishment_train.xlsx',ROOT/'data/replenishment_test.xlsx'] if not p.is_file()]
    if missing:parser.error('缺少授权原始数据：'+', '.join(missing)+'；未生成虚构数据。')
    # Aggregate old comparison rows are reference evidence, never training labels.
    references=json.loads((ROOT/'results/reference_tables.json').read_text(encoding='utf-8'))
    for relative,content in references.items():
        target=(ROOT/relative).resolve()
        assert target.is_relative_to((ROOT/'results/generated').resolve())
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);target.write_text(content,encoding='utf-8')
    from common import prepare
    if name in ML or name=='baseline':
        if args.protocol=='legacy':
            if name=='baseline':call(namespace['train_legacy'])
            else:
                from common.ml_training import train
                call(train,['--models',name,'--n-jobs',str(args.n_jobs)])
        elif args.protocol=='ready_immediate':
            prepare.current()
            from common.ready_training import train
            call(train,['--models']+([] if name=='baseline' else [name]))
        else:
            from common.lead_one_training import train
            train(name,args.n_jobs)
    elif name in TS:
        if args.protocol!='legacy':parser.error('本项目时间序列历史训练仅有legacy口径，不伪装为其他已调优方案。')
        from common.ts_runner import train_one
        train_one(name,args.window,args.n_jobs)
    elif name.startswith('dro_'):
        if args.protocol!='legacy':parser.error('这三种DRO历史方案只有legacy口径。')
        label={'dro_moment':'moment_dro_ma12','dro_wasserstein':'wasserstein_dro_ma12','dro_chi_square':'chi_square_dro_ma12'}[name]
        call(namespace['train_legacy'],['--models',label])
    elif name in ('nn_scenario','dfl_improved'):
        if args.protocol!='ready_immediate':parser.error('该历史原型采用立即到货/当期x已知，请使用ready_immediate。')
        prepare.current();call(namespace['train_legacy'])
    elif name=='service_policy':
        if args.protocol!='legacy':parser.error('原始—对偶历史方案仅legacy。')
        prepare.service();call(namespace['train_legacy'],['--threads',str(args.n_jobs)])
    elif name=='online_learning':
        if args.protocol!='lead_one':parser.error('在线原内层方案仅lead_one。')
        prepare.current();prepare.lagged();call(namespace['train_legacy'])
    elif name=='turnover_dfl':
        if args.protocol!='lead_one':parser.error('原总量周转DFL仅lead_one。')
        prepare.current();prepare.lagged()
        checkpoint=ROOT/'results/generated/online_learning/initial_model.pt'
        if not checkpoint.is_file():
            from models.online_learning import train_legacy
            call(train_legacy)
        prepare.turnover();call(namespace['train_legacy'])
    elif name=='adaptive_mpc':
        if args.protocol!='lead_one':parser.error('分群MPC仅lead_one。')
        prepare.current();prepare.mpc()
        if args.round in ('all','1'):namespace['train_round1']()
        if args.round in ('all','2'):namespace['refine']()
        if args.round in ('all','3'):namespace['train_round3']()
        if args.round in ('all','4'):namespace['train_round4']()
    else:
        if args.protocol!='legacy':parser.error('此SPO成本/直接动作实现仅legacy。')
        values=[] if name=='spo_linear' else ['--n-jobs',str(args.n_jobs)]
        call(namespace['train_legacy'],values)
