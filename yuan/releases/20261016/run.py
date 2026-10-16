"""唯一命令入口。先解析目录，再导入运行模块，不依赖原项目路径。"""
import argparse
import os
import warnings
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Yuan补货模型独立提交包')
    parser.add_argument('--workdir', default='runs/main', help='当前实验的输入、模型、结果目录')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='列出全部29个模型')
    p=sub.add_parser('reference', help='只读查看已核验的历史正式指标，不冒充新训练结果')
    p.add_argument('--owner',choices=['all','yuan','kristina'],default='all')
    p=sub.add_parser('replay', help='把本地审阅附件中某模型某SKU的真实逐期决策导出CSV')
    p.add_argument('--model',required=True)
    p.add_argument('--sku',required=True)
    p.add_argument('--through',type=int,choices=range(1,29),default=28)
    p.add_argument('--output',default='runs/replay/decision.csv')
    p=sub.add_parser('serve', help='仅在本机启动离线网站，不发布到互联网')
    p.add_argument('--port',type=int,default=8793)
    p = sub.add_parser('prepare', help='导入数据，禁止覆盖已有输入')
    p.add_argument('--train', type=Path)
    p.add_argument('--test', type=Path)
    p.add_argument('--npz', type=Path, help='已有规范input.npz，与Excel方式二选一')
    for command in ['train', 'evaluate']:
        p = sub.add_parser(command)
        p.add_argument('--models', nargs='+', required=True, help='模型ID，或all')
        if command == 'train':
            p.add_argument('--resume', action='store_true', help='跳过已完成的模型，不覆盖')
    sub.add_parser('compare', help='汇总当前运行的实际测试结果')
    args = parser.parse_args()
    # LightGBM给NumPy列自动起名后会产生此非错误提示；仅屏蔽这一条，不吞其他警告。
    warnings.filterwarnings('ignore', message='X does not have valid feature names, but LGBMRegressor was fitted with feature names')
    os.environ['REPLENISHMENT_RUN_DIR'] = str(Path(args.workdir).resolve())
    from replenishment.registry import MODELS, TITLES, group
    if args.command=='reference':
        from replenishment.reference import compare
        compare(args.owner)
        return
    if args.command=='replay':
        from replenishment.reference import replay
        replay(args.model,args.sku,args.through,args.output)
        return
    if args.command=='serve':
        from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
        from functools import partial
        folder=Path(__file__).resolve().parent/'website/offline'
        if not (folder/'index.html').exists(): folder=Path(__file__).resolve().parent/'website/app/dist'
        server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(SimpleHTTPRequestHandler,directory=str(folder)))
        print(f'离线本地预览：http://127.0.0.1:{args.port}/；Ctrl+C停止，不对外发布。',flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return
    if args.command == 'list':
        for name in MODELS:
            print(f'{name:24} {group(name):10} {TITLES[name]}')
        return
    if args.command == 'prepare':
        if bool(args.npz) == bool(args.train or args.test) or (not args.npz and not (args.train and args.test)):
            parser.error('请提供--npz，或者同时提供--train和--test。')
        from replenishment.data import prepare
        prepare(args.train, args.test, args.npz)
        print('数据核验与导入完成。')
        return
    if args.command == 'compare':
        import json
        import pandas as pd
        from replenishment.runtime import ROOT
        rows = [json.loads(p.read_text()) for p in (ROOT / 'results').glob('*/test/summary.json')]
        if not rows:
            parser.error('当前运行没有测试结果，请先evaluate。')
        cols = ['model', 'ready_rate', 'fill_rate', 'inventory_demand_ratio',
                'average_ending_total_inventory', 'interval_feasible']
        frame = pd.DataFrame(rows)[cols].sort_values('inventory_demand_ratio')
        frame.to_csv(ROOT / 'results/comparison.csv', index=False, encoding='utf-8-sig')
        print(frame.to_string(index=False))
        return
    names = MODELS if args.models == ['all'] else list(dict.fromkeys(args.models))
    unknown = set(names) - set(MODELS)
    if unknown:
        parser.error(f'未知模型：{sorted(unknown)}；先运行list查看ID。')
    if args.command == 'train':
        from replenishment.training.pipeline import train_models
        train_models(names, args.resume)
    else:
        from replenishment.inference import evaluate_models
        evaluate_models(names)


if __name__ == '__main__':
    main()
