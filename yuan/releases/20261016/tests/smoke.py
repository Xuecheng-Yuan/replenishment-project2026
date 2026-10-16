"""合成数据端到端验收；不属于正式6000SKU实验。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from .test_core import fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['baseline', 'ridge', 'spo_linear'])
    parser.add_argument('--output', type=Path, help='可选保存验收记录，默认仅显示')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    # 临时目录保留本次输出便于审计；不写入原模型工件目录。
    temp = Path(tempfile.mkdtemp(prefix='yuan_smoke_'))
    source = temp / 'synthetic.npz'
    np.savez_compressed(source, **fixture())
    work = temp / 'run'
    def run(*parts):
        subprocess.run([sys.executable, str(root / 'run.py'), '--workdir', str(work), *parts],
                       cwd=root, check=True)
    run('prepare', '--npz', str(source))
    run('train', '--models', *args.models)
    run('evaluate', '--models', *args.models)
    run('compare')
    rows = [json.loads(p.read_text()) for p in (work / 'results').glob('*/test/summary.json')]
    expected = 29 if args.models == ['all'] else len(args.models)
    assert len(rows) == expected, (len(rows), expected)
    for row in rows:
        out = work / 'results' / row['model'] / 'test'
        assert (out / 'sku_metrics.csv').exists()
        assert (out / 'period_metrics.csv').exists()
        assert row['skus'] == 24 and row['periods'] == 28
    record = dict(status='passed', scope='合成24SKU、60/28期端到端检查，不能作为正式成绩',
                  models=[r['model'] for r in rows], workdir=str(work), python=sys.version)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
