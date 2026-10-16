"""工作簿或标准NPZ导入；不把原始数据打包进提交材料。"""
import re
from pathlib import Path

import numpy as np

from .runtime import ROOT, save, sha


def workbook(path):
    """每张Period_k表含同一组SKU；显式检查排序、类别和初始库存。"""
    from openpyxl import load_workbook
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        sheets = sorted((int(m[1]), name) for name in book.sheetnames
                        if (m := re.fullmatch(r'Period_(\d+)', name, re.I)))
        if not sheets or [k for k, _ in sheets] != list(range(1, len(sheets) + 1)):
            raise ValueError('工作表必须从Period_1连续编号。')
        demands, features = [], []
        ids = categories = initial = None
        for k, name in sheets:
            rows = book[name].iter_rows(values_only=True)
            positions = {str(v).strip(): j for j, v in enumerate(next(rows))}
            columns = ['item_id', 'category', 'initial_inventory', f'demand_{k}']
            columns += [f'x{j}' for j in range(1, 21)]
            missing = set(columns) - positions.keys()
            if missing:
                raise ValueError(f'{name}缺少字段：{sorted(missing)}')
            records = list(rows)
            current_ids = np.array([str(r[positions['item_id']]) for r in records])
            current_cat = np.array([str(r[positions['category']]) for r in records])
            current_initial = np.array([r[positions['initial_inventory']] for r in records], float)
            if ids is None:
                ids, categories, initial = current_ids, current_cat, current_initial
            elif not np.array_equal(ids, current_ids):
                raise ValueError(f'{name}商品顺序不一致。')
            elif not np.array_equal(categories, current_cat):
                raise ValueError(f'{name}商品类别发生变化。')
            demands.append([r[positions[f'demand_{k}']] for r in records])
            features.append([[r[positions[f'x{j}']] for j in range(1, 21)] for r in records])
        return dict(item_ids=ids, categories=categories, initial=initial,
                    demand=np.array(demands, float).T,
                    x=np.array(features, float).transpose(1, 0, 2))
    finally:
        book.close()


def validate(raw):
    """固定本研究的数据长度60/28期，允许用更少SKU做小样本链路测试。"""
    n = len(raw['item_ids'])
    for key, shape in [('train_demand', (n, 60)), ('test_demand', (n, 28)),
                       ('train_x', (n, 60, 20)), ('test_x', (n, 28, 20)), ('initial', (n,))]:
        array = np.asarray(raw[key])
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f'{key}维度应为{shape}，且不能有缺失或无穷值。')
        if key.endswith('demand') or key == 'initial':
            if (array < 0).any():
                raise ValueError(f'{key}不能为负。')
    if n < 8 or len(np.unique(raw['item_ids'])) != n or len(raw['categories']) != n:
        raise ValueError('SKU至少8个，ID须唯一，类别须逐行对应。')


def prepare(train=None, test=None, npz=None):
    """导入一次，以后所有模型共用同一个规范输入。"""
    target = ROOT / 'data/input.npz'
    if target.exists():
        raise FileExistsError('当前运行已有输入，请换一个--workdir，避免覆盖已有实验。')
    if npz:
        with np.load(npz, allow_pickle=False) as source:
            raw = dict(source)
        sources = [npz]
    else:
        a, b = workbook(train), workbook(test)
        if not np.array_equal(a['item_ids'], b['item_ids']):
            raise ValueError('训练集和测试集SKU顺序不一致。')
        if not np.array_equal(a['categories'], b['categories']):
            raise ValueError('训练集和测试集商品类别不一致。')
        raw = dict(train_demand=a['demand'], test_demand=b['demand'],
                   train_x=a['x'], test_x=b['x'], initial=a['initial'],
                   item_ids=a['item_ids'], categories=a['categories'])
        sources = [train, test]
    validate(raw)
    # 消除object数组，使NPZ不需要启用不安全的pickle加载。
    raw['item_ids'] = np.asarray(raw['item_ids']).astype(str)
    raw['categories'] = np.asarray(raw['categories']).astype(str)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **raw)
    save(ROOT / 'data/manifest.json', dict(
        sources=[dict(file=Path(p).name, sha256=sha(p)) for p in sources],
        input_sha256=sha(target), skus=len(raw['initial']), train_periods=60, test_periods=28,
        initial_convention='训练工作簿的初始库存；验证预热和测试回放均按统一约定重置，无初始在途'))
