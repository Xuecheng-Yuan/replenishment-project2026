"""当前运行的输入输出；默认不读取原工程或历史模型。"""
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('REPLENISHMENT_RUN_DIR', 'runs/main')).resolve()


def sha(path):
    """流式计算哈希，避免为了校验大文件额外占用内存。"""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    """统一UTF-8 JSON输出，拒绝写入NaN和无穷值。"""
    def convert(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, Path):
            return str(v)
        raise TypeError(f'不能序列化：{type(v).__name__}')

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False, default=convert), encoding='utf-8')


@lru_cache(maxsize=1)
def data():
    """每次从当前运行目录载入同一份已核验数据。"""
    path = ROOT / 'data/input.npz'
    if not path.exists():
        raise FileNotFoundError('请先执行 python run.py prepare，详见README。')
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    if sha(path) != manifest['input_sha256']:
        raise ValueError('输入数据在导入后被修改，请新建运行目录重新导入。')
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def freeze_selection(name, value):
    """保存验证期选择；测试评价不允许改写此文件。"""
    save(ROOT / 'models' / name / 'selection.json', dict(
        model=name, protocol='service_95_96_lead1',
        artifact_schema_version=2, source_code_sha256=source_fingerprint(),
        input_sha256=json.loads((ROOT/'data/manifest.json').read_text())['input_sha256'],
        selected_only_on_train_validation=True, **value))


def training_data():
    """训练分支屏蔽测试需求和原始x，评价分支才允许按时点读取真实测试。"""
    raw = data()
    return dict(raw, test_demand=np.zeros_like(raw['test_demand']),
                test_x=np.zeros_like(raw['test_x']))


def source_fingerprint():
    """用算法源码身份阻止改了配方却resume复用旧模型，文档修改不影响。"""
    package=Path(__file__).resolve().parent
    digest=hashlib.sha256()
    for file in sorted(package.rglob('*.py')):
        digest.update(str(file.relative_to(package)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()
