"""生成本地审阅文件清单和ZIP；不提交Git、不上传、不修改旧工程。"""
from pathlib import Path
import hashlib
import json
import platform
import sys
import zipfile
from importlib.metadata import version, PackageNotFoundError

ROOT=Path(__file__).resolve().parents[1]


def included(file):
    path=file.relative_to(ROOT)
    if any(p in {'__pycache__','.venv','node_modules','.git','.openai'} for p in path.parts): return False
    if file.suffix in {'.pyc','.zip'} or file.name=='.DS_Store': return False
    if path.parts[0]=='runs': return False
    # 单文件离线网页已包含构建快照，ZIP不再重复三份完整数据。
    if str(path).startswith(('website/app/dist/','website/app/.data-app-offline/')): return False
    if str(path)=='website/app/src/data.json': return False
    return str(path)!='verification/文件清单.json'


def main():
    dependencies={}
    for name in ['numpy','pandas','scipy','scikit-learn','xgboost','lightgbm','catboost','torch','openpyxl','joblib']:
        try: dependencies[name]=version(name)
        except PackageNotFoundError: dependencies[name]='当前解释器不可用'
    environment=dict(python=sys.version,platform=platform.platform(),architecture=platform.machine(),
        dependencies=dependencies,
        note='本机QA环境记录，不是精确可移植锁文件；ds解释器优先，torch环境site-packages补充。未修改系统安装。')
    (ROOT/'verification/环境.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2))
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and included(p))
    manifest=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,
                   sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    target=ROOT/'verification/文件清单.json'
    target.write_text(json.dumps(dict(scope='本地审阅ZIP内容；不等于批准公开清单',files=manifest),ensure_ascii=False,indent=2))
    archive=ROOT.parent/(ROOT.name+'_本地审阅.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for file in files+[target]: bundle.write(file,ROOT.name+'/'+str(file.relative_to(ROOT)))
    print(json.dumps(dict(archive=str(archive),files=len(files)+1,bytes=archive.stat().st_size,
                          sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
