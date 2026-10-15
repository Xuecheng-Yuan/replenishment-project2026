"""Build the rolling-validation/full-60 package; refuse existing destinations."""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess
from extend_with_ro import ROOT
from constraint_ablation import HOME

def main(dest):
    dest=dest.resolve()
    if dest.exists():raise FileExistsError(dest)
    project=dest/'项目';project.mkdir(parents=True)
    prefixes=('experiments/demand_fill_v3/','experiments/rolling_dro_v3/',
        'experiments/ro_extension/','experiments/version_comparison/','docs/rolling_dro_v3/',
        '汇报成果/2026-10-15/','scripts/')
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf-8').split('\0')
    paths={Path(p) for p in tracked if p.startswith(prefixes)}
    paths.add(Path('experiments/version_comparison/HISTORICAL_COMPARISON_32.md'))
    paths.update(p.relative_to(ROOT) for p in (ROOT/'experiments/ro_extension').iterdir() if p.is_file())
    for folder in ('scripts','汇报成果/2026-10-15'):
        paths.update(p.relative_to(ROOT) for p in (ROOT/folder).rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.name!='最近启动状态.json')
    for p in HOME.rglob('*'):
        if not p.is_file() or p.name.endswith('.inspect.ndjson') or '__pycache__' in p.parts:continue
        if p.suffix=='.csv' and (p.name.startswith('detail_') or p.name=='decision_comparison_full.csv'):continue
        paths.add(p.relative_to(ROOT))
    paths.update(Path('data/raw')/name for name in ('replenishment_train.xlsx','replenishment_test.xlsx'))
    for relative in sorted(paths):
        target=project/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target)
    cache=ROOT/'experiments/rolling_dro_v3/.cache'
    if cache.exists():shutil.copytree(cache,project/'experiments/rolling_dro_v3/.cache')
    copies={ROOT/'experiments/rolling_dro_v3/requirements_v3.txt':'requirements.txt',
        ROOT/'experiments/demand_fill_v3/with_ro/outputs_v4/replenishment_comparison.xlsx':'旧版_需求满足率_六方法.xlsx',
        HOME/'new_positive/outputs_v4/replenishment_comparison.xlsx':'新版_全60期_六方法.xlsx',
        HOME/'outputs/新旧及单因素总对照.xlsx':'新旧及单因素总对照.xlsx',
        HOME/'REPORT.md':'单因素实验报告.md',HOME/'CHANGE_REPORT.md':'任务完成报告.md',
        HOME/'DELIVERY_README.md':'00_阅读说明.md'}
    for source,name in copies.items():shutil.copy2(source,dest/name)
    charts=dest/'图表';charts.mkdir()
    shutil.copy2(HOME/'outputs/constraint_only_comparison.png',charts/'仅服务约束变化_全60期.png')
    shutil.copy2(HOME/'new_positive/visualization/six_method_comparison.png',charts/'新版全60期六方法.png')
    shutil.copy2(ROOT/'experiments/demand_fill_v3/with_ro/visualization/six_method_comparison.png',charts/'旧版六方法.png')
    animation=dest/'动画';shutil.copytree(ROOT/'汇报成果/2026-10-15/动画',animation,
        ignore=shutil.ignore_patterns('最近启动状态.json'))
    (project/'README.md').write_text('# 三轮滚动验证与完整60期训练版\n\n请阅读上一级00_阅读说明.md及任务完成报告.md。当前结果在experiments/full60_revision，rolling_dro_v3中原32期结果仅作历史追溯。动画在汇报成果/2026-10-15/动画/index.html，可切换全部6000、爆品和长尾品。\n',encoding='utf-8')
    (dest/'复核全部结果.bat').write_text('@echo off\r\ncd /d "%~dp0项目"\r\npython experiments/ro_extension/verify_ablation.py\r\nif errorlevel 1 exit /b 1\r\npython scripts/verify_animation.py\r\nif errorlevel 1 exit /b 1\r\npython experiments/ro_extension/verify_ro.py\r\nif errorlevel 1 exit /b 1\r\npython experiments/demand_fill_v3/verify_archive.py\r\npause\r\n',encoding='gbk')
    manifest={str(p.relative_to(dest)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in dest.rglob('*') if p.is_file()}
    (dest/'文件清单_SHA256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(path=str(dest),files=len(manifest),bytes=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())),ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--destination',type=Path,required=True);args=ap.parse_args();main(args.destination)
