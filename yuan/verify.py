"""Audit the flat model repository. --runtime adds import and CLI checks.

This validates restructuring, never fits all real-data models. Exact numerical
and time-order logic is checked separately with unittest. All evidence remains
historical; workflow-level private-data retraining has not been claimed.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',action='store_true')
    args=parser.parse_args()
    catalog=json.loads((ROOT/'configs/models.json').read_text())
    records=json.loads((ROOT/'results/variants.json').read_text())
    assert len(catalog)==28 and len(records)==125
    model_names={m['name'] for m in catalog}
    assert {p.stem for p in (ROOT/'models').glob('*.py') if p.stem!='__init__'}==model_names
    assert not any((ROOT/n).exists() for n in ('src','experiments','schemes','outputs','程序'))
    py_files=[p for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts]
    for path in py_files:
        source=path.read_text();tree=ast.parse(source)
        for n in ast.walk(tree):
            if isinstance(n,ast.ImportFrom):
                assert not (n.module or '').startswith(('src.','experiments.')),(path,n.module)
                if (n.module or '').startswith(('models.','common.')):
                    assert (ROOT/(n.module.replace('.','/')+'.py')).is_file(),(path,n.module)
        if path.parent==ROOT/'models' and path.stem!='__init__':
            own=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
            assert own,(path,'thin launcher instead of implementation')
            assert 'subprocess' not in source and 'spec_from_file_location' not in source,path
    for path in ROOT.rglob('*.md'):
        value=path.read_text()
        assert '\ufffd' not in value
        assert len(re.findall(r'^\$\$$',value,re.M))%2==0,path
        for target in re.findall(r'\]\(([^)]+)\)',value):
            if target.startswith(('http:','https:','mailto:','#')):continue
            target=unquote(target.strip('<>').split('#')[0])
            assert (path.parent/target).exists(),(path,target)
    for entry in catalog:
        assert (ROOT/entry['code']).is_file() and (ROOT/entry['doc']).is_file()
        assert len([r for r in records if r['model_file']==entry['code']])==len(entry['historical_entries'])
    for row in records:
        for key,val in row['metrics'].items():
            assert math.isfinite(val) and val>=0
            if key in ('fill_rate','ready_rate','sku_pass_share'):assert val<=1+1e-12
    runtime=[]
    if args.runtime:
        env=dict(os.environ)
        env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        def check(entry):
            name=entry['name'];out=[]
            for flags in ([],['--help']):
                r=subprocess.run([sys.executable,'-m','models.'+name,*flags],cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
                assert r.returncode==0,(name,r.stderr)
                out.append('default' if not flags else 'help')
            return dict(model=name,passed=out)
        with ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(check,catalog):runtime.append(result);print('CHECK',result['model'],flush=True)
    manifest=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),lines=len(p.read_text().splitlines())) for p in sorted(py_files)]
    (ROOT/'configs/code_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    report=dict(kind='source_restructuring_not_new_model_training',models=len(catalog),result_variants=125,
        syntax_checked=len(py_files),runtime=runtime,passed=True,
        limitation='No full private-data training of all models repeated; old service failures remain failures.')
    (ROOT/'results/verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='runtime'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
