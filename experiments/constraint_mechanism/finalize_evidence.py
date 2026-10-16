"""Create the experiment's portable, complete SHA256 file manifest."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent

def main():
    files={}
    for p in sorted(HERE.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or 'rerun_outputs' in p.parts or p.name=='SHA256SUMS.json':continue
        files[p.relative_to(HERE).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    (HERE/'SHA256SUMS.json').write_text(json.dumps(files,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Manifest:',len(files),'files')

if __name__=='__main__':main()
