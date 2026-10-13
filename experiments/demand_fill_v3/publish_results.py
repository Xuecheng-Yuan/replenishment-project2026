"""Compress complete numerical traces for Git and record SHA256 provenance."""
import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT/'outputs_v4')
    parser.add_argument('--destination', type=Path, default=ROOT/'outputs_v4')
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name in ['decision_comparison_full.csv'] + [f'detail_{m}.csv' for m in
            ('DRO_cluster', 'DRO_global', 'DRO_category', 'sS', 'SAA')]:
        path = source/name
        if path.exists():
            with path.open('rb') as inp, (destination/(name+'.gz')).open('wb') as out:
                with gzip.GzipFile(filename='', mode='wb', fileobj=out, mtime=0) as packed:
                    shutil.copyfileobj(inp, packed)
        elif not (destination/(name+'.gz')).exists():
            raise FileNotFoundError(path)
    files = []
    for path in sorted(destination.iterdir()):
        if not path.is_file() or path.name == 'artifact_manifest.json':
            continue
        if path.suffix not in ('.csv', '.gz', '.json', '.npz', '.xlsx', '.png', '.md'):
            continue
        if path.name.startswith('detail_') and path.suffix == '.csv':
            continue
        if path.name == 'decision_comparison_full.csv':
            continue
        files.append(dict(file=path.name, bytes=path.stat().st_size,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest = dict(schema=1, source='run_v3.py outputs; full CSV traces published as gzip',
                    compression='gzip mtime=0; numeric contents unchanged', files=files)
    (destination/'artifact_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Published {len(files)} result artifacts')


if __name__ == '__main__':
    main()
