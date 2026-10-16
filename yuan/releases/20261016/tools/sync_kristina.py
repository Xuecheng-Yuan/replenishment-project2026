"""只读下载固定GitHub提交：不执行上游代码、不写入远端、不读取凭据内容。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import base64
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'Xuecheng-Yuan/replenishment-project2026'
COMMIT = 'aec11c863b6d8a71a1e0874cfb60594cc034cf51'


def api(endpoint, raw=False):
    command = ['gh', 'api', endpoint]
    if raw:
        command += ['-H', 'Accept: application/vnd.github.raw+json']
    return subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', default=COMMIT)
    args = parser.parse_args()
    destination = ROOT / 'partners/kristina/upstream'
    tree = json.loads(api(f'repos/{REPOSITORY}/git/trees/{args.commit}?recursive=1'))
    assert not tree.get('truncated'), '仓库树被截断，不能把不完整快照当成完整来源。'
    prefix = 'experiments/service95_revision/outputs_v4/'
    outputs = {'summary.csv', 'summary_by_segment.csv', 'frozen_parameters.json',
               'input_audit.json', 'test_audit.json', 'results_by_period.csv',
               'service_feedback.csv', 'forecast_h1.csv', 'selected_validation_by_fold.csv',
               'version_comparison.csv', *[f'detail_{m}.csv.gz' for m in
                   ['DRO_cluster', 'DRO_global', 'DRO_category', 'sS', 'SAA', 'RO']]}
    sources = ['src/replenishment/', 'experiments/rolling_dro_v3/',
               'experiments/ro_extension/', 'experiments/constraint_mechanism/']
    chosen = []
    for row in tree['tree']:
        name = row['path']
        if row['type'] != 'blob':
            continue
        source = (any(name.startswith(p) for p in sources)
                  and not any(p in name for p in ['/outputs', '/history/', '/visualization/'])
                  and Path(name).suffix in {'.py', '.txt', '.md'})
        result = name.startswith(prefix) and name[len(prefix):] in outputs
        report = name in {'experiments/service95_revision/REPORT.md',
                         'experiments/service95_revision/README.md',
                         'experiments/service95_revision/history/conservative_99_percent/frozen_parameters.json',
                         'experiments/constraint_mechanism/outputs/published_summary.csv',
                         'experiments/constraint_mechanism/outputs/verification.json',
                         'experiments/constraint_mechanism/reports/机制探索报告.md'}
        if source or result or report:
            chosen.append(row)

    def download(row):
        name = row['path']
        file = destination / name
        if file.exists():
            content = file.read_bytes()
        else:
            if Path(name).suffix in {'.gz', '.npz'}:
                blob = json.loads(api(f'repos/{REPOSITORY}/git/blobs/{row["sha"]}'))
                assert blob['encoding'] == 'base64'
                content = base64.b64decode(blob['content'])
            else:
                content = api(f'repos/{REPOSITORY}/contents/{name}?ref={args.commit}', True)
        digest = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        assert digest == row['sha'], f'Git blob身份不一致：{name}'
        file.parent.mkdir(parents=True, exist_ok=True)
        if not file.exists():
            file.write_bytes(content)
        print('SNAPSHOT', name, len(content), flush=True)
        return dict(path=name, bytes=len(content), git_blob_sha1=digest,
                    sha256=hashlib.sha256(content).hexdigest())

    with ThreadPoolExecutor(max_workers=4) as executor:
        files = list(executor.map(download, chosen))
    manifest = dict(repository=REPOSITORY, commit=args.commit, mode='read_only_fixed_commit',
                    files=files, model_release='service95_revision/outputs_v4',
                    mechanism_experiment='diagnostic_only_not_replacing_published_models')
    (destination.parent/'SOURCE.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print('COMPLETE', len(files), 'files; no remote writes')


if __name__ == '__main__':
    main()
