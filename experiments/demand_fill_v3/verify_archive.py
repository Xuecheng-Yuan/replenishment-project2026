"""Verify historical archive file provenance without needing source Excel."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
provenance=json.loads((root/'ARCHIVE_PROVENANCE.json').read_text(encoding='utf-8'))
for item in provenance['files']:
    path=root/item['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['archived_sha256'],path
    if item['file'].startswith('outputs_v4/'):
        assert item['archived_sha256']==item['source_sha256'],path
print(f"Archive provenance: {len(provenance['files'])} files pass; all original outputs preserved")
