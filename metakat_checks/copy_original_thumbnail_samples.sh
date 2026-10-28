#!/usr/bin/env bash
# Run locally; rsync reads original public thumbnails from Sophie2 over SSH.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
remote=${SOPHIE_HOST:-ikohut@sophie2}
source_root=/homes/ikohut/naki.images/public.256
output_dir=${1:-/home/ikohut/tmp/metakat-evaluation-parity/original-thumbnails}

command -v rsync >/dev/null || { echo 'rsync is required' >&2; exit 1; }
mkdir -p -- "$output_dir"
cp -- "$script_dir/original_thumbnail_samples.paths" "$output_dir/samples.paths"
cp -- "$script_dir/original_thumbnail_samples.json" "$output_dir/samples.json"

rsync --archive --relative --protect-args --checksum --partial \
  --rsh="ssh -4" \
  --info=progress2 \
  --files-from="$script_dir/original_thumbnail_samples.paths" \
  "$remote:$source_root/" "$output_dir/"

python3 - "$output_dir" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = json.loads((root / 'samples.json').read_text())
checksums = []
missing = []
for sample in manifest['samples']:
    path = root / sample['relative_path']
    if not path.is_file() or not path.stat().st_size:
        missing.append(sample['relative_path'])
        continue
    with path.open('rb') as source:
        checksum = hashlib.sha256(source.read()).hexdigest()
    checksums.append(f"{checksum}  {sample['relative_path']}\n")
if missing:
    raise SystemExit('Missing or empty thumbnails:\n' + '\n'.join(missing))
(root / 'SHA256SUMS').write_text(''.join(checksums))
print(f"Copied and checked {len(checksums)} original thumbnails in {root}")
PY
