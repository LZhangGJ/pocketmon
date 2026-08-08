#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ARCHIVE="${REPO_ROOT}/experiment7/source/experiment7_code_for_gpt_2026-08-08.zip"
ARCHIVE="${EXPERIMENT7_ARCHIVE:-${DEFAULT_ARCHIVE}}"
DEST="${1:-${REPO_ROOT}/runs/experiment7/source}"
EXPECTED="9c0d24067eacee8abc38223dba28d893e5d1e4e9b75204a9ce92a03093558229"
EXPECTED_BYTES=94038
PYTHON_BIN="${PYTHON:-python}"

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "Experiment 7 archive not found: ${ARCHIVE}" >&2
  echo "Set EXPERIMENT7_ARCHIVE to the verified 94,038-byte teammate ZIP copied to the Linux server." >&2
  exit 2
fi

actual_bytes="$(stat -c '%s' "${ARCHIVE}")"
actual="$(sha256sum "${ARCHIVE}" | awk '{print $1}')"
if [[ "${actual_bytes}" != "${EXPECTED_BYTES}" || "${actual}" != "${EXPECTED}" ]]; then
  echo "Experiment 7 archive integrity failure" >&2
  echo "archive=${ARCHIVE}" >&2
  echo "expected_bytes=${EXPECTED_BYTES} actual_bytes=${actual_bytes}" >&2
  echo "expected_sha256=${EXPECTED} actual_sha256=${actual}" >&2
  if [[ "${ARCHIVE}" == "${DEFAULT_ARCHIVE}" ]]; then
    cat >&2 <<'EOF'
The repository copy may have been truncated by binary transport. Do not train from it.
From the Windows control host, copy the original verified ZIP to a non-Git server path, for example:

  scp <windows-path-to-zip> doraemon02:/homes/lzhang/pocketmon/data/imports/experiment7_code_for_gpt_2026-08-08.zip

Then run remotely with:

  export EXPERIMENT7_ARCHIVE=/homes/lzhang/pocketmon/data/imports/experiment7_code_for_gpt_2026-08-08.zip
  bash experiment7/unpack_source.sh
EOF
  fi
  exit 2
fi

rm -rf "${DEST}"
mkdir -p "${DEST}"
unzip -q "${ARCHIVE}" -d "${DEST}"

"${PYTHON_BIN}" - "${DEST}" "${ARCHIVE}" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
archive = Path(sys.argv[2])
manifest = root / "PACKAGE_MANIFEST.csv"
rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig", newline="")))
errors: list[str] = []
for row in rows:
    normalized = {str(key).strip().lower(): value for key, value in row.items()}
    relative = normalized["path"]
    path = root / relative
    if not path.is_file():
        errors.append(f"missing:{relative}")
        continue
    if path.stat().st_size != int(normalized["bytes"]):
        errors.append(f"size:{relative}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest.lower() != normalized["sha256"].lower():
        errors.append(f"sha256:{relative}")
if errors:
    raise SystemExit("\n".join(errors))
print(json.dumps({
    "archive": str(archive),
    "archive_bytes": archive.stat().st_size,
    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    "destination": str(root),
    "manifest_files": len(rows),
    "errors": 0,
}, sort_keys=True))
PY

"${PYTHON_BIN}" -m compileall -q "${DEST}"
echo "Experiment 7 source unpacked and verified at ${DEST}"
