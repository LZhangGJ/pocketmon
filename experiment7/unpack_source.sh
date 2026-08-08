#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="${REPO_ROOT}/experiment7/source/experiment7_code_for_gpt_2026-08-08.zip"
DEST="${1:-${REPO_ROOT}/runs/experiment7/source}"
EXPECTED="9c0d24067eacee8abc38223dba28d893e5d1e4e9b75204a9ce92a03093558229"
PYTHON_BIN="${PYTHON:-python}"

actual="$(sha256sum "${ARCHIVE}" | awk '{print $1}')"
if [[ "${actual}" != "${EXPECTED}" ]]; then
  echo "archive SHA-256 mismatch: expected=${EXPECTED} actual=${actual}" >&2
  exit 2
fi

rm -rf "${DEST}"
mkdir -p "${DEST}"
unzip -q "${ARCHIVE}" -d "${DEST}"

"${PYTHON_BIN}" - <<'PY' "${DEST}"
import csv
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = root / "PACKAGE_MANIFEST.csv"
rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig", newline="")))
errors: list[str] = []
for row in rows:
    path = root / row["path"]
    if not path.is_file():
        errors.append(f"missing:{row['path']}")
        continue
    if path.stat().st_size != int(row["bytes"]):
        errors.append(f"size:{row['path']}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest.lower() != row["sha256"].lower():
        errors.append(f"sha256:{row['path']}")
if errors:
    raise SystemExit("\n".join(errors))
print({"destination": str(root), "manifest_files": len(rows), "errors": 0})
PY

"${PYTHON_BIN}" -m compileall -q "${DEST}"
echo "Experiment 7 source unpacked and verified at ${DEST}"
