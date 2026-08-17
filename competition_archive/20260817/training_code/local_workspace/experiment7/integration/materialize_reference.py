from __future__ import annotations

import base64
import csv
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

EXPECTED_BYTES = 94_038
EXPECTED_SHA256 = "9c0d24067eacee8abc38223dba28d893e5d1e4e9b75204a9ce92a03093558229"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    chunks = repo / "experiment7" / "reference_archive"
    parts = sorted(chunks.glob("part-*"))
    if not parts:
        raise RuntimeError(f"no reference chunks under {chunks}")
    encoded = b"".join(path.read_bytes() for path in parts)
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != EXPECTED_BYTES:
        raise RuntimeError(f"reference ZIP size mismatch: {len(raw)} != {EXPECTED_BYTES}")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"reference ZIP SHA mismatch: {digest} != {EXPECTED_SHA256}")

    with tempfile.TemporaryDirectory() as temporary_text:
        temporary = Path(temporary_text)
        archive = temporary / "reference.zip"
        archive.write_bytes(raw)
        extracted = temporary / "source"
        extracted.mkdir()
        with zipfile.ZipFile(archive) as handle:
            bad = handle.testzip()
            if bad is not None:
                raise RuntimeError(f"corrupt ZIP member: {bad}")
            handle.extractall(extracted)

        manifest = extracted / "PACKAGE_MANIFEST.csv"
        rows = list(csv.DictReader(manifest.open("r", encoding="utf-8-sig", newline="")))
        errors: list[str] = []
        for row in rows:
            normalized = {str(key).strip().lower(): value for key, value in row.items()}
            relative = normalized["path"]
            path = extracted / relative
            if not path.is_file():
                errors.append(f"missing:{relative}")
                continue
            if path.stat().st_size != int(normalized["bytes"]):
                errors.append(f"bytes:{relative}")
            if sha256(path).lower() != normalized["sha256"].lower():
                errors.append(f"sha256:{relative}")
        if errors:
            raise RuntimeError("reference manifest failure: " + ", ".join(errors[:10]))

        destination = repo / "experiment7" / "reference"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(extracted, destination)
        receipt = {
            "schemaVersion": 1,
            "archiveBytes": EXPECTED_BYTES,
            "archiveSha256": EXPECTED_SHA256,
            "archiveChunks": len(parts),
            "manifestFiles": len(rows),
            "manifestErrors": 0,
            "materializedPath": "experiment7/reference",
        }
        (destination / "IMPORT_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    shutil.rmtree(chunks)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
