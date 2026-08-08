from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path


REFERENCE_PATH = "data_pipeline/build_token_cache.py"
ORIGINAL_GUARD = (
    "    if max_actions > 64:\n"
    "        raise RuntimeError(f\"unexpected max action count {max_actions}\")\n"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_reference_script(reference_root: Path) -> Path:
    script = reference_root / REFERENCE_PATH
    manifest = reference_root / "PACKAGE_MANIFEST.csv"
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {}
        for raw_row in csv.DictReader(handle):
            row = {str(key).strip().lower(): value for key, value in raw_row.items()}
            rows[str(row["path"]).replace("\\", "/")] = row
    if REFERENCE_PATH not in rows:
        raise RuntimeError(f"reference manifest does not list {REFERENCE_PATH}")
    row = rows[REFERENCE_PATH]
    actual_bytes = script.stat().st_size
    actual_sha256 = sha256_file(script)
    if actual_bytes != int(row["bytes"]) or actual_sha256 != row["sha256"].lower():
        raise RuntimeError(
            "reference token-cache source differs from PACKAGE_MANIFEST.csv: "
            f"bytes={actual_bytes} sha256={actual_sha256}"
        )
    return script


def adapt_reference_source(source: str, maximum_supported_actions: int) -> str:
    if maximum_supported_actions < 64:
        raise ValueError("maximum_supported_actions must be at least 64")
    if source.count(ORIGINAL_GUARD) != 1:
        raise RuntimeError("reference max-action guard did not match exactly once")
    replacement = (
        f"    if max_actions > {maximum_supported_actions}:\n"
        "        raise RuntimeError(f\"unexpected max action count {max_actions}\")\n"
    )
    return source.replace(ORIGINAL_GUARD, replacement, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the manifest-verified reference token cache with a wider legal-option guard"
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--maximum-supported-actions", type=int, default=128)
    args, reference_args = parser.parse_known_args()

    reference_root = args.reference_root.resolve()
    script = verify_reference_script(reference_root)
    source = adapt_reference_source(
        script.read_text(encoding="utf-8"), args.maximum_supported_actions
    )
    pipeline = reference_root / "data_pipeline"
    if str(pipeline) not in sys.path:
        sys.path.insert(0, str(pipeline))
    sys.argv = [str(script), *reference_args]
    namespace = {
        "__name__": "__main__",
        "__file__": str(script),
        "__package__": None,
    }
    exec(compile(source, str(script), "exec"), namespace)


if __name__ == "__main__":
    main()
