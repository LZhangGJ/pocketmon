from __future__ import annotations

import argparse
import json
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

from common import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "experiment7" / "reference_impl" / "runtime_agent"
RUNTIME_FILES = (
    "main.py",
    "features.py",
    "tokenizer.py",
    "portable.py",
    "deck_identity_portable.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize one self-contained Experiment 7 Agent per selected deck")
    parser.add_argument("--selected-decks", type=Path, required=True)
    parser.add_argument("--portable", type=Path, required=True)
    parser.add_argument("--engine-catalog", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()

    payload = json.loads(args.selected_decks.read_text(encoding="utf-8"))
    selected = payload.get("selected") or []
    if not selected:
        raise ValueError("selected deck manifest is empty")
    for path in [args.portable, args.engine_catalog, *[RUNTIME / name for name in RUNTIME_FILES]]:
        if not path.is_file():
            raise FileNotFoundError(path)

    packages = []
    for deck in selected:
        package = args.output_root / deck["name"] / args.model_id
        if package.exists():
            raise FileExistsError(package)
        package.mkdir(parents=True)
        for filename in RUNTIME_FILES:
            shutil.copy2(RUNTIME / filename, package / filename)
            py_compile.compile(str(package / filename), doraise=True)
        shutil.copy2(args.portable, package / "deck_identity_bc.npz")
        shutil.copy2(args.engine_catalog, package / "engine_catalog.json")
        shutil.copy2(Path(deck["deck_path"]), package / "deck.csv")
        # Importing the module is a strict package-load smoke.  It does not call
        # the game engine or open hidden state.
        subprocess.run(
            [sys.executable, "-c", "import main; print(main.DECK.shape, main.POLICY.config['history_length'])"],
            cwd=package,
            check=True,
            capture_output=True,
            text=True,
        )
        receipt = {
            "schema_version": 1,
            "deck_name": deck["name"],
            "archetype_id": deck["archetype_id"],
            "archetype_label": deck["archetype_label"],
            "model_id": args.model_id,
            "package": str(package.resolve()),
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(package.iterdir())
                if path.is_file()
            },
            "load_smoke": "passed",
        }
        write_json(package / "receipt.json", receipt)
        packages.append(receipt)
    write_json(
        args.output_root / f"package_manifest_{args.model_id}.json",
        {"schema_version": 1, "model_id": args.model_id, "packages": packages},
    )
    print(args.output_root / f"package_manifest_{args.model_id}.json")


if __name__ == "__main__":
    main()
