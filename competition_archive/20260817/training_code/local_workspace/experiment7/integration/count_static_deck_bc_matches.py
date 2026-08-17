from __future__ import annotations

import argparse
import json
from pathlib import Path

from static_deck_bc_common import count_catalog, load_json, read_deck_card_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_json(args.config.resolve())
    manifest_path = (args.manifest or Path(config["sourceManifest"])).resolve()
    manifest = load_json(manifest_path)
    engine_catalog = Path(manifest["engineCatalog"]["path"])
    rows = []
    for dataset in manifest["datasets"]:
        catalog_dir = Path(dataset["root"]).parent / "catalog"
        catalog_path = catalog_dir / "replay_catalog.csv"
        deck_names = read_deck_card_names(catalog_dir, engine_catalog)
        result = count_catalog(catalog_path, deck_names, config)
        result["day"] = dataset["name"]
        rows.append(result)
    payload = {
        "schemaVersion": 1,
        "kind": "experiment7_static_deck_bc_match_counts",
        "sourceManifest": str(manifest_path),
        "days": rows,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
