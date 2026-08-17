#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("manifests", nargs="+", type=Path)
    args = parser.parse_args()

    loaded = []
    for path in args.manifests:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("kind") != "experiment7_universal_bc":
            raise SystemExit(f"not a Universal BC source manifest: {path}")
        dataset = dict(payload["dataset"])
        date = next((part for part in path.parts if part.startswith("2026-08-")), path.parent.name)
        dataset["name"] = date
        loaded.append((path, payload, dataset))

    base = loaded[-1][1]
    card_vocab = base["engineCatalog"]["cardVocab"]
    for path, payload, dataset in loaded:
        if payload["engineCatalog"]["cardVocab"] != card_vocab:
            raise SystemExit(f"card vocabulary mismatch: {path}")
        summary = dataset.get("summary", {})
        if summary.get("stateFeatureDimension") not in (None, 320):
            raise SystemExit(f"state dimension mismatch: {path}")
        if summary.get("optionFeatureDimension") not in (None, 176):
            raise SystemExit(f"option dimension mismatch: {path}")

    output = {
        "schemaVersion": 2,
        "kind": "experiment7_universal_bc",
        "referenceRoot": base["referenceRoot"],
        "engineCatalog": base["engineCatalog"],
        "datasets": [row[2] for row in loaded],
        "policySource": "winners",
        "minGameScoreExclusive": 900.0,
        "moduleVersions": "*",
        "privacyBoundary": base.get("privacyBoundary"),
        "capacityComparison": {
            "dates": [row[2]["name"] for row in loaded],
            "manifestPaths": [str(row[0]) for row in loaded],
            "hashVerificationRequired": False,
            "splitRule": "reuse each daily cache train/validation split",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "shards": len(loaded),
        "dates": output["capacityComparison"]["dates"],
        "decisions": sum(int(row[2].get("summary", {}).get("decisions", 0)) for row in loaded),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
