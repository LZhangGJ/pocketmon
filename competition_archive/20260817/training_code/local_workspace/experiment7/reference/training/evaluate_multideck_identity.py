from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from deck_identity_model import DeckIdentityModelConfig, PTCGDeckIdentityTransformerPolicy
from sequence_model import PTCGSequenceTransformerPolicy, SequenceModelConfig
from train_sequence import evaluate as evaluate_sequence
from train_multideck_identity import IdentityBundle, evaluate_identity


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--source",
        nargs=5,
        action="append",
        metavar=("NAME", "FEATURES", "TOKEN_CACHE", "SEQUENCE_CACHE", "IDENTITY_CACHE"),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = PTCGDeckIdentityTransformerPolicy(
        DeckIdentityModelConfig(**checkpoint["config"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    baseline_checkpoint = torch.load(
        args.baseline_checkpoint, map_location=device, weights_only=False
    )
    baseline = PTCGSequenceTransformerPolicy(
        SequenceModelConfig(**baseline_checkpoint["config"])
    ).to(device)
    baseline.load_state_dict(baseline_checkpoint["state_dict"])

    candidate_results = {}
    baseline_results = {}
    source_receipts = []
    for name, features_text, token_text, sequence_text, identity_text in args.source:
        features = Path(features_text)
        token_cache = Path(token_text)
        sequence_cache = Path(sequence_text)
        identity_cache = Path(identity_text)
        bundle = IdentityBundle.load(
            name, features, token_cache, sequence_cache, identity_cache
        )
        holdout = np.flatnonzero(bundle.sequence.base.data["validation"] == 1)
        if not len(holdout):
            raise RuntimeError(f"{name}: no chronological holdout")
        candidate_results[name] = evaluate_identity(
            model, bundle, holdout, device, args.batch_size
        )
        baseline_results[name] = evaluate_sequence(
            baseline, bundle.sequence, holdout, device, args.batch_size
        )
        source_receipts.append(
            {
                "name": name,
                "features": str(features.resolve()),
                "featuresSha256": sha256(features),
                "tokenCacheManifestSha256": sha256(token_cache / "manifest.json"),
                "sequenceCacheManifestSha256": sha256(sequence_cache / "manifest.json"),
                "identityCacheManifestSha256": sha256(identity_cache / "manifest.json"),
                "holdoutDecisions": int(len(holdout)),
                "holdoutEpisodes": len(
                    set(
                        int(value)
                        for value in bundle.sequence.base.data["episode_ids"][holdout]
                    )
                ),
            }
        )
    candidate_macro = float(
        np.mean([value["exactSemantic"] for value in candidate_results.values()])
    )
    baseline_macro = float(
        np.mean([value["exactSemantic"] for value in baseline_results.values()])
    )
    payload = {
        "schemaVersion": 1,
        "evaluationBoundary": "single opening of each source's chronological validation holdout after checkpoint freeze",
        "device": str(device),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256(args.checkpoint),
            "metadata": checkpoint.get("metadata"),
        },
        "baselineCheckpoint": {
            "path": str(args.baseline_checkpoint.resolve()),
            "sha256": sha256(args.baseline_checkpoint),
            "metadata": baseline_checkpoint.get("metadata"),
        },
        "candidate": candidate_results,
        "baseline": baseline_results,
        "candidateMacroExactSemantic": candidate_macro,
        "baselineMacroExactSemantic": baseline_macro,
        "macroDelta": candidate_macro - baseline_macro,
        "sources": source_receipts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
