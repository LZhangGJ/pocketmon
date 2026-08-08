from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from deck_identity_model import DeckIdentityModelConfig, PTCGDeckIdentityTransformerPolicy
from deck_identity_portable import PortableDeckIdentityTransformerPolicy
from train_multideck_identity import IdentityBundle, forward_identity, make_identity_batch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--portable", type=Path, required=True)
    parser.add_argument(
        "--source",
        nargs=5,
        action="append",
        metavar=("NAME", "FEATURES", "TOKEN_CACHE", "SEQUENCE_CACHE", "IDENTITY_CACHE"),
        required=True,
    )
    parser.add_argument("--decisions-per-source", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PTCGDeckIdentityTransformerPolicy(
        DeckIdentityModelConfig(**checkpoint["config"])
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    portable = PortableDeckIdentityTransformerPolicy(args.portable)
    rng = np.random.default_rng(args.seed)
    source_reports = []
    total = mismatches = 0
    max_option_delta = max_count_delta = max_value_delta = max_opponent_delta = 0.0
    timings = []
    for name, features, token_cache, sequence_cache, identity_cache in args.source:
        bundle = IdentityBundle.load(
            name,
            Path(features),
            Path(token_cache),
            Path(sequence_cache),
            Path(identity_cache),
        )
        eligible = np.flatnonzero(bundle.sequence.base.nontrivial_mask())
        chosen = rng.choice(
            eligible,
            size=min(args.decisions_per_source, len(eligible)),
            replace=False,
        )
        source_mismatches = 0
        for decision_value in chosen:
            decision = int(decision_value)
            batch = make_identity_batch(bundle, np.asarray([decision]), torch.device("cpu"))
            with torch.inference_mode():
                torch_outputs = forward_identity(model, batch)
            begin = int(bundle.sequence.base.data["option_offsets"][decision])
            end = int(bundle.sequence.base.data["option_offsets"][decision + 1])
            entity_count = int(bundle.sequence.base.entity_mask[decision].sum())
            started = time.perf_counter()
            portable_outputs = portable.forward(
                bundle.sequence.base.data["state_features"][decision],
                batch["history_state"][0].numpy(),
                batch["history_action"][0].numpy(),
                batch["history_mask"][0].numpy(),
                bundle.own_deck_cards[decision],
                bundle.sequence.base.entity_cat[decision, :entity_count],
                bundle.sequence.base.entity_num[decision, :entity_count],
                bundle.sequence.base.entity_mask[decision, :entity_count],
                bundle.sequence.base.data["option_features"][begin:end],
            )
            timings.append(time.perf_counter() - started)
            torch_option = torch_outputs[0][0, : end - begin].detach().numpy()
            torch_count = torch_outputs[1][0].detach().numpy()
            torch_value = float(torch_outputs[2][0])
            torch_opponent = torch_outputs[3][0].detach().numpy()
            option_delta = float(np.max(np.abs(torch_option - portable_outputs[0])))
            count_delta = float(np.max(np.abs(torch_count - portable_outputs[1])))
            value_delta = abs(torch_value - portable_outputs[2])
            opponent_delta = float(
                np.max(np.abs(torch_opponent - portable_outputs[3]))
            )
            max_option_delta = max(max_option_delta, option_delta)
            max_count_delta = max(max_count_delta, count_delta)
            max_value_delta = max(max_value_delta, value_delta)
            max_opponent_delta = max(max_opponent_delta, opponent_delta)
            torch_action = np.argsort(-torch_option, kind="stable")
            portable_action = np.argsort(-portable_outputs[0], kind="stable")
            mismatch = not np.array_equal(torch_action, portable_action)
            source_mismatches += int(mismatch)
            mismatches += int(mismatch)
            total += 1
        source_reports.append(
            {"name": name, "decisions": len(chosen), "rankingMismatches": source_mismatches}
        )

    payload = {
        "schemaVersion": 1,
        "checkpointSha256": sha256(args.checkpoint),
        "portableSha256": sha256(args.portable),
        "decisions": total,
        "rankingMismatches": mismatches,
        "maxOptionLogitDelta": max_option_delta,
        "maxCountLogitDelta": max_count_delta,
        "maxValueLogitDelta": max_value_delta,
        "maxOpponentLogitDelta": max_opponent_delta,
        "meanPortableMilliseconds": float(np.mean(timings) * 1000.0),
        "p95PortableMilliseconds": float(np.quantile(timings, 0.95) * 1000.0),
        "sources": source_reports,
    }
    if mismatches:
        raise RuntimeError(json.dumps(payload))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
