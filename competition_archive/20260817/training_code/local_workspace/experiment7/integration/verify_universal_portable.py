from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import Experiment7Error, read_json, sha256_file, write_json
from feature_tensor_store import install_memmap_bundle_loader
from universal_deck_portable import (
    GREEDY_TIE_TOLERANCE,
    PortableUniversalDeckTransformerPolicy,
    _stable_argmax,
)


def stable_order(logits: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Order logits with a deterministic low-index tie break for numerical near-ties."""

    remaining = np.flatnonzero(valid)
    ordered: list[int] = []
    while len(remaining):
        maximum = logits[remaining].max()
        tied = remaining[
            logits[remaining] >= maximum - float(GREEDY_TIE_TOLERANCE)
        ]
        ordered.extend(sorted(int(index) for index in tied))
        remaining = remaining[~np.isin(remaining, tied)]
    return np.asarray(ordered, dtype=np.int64)


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def setup(reference_root: Path):
    for path in (reference_root / "training", reference_root / "data_pipeline"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from train import Bundle
    install_memmap_bundle_loader(Bundle)
    from train_multideck_identity import IdentityBundle, make_identity_batch
    from universal_deck_model import (
        UniversalDeckModelConfig,
        UniversalDeckTransformerPolicy,
    )

    return IdentityBundle, make_identity_batch, UniversalDeckModelConfig, UniversalDeckTransformerPolicy


def verify(args: argparse.Namespace) -> dict[str, Any]:
    sources = read_json(args.sources)
    if sources.get("kind") != "experiment7_universal_bc":
        raise Experiment7Error("expected an experiment7_universal_bc source manifest")
    IdentityBundle, make_identity_batch, Config, Model = setup(args.reference_root)
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint.get("architecture") != "experiment7_universal_deck8_autoregressive_stop":
        raise Experiment7Error(f"unexpected checkpoint architecture: {checkpoint.get('architecture')}")
    model = Model(Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    portable = PortableUniversalDeckTransformerPolicy(args.portable)
    if portable.config != checkpoint["config"]:
        raise Experiment7Error("portable config differs from checkpoint config")

    rng = np.random.default_rng(args.seed)
    source_reports: list[dict[str, Any]] = []
    total = action_mismatches = ranking_mismatches = full_ranking_mismatches = illegal = 0
    mismatch_details: list[dict[str, Any]] = []
    full_ranking_mismatch_details: list[dict[str, Any]] = []
    max_global_delta = max_option_delta = max_deck_delta = 0.0
    max_value_delta = max_opponent_delta = max_decoder_delta = 0.0
    timings: list[float] = []
    for row in sources["datasets"]:
        bundle = IdentityBundle.load(
            row["name"],
            Path(row["features"]),
            Path(row["tokenCache"]),
            Path(row["sequenceCache"]),
            Path(row["identityCache"]),
        )
        eligible = np.flatnonzero(bundle.sequence.base.nontrivial_mask())
        chosen = rng.choice(
            eligible,
            size=min(args.decisions_per_source, len(eligible)),
            replace=False,
        )
        source_action_mismatches = 0
        source_ranking_mismatches = 0
        source_full_ranking_mismatches = 0
        for decision_value in chosen:
            decision = int(decision_value)
            batch = make_identity_batch(
                bundle, np.asarray([decision]), torch.device("cpu")
            )
            with torch.inference_mode():
                torch_encoding = model(
                    batch["state"],
                    batch["history_state"],
                    batch["history_action"],
                    batch["history_mask"],
                    batch["own_deck_cards"],
                    batch["entity_cat"],
                    batch["entity_num"],
                    batch["entity_mask"],
                    batch["options"],
                    batch["option_mask"],
                )
                torch_action = model.greedy_actions(
                    torch_encoding, batch["min_count"], batch["max_count"]
                )[0]

            begin = int(bundle.sequence.base.data["option_offsets"][decision])
            end = int(bundle.sequence.base.data["option_offsets"][decision + 1])
            option_count = end - begin
            entity_count = int(bundle.sequence.base.entity_mask[decision].sum())
            started = time.perf_counter()
            portable_encoding = portable.encode(
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
            minimum = int(batch["min_count"][0])
            maximum = int(batch["max_count"][0])
            portable_action = portable.greedy_actions(
                portable_encoding, minimum, maximum
            )
            timings.append(time.perf_counter() - started)

            action_mismatch = torch_action != portable_action
            source_action_mismatches += int(action_mismatch)
            action_mismatches += int(action_mismatch)
            if not (
                minimum <= len(portable_action) <= maximum
                and len(portable_action) == len(set(portable_action))
                and all(0 <= index < option_count for index in portable_action)
            ):
                illegal += 1

            torch_global = torch_encoding.global_hidden[0].numpy()
            torch_option = torch_encoding.option_hidden[0, :option_count].numpy()
            torch_deck = torch_encoding.deck_hidden[0].numpy()
            torch_value = float(torch_encoding.value_logits[0])
            torch_opponent = torch_encoding.opponent_logits[0].numpy()
            max_global_delta = max(
                max_global_delta,
                float(np.max(np.abs(torch_global - portable_encoding["global_hidden"]))),
            )
            max_option_delta = max(
                max_option_delta,
                float(np.max(np.abs(torch_option - portable_encoding["option_hidden"]))),
            )
            max_deck_delta = max(
                max_deck_delta,
                float(np.max(np.abs(torch_deck - portable_encoding["deck_hidden"]))),
            )
            max_value_delta = max(
                max_value_delta, abs(torch_value - float(portable_encoding["value_logits"]))
            )
            max_opponent_delta = max(
                max_opponent_delta,
                float(np.max(np.abs(torch_opponent - portable_encoding["opponent_logits"]))),
            )

            selected = np.zeros(option_count, dtype=bool)
            trace_mismatch = False
            full_trace_mismatch = False
            trace_details: list[dict[str, Any]] = []
            for decoder_step in range(option_count + 1):
                with torch.inference_mode():
                    torch_logits = model.decoder_logits(
                        torch_encoding,
                        torch.from_numpy(selected[None, :]),
                        batch["min_count"],
                        batch["max_count"],
                    )[0].numpy()
                portable_logits = portable.decoder_logits(
                    portable_encoding, selected, minimum, maximum
                )
                valid = (torch_logits > -9999.0) | (portable_logits > -9999.0)
                if bool(valid.any()):
                    max_decoder_delta = max(
                        max_decoder_delta,
                        float(np.max(np.abs(torch_logits[valid] - portable_logits[valid]))),
                    )
                    torch_order = stable_order(torch_logits, valid)
                    portable_order = stable_order(portable_logits, valid)
                    order_mismatch = not np.array_equal(torch_order, portable_order)
                    full_trace_mismatch |= order_mismatch
                    if order_mismatch:
                        torch_top = torch_order[:8]
                        portable_top = portable_order[:8]
                        trace_details.append(
                            {
                                "decoderStep": decoder_step,
                                "selectedBeforeStep": np.flatnonzero(selected).astype(int).tolist(),
                                "torchTopIndices": torch_top.astype(int).tolist(),
                                "torchTopLogits": torch_logits[torch_top].astype(float).tolist(),
                                "portableTopIndices": portable_top.astype(int).tolist(),
                                "portableTopLogits": portable_logits[portable_top].astype(float).tolist(),
                            }
                        )
                choice = _stable_argmax(torch_logits)
                portable_choice = _stable_argmax(portable_logits)
                trace_mismatch |= choice != portable_choice
                if choice == option_count:
                    break
                selected[choice] = True
            source_ranking_mismatches += int(trace_mismatch)
            ranking_mismatches += int(trace_mismatch)
            source_full_ranking_mismatches += int(full_trace_mismatch)
            full_ranking_mismatches += int(full_trace_mismatch)
            if action_mismatch or trace_mismatch:
                mismatch_details.append(
                    {
                        "source": row["name"],
                        "decision": decision,
                        "optionCount": option_count,
                        "minimum": minimum,
                        "maximum": maximum,
                        "torchAction": [int(index) for index in torch_action],
                        "portableAction": [int(index) for index in portable_action],
                        "actionMismatch": action_mismatch,
                        "rankingMismatch": trace_mismatch,
                        "trace": trace_details,
                    }
                )
            if full_trace_mismatch:
                full_ranking_mismatch_details.append(
                    {
                        "source": row["name"],
                        "decision": decision,
                        "optionCount": option_count,
                        "minimum": minimum,
                        "maximum": maximum,
                        "torchAction": [int(index) for index in torch_action],
                        "portableAction": [int(index) for index in portable_action],
                        "trace": trace_details,
                    }
                )
            total += 1
        source_reports.append(
            {
                "name": row["name"],
                "decisions": len(chosen),
                "actionMismatches": source_action_mismatches,
                "stableRankingMismatches": source_ranking_mismatches,
                "fullRankingMismatches": source_full_ranking_mismatches,
            }
        )

    payload = {
        "schemaVersion": 1,
        "architecture": checkpoint["architecture"],
        "checkpointSha256": sha256_file(args.checkpoint),
        "portableSha256": sha256_file(args.portable),
        "decisions": total,
        "actionMismatches": action_mismatches,
        "stableRankingMismatches": ranking_mismatches,
        "fullRankingMismatches": full_ranking_mismatches,
        "illegalPredictionCount": illegal,
        "mismatchDetails": mismatch_details,
        "fullRankingMismatchDetails": full_ranking_mismatch_details,
        "maxGlobalHiddenDelta": max_global_delta,
        "maxOptionHiddenDelta": max_option_delta,
        "maxDeckHiddenDelta": max_deck_delta,
        "maxDecoderLogitDelta": max_decoder_delta,
        "maxValueLogitDelta": max_value_delta,
        "maxOpponentLogitDelta": max_opponent_delta,
        "meanPortableMilliseconds": float(np.mean(timings) * 1000.0),
        "p95PortableMilliseconds": float(np.quantile(timings, 0.95) * 1000.0),
        "sources": source_reports,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    if action_mismatches or ranking_mismatches or illegal:
        raise Experiment7Error(
            "universal portable parity failed: "
            f"actions={action_mismatches} rankings={ranking_mismatches} illegal={illegal}"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify PyTorch/NumPy parity for Experiment 7 Universal Deck-8 BC"
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--portable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions-per-source", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    verify(args)


if __name__ == "__main__":
    main()
