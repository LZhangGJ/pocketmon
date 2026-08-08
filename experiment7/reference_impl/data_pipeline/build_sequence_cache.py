from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history-length", type=int, default=8)
    args = parser.parse_args()
    if args.history_length <= 0:
        raise ValueError("history length must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.features) as archive:
        episode_ids = archive["episode_ids"]
        player_indices = archive["player_indices"]
        source_steps = archive["source_steps"]
        option_features = archive["option_features"]
        option_labels = archive["option_labels"]
        option_offsets = archive["option_offsets"]

    decisions = len(episode_ids)
    option_dim = int(option_features.shape[1])
    history_path = args.output_dir / "history_indices.npy"
    action_path = args.output_dir / "expert_action_features.npy"
    history = np.lib.format.open_memmap(
        history_path,
        mode="w+",
        dtype=np.int32,
        shape=(decisions, args.history_length),
    )
    history[:] = -1
    actions = np.lib.format.open_memmap(
        action_path,
        mode="w+",
        dtype=np.float16,
        shape=(decisions, option_dim),
    )
    actions[:] = 0

    previous: dict[tuple[int, int], deque[int]] = defaultdict(
        lambda: deque(maxlen=args.history_length)
    )
    last_step: dict[tuple[int, int], int] = {}
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    history_counts = np.zeros(decisions, dtype=np.int16)
    empty_actions = 0
    for decision in range(decisions):
        key = (int(episode_ids[decision]), int(player_indices[decision]))
        step = int(source_steps[decision])
        if key in last_step and step <= last_step[key]:
            raise RuntimeError(
                f"non-monotonic sourceStep for {key}: {step} after {last_step[key]}"
            )
        prior = list(previous[key])
        if prior:
            history[decision, -len(prior) :] = prior
        history_counts[decision] = len(prior)

        begin = int(option_offsets[decision])
        end = int(option_offsets[decision + 1])
        selected = np.flatnonzero(option_labels[begin:end]) + begin
        if len(selected):
            actions[decision] = option_features[selected].mean(axis=0).astype(np.float16)
        else:
            empty_actions += 1
        previous[key].append(decision)
        last_step[key] = step
        pair_counts[key] += 1

    history.flush()
    actions.flush()
    del history, actions

    pair_lengths = np.asarray(list(pair_counts.values()), dtype=np.int32)
    manifest = {
        "schemaVersion": 1,
        "features": str(args.features.resolve()),
        "featuresSha256": sha256(args.features),
        "decisions": decisions,
        "episodePlayerTrajectories": len(previous),
        "historyLength": args.history_length,
        "optionDimension": option_dim,
        "decisionsWithFullHistory": int(np.sum(history_counts == args.history_length)),
        "meanAvailableHistory": float(history_counts.mean()),
        "emptyExpertActions": empty_actions,
        "trajectoryDecisionMin": int(pair_lengths.min()) if len(pair_lengths) else 0,
        "trajectoryDecisionMax": int(pair_lengths.max()) if len(pair_lengths) else 0,
        "historyIndicesSha256": sha256(history_path),
        "expertActionFeaturesSha256": sha256(action_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
