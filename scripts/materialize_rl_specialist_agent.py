from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_deck(path: Path) -> list[int]:
    deck = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"deck must contain exactly 60 cards, got {len(deck)}")
    return deck


def materialize(
    checkpoint: Path, deck_path: Path, output: Path, name: str,
    q_checkpoint: Path | None = None,
    q_top_k: int = 4,
    q_uncertainty_penalty: float = 0.50,
    q_min_margin: float = 0.20,
    q_max_uncertainty: float = 0.10,
    q_max_override_rate: float = 0.15,
    q_min_validation_rows: int = 500,
    q_max_validation_mae: float = 0.30,
) -> dict[str, object]:
    checkpoint = checkpoint.resolve(strict=True)
    deck_path = deck_path.resolve(strict=True)
    q_checkpoint = q_checkpoint.resolve(strict=True) if q_checkpoint is not None else None
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing package: {output}")
    deck = read_deck(deck_path)
    q_policy: dict[str, object] | None = None
    if q_checkpoint is not None:
        if q_top_k <= 0:
            raise ValueError("Q top-k must be positive")
        confidence_values = (q_uncertainty_penalty, q_min_margin, q_max_uncertainty)
        if any(not math.isfinite(value) or value < 0 for value in confidence_values):
            raise ValueError("Q confidence thresholds must be non-negative")
        if not math.isfinite(q_max_override_rate) or not 0.0 <= q_max_override_rate <= 1.0:
            raise ValueError("Q maximum override rate must be in [0, 1]")
        if q_min_validation_rows < 0 or not math.isfinite(q_max_validation_mae) or q_max_validation_mae < 0:
            raise ValueError("Q validation requirements must be non-negative")
        try:
            q_payload = torch.load(q_checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            q_payload = torch.load(q_checkpoint, map_location="cpu")
        actor_hash = sha256(checkpoint)
        if q_payload.get("actor_checkpoint_sha256") != actor_hash:
            raise ValueError("action-Q checkpoint was trained for a different actor")
        q_kind = q_payload.get("kind", "counterfactual_action_q_ensemble")
        if q_kind not in {
            "counterfactual_action_q_ensemble",
            "counterfactual_dueling_action_q_ensemble",
        }:
            raise ValueError(f"unsupported action-Q checkpoint kind: {q_kind!r}")
        validation = q_payload.get("validation") or {}
        validation_rows = int(validation.get("rows", 0))
        validation_mae = float(validation.get("mae", float("inf")))
        if validation_rows < q_min_validation_rows:
            raise ValueError(
                f"action-Q validation rows {validation_rows} < {q_min_validation_rows}"
            )
        if not math.isfinite(validation_mae) or validation_mae > q_max_validation_mae:
            raise ValueError(
                f"action-Q validation MAE {validation_mae} > {q_max_validation_mae}"
            )
        q_policy = {
            "schema_version": 1,
            "mode": "conservative_lower_confidence_reranker",
            "top_k": q_top_k,
            "uncertainty_penalty": q_uncertainty_penalty,
            "min_margin": q_min_margin,
            "max_uncertainty": q_max_uncertainty,
            "max_override_rate": q_max_override_rate,
            "min_validation_rows": q_min_validation_rows,
            "max_validation_mae": q_max_validation_mae,
            "observed_validation_rows": validation_rows,
            "observed_validation_mae": validation_mae,
            "head_kind": q_kind,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        shutil.copy2(ROOT / "agents" / "rl_bc_specialist" / "main.py", staging / "main.py")
        shutil.copytree(ROOT / "rl", staging / "rl", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(checkpoint, staging / "checkpoint.pt")
        if q_checkpoint is not None:
            shutil.copy2(q_checkpoint, staging / "action_q.pt")
            (staging / "q_policy.json").write_text(
                json.dumps(q_policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        (staging / "deck.csv").write_text("".join(f"{card_id}\n" for card_id in deck), encoding="utf-8")
        manifest: dict[str, object] = {
            "schema_version": 1,
            "name": name,
            "kind": "our_rl_bc_specialist",
            "checkpoint_sha256": sha256(staging / "checkpoint.pt"),
            "deck_sha256": sha256(staging / "deck.csv"),
            "main_sha256": sha256(staging / "main.py"),
            "card_count": len(deck),
            "confidence_threshold": 0.0,
            "action_q_sha256": sha256(staging / "action_q.pt") if q_checkpoint is not None else None,
            "q_policy": q_policy,
        }
        (staging / "agent_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an immutable local League package for a trained RL specialist")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--q-checkpoint", type=Path)
    parser.add_argument("--q-top-k", type=int, default=4)
    parser.add_argument("--q-uncertainty-penalty", type=float, default=0.50)
    parser.add_argument("--q-min-margin", type=float, default=0.20)
    parser.add_argument("--q-max-uncertainty", type=float, default=0.10)
    parser.add_argument("--q-max-override-rate", type=float, default=0.15)
    parser.add_argument("--q-min-validation-rows", type=int, default=500)
    parser.add_argument("--q-max-validation-mae", type=float, default=0.30)
    args = parser.parse_args()
    print(json.dumps(materialize(
        args.checkpoint,
        args.deck,
        args.output,
        args.name,
        q_checkpoint=args.q_checkpoint,
        q_top_k=args.q_top_k,
        q_uncertainty_penalty=args.q_uncertainty_penalty,
        q_min_margin=args.q_min_margin,
        q_max_uncertainty=args.q_max_uncertainty,
        q_max_override_rate=args.q_max_override_rate,
        q_min_validation_rows=args.q_min_validation_rows,
        q_max_validation_mae=args.q_max_validation_mae,
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
