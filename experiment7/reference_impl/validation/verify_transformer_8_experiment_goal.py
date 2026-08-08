#!/usr/bin/env python3
"""Fail-closed final verifier for the eight local Transformer experiments."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTROL_SHA = "3c0ce5024e94f429dbbef8eb9c544deec2a953aea358e933624a78447ad26523"
BEST_SHA = "fb448ada8a2a7dabeb09563c7a229cb2a5cc64558e4c2a9f8834f17942523bb9"

RESULT_PATHS = {
    1: ROOT / ".cache/validation_pool/experiment1_public_events/experiment_result.json",
    2: ROOT / ".cache/validation_pool/experiment2_deck_knowledge/experiment_result.json",
    3: ROOT / ".cache/real_loss_context0_q_23x64_merged_v1/experiment_result.json",
    4: ROOT / ".cache/real_loss_action_value_head_v1/experiment_result.json",
    5: ROOT / ".cache/validation_pool/experiment5_frozen_pool/experiment_result.json",
    6: ROOT / ".cache/validation_pool/experiment6_multienv_ppo/experiment_result.json",
    7: ROOT / ".cache/validation_pool/experiment7_multideck_identity/experiment_result.json",
    8: ROOT / ".cache/validation_pool/experiment8_jointdeck/experiment_result.json",
}

EXPECTED_STATUS = {
    1: "REJECTED_AT_200",
    2: "REJECTED_AT_200_NEAR_MISS",
    3: "COMPLETE_Q_DATA_TOO_SPARSE_FOR_SAFE_OVERRIDE",
    4: "REJECT_NO_SAFE_CALIBRATION_COVERAGE",
    5: "PASS_INFRASTRUCTURE",
    6: "REJECT_200_CELL_HARM_AND_PARENT_REGRESSION",
    7: "PASS_DECK_CONDITIONED_POLICY_REJECT_AUXILIARY_IDENTIFIER",
    8: "REJECT_NOT_BETTER_THAN_CURRENT_BEST",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_receipt(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def close(actual: float, expected: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(actual), float(expected), abs_tol=tolerance, rel_tol=0.0)


def resolve_receipt_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_artifact_receipts(result: dict[str, Any], failures: list[str]) -> int:
    checked = 0
    for label, item in result.get("artifacts", {}).items():
        if not isinstance(item, dict) or "path" not in item or "sha256" not in item:
            continue
        path = resolve_receipt_path(str(item["path"]))
        if not path.is_file():
            failures.append(f"experiment {result['experiment']} missing artifact {label}: {path}")
            continue
        actual = sha256(path)
        if actual != item["sha256"]:
            failures.append(
                f"experiment {result['experiment']} artifact hash drift {label}: "
                f"{actual} != {item['sha256']}"
            )
        checked += 1
    return checked


def verify_matrix(
    matrix: dict[str, Any],
    *,
    games_per_opponent: int,
    package_sha: str,
    failures: list[str],
) -> None:
    if matrix.get("gamesPerOpponent") != games_per_opponent:
        failures.append(f"matrix gamesPerOpponent != {games_per_opponent}")
    if matrix.get("candidatePackageSha256") != package_sha:
        failures.append("matrix candidate package hash mismatch")
    if matrix.get("opponents") != 3 or len(matrix.get("cells", [])) != 3:
        failures.append("matrix does not contain exactly three promotion opponents")
    if matrix.get("pooled", {}).get("errors") != 0:
        failures.append("matrix contains engine errors")
    for cell in matrix.get("cells", []):
        if cell.get("total", {}).get("games") != games_per_opponent:
            failures.append(f"{cell.get('opponent')} total games mismatch")
        if cell.get("seat0", {}).get("games") != games_per_opponent // 2:
            failures.append(f"{cell.get('opponent')} seat0 is not balanced")
        if cell.get("seat1", {}).get("games") != games_per_opponent // 2:
            failures.append(f"{cell.get('opponent')} seat1 is not balanced")
        if cell.get("total", {}).get("errors") != 0:
            failures.append(f"{cell.get('opponent')} has engine errors")
        if cell.get("candidateAgentStats", {}).get("fallbackCalls") != 0:
            failures.append(f"{cell.get('opponent')} has candidate fallbacks")


def main() -> int:
    failures: list[str] = []
    results: dict[int, dict[str, Any]] = {}
    artifact_receipts_checked = 0

    for experiment, path in RESULT_PATHS.items():
        if not path.is_file():
            failures.append(f"missing experiment {experiment} result: {path}")
            continue
        result = load_json(path)
        results[experiment] = result
        if result.get("experiment") != experiment:
            failures.append(f"experiment id mismatch in {path}")
        if result.get("status") != EXPECTED_STATUS[experiment]:
            failures.append(f"experiment {experiment} unexpected status: {result.get('status')}")
        if result.get("kaggleSubmissionAuthorized") is not False:
            failures.append(f"experiment {experiment} lacks explicit local-only boundary")
        artifact_receipts_checked += verify_artifact_receipts(result, failures)

    manifest = load_json(ROOT / "behavior_cloning/validation_pool_manifest.json")
    control = load_json(ROOT / "behavior_cloning/validation_control_matrix_200.json")
    protocol = ROOT / "behavior_cloning/VALIDATION_POOL_PROTOCOL.md"
    if manifest.get("promotionOpponentCount") != 3:
        failures.append("promotion pool does not contain three opponents")
    if manifest.get("promotionDistinctDeckCount") != 3:
        failures.append("promotion pool does not contain three distinct decks")
    if control.get("candidatePackageSha256") != CONTROL_SHA:
        failures.append("frozen control receipt mismatch")
    if control.get("gamesPerOpponent") != 200 or control.get("pooled", {}).get("errors") != 0:
        failures.append("frozen control 200 matrix is incomplete or contains errors")
    if not protocol.is_file() or "400 fresh games per promotion opponent" not in protocol.read_text(encoding="utf-8"):
        failures.append("frozen promotion protocol is missing or incomplete")

    control_root = ROOT / "behavior_cloning/agents/grimmsnarl_sequence_bc_h8_pre12_v1"
    best_root = ROOT / "behavior_cloning/agents/grimmsnarl_multideck_identity_h8_pre12_v1"
    actual_control_sha = directory_receipt(control_root)
    actual_best_sha = directory_receipt(best_root)
    if actual_control_sha != CONTROL_SHA:
        failures.append(f"control package drift: {actual_control_sha}")
    if actual_best_sha != BEST_SHA:
        failures.append(f"best package drift: {actual_best_sha}")

    gate200_path = ROOT / ".cache/validation_pool/experiment7_multideck_identity/gate_200.json"
    gate400_path = ROOT / ".cache/validation_pool/experiment7_multideck_identity/gate_400_replication.json"
    matrix200_path = ROOT / ".cache/validation_pool/experiment7_multideck_identity/matrix_200.json"
    matrix400_path = ROOT / ".cache/validation_pool/experiment7_multideck_identity/matrix_400_replication.json"
    gate200 = load_json(gate200_path)
    gate400 = load_json(gate400_path)
    matrix200 = load_json(matrix200_path)
    matrix400 = load_json(matrix400_path)
    verify_matrix(matrix200, games_per_opponent=200, package_sha=BEST_SHA, failures=failures)
    verify_matrix(matrix400, games_per_opponent=400, package_sha=BEST_SHA, failures=failures)
    for label, gate in (("200", gate200), ("400", gate400)):
        if gate.get("passesPromotionGate") is not True:
            failures.append(f"best candidate failed the {label} promotion gate")
        if gate.get("zeroErrorsAndFallbacks") is not True:
            failures.append(f"best candidate {label} gate has errors or fallbacks")
        if float(gate.get("macroDelta", -1.0)) < 0.03:
            failures.append(f"best candidate {label} macro delta is below 3pp")
        if not all(cell.get("passesHarmGate") is True for cell in gate.get("cells", [])):
            failures.append(f"best candidate {label} failed a cell-harm gate")

    if sha256(matrix200_path) != gate200.get("candidateMatrix", {}).get("sha256"):
        failures.append("200 matrix hash does not match its gate")
    if sha256(matrix400_path) != gate400.get("candidateMatrix", {}).get("sha256"):
        failures.append("400 matrix hash does not match its gate")

    screen_root = ROOT / ".cache/validation_pool/experiment7_multideck_identity/20"
    expected_labels = {item["label"] for item in manifest["opponents"] if item.get("promotionPool")}
    actual_labels: set[str] = set()
    for summary_path in screen_root.glob("*/summary.json"):
        actual_labels.add(summary_path.parent.name)
        summary = load_json(summary_path)
        if summary.get("games") != 20 or summary.get("gamesPerSeat") != 10:
            failures.append(f"bad 20-game screen: {summary_path.parent.name}")
        if summary.get("errors") != 0 or summary.get("candidateAgentStats", {}).get("fallbackCalls") != 0:
            failures.append(f"20-game screen error/fallback: {summary_path.parent.name}")
        if summary.get("candidatePackageSha256") != BEST_SHA:
            failures.append(f"20-game screen package mismatch: {summary_path.parent.name}")
    if actual_labels != expected_labels:
        failures.append("20-game screen opponent set differs from the frozen promotion pool")

    if 3 in results:
        exp3 = results[3]
        if exp3.get("sourceStates") != 23 or exp3.get("pairedTerminalSamplesPerState") != 64:
            failures.append("experiment 3 Q dataset dimensions changed")
        if exp3.get("searchProvenPositive") != 1:
            failures.append("experiment 3 strict-positive count changed")
    if 4 in results:
        exp4 = results[4]
        if exp4.get("calibrationGatePass") is not False or exp4.get("holdoutOpened") is not False:
            failures.append("experiment 4 fail-closed calibration boundary changed")
    if 6 in results and results[6].get("arena200", {}).get("passesFrozenGate") is not False:
        failures.append("experiment 6 rejection gate changed")
    if 7 in results:
        exp7 = results[7]
        if exp7.get("candidate", {}).get("packageSha256") != BEST_SHA:
            failures.append("experiment 7 result points to a different package")
        if not close(exp7.get("arena400IndependentReplication", {}).get("macro", -1), 0.6533333333333333):
            failures.append("experiment 7 independent 400 macro changed")
        if exp7.get("opponentDeckIdentification", {}).get("status") != "REJECT_UNRELIABLE_GENERALIZATION":
            failures.append("experiment 7 auxiliary identifier status changed")
    if 8 in results:
        exp8 = results[8]
        if float(exp8.get("arena200", {}).get("currentBestMacroDelta", 0.0)) >= 0.0:
            failures.append("experiment 8 no longer trails the frozen best")

    receipt = {
        "schemaVersion": 1,
        "status": "PASS" if not failures else "FAIL",
        "localOnly": True,
        "kaggleSubmissionAuthorized": False,
        "experimentsChecked": sorted(results),
        "resultSha256": {
            str(experiment): sha256(path)
            for experiment, path in RESULT_PATHS.items()
            if path.is_file()
        },
        "artifactReceiptsChecked": artifact_receipts_checked,
        "frozenControl": {
            "packageSha256": actual_control_sha,
            "macro200": control.get("macroScoreRate"),
        },
        "bestCandidate": {
            "experiment": 7,
            "packageSha256": actual_best_sha,
            "macro200": matrix200.get("macroScoreRate"),
            "delta200": gate200.get("macroDelta"),
            "macro400Independent": matrix400.get("macroScoreRate"),
            "delta400Independent": gate400.get("macroDelta"),
            "gate200": gate200.get("passesPromotionGate"),
            "gate400Independent": gate400.get("passesPromotionGate"),
        },
        "failures": failures,
    }
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
