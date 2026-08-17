from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def acquire_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f"static deck controller lock already exists: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"host": socket.gethostname(), "pid": os.getpid(), "createdAt": now()}, handle)
        handle.write("\n")


def verify_strict_window(root: Path, expected_days: int, threshold: float) -> dict[str, Any]:
    success = root / "SUCCESS"
    manifest_path = root / "tensordict-sources.json"
    if not success.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("strict final SUCCESS/manifest is not ready")
    manifest = load_json(manifest_path)
    datasets = manifest.get("datasets", [])
    names = [str(row["name"]) for row in datasets]
    if len(datasets) != expected_days or len(set(names)) != expected_days:
        raise ValueError(f"strict final day parity failed: {len(datasets)} != {expected_days}")
    receipts = []
    for row in datasets:
        day_root = Path(row["root"])
        receipt_path = Path(row.get("auditReceipt") or day_root / "audit-receipt.json")
        if not receipt_path.is_file():
            receipt_path = day_root / "audit-receipt.json"
        receipt = load_json(receipt_path)
        if not receipt.get("parity", {}).get("passed"):
            raise ValueError(f"strict day parity is not passing: {row['name']}")
        catalog = receipt.get("catalog", {})
        if int(catalog.get("strictRows", 0)) <= 0:
            raise ValueError(f"strict day contains no episodes: {row['name']}")
        if float(receipt.get("summary", {}).get("minGameScoreExclusive", -1)) != threshold:
            raise ValueError(f"strict day threshold mismatch: {row['name']}")
        receipts.append(str(receipt_path))
    return {
        "strictRoot": str(root),
        "strictSuccess": str(success),
        "manifest": str(manifest_path),
        "days": names,
        "receipts": receipts,
        "parity": True,
    }


def verify_remote_local_authority(
    host: str,
    root: Path,
    expected_days: int,
    expected_manifest_sha256: str,
    parity_receipt: Path,
) -> dict[str, Any]:
    """Verify a host-local final without requiring its 50+ GB shared publish."""
    command = (
        f"test -f '{root}/SUCCESS' && test -f '{root}/tensordict-sources.json' "
        f"&& sha256sum '{root}/tensordict-sources.json'"
    )
    completed = subprocess.run(
        ["ssh", host, command], check=False, text=True, capture_output=True,
    )
    if completed.returncode:
        raise FileNotFoundError("remote local-authority SUCCESS/manifest is not ready")
    manifest_sha256 = completed.stdout.strip().split()[0]
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            f"remote local-authority manifest SHA mismatch: {manifest_sha256}"
        )
    parity = load_json(parity_receipt)
    days = parity.get("days", [])
    if parity.get("status") != "passed" or len(days) != expected_days:
        raise ValueError("remote local-authority parity receipt is not passing")
    return {
        "strictRoot": str(root),
        "authorityHost": host,
        "strictSuccess": str(root / "SUCCESS"),
        "manifest": str(root / "tensordict-sources.json"),
        "manifestSha256": manifest_sha256,
        "parityReceipt": str(parity_receipt),
        "days": [str(row.get("name", row)) if isinstance(row, dict) else str(row) for row in days],
        "parity": True,
        "localAuthority": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--strict-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--launch-command", type=Path, required=True)
    parser.add_argument("--authority-host")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--parity-receipt", type=Path)
    parser.add_argument(
        "--launch-mode", choices=("formal_profiles", "authority_build_only"),
        default="formal_profiles",
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args()
    config = load_json(args.config.resolve())
    strict_root = args.strict_root.resolve()
    control = args.control_root.resolve()
    control.mkdir(parents=True, exist_ok=True)
    acquire_lock(control / "controller.lock")
    state_path = control / "state.json"
    state = {
        "schemaVersion": 1,
        "kind": "experiment7_static_deck_bc_post_strict_controller",
        "status": "waiting_strict_10_of_10_success_and_parity",
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "createdAt": now(),
        "strictRoot": str(strict_root),
        "expectedDays": int(config["expectedDays"]),
        "strictPredicate": config["strictPredicate"],
        "standardAllowed": False,
        "formalTrainingStarted": False,
        "launchCommand": str(args.launch_command.resolve()),
    }
    atomic_json(state_path, state)
    while True:
        try:
            if args.authority_host:
                if not args.expected_manifest_sha256 or not args.parity_receipt:
                    raise ValueError("remote authority requires manifest SHA and parity receipt")
                verification = verify_remote_local_authority(
                    args.authority_host,
                    strict_root,
                    int(config["expectedDays"]),
                    args.expected_manifest_sha256,
                    args.parity_receipt.resolve(),
                )
            else:
                verification = verify_strict_window(
                    strict_root,
                    int(config["expectedDays"]),
                    float(config["minScoreExclusive"]),
                )
        except FileNotFoundError:
            state["observedAt"] = now()
            atomic_json(state_path, state)
            time.sleep(args.poll_seconds)
            continue
        state.update({
            "status": "strict_verified_launching",
            "strictVerification": verification,
            "observedAt": now(),
        })
        atomic_json(state_path, state)
        command = args.launch_command.resolve()
        if not command.is_file():
            raise FileNotFoundError(f"launch command is missing: {command}")
        completed = subprocess.run(
            ["bash", str(command), str(strict_root), str(control)],
            check=False,
            text=True,
            capture_output=True,
        )
        atomic_json(control / "launch-receipt.json", {
            "schemaVersion": 1,
            "launchedAt": now(),
            "command": str(command),
            "returnCode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "strictVerification": verification,
        })
        if completed.returncode != 0:
            state.update({"status": "launch_failed", "returnCode": completed.returncode})
            atomic_json(state_path, state)
            raise SystemExit(completed.returncode)
        formal = args.launch_mode == "formal_profiles"
        state.update({
            "status": "profiles_launched" if formal else "authority_builders_launched",
            "formalTrainingStarted": formal,
            "launchMode": args.launch_mode,
            "launchedAt": now(),
        })
        atomic_json(state_path, state)
        return


if __name__ == "__main__":
    main()
