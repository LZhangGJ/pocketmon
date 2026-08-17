from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from stage_and_run_scaled_universal_bc import stage_sources


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with source.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer, 4 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temporary, target)


def process_command(pid: int) -> str | None:
    path = Path(f"/proc/{pid}/cmdline")
    if not path.is_file():
        return None
    raw = path.read_bytes()
    if not raw:
        return None
    return raw.replace(b"\0", b" ").decode("utf-8", "replace")


def terminate_verified(pid: int, needle: str) -> dict:
    command = process_command(pid)
    if command is None:
        return {"pid": pid, "status": "already_exited"}
    if needle not in command:
        raise RuntimeError(f"PID {pid} command mismatch: {command}")
    os.kill(pid, signal.SIGTERM)
    os.kill(pid, signal.SIGCONT)
    for _ in range(50):
        if process_command(pid) is None:
            return {"pid": pid, "status": "terminated", "command": command}
        time.sleep(0.2)
    raise RuntimeError(f"PID {pid} did not terminate after SIGTERM")


def write_state(control: Path, state: dict) -> None:
    payload = {"schemaVersion": 1, "controllerPid": os.getpid(), "host": socket.gethostname(), **state}
    atomic_json(control / "state.json", payload)
    atomic_json(control / "latest.json", payload)


def archive_epoch2(args: argparse.Namespace, control: Path) -> dict:
    while not args.epoch2_result.is_file():
        if process_command(args.old_validator_pid) is None:
            raise RuntimeError("epoch2 validator exited before result publication")
        write_state(control, {
            "status": "waiting_for_epoch2_validation",
            "checkpoint": str(args.epoch2_checkpoint),
            "checkpointSha256": args.epoch2_sha256,
            "result": str(args.epoch2_result),
        })
        time.sleep(args.poll_seconds)
    if sha256(args.epoch2_checkpoint) != args.epoch2_sha256:
        raise RuntimeError("epoch2 checkpoint SHA changed before archive")
    archive = args.profile_root / "frozen" / "epoch_000002"
    archive_checkpoint = archive / "checkpoint.pt"
    atomic_copy(args.epoch2_checkpoint, archive_checkpoint)
    if sha256(archive_checkpoint) != args.epoch2_sha256:
        raise RuntimeError("archived epoch2 checkpoint SHA mismatch")
    files = {
        "validationResult": (args.epoch2_result, "validation-result.json"),
        "trainingReport": (args.epoch2_output / "async-training-report.json", "async-training-report.json"),
        "validationReport": (args.epoch2_output / "async-validation-report.json", "async-validation-report.json"),
        "validationQueue": (args.epoch2_output / "validation-queue" / "epoch_000002.json", "validation-queue.json"),
    }
    archived_files = {}
    for name, (source, target_name) in files.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = archive / target_name
        atomic_copy(source, target)
        archived_files[name] = {"path": str(target), "sha256": sha256(target)}
    result = read_json(args.epoch2_result)
    receipt = {
        "schemaVersion": 1,
        "status": "epoch2_validation_archived",
        "archivedAt": now(),
        "checkpoint": str(archive_checkpoint),
        "checkpointSha256": args.epoch2_sha256,
        "validation": result,
        "files": archived_files,
    }
    atomic_json(archive / "archive-receipt.json", receipt)
    cleanup = [
        terminate_verified(args.old_trainer_pid, "train_universal_bc_async_scaled.py"),
        terminate_verified(args.old_validator_pid, "validate_universal_bc_async_scaled.py"),
    ]
    for _ in range(50):
        if process_command(args.old_controller_pid) is None:
            break
        time.sleep(0.2)
    receipt["oldPersistentBoundaryStop"] = {
        "completedAt": now(),
        "children": cleanup,
        "controllerPid": args.old_controller_pid,
        "controllerExited": process_command(args.old_controller_pid) is None,
        "safety": "epoch2 archived before exact-PID termination; no PPO process signaled",
    }
    atomic_json(archive / "archive-receipt.json", receipt)
    return receipt


def strict_ready(args: argparse.Namespace) -> tuple[bool, dict]:
    evidence = {
        "success": args.strict_success.is_file(),
        "parity": args.parity_receipt.is_file(),
        "manifest": args.strict_manifest.is_file(),
    }
    if not all(evidence.values()):
        return False, evidence
    parity = read_json(args.parity_receipt)
    manifest = read_json(args.strict_manifest)
    manifest_sha256 = sha256(args.strict_manifest)
    evidence.update({
        "parityStatus": parity.get("status"),
        "parityDays": len(parity.get("days", [])),
        "minGameScoreExclusive": manifest.get("minGameScoreExclusive"),
        "policySource": manifest.get("policySource"),
        "sourceDays": len(manifest.get("datasets", [])),
        "cardVocab": manifest.get("engineCatalog", {}).get("cardVocab"),
        "manifestSha256": manifest_sha256,
        "expectedManifestSha256": args.expected_strict_manifest_sha256,
    })
    ready = (
        parity.get("status") == "passed"
        and len(parity.get("days", [])) == 10
        and manifest.get("minGameScoreExclusive") == 1000.0
        and manifest.get("policySource") == "winners"
        and len(manifest.get("datasets", [])) == 10
        and int(manifest.get("engineCatalog", {}).get("cardVocab", -1)) == 1268
        and (
            args.expected_strict_manifest_sha256 is None
            or manifest_sha256 == args.expected_strict_manifest_sha256
        )
    )
    return ready, evidence


def gpu_rows() -> list[dict]:
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    busy_uuids = {line.split(",", 1)[0].strip() for line in apps.splitlines() if line.strip()}
    rows = []
    for line in output.splitlines():
        index, uuid, util, memory = (value.strip() for value in line.split(","))
        rows.append({
            "index": int(index),
            "uuid": uuid,
            "utilization": int(util),
            "memoryMiB": int(memory),
            "hasComputeProcess": uuid in busy_uuids,
        })
    return rows


def choose_gpu_pair(args: argparse.Namespace, control: Path) -> tuple[int, int, list[dict]]:
    consecutive = 0
    last = []
    while consecutive < 2:
        last = gpu_rows()
        free = [
            row["index"] for row in last
            if not row["hasComputeProcess"]
            and row["utilization"] < 5
            and row["memoryMiB"] < 512
        ]
        if args.gpu_pair is not None:
            requested = list(args.gpu_pair)
            free = requested if all(index in free for index in requested) else []
        if len(free) >= 2:
            consecutive += 1
        else:
            consecutive = 0
        write_state(control, {
            "status": "waiting_for_two_conflict_free_gpus",
            "gpuRows": last,
            "consecutiveIdleSamples": consecutive,
            "standardStarted": False,
        })
        if consecutive < 2:
            time.sleep(args.gpu_poll_seconds)
    return free[0], free[1], last


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--epoch2-output", type=Path, required=True)
    parser.add_argument("--epoch2-checkpoint", type=Path, required=True)
    parser.add_argument("--epoch2-result", type=Path, required=True)
    parser.add_argument("--epoch2-sha256", required=True)
    parser.add_argument("--strict-success", type=Path, required=True)
    parser.add_argument("--parity-receipt", type=Path, required=True)
    parser.add_argument("--strict-manifest", type=Path, required=True)
    parser.add_argument("--expected-strict-manifest-sha256")
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--old-controller-pid", type=int, required=True)
    parser.add_argument("--old-trainer-pid", type=int, required=True)
    parser.add_argument("--old-validator-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=20)
    parser.add_argument("--gpu-poll-seconds", type=float, default=10)
    parser.add_argument("--gpu-pair", nargs=2, type=int)
    args = parser.parse_args()

    control = args.profile_root / "post-strict-31m-controller"
    control.mkdir(parents=True, exist_ok=True)
    lock = (control / "controller.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("31M_POST_STRICT_CONTROLLER_ALREADY_RUNNING")
    (control / "controller.pid").write_text(f"{os.getpid()}\n")
    try:
        archive = archive_epoch2(args, control)
        write_state(control, {
            "status": "waiting_for_strict_success_and_parity",
            "epoch2Archive": archive,
            "strictSuccess": str(args.strict_success),
            "parityReceipt": str(args.parity_receipt),
            "standardStarted": False,
        })
        while True:
            ready, evidence = strict_ready(args)
            if ready:
                break
            write_state(control, {
                "status": "waiting_for_strict_success_and_parity",
                "epoch2Archive": archive,
                "strictEvidence": evidence,
                "standardStarted": False,
            })
            time.sleep(args.poll_seconds)

        final = args.output_root.resolve() / "ultra_512x8_31m"
        attempt = final / "attempts" / args.attempt_id
        final.mkdir(parents=True, exist_ok=True)
        attempt.mkdir(parents=True, exist_ok=True)
        write_state(control, {"status": "staging_strict_window", "startedAt": now(), "standardStarted": False})
        local_sources = stage_sources(args.strict_manifest, args.stage_root, 4, final, attempt)
        train_gpu, validation_gpu, gpu_evidence = choose_gpu_pair(args, control)
        initialize_from = Path(archive["checkpoint"])
        command = [
            str(args.python), "-s", str(args.runtime / "stage_and_run_scaled_universal_bc.py"),
            "--sources", str(args.strict_manifest),
            "--stage-root", str(args.stage_root),
            "--scratch-root", str(args.scratch_root),
            "--runtime", str(args.runtime),
            "--python", str(args.python),
            "--output-root", str(args.output_root),
            "--baseline-report", str(args.baseline_report),
            "--initialize-from", str(initialize_from),
            "--train-gpu", str(train_gpu),
            "--validation-gpu", str(validation_gpu),
            "--copy-workers", "4",
            "--lock-name", "controller-post-strict-31m.lock",
            "--attempt-id", args.attempt_id,
            "--expected-card-vocab", "1268",
        ]
        log = control / "incremental-controller.log"
        environment = dict(os.environ, PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with log.open("a") as handle:
            child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=environment)
        time.sleep(5)
        if child.poll() is not None:
            raise RuntimeError(f"incremental controller exited early rc={child.returncode}")
        receipt = {
            "schemaVersion": 1,
            "status": "launched",
            "launchedAt": now(),
            "controllerPid": child.pid,
            "trainGpu": train_gpu,
            "validationGpu": validation_gpu,
            "gpuIdleEvidence": gpu_evidence,
            "sources": str(local_sources),
            "strictManifest": str(args.strict_manifest),
            "strictManifestSha256": sha256(args.strict_manifest),
            "initializeFrom": str(initialize_from),
            "initializeFromSha256": args.epoch2_sha256,
            "parameterCount": 30_724_612,
            "cardVocab": 1268,
            "attemptId": args.attempt_id,
            "command": command,
            "standardStarted": False,
        }
        atomic_json(control / "launch-receipt.json", receipt)
        write_state(control, {"status": "incremental_controller_launched", "launch": receipt})
        return 0
    except Exception as error:
        write_state(control, {
            "status": "failed",
            "failedAt": now(),
            "error": f"{type(error).__name__}: {error}",
            "standardStarted": False,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
