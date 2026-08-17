from __future__ import annotations

import argparse
import ctypes
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARCHETYPE = "grimmsnarl_froslass_munkidori"
REMOTE_HOST = "doraemon17"
REMOTE_ROOT = f"/dev/shm/lzhang-static-deck-bc-10d-20260815-build/profiles/{ARCHETYPE}"
SHARED_ROOT = "/dataT0/Free/lzhang/pocketmon-runs/experiment7-static-deck-bc-10d-20260815"
STRICT_MANIFEST = "/tmp/lzhang-strict-scoregt1000-window-20260815T1200Z/tensordict-sources.json"
REMOTE_PYTHON = "/homes/lzhang/mypath/new/envs/trans/bin/python"


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def available_memory_mib() -> int:
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return int(status.ullAvailPhys // (1024 * 1024))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, capture_output=True)


def remote_build(control_state: Path) -> int:
    runtime = f"{SHARED_ROOT}/control/runtime"
    control = f"{SHARED_ROOT}/control/post-strict"
    duplicate = run([
        "ssh", REMOTE_HOST,
        f"pgrep -af '[r]un_static_deck_bc_profile.py.*--archetype {ARCHETYPE}'",
    ], check=False)
    if duplicate.returncode == 0 and duplicate.stdout.strip():
        fields = duplicate.stdout.strip().split(maxsplit=1)
        return int(fields[0])
    command = (
        f"mkdir -p '{REMOTE_ROOT}/logs'; "
        f"nohup '{REMOTE_PYTHON}' -s '{runtime}/integration/run_static_deck_bc_profile.py' "
        f"--config '{runtime}/config/static_deck_bc_10d_20260815.json' "
        f"--strict-manifest '{STRICT_MANIFEST}' --archetype '{ARCHETYPE}' "
        f"--local-root '{REMOTE_ROOT}' --control-root '{control}' --runtime-root '{runtime}' "
        f"--device 0 --batch-size 512 --python '{REMOTE_PYTHON}' --build-only "
        f"> '{REMOTE_ROOT}/logs/controller.log' 2>&1 & echo $!"
    )
    launched = run(["ssh", REMOTE_HOST, command])
    pid = int(launched.stdout.strip().splitlines()[-1])
    atomic_json(control_state, {"status": "remote_grim_shards_building", "remoteHost": REMOTE_HOST, "remotePid": pid, "observedAt": now()})
    return pid


def wait_remote_sources(pid: int, state_path: Path) -> None:
    while True:
        ready = run(["ssh", REMOTE_HOST, f"test -f '{REMOTE_ROOT}/tensordict-sources.json'"], check=False)
        if ready.returncode == 0:
            return
        alive = run(["ssh", REMOTE_HOST, f"kill -0 {pid}"], check=False)
        if alive.returncode != 0:
            tail = run(["ssh", REMOTE_HOST, f"tail -80 '{REMOTE_ROOT}/logs/controller.log'"], check=False)
            raise RuntimeError(f"remote Grim shard builder exited before sources: {tail.stdout}{tail.stderr}")
        atomic_json(state_path, {"status": "remote_grim_shards_building", "remoteHost": REMOTE_HOST, "remotePid": pid, "observedAt": now()})
        time.sleep(15)


def sources_manifest(dataset_root: Path, reference_root: Path) -> dict[str, Any]:
    datasets = []
    for day_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        shard = day_dir / ARCHETYPE
        receipt_path = shard / "specialist-receipt.json"
        if not receipt_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        datasets.append({
            "name": day_dir.name,
            "root": str(shard),
            "features": str(shard / "features_tensordict"),
            "decisions": str(shard / "decisions.jsonl.gz"),
            "tokenCache": str(shard / "token_cache"),
            "sequenceCache": str(shard / "sequence_cache"),
            "identityCache": str(shard / "identity_cache"),
            "specialistReceipt": str(receipt_path),
            "summary": {"sourceEpisodes": receipt["episodes"], "decisions": receipt["decisions"]},
        })
    if len(datasets) != 10:
        raise ValueError(f"Windows Grim source parity failed: {len(datasets)} != 10")
    return {
        "schemaVersion": 1,
        "kind": "experiment7_universal_bc",
        "staticProfile": f"10d-deck-specialist-static-bc:{ARCHETYPE}",
        "referenceRoot": str(reference_root),
        "strictPredicate": "is_clean == 1 and float(min_score) > 1000.0",
        "minGameScoreExclusive": 1000.0,
        "datasets": datasets,
    }


def gpu_idle() -> tuple[int, int]:
    result = run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits", "-i", "0"])
    memory, utilization = (int(value.strip()) for value in result.stdout.strip().split(","))
    return memory, utilization


def gpu_sample_is_idle(sample: tuple[int, int]) -> bool:
    """Allow the measured WDDM desktop baseline while rejecting GPU workloads."""
    memory, utilization = sample
    return memory <= 3072 and utilization <= 20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persistent-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    persistent = args.persistent_root.resolve()
    scratch = args.scratch_root.resolve()
    state_path = args.state.resolve()
    remote_pid = remote_build(state_path)
    wait_remote_sources(remote_pid, state_path)
    dataset_root = scratch / "datasets" / "grim-10d-20260815" / "day-shards"
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    integration = persistent / "runtime" / "source" / "experiment7" / "integration"
    reference = persistent / "runtime" / "source" / "experiment7" / "reference"
    config = persistent / "runtime" / "source" / "experiment7" / "config" / "static_deck_bc_10d_20260815.json"
    checkpoint = persistent / "checkpoints" / "universal-large-g51.pt"
    sources_path = dataset_root.parent / "tensordict-sources.windows.json"
    try:
        local_sources = sources_manifest(dataset_root, reference)
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
        local_sources = None
    if local_sources is None:
        atomic_json(state_path, {"status": "pulling_grim_shards_to_windows_scratch", "remotePid": remote_pid, "target": str(dataset_root), "observedAt": now()})
        run(["scp", "-r", "-o", "ProxyJump=taka2", f"lzhang@10.113.13.75:{REMOTE_ROOT}/day-shards", str(dataset_root.parent)])
        local_sources = sources_manifest(dataset_root, reference)
    atomic_json(sources_path, local_sources)
    first = gpu_idle()
    time.sleep(10)
    second = gpu_idle()
    if not all(gpu_sample_is_idle(sample) for sample in (first, second)):
        raise RuntimeError(f"Windows GPU0 is not idle after two samples: {first}, {second}")
    profile = persistent / "profiles" / ARCHETYPE / "training"
    profile.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": str(integration),
        "PIP_CACHE_DIR": str(scratch / "cache" / "pip"),
        "TMP": str(scratch / "tmp"),
        "TEMP": str(scratch / "tmp"),
        "CUDA_VISIBLE_DEVICES": "0",
    })
    python = persistent / "runtime" / "venv" / "Scripts" / "python.exe"
    logs = persistent / "profiles" / ARCHETYPE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    available_mib = available_memory_mib()
    batch_size = 256 if available_mib >= 8192 else 128
    validator_log = (logs / "validator.log").open("a", encoding="utf-8")
    validator = subprocess.Popen([
        str(python), str(integration / "validate_static_deck_bc_async.py"),
        "--config", str(config), "--sources", str(sources_path), "--output-dir", str(profile),
        "--initialize-from", str(checkpoint), "--device", "cuda:0", "--batch-size", str(batch_size),
    ], stdout=validator_log, stderr=subprocess.STDOUT, env=environment)
    while not (profile / "async-validation-report.json").is_file():
        if validator.poll() is not None:
            raise RuntimeError(f"Windows validator exited before baseline: {validator.returncode}")
        time.sleep(5)
    trainer_log = (logs / "trainer.log").open("a", encoding="utf-8")
    progress_checkpoint = profile / "recovery" / "epoch_000001_latest.pt"
    resume_args = ["--resume-progress", str(progress_checkpoint)] if progress_checkpoint.is_file() else []
    recovery_receipt = {
        "schemaVersion": 1,
        "status": "windows_grim_ram_recovery_launched",
        "reason": "host_ram_allocation_failure",
        "availableMemoryMiB": available_mib,
        "batchSize": batch_size,
        "prefetchBatches": 0,
        "prefetchWorkers": 1,
        "pinMemory": False,
        "resumeProgress": str(progress_checkpoint) if progress_checkpoint.is_file() else None,
        "observedAt": now(),
    }
    atomic_json(persistent / "control" / "windows-grim-ram-recovery.json", recovery_receipt)
    trainer = subprocess.Popen([
        str(python), str(integration / "train_static_deck_bc_async.py"),
        "--config", str(config), "--sources", str(sources_path), "--output-dir", str(profile),
        "--initialize-from", str(checkpoint), "--device", "cuda:0", "--batch-size", str(batch_size),
        "--prefetch-batches", "0", "--prefetch-workers", "1", *resume_args,
    ], stdout=trainer_log, stderr=subprocess.STDOUT, env=environment)
    atomic_json(state_path, {
        "status": "windows_grim_training_with_async_validation",
        "trainerPid": trainer.pid,
        "validatorPid": validator.pid,
        "gpu": 0,
        "batchSize": batch_size,
        "prefetchBatches": 0,
        "resumeProgress": str(progress_checkpoint) if progress_checkpoint.is_file() else None,
        "sources": str(sources_path),
        "formalTrainingStarted": True,
        "observedAt": now(),
    })
    trainer_rc = trainer.wait()
    validator_rc = validator.wait()
    trainer_log.close()
    validator_log.close()
    best = profile / "best_model.pt"
    if trainer_rc or validator_rc or not best.is_file():
        raise RuntimeError(f"Windows Grim training failed trainer={trainer_rc} validator={validator_rc} best={best.is_file()}")
    frozen = persistent / "frozen" / ARCHETYPE / "best_model.pt"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    temporary = frozen.with_name(f".{frozen.name}.{os.getpid()}.tmp")
    shutil.copyfile(best, temporary)
    os.replace(temporary, frozen)
    final = {
        "status": "windows_grim_training_complete",
        "bestModel": str(best),
        "bestSha256": sha256(best),
        "frozenCopy": str(frozen),
        "frozenSha256": sha256(frozen),
        "copies": 2,
        "staticFrozen": True,
        "ppoUpdatesAllowed": False,
        "completedAt": now(),
    }
    atomic_json(state_path, final)
    atomic_json(persistent / "control" / "windows-grim-completion.json", final)


if __name__ == "__main__":
    main()
