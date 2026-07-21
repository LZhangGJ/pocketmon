from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl.unseeded_eval import alternating_schedule, require_sha256, resource_peak_fields, run_game_subprocess, summarize_stage_a

ARCHIVE_SHA = "09ad210b15476f5064c1509addb32a459c777d92d4e4e7db470f9d0c039c3282"
API_SHA = "593f1298e52a635f90f8f505a52113e9af114f444c293404e37906f18ee06ced"
GAME_SHA = "3bd3d4f4a369a11e6d2f5da9094cf15ebc410a2221835e6417b7cff4883f1fc2"
SIM_SHA = "1555f57f5d22bf4c09d70e0e667a916e575e68c9dd1de9ead34ba5e7e4968655"
NATIVE_SHA = "feafd4046b2f688bdb33a4972c139b78e13e243ab5707ece52c43cf39a34b887"
WHEEL_SHA = "359226741a04fbe1dbbc10121aef140fd96ec4fa31bace2037d05e7ef2bbf4e8"
RUNTIME_NATIVE_SHA = "7acbfc7bc61d4f8233515c63debcfa454b8f804f138a6c395c599decc3dd17d0"
CHECKPOINT_SHA = "2faac94de9e937dee77cd6d5d44036d7f45bb2dc4cc6491c1c97c0091f4fb216"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_state() -> tuple[str, bool]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return sha, dirty


def main() -> int:
    parser = argparse.ArgumentParser(description="EVAL-UNSEEDED-001 isolated Stage A parent")
    for name in ("archive-zip", "archive-root", "runtime-wheel", "runtime-root", "checkpoint"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--act-timeout", type=int, default=5)
    parser.add_argument("--run-timeout", type=int, default=120)
    parser.add_argument("--episode-steps", type=int, default=100000)
    parser.add_argument("--hard-timeout", type=float, default=180)
    parser.add_argument("--isolation-prefix", required=True,
                        help="OS isolation command, e.g. 'unshare --user --map-root-user --net --'")
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument("--games-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    if args.games != 20:
        raise SystemExit("Stage A requires exactly 20 games")
    isolation = shlex.split(args.isolation_prefix)
    if not isolation or not any(token in isolation for token in ("--net", "--unshare-net", "--network=none")):
        raise SystemExit("formal run requires an explicit OS-level no-network isolation prefix")
    for output in (args.gate_output, args.games_output, args.summary_output):
        if output.exists():
            raise SystemExit(f"refusing to overwrite existing evidence: {output}")
    commit, dirty = git_state()
    if dirty:
        raise SystemExit("formal Stage A requires a clean worktree")
    archive_cg = args.archive_root.resolve() / "cg"
    hashes = {
        "archive_zip": require_sha256(args.archive_zip, ARCHIVE_SHA, "competition archive"),
        "archive_api": require_sha256(archive_cg / "api.py", API_SHA, "competition api.py"),
        "archive_game": require_sha256(archive_cg / "game.py", GAME_SHA, "competition game.py"),
        "archive_sim": require_sha256(archive_cg / "sim.py", SIM_SHA, "competition sim.py"),
        "archive_native": require_sha256(archive_cg / "libcg.so", NATIVE_SHA, "competition libcg.so"),
        "runtime_wheel": require_sha256(args.runtime_wheel, WHEEL_SHA, "runtime wheel"),
        "runtime_bundled_native": require_sha256(
            args.runtime_root / "kaggle_environments/envs/cabt/cg/libcg.so", RUNTIME_NATIVE_SHA, "runtime native"),
        "checkpoint": require_sha256(args.checkpoint, CHECKPOINT_SHA, "frozen checkpoint"),
    }
    base = {
        "archive_root": str(args.archive_root.resolve()), "runtime_root": str(args.runtime_root.resolve()),
        "act_timeout": args.act_timeout, "run_timeout": args.run_timeout,
        "episode_steps": args.episode_steps, "hashes": hashes,
    }
    gate = {
        "experiment_id": "EVAL-UNSEEDED-001", "stage": "A", "status": "running",
        "code_commit": commit, "dirty_at_start": dirty, "hashes": hashes,
        "host": platform.node(), "python": sys.version.replace("\n", " "),
        "command": [sys.executable, *sys.argv], "os_network_isolation": {
            "required": True, "prefix": isolation, "full_command_recorded_per_game": True,
            "python_network_blocker_is_secondary": True,
        },
        "nominal_python_seed_used_as_engine_seed": False, "pairing_key_used": False,
    }
    write_json(args.gate_output, gate)
    records = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="eval-unseeded-001-") as directory:
        temp = Path(directory)
        for item in alternating_schedule(20):
            spec_path = temp / f"game-{item['game_id']}.spec.json"
            result_path = temp / f"game-{item['game_id']}.result.json"
            write_json(spec_path, {**base, **item})
            command = isolation + [sys.executable, str(ROOT / "scripts/evaluate_unseeded_game_worker.py"),
                                   "--spec", str(spec_path), "--result", str(result_path)]
            row = {**item, **run_game_subprocess(command, result_path, args.hard_timeout)}
            row.setdefault("statuses", [])
            row.setdefault("rewards", [])
            row.setdefault("native_hash_verified", False)
            row.setdefault("network_attempts", 0)
            records.append(row)
            append_jsonl(args.games_output, row)
    network_attempts = sum(int(row.get("network_attempts", 0)) for row in records)
    summary = summarize_stage_a(records, 20, network_attempts)
    resources = resource_peak_fields(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                     [row.get("peak_rss_kb", 0) for row in records])
    summary.update({
        "experiment_id": "EVAL-UNSEEDED-001", "stage": "A", "code_commit": commit,
        "elapsed_seconds": time.perf_counter() - started,
        **resources,
        "peak_vram_mb": 0, "model_games": 0,
        "stage_b_authorized": summary["gate_passed"], "hashes": hashes,
        "os_network_isolation": gate["os_network_isolation"],
    })
    gate["status"] = "passed" if summary["gate_passed"] else "failed"
    gate["summary"] = summary
    write_json(args.gate_output, gate)
    write_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
