from __future__ import annotations

import argparse, importlib.util, json, os, platform, resource, shlex, socket, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl.unseeded_eval import (OfficialCabtModuleFinder, install_agent_cg_alias, loaded_native_libraries,
    preflight_gates, require_isolation_prefix, require_sha256, run_monitored_subprocess)
from scripts.evaluate_unseeded_runtime import CHECKPOINT_SHA, NATIVE_SHA, write_json


def worker(checkpoint: Path, archive_root: Path, runtime_root: Path, output: Path) -> int:
    result = {"exception": None, "urandom_readable": False, "os_urandom_successful": False,
              "torch_import_successful": False, "torch_cpu_tensor_successful": False,
              "checkpoint_loaded": False, "checkpoint_hash_verified": False,
              "native_loaded": False, "native_hash_verified": False,
              "eth0_absent": False, "tcp_unavailable": False, "dns_unavailable": False}
    try:
        with Path("/dev/urandom").open("rb") as handle:
            result["urandom_readable"] = len(handle.read(1)) == 1
        result["os_urandom_successful"] = len(os.urandom(32)) == 32
        import torch
        result["torch_import_successful"] = True
        result["torch_cpu_tensor_successful"] = torch.tensor([1.0], device="cpu").item() == 1.0
        result["checkpoint_sha256"] = require_sha256(checkpoint, CHECKPOINT_SHA, "RL-BC-002-A checkpoint")
        result["checkpoint_hash_verified"] = True
        torch.load(checkpoint, map_location="cpu", weights_only=False)
        result["checkpoint_loaded"] = True
        archive_cg = archive_root / "cg"
        sys.path[:0] = [str(archive_root), str(runtime_root)]
        sys.meta_path.insert(0, OfficialCabtModuleFinder(archive_cg))
        from kaggle_environments.envs.cabt.cg import sim as loaded_sim
        install_agent_cg_alias(archive_cg, loaded_sim)
        native_path = Path(loaded_sim.lib._name).resolve()
        result["mapped_native_path"] = str(native_path)
        result["mapped_native_sha256"] = require_sha256(native_path, NATIVE_SHA, "loaded native")
        result["mapped_native_libraries"] = {str(path): require_sha256(path, NATIVE_SHA, "mapped native")
                                             for path in loaded_native_libraries()}
        result["native_loaded"] = True
        result["native_hash_verified"] = bool(result["mapped_native_libraries"])
        result["eth0_absent"] = not Path("/sys/class/net/eth0").exists()
        try:
            socket.create_connection(("1.1.1.1", 53), timeout=0.5)
        except OSError:
            result["tcp_unavailable"] = True
        try:
            socket.getaddrinfo("eval-isolation-network-test.invalid", 80)
        except socket.gaierror:
            result["dns_unavailable"] = True
    except Exception as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
    result["child_peak_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(output, result)
    return 0 if result["exception"] is None else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="EVAL-ISOLATION-001 no-game preflight")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--isolation-prefix", required=True)
    parser.add_argument("--preflight-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args.checkpoint.resolve(), args.archive_root.resolve(), args.runtime_root.resolve(), args.worker_output)
    for output in (args.preflight_output, args.summary_output):
        if output.exists(): raise SystemExit(f"refusing to overwrite evidence: {output}")
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    if dirty: raise SystemExit("formal preflight requires a clean worktree")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    prefix = shlex.split(args.isolation_prefix); require_isolation_prefix(prefix)
    child_output = args.preflight_output.with_suffix(".child.json")
    command = prefix + [sys.executable, str(Path(__file__).resolve()), "--worker",
        "--checkpoint", str(args.checkpoint.resolve()), "--archive-root", str(args.archive_root.resolve()),
        "--runtime-root", str(args.runtime_root.resolve()),
        "--isolation-prefix", args.isolation_prefix, "--preflight-output", str(args.preflight_output),
        "--summary-output", str(args.summary_output), "--worker-output", str(child_output)]
    process = run_monitored_subprocess(command, args.timeout)
    child = json.loads(child_output.read_text()) if child_output.exists() else {"exception":"child evidence missing"}
    if child_output.exists(): child_output.unlink()
    gates = preflight_gates(child, process)
    parent_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    preflight = {"experiment_id":"EVAL-ISOLATION-001","code_commit":commit,"dirty_at_start":dirty,
        "host":platform.node(),"isolation_prefix":prefix,"child":child,"process":process,"gates":gates,
        "gate_passed":all(gates.values()),"parent_peak_rss_kb":parent_rss,
        "max_child_peak_rss_kb":max(process["max_child_peak_rss_kb"], child.get("child_peak_rss_kb",0)),
        "max_process_tree_peak_rss_kb":process["max_process_tree_peak_rss_kb"]}
    preflight["overall_peak_rss_kb"] = max(preflight["parent_peak_rss_kb"],preflight["max_process_tree_peak_rss_kb"])
    preflight["overall_peak_rss_definition"] = "max(parent ru_maxrss, sampled sum of child plus descendants VmRSS)"
    write_json(args.preflight_output,preflight)
    summary={key:preflight[key] for key in ("experiment_id","code_commit","gates","gate_passed","parent_peak_rss_kb",
        "max_child_peak_rss_kb","max_process_tree_peak_rss_kb","overall_peak_rss_kb","overall_peak_rss_definition")}
    summary.update({"elapsed_seconds":process["elapsed_seconds"],"exit_code":process["exit_code"],
        "signal":process["signal"],"hard_timeout":process["hard_timeout"],"stdout":process["stdout"],"stderr":process["stderr"],
        "checkpoint_sha256":child.get("checkpoint_sha256"),"native_sha256":child.get("mapped_native_sha256")})
    write_json(args.summary_output,summary); print(json.dumps(summary,indent=2,sort_keys=True))
    return 0 if summary["gate_passed"] else 2

if __name__ == "__main__": raise SystemExit(main())
