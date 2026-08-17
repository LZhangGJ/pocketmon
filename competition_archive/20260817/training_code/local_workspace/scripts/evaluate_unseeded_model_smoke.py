from __future__ import annotations

import argparse, json, os, resource, shlex, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl.unseeded_eval import require_sha256, run_game_subprocess, stage_b_schedule, summarize_stage_b
from scripts.evaluate_unseeded_runtime import (API_SHA, ARCHIVE_SHA, CHECKPOINT_SHA, GAME_SHA, NATIVE_SHA,
    RUNTIME_NATIVE_SHA, SIM_SHA, WHEEL_SHA, append_jsonl, write_json)


def git_state():
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return sha, dirty


def main() -> int:
    p = argparse.ArgumentParser(description="isolated EVAL-UNSEEDED-001 Stage B")
    for name in ("archive-zip", "archive-root", "runtime-wheel", "runtime-root", "checkpoint"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--stage-a-gate", type=Path, required=True)
    p.add_argument("--stage-a-games", type=Path, required=True)
    p.add_argument("--games-output", type=Path, required=True)
    p.add_argument("--summary-output", type=Path, required=True)
    p.add_argument("--isolation-prefix", required=True)
    p.add_argument("--hard-timeout", type=float, default=180)
    p.add_argument("--act-timeout", type=int, default=5)
    p.add_argument("--run-timeout", type=int, default=120)
    p.add_argument("--episode-steps", type=int, default=100000)
    args = p.parse_args()
    for output in (args.games_output, args.summary_output):
        if output.exists(): raise SystemExit(f"refusing to overwrite evidence: {output}")
    commit, dirty = git_state()
    if dirty: raise SystemExit("formal Stage B requires a clean worktree")
    gate = json.loads(args.stage_a_gate.read_text())
    rows_a = [json.loads(line) for line in args.stage_a_games.read_text().splitlines()]
    if gate.get("status") != "passed" or not gate.get("summary", {}).get("gate_passed") or len(rows_a) != 20:
        raise SystemExit("Stage B not authorized by complete passing Stage A evidence")
    isolation = shlex.split(args.isolation_prefix)
    if not any(x in isolation for x in ("--net", "--unshare-net", "--network=none")):
        raise SystemExit("OS-level network isolation required")
    archive_cg = args.archive_root.resolve() / "cg"
    hashes = {"archive_zip": require_sha256(args.archive_zip, ARCHIVE_SHA, "archive"),
        "archive_api": require_sha256(archive_cg/"api.py", API_SHA, "api"),
        "archive_game": require_sha256(archive_cg/"game.py", GAME_SHA, "game"),
        "archive_sim": require_sha256(archive_cg/"sim.py", SIM_SHA, "sim"),
        "archive_native": require_sha256(archive_cg/"libcg.so", NATIVE_SHA, "native"),
        "runtime_wheel": require_sha256(args.runtime_wheel, WHEEL_SHA, "wheel"),
        "runtime_bundled_native": require_sha256(args.runtime_root/"kaggle_environments/envs/cabt/cg/libcg.so", RUNTIME_NATIVE_SHA, "runtime native"),
        "checkpoint": require_sha256(args.checkpoint, CHECKPOINT_SHA, "checkpoint")}
    records=[]; started=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="eval-unseeded-stage-b-") as directory:
        for item in stage_b_schedule():
            spec=Path(directory)/f"{item['game_id']}.json"; result=Path(directory)/f"{item['game_id']}.result.json"
            write_json(spec, {**item, "stage":"B", "archive_root":str(args.archive_root.resolve()),
                "runtime_root":str(args.runtime_root.resolve()), "checkpoint":str(args.checkpoint.resolve()),
                "act_timeout":args.act_timeout, "run_timeout":args.run_timeout,
                "episode_steps":args.episode_steps, "hashes":hashes})
            command=isolation+[sys.executable,str(ROOT/"scripts/evaluate_unseeded_game_worker.py"),"--spec",str(spec),"--result",str(result)]
            row={**item,**run_game_subprocess(command,result,args.hard_timeout)}
            row.setdefault("candidate_diagnostics",{}); row.setdefault("model_decision_latency_ms",[])
            records.append(row); append_jsonl(args.games_output,row)
    summary=summarize_stage_b(records)
    summary.update({"experiment_id":"EVAL-UNSEEDED-001","stage":"B","code_commit":commit,
        "hashes":hashes,"model_games":len(records),"elapsed_seconds":time.perf_counter()-started,
        "peak_rss_kb":max([resource.getrusage(resource.RUSAGE_SELF).ru_maxrss]+[r.get("peak_rss_kb",0) for r in records]),
        "peak_vram_mb":max([0]+[r.get("peak_vram_mb",0) for r in records]),
        "os_network_isolation":{"prefix":isolation,"python_network_blocker_is_secondary":True}})
    write_json(args.summary_output,summary); print(json.dumps(summary,indent=2,sort_keys=True))
    return 0 if summary["gate_passed"] else 2

if __name__ == "__main__": raise SystemExit(main())
