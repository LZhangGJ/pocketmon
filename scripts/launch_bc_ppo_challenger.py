from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def wait_for_formal_metrics(metrics_path: Path, timeout_seconds: float, poll_seconds: float) -> dict:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            status = metrics.get("status")
            if status == "completed_formal":
                return metrics
            if status not in (None, "running"):
                raise RuntimeError(f"training did not complete formally: status={status!r}")
        print(f"{now()} waiting for {metrics_path}", flush=True)
        time.sleep(poll_seconds)
    raise TimeoutError(f"timed out waiting for formal metrics: {metrics_path}")


def validate_metrics(metrics: dict, checkpoint: Path) -> dict:
    failures: list[str] = []
    actual = metrics.get("actual_config") or {}
    validation = metrics.get("validation") or {}
    recorded_checkpoint = metrics.get("checkpoint") or {}
    if metrics.get("status") != "completed_formal":
        failures.append("status is not completed_formal")
    if actual.get("config_matched") is not True:
        failures.append("planned and actual configs do not match")
    if metrics.get("failures"):
        failures.append(f"training reported failures: {metrics['failures']}")
    if validation.get("decode_legal_rate") != 1.0:
        failures.append(f"decode_legal_rate is {validation.get('decode_legal_rate')!r}, expected 1.0")
    if validation.get("invalid_actions") != 0:
        failures.append(f"invalid_actions is {validation.get('invalid_actions')!r}, expected 0")
    if not checkpoint.is_file():
        failures.append(f"checkpoint is missing: {checkpoint}")
        actual_sha = None
    else:
        actual_sha = sha256(checkpoint)
        if recorded_checkpoint.get("sha256") != actual_sha:
            failures.append("checkpoint SHA-256 does not match metrics")
    report = {
        "checked_at": now(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "status": metrics.get("status"),
        "config_matched": actual.get("config_matched"),
        "decode_legal_rate": validation.get("decode_legal_rate"),
        "invalid_actions": validation.get("invalid_actions"),
        "failures": failures,
        "passed": not failures,
    }
    if failures:
        raise RuntimeError("; ".join(failures))
    return report


def run_smoke(
    *, python: str, code_root: Path, candidate: Path, parent: Path, cg_dir: Path, seed: int
) -> list[dict]:
    rows = []
    for candidate_seat in (0, 1):
        agents = [candidate, parent] if candidate_seat == 0 else [parent, candidate]
        command = [
            python,
            str(code_root / "scripts" / "run_local_match.py"),
            "--agent0",
            str(agents[0]),
            "--agent1",
            str(agents[1]),
            "--cg-dir",
            str(cg_dir),
            "--max-decisions",
            "5000",
            "--seed",
            str(seed + candidate_seat),
        ]
        completed = subprocess.run(
            command, cwd=code_root, check=True, capture_output=True, text=True, timeout=900
        )
        row = json.loads(completed.stdout.strip().splitlines()[-1])
        row["candidate_seat"] = candidate_seat
        diagnostics = row.get("agent_diagnostics") or []
        for diagnostic in diagnostics:
            if diagnostic.get("load_error") or diagnostic.get("inference_error"):
                raise RuntimeError(f"agent diagnostic error: {diagnostic}")
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wait for a formal BC refresh, smoke it, then launch its finite PPO challenger branch"
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--agent-output", type=Path, required=True)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--cg-dir", type=Path, required=True)
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout-hours", type=float, default=8.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pipeline-log", type=Path, required=True)
    args = parser.parse_args()

    metrics = wait_for_formal_metrics(
        args.metrics, args.timeout_hours * 3600.0, args.poll_seconds
    )
    report = validate_metrics(metrics, args.checkpoint)

    from materialize_rl_specialist_agent import materialize

    if args.agent_output.exists():
        manifest_path = args.agent_output / "agent_manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(f"existing output is not an agent package: {args.agent_output}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("checkpoint_sha256") != report["checkpoint_sha256"]:
            raise FileExistsError("existing package was built from a different checkpoint")
    else:
        manifest = materialize(
            args.checkpoint, args.deck, args.agent_output, args.agent_name
        )
    report["agent_manifest"] = manifest
    report["smoke_matches"] = run_smoke(
        python=args.python,
        code_root=args.code_root,
        candidate=args.agent_output,
        parent=args.parent,
        cg_dir=args.cg_dir,
        seed=args.seed,
    )

    config = json.loads(args.pipeline_config.read_text(encoding="utf-8"))
    if Path(config["initial_champion_package"]) != args.agent_output:
        raise ValueError("pipeline initial_champion_package does not match materialized agent")
    run_root = Path(config["run_root"])
    state_path = run_root / "state.json"
    if state_path.exists():
        raise FileExistsError(f"PPO challenger state already exists: {state_path}")

    args.pipeline_log.parent.mkdir(parents=True, exist_ok=True)
    log_handle = args.pipeline_log.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [args.python, str(args.code_root / "scripts" / "continuous_rl_pipeline.py"), "--config", str(args.pipeline_config)],
        cwd=args.code_root,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    report.update({
        "passed": True,
        "ppo_launched_at": now(),
        "ppo_pid": process.pid,
        "pipeline_config": str(args.pipeline_config),
        "pipeline_log": str(args.pipeline_log),
        "run_root": str(run_root),
    })
    atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
