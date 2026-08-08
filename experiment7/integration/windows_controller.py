from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_SERVERS = [
    "doraemon02",
    "doraemon03",
    "doraemon15",
    "doraemon16",
    "doraemon19",
    "doraemon20",
]
REMOTE_REPO = "/homes/lzhang/pocketmon"
REMOTE_PYTHON = "/homes/lzhang/mypath/new/envs/trans/bin/python"
REPLAY_DIR = "/homes/lzhang/pocketmon/data/raw/replays/2026-08-06"
ANALYSIS_DIR = "/homes/lzhang/pocketmon/analysis/outputs/top_ladder_2026_08_07_20260808"
RUN_ROOT = "/homes/lzhang/pocketmon/runs/experiment7-multideck-20260808"


@dataclass(frozen=True)
class GPU:
    host: str
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_percent: int


def checked(command: list[str], *, cwd: Path | None = None, capture: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=capture,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command}\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    return result


def ssh(host: str, script: str, *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return checked(["ssh", host, f"bash -lc {shlex.quote(script)}"], capture=capture)


def git_value(repo: Path, *args: str) -> str:
    return checked(["git", *args], cwd=repo).stdout.strip()


def probe(servers: Iterable[str]) -> list[GPU]:
    gpus: list[GPU] = []
    query = (
        "set -euo pipefail; hostname; "
        "test -d /homes/lzhang/pocketmon; "
        "test -x /homes/lzhang/mypath/new/envs/trans/bin/python; "
        "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu "
        "--format=csv,noheader,nounits"
    )
    for host in servers:
        result = ssh(host, query)
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError(f"{host}: empty probe response")
        actual_host = lines[0]
        for line in lines[1:]:
            fields = [value.strip() for value in line.split(",")]
            if len(fields) != 6:
                raise RuntimeError(f"{host}: unexpected nvidia-smi row: {line}")
            gpus.append(
                GPU(
                    host=actual_host,
                    index=int(fields[0]),
                    name=fields[1],
                    memory_total_mb=int(fields[2]),
                    memory_used_mb=int(fields[3]),
                    memory_free_mb=int(fields[4]),
                    utilization_percent=int(fields[5]),
                )
            )
    return gpus


def worktree_path(host: str, commit: str) -> str:
    return f"/homes/lzhang/worktrees/experiment7-{host}-{commit[:12]}"


def bootstrap_host(host: str, commit: str) -> str:
    worktree = worktree_path(host, commit)
    script = f"""
set -euo pipefail
cd {shlex.quote(REMOTE_REPO)}
git fetch origin --prune
git cat-file -e {shlex.quote(commit + '^{commit}')}
if [[ ! -d {shlex.quote(worktree)} ]]; then
  git worktree add --detach {shlex.quote(worktree)} {shlex.quote(commit)}
fi
cd {shlex.quote(worktree)}
test "$(git rev-parse HEAD)" = {shlex.quote(commit)}
{shlex.quote(REMOTE_PYTHON)} -m compileall -q experiment7/integration experiment7/reference_impl
printf '%s\n' {shlex.quote(worktree)}
"""
    return ssh(host, script).stdout.strip().splitlines()[-1]


def ensure_local_context(repo: Path) -> tuple[str, str]:
    if not (repo / ".git").exists():
        raise FileNotFoundError(f"not a Git repository: {repo}")
    full_name = "LZhangGJ/pocketmon"
    remote = git_value(repo, "remote", "get-url", "origin")
    if "LZhangGJ/pocketmon" not in remote:
        raise RuntimeError(f"unexpected origin: {remote}")
    branch = git_value(repo, "branch", "--show-current")
    commit = git_value(repo, "rev-parse", "HEAD")
    if git_value(repo, "status", "--porcelain"):
        raise RuntimeError("Windows working tree is dirty; commit and push before remote execution")
    checked(["git", "push", "origin", branch], cwd=repo)
    return branch, commit


def write_local(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def command_probe(args: argparse.Namespace) -> None:
    gpus = probe(args.servers)
    payload = {"schema_version": 1, "gpus": [asdict(gpu) for gpu in gpus]}
    write_local(args.output, payload)
    print(json.dumps(payload, indent=2))


def command_bootstrap(args: argparse.Namespace) -> None:
    _, commit = ensure_local_context(args.repo)
    rows = []
    for host in args.servers:
        rows.append({"host": host, "worktree": bootstrap_host(host, commit), "commit": commit})
    write_local(args.output, {"schema_version": 1, "workers": rows})
    print(json.dumps(rows, indent=2))


def command_prepare(args: argparse.Namespace) -> None:
    _, commit = ensure_local_context(args.repo)
    worktree = bootstrap_host(args.coordinator, commit)
    script = f"""
set -euo pipefail
export PYTHON={shlex.quote(REMOTE_PYTHON)}
bash {shlex.quote(worktree + '/experiment7/integration/remote_prepare_data.sh')} \
  {shlex.quote(worktree)} \
  {shlex.quote(REPLAY_DIR)} \
  {shlex.quote(ANALYSIS_DIR)} \
  {shlex.quote(args.run_root)} \
  --desired-decks {args.desired_decks} \
  --minimum-decks {args.minimum_decks} \
  --min-actor-episodes {args.min_actor_episodes} \
  --min-policy-decisions {args.min_policy_decisions}
"""
    result = ssh(args.coordinator, script, capture=False)
    if result.returncode:
        raise RuntimeError("remote data preparation failed")
    print(f"prepared: {args.run_root}/data/datasets/dataset_manifest.json")


def command_train(args: argparse.Namespace) -> None:
    _, commit = ensure_local_context(args.repo)
    gpus = [
        gpu
        for gpu in probe(args.servers)
        if gpu.utilization_percent <= args.max_utilization
        and gpu.memory_free_mb >= args.min_free_memory_mb
    ]
    gpus.sort(key=lambda gpu: (gpu.memory_free_mb, -gpu.utilization_percent), reverse=True)
    if len(gpus) < len(args.seeds):
        raise RuntimeError(
            f"need {len(args.seeds)} idle GPUs, found {len(gpus)}; do not oversubscribe existing jobs"
        )
    jobs = []
    dataset_manifest = f"{args.run_root}/data/datasets/dataset_manifest.json"
    for seed, gpu in zip(args.seeds, gpus):
        worktree = bootstrap_host(gpu.host, commit)
        run_dir = f"{args.run_root}/training/seed-{seed}"
        script = f"""
set -euo pipefail
mkdir -p {shlex.quote(run_dir)}
nohup env PYTHON={shlex.quote(REMOTE_PYTHON)} \
  bash {shlex.quote(worktree + '/experiment7/integration/remote_train_job.sh')} \
  {shlex.quote(worktree)} {shlex.quote(dataset_manifest)} {shlex.quote(run_dir)} {seed} {gpu.index} \
  > {shlex.quote(run_dir + '/launcher.log')} 2>&1 < /dev/null &
echo $!
"""
        pid = int(ssh(gpu.host, script).stdout.strip().splitlines()[-1])
        jobs.append(
            {
                "seed": seed,
                "host": gpu.host,
                "gpu": gpu.index,
                "gpu_name": gpu.name,
                "pid": pid,
                "commit": commit,
                "worktree": worktree,
                "run_dir": run_dir,
                "receipt": f"{run_dir}/job_receipt.json",
            }
        )
    write_local(args.output, {"schema_version": 1, "jobs": jobs})
    print(json.dumps(jobs, indent=2))


def command_status(args: argparse.Namespace) -> None:
    payload = json.loads(args.jobs.read_text(encoding="utf-8"))
    rows = []
    for job in payload["jobs"]:
        script = f"""
set -euo pipefail
if [[ -f {shlex.quote(job['receipt'])} ]]; then
  cat {shlex.quote(job['receipt'])}
else
  printf '{{"status":"receipt_missing","pid":%s}}\n' {int(job['pid'])}
fi
"""
        result = ssh(job["host"], script)
        receipt = json.loads(result.stdout.strip().splitlines()[-1])
        rows.append({**job, "remote": receipt})
    print(json.dumps(rows, indent=2, ensure_ascii=False))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Windows control-plane for Experiment 7 jobs on doraemon Linux servers"
    )
    result.add_argument("--repo", type=Path, default=Path.cwd())
    result.add_argument("--servers", nargs="+", default=DEFAULT_SERVERS)
    result.add_argument("--run-root", default=RUN_ROOT)
    sub = result.add_subparsers(dest="command", required=True)

    probe_parser = sub.add_parser("probe")
    probe_parser.add_argument("--output", type=Path, default=Path("experiment7_gpu_inventory.json"))
    probe_parser.set_defaults(func=command_probe)

    bootstrap = sub.add_parser("bootstrap")
    bootstrap.add_argument("--output", type=Path, default=Path("experiment7_workers.json"))
    bootstrap.set_defaults(func=command_bootstrap)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--coordinator", default="doraemon02")
    prepare.add_argument("--desired-decks", type=int, default=6)
    prepare.add_argument("--minimum-decks", type=int, default=4)
    prepare.add_argument("--min-actor-episodes", type=int, default=40)
    prepare.add_argument("--min-policy-decisions", type=int, default=800)
    prepare.set_defaults(func=command_prepare)

    train = sub.add_parser("train")
    train.add_argument("--seeds", nargs="+", type=int, default=[20260808, 20260809, 20260810])
    train.add_argument("--max-utilization", type=int, default=20)
    train.add_argument("--min-free-memory-mb", type=int, default=12000)
    train.add_argument("--output", type=Path, default=Path("experiment7_jobs.json"))
    train.set_defaults(func=command_train)

    status = sub.add_parser("status")
    status.add_argument("--jobs", type=Path, default=Path("experiment7_jobs.json"))
    status.set_defaults(func=command_status)
    return result


def main() -> None:
    args = parser().parse_args()
    args.repo = args.repo.resolve()
    args.func(args)


if __name__ == "__main__":
    main()
