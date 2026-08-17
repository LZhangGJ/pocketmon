from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(
    "/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/"
    "0812-d14-ram-npz-fast-20260813/replacement-screening"
)
PYTHON = "/homes/lzhang/mypath/new/envs/trans/bin/python"
CONTROLLER = "/homes/lzhang/run_bc_replacement_screening_20260813.py"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def wait_complete(profile: str) -> None:
    marker = ROOT / "direct-new-old" / profile / "complete.json"
    while not marker.is_file() or load(marker).get("status") != "complete":
        time.sleep(20)


def run_stage(profile: str, portable: Path) -> None:
    command = [
        PYTHON,
        "-c",
        (
            "import importlib.util;"
            f"s=importlib.util.spec_from_file_location('screening',{CONTROLLER!r});"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            f"m.run_stage({(profile + '-frozen40-round2')!r},{str(portable)!r} and "
            f"m.Path({str(portable)!r}),40,45)"
        ),
    ]
    log = ROOT / "logs" / f"{profile}-frozen40-round2-watcher.log"
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    profiles = {
        "current_bc": Path(
            "/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/"
            "final_7d/downstream-e3cb2936afb5/seed-20260812/universal_bc.npz"
        ),
        "large_256x6": ROOT / "large_256x6/universal_bc.npz",
        "standard_1m": ROOT / "standard_1m/universal_bc.npz",
    }
    wait_complete("large_256x6")
    wait_complete("standard_1m")
    # The baseline and priority large candidate are the critical comparison.
    # Each shard is independently guarded, so excess parallel work waits rather
    # than bypassing CPU/I/O boundaries.
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_stage, profile, profiles[profile])
            for profile in ("current_bc", "large_256x6")
        ]
        for future in futures:
            future.result()
    run_stage("standard_1m", profiles["standard_1m"])
    (ROOT / "FROZEN40_ROUND2_COMPLETE.txt").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
