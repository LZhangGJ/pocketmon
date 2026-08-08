from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from common import Experiment7Error, read_json, utc_now, write_csv, write_json


DEFAULT_HOSTS = ["doraemon02", "doraemon03", "doraemon15", "doraemon16", "doraemon19", "doraemon20"]


@dataclass(frozen=True)
class GPU:
    host: str
    index: int
    name: str
    total_mib: int
    used_mib: int
    free_mib: int
    utilization: int

    @property
    def idle_score(self) -> tuple[int, int, int]:
        return (self.utilization, -self.free_mib, self.index)


def ssh(host: str, command: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", host, command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def inventory(hosts: list[str], output: Path, minimum_free_mib: int, maximum_utilization: int) -> dict[str, Any]:
    rows = []
    errors = []
    query = (
        "set -e; hostname; "
        "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu "
        "--format=csv, noheader,nounits"
    )
    for host in hosts:
        try:
            result = ssh(host, query)
        except subprocess.TimeotExpired:
            errors.append({"host": host, "error": "ssh_timeout"})
            continue
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if result.returncode != 0 or len(lines) < 2:
            errors.append({"host": host, "error": result.stdout[-1000:], "returnCode": result.returncode})
            continue
        remote_name = lines[0]
        for line in lines[1:]:
            values = [value.strip() for value in line.split(",")]
            if len(values) != 6:
                errors.append({"host": host, "error": f"unexpected nvidia-smi row: {line}"})
                continue
            index, name, total, used, free, utilization_value = values
            row = {
                "host": host,
                "remoteHost": remote_name,
                "gpuIndex": int(index),
                "name": name,
                "totalMiB": int(total),
                "usedMiB": int(used),
                "freeMiB": int(free),
                "utilizationPercent": int(utilization_value),
            }
            row["eligible"] = row["freeMiB"] >= minimum_free_mib and row["utilizationPercent"] <= maximum_utilization
            rows.append(row)
    payload = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "minimumFreeMiB": minimum_free_mib,
        "maximumUtilizationPercent": maximum_utilization,
        "gpus": rows,
        "errors": errors,
    }
    write_json(output, payload)
    write_csv(output.with_suffix(".csv"), rows)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def eligible_gpus(inventory_payload: dict[str, Any]) -> list[GPU]:
    values = [
        GPU(host=row["host"], index=int(row["gpuIndex"]), name=row["name"], total_mib=int(row["totalMiB"]), used_mib=int(row["usedMiB"]), free_mib=int(row["freeMiB"]), utilization=int(row["utilizationPercent"]))
        for row in inventory_payload["gpus"]
        if row.get("eligible")
    ]
    values.sort(key=lambda gpu: gpu.idle_score)
    return values


def make_training_plan(
    inventory_path: Path,
    output: Path,
    worktree: PurePosixPath,
    commit: str,
    python: str,
    sources: PurePosixPath,
    run_root: PurePosixPath,
    stage: str,
    pretrain_checkpoint: PurePosixPath | None,
    seeds: list[int],
) -> dict[str, Any]:
    gpus = eligible_gpus(read_json(inventory_path))
    required = 1 if stage in {"pretrain", "smoke"} else len(seeds)
    if len(gpus) < required:
        raise Experiment7Error(f"stage {stage} needs {required} idle GPUs, found {len(gpus)}")
    integration = worktree / "experiment7" / "integration"
    jobs = []
    if stage == "pretrain":
        assignments = [("pretrain", 20260808, gpus[0])]
    elif stage == "smoke":
        assignments = [("smoke", 20260808, gpus[0])]
    elif stage == "finetune":
        if pretrain_checkpoint is None:
            raise ValueError("finetune requires pretrain_checkpoint")
        assignments = [(f"finetune-seed-{seed}", seed, gpus[index]) for index, seed in enumerate(seeds)]
    else:
        raise ValueError(stage)
    for name, seed, gpu in assignments:
        run_dir = run_root / name
        if stage == "pretrain":
            command = [python, str(integration / "train_driver.py"), "pretrain", "--sources", str(sources), "--output-dir", str(run_dir), "--seed", str(seed)]
        elif stage == "smoke":
            command = [python, str(integration / "train_driver.py"), "smoke", "--sources", str(sources), "--output-dir", str(run_dir), "--seed", str(seed)]
        else:
            command = [python, str(integration / "train_driver.py"), "finetune", "--sources", str(sources), "--pretrain-checkpoint", str(pretrain_checkpoint), "--output-dir", str(run_dir), "--seed", str(seed)]
        job_path = run_root / "jobs" / f"{name}.json"
        jobs.append({"jobId": name, "stage": stage, "host": gpu.host, "gpuIndex": gpu.index, "commit": commit, "cwd": str(worktree), "command": command, "runDir": str(run_dir), "logPath": str(run_dir / "train.log"), "receiptPath": str(run_dir / "job_receipt.json"), "jobPath": str(job_path), "env": {"PYTHON": python}})
    payload = {"schemaVersion": 1, "createdAt": utc_now(), "stage": stage, "commit": commit, "worktree": str(worktree), "sources": str(sources), "jobs": jobs}
    write_json(output, payload)
    write_csv(output.with_suffix(".csv"), [{"job_id": row["jobId"], "stage": row["stage"], "host": row["host"], "gpu_index": row["gpuIndex"], "run_dir": row["runDir"], "receipt": row["receipuA…Ñ ‰t°€‰½µµ…¹ˆè©Í½¸¹‘ÕµÁÌ¡É½Ýl‰½µµ…¹‰t°•¹ÍÕÉ•}…Í¥¤õ…±Í”¥ô™½ÈÉ½Ü¥¸©½‰Ít¤(€€€ÁÉ¥¹Ð¡©Í½¸¹‘ÕµÁÌ¡Á…å±½…°•¹ÍÕÉ•}…Í¥¤õ…±Í”¤°™±ÕÍ õQÉÕ”¤(€€€É•ÑÕÉ¸Á…å±½…(()‘•˜}ÕÁ±½…‘}©Í½¸¡¡½ÍÐèÍÑÈ°É•µ½Ñ•}Á…Ñ èÍÑÈ°Á…å±½…è‘¥ÑmÍÑÈ°¹åt¤€´ø9½¹”è(€€€•¹½‘•€ô‰…Í”ØÐ¹ˆØÑ•¹½‘”¡©Í½¸¹‘ÕµÁÌ¡Á…å±½…°•¹ÍÕÉ•}…Í¥¤õ…±Í”¤¹•¹½‘” ‰ÕÑ˜´àˆ¤¤¹‘•½‘” ‰…Í¥¤ˆ¤(€€€½µµ…¹€ô˜‰µ­‘¥È€µÀíÍ¡•±±}ÅÕ½Ñ”¡ÍÑÈ¡AÕÉ•A½Í¥áA…Ñ ¡É•µ½Ñ•}Á…Ñ ¤¹Á…É•¹Ð¤¥ôìÁÉ¥¹Ñ˜€•ÌíÍ¡•±±}ÅÕ½Ñ”¡•¹½‘•¥ôð‰…Í”ØÐ€µ€øíÍ¡•±±}ÅÕ½Ñ”¡É•µ½Ñ•}Á…Ñ ¥ôˆ(€€€É•ÍÕ±Ð€ôÍÍ ¡¡½ÍÐ°½µµ…¹¤(€€€¥˜É•ÍÕ±Ð¹É•ÑÕÉ¹½‘”€„ô€Àè(€€€€€€€É…¥Í”áÁ•É¥µ•¹ÐÝÉÉ½È¡˜‰™…¥±•Ñ¼ÕÁ±½…©½ˆÑ¼í¡½ÍÑôèíÉ•ÍÕ±Ð¹ÍÑ‘½ÕÑôˆ¤(()‘•˜Í¡•±±}ÅÕ½Ñ”¡Ù…±Õ”èÍÑÈ¤€´øÍÑÈè(€€€¥µÁ½ÉÐÍ¡±•à((€€€É•ÑÕÉ¸Í¡±•à¹ÅÕ½Ñ”¡Ù…±Õ”¤(()‘•˜±…Õ¹¡}Á±…¸¡Á±…¹}Á…Ñ èA…Ñ °É•µ½Ñ•}ÁåÑ¡½¸èÍÑÈ°Ý½É­•É}Á…Ñ èAÕÉ•A½Í¥áA…Ñ ¤€´ø‘¥ÑmÍÑÈ°¹åtè(€€€Á±…¸€ôÉ•…‘}©Í½¸¡Á±…¹}Á…Ñ ¤(€€€±…Õ¹¡•€ômt(€€€™½È©½ˆ¥¸Á±…¹l‰©½‰Ì‰tè(€€€€€€€}ÕÁ±½…‘}©Í½¸¡©½‰l‰¡½ÍÐ‰t°©½‰l‰©½‰A…Ñ ‰t°©½ˆ¤(€€€€€€€½µµ…¹€ô˜‰íÍ¡•±±}ÅÕ½Ñ”¡É•µ½Ñ•}ÁåÑ¡½¸¥ôíÍ¡•±±}ÅÕ½Ñ”¡ÍÑÈ¡Ý½É­•É}Á…Ñ ¤¥ô±…Õ¹ €´µ©½ˆíÍ¡•±±}ÅÕ½Ñ”¡©½‰l©½‰A…Ñ t¥ôˆ(€€€€€€€É•ÍÕ±Ð€ôÍÍ ¡©½‰l‰¡½ÍÐ‰t°½µµ…¹¤(€€€€€€€¥˜É•ÍÕ±Ð¹É•ÑÕÉ¹½‘”€„ô€Àè(€€€€€€€€€€€É…¥Í”áÁ•É¥µ•¹ÐÝÉÉ½È¡˜‰±…Õ¹ ™…¥±•½¸í©½‰l¡½ÍÐuôèíÉ•ÍÕ±Ð¹ÍÑ‘½ÕÑôˆ¤(€€€€€€€ÑÉäè(€€€€€€€€€€€É••¥ÁÐ€ô©Í½¸¹±½…‘Ì¡É•ÍÕ±Ð¹ÍÑ‘½ÕÐ¹ÍÑÉ¥À ¤¹ÍÁ±¥Ñ±¥¹•Ì ¥l´Åt¤(€€€€€€€•á•ÁÐ€¡©Í½¸¹)M=9•½‘•ÉÉ½È°%¹‘•áÉÉ½È¤…Ì•áŒè(€€€€€€€€€€€É…¥Í”áÁ•É¥µ•¹ÐÝÉÉ½È¡˜‰¥¹Ù…±¥±…Õ¹ É••¥ÁÐ™É½´í©½‰l¡½ÍÐuôèíÉ•ÍÕ±Ð¹ÍÑ‘½ÕÑôˆ¤™É½´•áŒ(€€€€€€€±…Õ¹¡•¹…ÁÁ•¹¡É••¥ÁÐ¤(€€€Á…å±½…€ôì‰Í¡•µ…Y•ÉÍ¥½¸ˆè€Ä°€‰É•…Ñ•‘ÐˆèÕÑ}¹½Ü ¤°€‰Á±…¸ˆèÍÑÈ¡Á±…¹}Á…Ñ ¹É•Í½±Ù” ¤¤°€‰±…Õ¹¡•ˆè±…Õ¹¡•‘ô(€€€½ÕÑÁÕÐ€ôÁ±…¹}Á…Ñ ¹Ý¥Ñ¡}¹…µ”¡Á±…¹}Á…Ñ ¹ÍÑ•´€¬€‰}±…Õ¹¡}É••¥ÁÐ¹©Í½¸ˆ¤(€€€ÝÉ¥Ñ•}©Í½¸¡½ÕÑÁÕÐ°Á…å±½…¤(€€€ÁÉ¥¹Ð¡©Í½¸¹‘ÕµÁÌ¡Á…å±½…°•¹ÍÕÉ•}…Í¥¤õ…±Í”¤°™±ÕÍ õQÉÕ”¤(€€€É•ÑÕÉ¸Á…å±½…(()‘•˜ÍÑ…ÑÕÍ}Á±…¸¡Á±…¹}Á…Ñ èA…Ñ °É•µ½Ñ•}ÁåÑ¡½¸èÍÑÈ°Ý½É­•É}Á…Ñ èAÕÉ•A½Í¥áA…Ñ ¤€´ø‘¥ÑmÍÑÈ°¹åtè(€€€Á±…¸€ôÉ•…‘}©Í½¸¡Á±…¹}Á…Ñ ¤(€€€ÍÑ…ÑÕÍ•Ì€ômt(€€€™½È©½ˆ¥¸Á±…¹l‰©½‰Ì‰tè(€€€€€€€½µµ…¹€ô˜‰íÍ¡•±±}ÅÕ½Ñ”¡É•µ½Ñ•}ÁåÑ¡½¸¥ôíÍ¡•±±}ÅÕ½Ñ”¡ÍÑÈ¡Ý½É­•É}Á…Ñ ¤¥ôÍÑ…ÑÕÌ€´µ©½ˆíÍ¡•±±}ÅÕ½Ñ”¡©½‰l©½‰A…Ñ t¥ôˆ(€€€€€€€É•ÍÕ±Ð€ôÍÍ ¡©½‰l‰¡½ÍÐ‰t°½µµ…¹¤(€€€€€€€¥˜É•ÍÕ±Ð¹É•ÑÕÉ¹½‘”€„ô€Àè(€€€€€€€€€€€ÍÑ…ÑÕÍ•Ì¹…ÁÁ•¹¡ì‰©½‰%ˆè©½‰l‰©½‰%‰t°€‰¡½ÍÐˆè©½‰l‰¡½ÍÐ‰t°€‰ÍÑ…ÑÕÌˆè€‰ÍÍ¡}•ÉÉ½Èˆ°€‰½ÕÑÁÕÐˆèÉ•ÍÕ±Ð¹ÍÑ‘½ÕÑl´ÄÀÀÀéuô¤(€€€€€€€€€€€½¹Ñ¥¹Õ”(€€€€€€€ÍÑ…ÑÕÍ•Ì¹…ÁÁ•¹¡©Í½¸¹±½…‘Ì¡É•ÍÕ±Ð¹ÍÑ‘½ÕÐ¹ÍÑÉ¥À ¤¹ÍÁ±¥Ñ±¥¹•Ì ¥l´Åt¤¤(€€€Á…å±½…€ôì‰Í¡•µ…Y•ÉÍ¥½¸ˆè€Ä°€‰É•…Ñ•‘ÐˆèÕÑ}¹½Ü ¤°€‰Á±…¸ˆèÍÑÈ¡Á±…¹}Á…Ñ ¹É•Í½±Ù” ¤¤°€‰ÍÑ…ÑÕÍ•ÌˆèÍÑ…ÑÕÍ•Íô(€€€ÁÉ¥¹Ð¡©Í½¸¹‘ÕµÁÌ¡Á…å±½…°•¹ÍÕÉ•}…Í¥¤õ…±Í”¤°™±ÕÍ õQÉÕ”¤(€€€É•ÑÕÉ¸Á…å±½…(()‘•˜µ…¥¸ ¤€´ø9½¹”è(€€€Á…ÉÍ•È€ô…ÉÁ…ÉÍ”¹ÉÕµ•¹ÑA…ÉÍ•È¡‘•ÍÉ¥ÁÑ¥½¸ô‰]¥¹‘½ÝÌ½1¥¹ÕàMM Í¡•‘Õ±•È™½ÈáÁ•É¥µ•¹Ð€ÜAT©½‰Ìˆ¤(€€€ÍÕˆ€ôÁ…ÉÍ•È¹…‘‘}ÍÕ‰Á…ÉÍ•ÉÌ¡‘•ÍÐô‰½µµ…¹ˆ°É•ÅÕ¥É•õQÉÕ”¤((€€€¥¹Ø€ôÍÕˆ¹…‘‘}Á…ÉÍ•È ‰¥¹Ù•¹Ñ½Éäˆ¤(€€€¥¹Ø¹…‘‘}…ÉÕµ•¹Ð ˆ´µ¡½ÍÑÌˆ°¹…ÉÌôˆ¨ˆ°‘•™…Õ±ÐõU1Q}!=MQL¤(€€€¥¹Ø¹…‘‘}…ÉÕµ•¹Ð ˆ´µ½ÕÑÁÕÐˆ°ÑåÁ”õA…Ñ °É•ÅÕ¥É•õQÉÕ”¤(€€€¥¹Ø¹…‘‘}…ÉÕµ•¹Ð ˆ´µµ¥¹¥µÕ´µ™É•”µµ¥ˆˆ°ÑåÁ”õ¥¹Ð°‘•™…Õ±ÐôÄÉ|ÀÀÀ¤(€€€¥¹Ø¹…‘‘}…ÉÕµ•¹Ð ˆ´µµ…á¥µÕ´µÕÑ¥±¥é…Ñ¥½¸ˆ°ÑåÁ”õ¥¹Ð°‘•™…Õ±ÐôÈÀ¤((€€€Á±…¸€ôÍÕˆ¹…‘‘}Á…ÉÍ•È ‰µ…­”µÑÉ…¥¹¥¹œµÁ±…¸ˆ¤(€€€Á±…¸¹…‘‘}…ÉÕµ•¹Ð ˆ´µ¥¹Ù•¹Ñ½Éäˆ°ÑåÁ”õA…Ñ °É•ÅÕ¥É•õQÉÕ”¤(€€€Á±…¸¹…‘‘}…ÉÕµ•¹Ð ˆ´µ½ÕÑÁÕÐˆ°ÑåÁ”õA…Ñ °É•ÅÕ¥É•õQÉÕ”¤(€€€Á±…¸¹…‘‘}…ÉÕµ•¹Ð ˆ´µÝ½É­ÑÉ•”ˆ°É•ÅÕ¥É•õQÉÕ”¤(€€€Á±…¸¹…‘‘}…ÉÕµ•¹Ð ˆ´µ½µµ¥Ðˆ°É•ÅÕ¥É•õQÉÕ”¤(€€€Á±…¸¹…‘‘}…ÉÕµ•¹Ð ˆ´µÁåÑ¡½¸ˆ°É•ÅÕ¥É•õQÉÕ”¤(€€€Á±…¸¹…‘‘}…ÉÕµ•¹Ð ˆ´µÍ½ÕÉ•Ìˆ°É•ÅÕ¥É•õQÉÕ”¤(€€€Á±…¸¹…‘‘}…ÉÕµ•¹Ð ˆ´µÉÕ¸µÉ½½Ðˆ°É•ÅÕ¥É•õQÉÕ”¤(€€€Á±…¸¹…‘‘}…ÉÕµ•¹Ð ˆ´µÍÑ…”ˆ°¡½¥•Ìô ‰Íµ½­”ˆ°€‰ÁÉ•ÑÉ…¥¸ˆ°€‰™¥¹•ÑÕ¹”ˆ¤°É•ÅÕ¥É•õQÉÕ”¤(€€€Á±…¸¹…‘‘}…ÉÕµ•¹Ð ˆ´µÁÉ•ÑÉ…¥¸µ¡•­Á½¥¹Ðˆ¤(€€€Á±…¸¹…‘‘}…ÉÕµ•¹Ð ˆ´µÍ••‘Ìˆ°ÑåÁ”õ¥¹Ð°¹…ÉÌôˆ¨ˆ°‘•™…Õ±ÐõlÈÀÈØÀàÀà°€ÈÀÈØÀàÀä°€ÈÀÈØÀàÄÁt¤((€€€±…Õ¹ €ôÍÕˆ¹…‘‘}Á…ÉÍ•È ‰±…Õ¹ ˆ¤(€€€±…Õ¹ ¹…‘‘}…ÉÕµ•¹Ð ˆ´µÁ±…¸ˆ°ÑåÁ”õA…Ñ °É•ÅÕ¥É•õQÉÕ”¤(€€€±…Õ¹ ¹…‘‘}…ÉÕµ•¹Ð ˆ´µÉ•µ½Ñ”µÁåÑ¡½¸ˆ°É•ÅÕ¥É•õQÉÕ”¤(€€€±…Õ¹ ¹…‘‘}…ÉÕµ•¹Ð ˆ´µÝ½É­•Èˆ°É•ÅÕ¥É•õQÉÕ”¤((€€€ÍÑ…ÑÕÌ€ôÍÕˆ¹…‘‘}Á…ÉÍ•È ‰ÍÑ…ÑÕÌˆ¤(€€€ÍÑ…ÑÕÌ¹…‘‘}…ÉÕµ•¹Ð ˆ´µÁ±…¸ˆ°ÑåÁ”õA…Ñ °É•ÅÕ¥É•õQÉÕ”¤(€€€ÍÑ…ÑÕÌ¹…‘‘}…ÉÕµ•¹Ð ˆ´µÉ•µ½Ñ”µÁåÑ¡½¸ˆ°É•ÅÕ¥É•õQÉÕ”¤(€€€ÍÑ…ÑÕÌ¹…‘‘}…ÉÕµ•¹Ð ˆ´µÝ½É­•Èˆ°É•ÅÕ¥É•õQÉÕ”¤((€€€…ÉÌ€ôÁ…ÉÍ•È¹Á…ÉÍ•}…ÉÌ ¤(€€€¥˜…ÉÌ¹½µµ…¹€ôô€‰¥¹Ù•¹Ñ½Éäˆè(€€€€€€€¥¹Ù•¹Ñ½Éä¡…ÉÌ¹¡½ÍÑÌ°…ÉÌ¹½ÕÑÁÕÐ¹É•Í½±Ù” ¤°…ÉÌ¹µ¥¹¥µÕµ}™É••}µ¥ˆ°…ÉÌ¹µ…á¥µÕµ}ÕÑ¥±¥é…Ñ¥½¸¤(€€€•±¥˜…ÉÌ¹½µµ…¹€ôô€‰µ…­”µÑÉ…¥¹¥¹œµÁ±…¸ˆè(€€€€€€€µ…­•}ÑÉ…¥¹¥¹}Á±…¸ (€€€€€€€€€€€…ÉÌ¹¥¹Ù•¹Ñ½Éä¹É•Í½±Ù” ¤°(€€€€€€€€€€€…ÉÌ¹½ÕÑÁÕÐ¹É•Í½±Ù” ¤°(€€€€€€€€€€€AÕÉ•A½Í¥áA…Ñ ¡…ÉÌ¹Ý½É­ÑÉ•”¤°(€€€€€€€€€€€…ÉÌ¹½µµ¥Ð°(€€€€€€€€€€€…ÉÌ¹ÁåÑ¡½¸°(€€€€€€€€€€€AÕÉ•A½Í¥áA…Ñ ¡…ÉÌ¹Í½ÕÉ•Ì¤°(€€€€€€€€€€€AÕÉ•A½Í¥áA…Ñ ¡…ÉÌ¹ÉÕ¹}É½½Ð¤°(€€€€€€€€€€€…ÉÌ¹ÍÑ…”°(€€€€€€€€€€€AÕÉ•A½Í¥áA…Ñ ¡…ÉÌ¹ÁÉ•ÑÉ…¥¹}¡•­Á½¥¹Ð¤¥˜…ÉÌ¹ÁÉ•ÑÉ…¥¹}¡•­Á½¥¹Ð•±Í”9½¹”°(€€€€€€€€€€€…ÉÌ¹Í••‘Ì°(€€€€€€€€¤(€€€•±¥˜…ÉÌ¹½µµ…¹€ôô€‰±…Õ¹ ˆè(€€€€€€€±…Õ¹¡}Á±…¸¡…ÉÌ¹Á±…¸¹É•Í½±Ù” ¤°…ÉÌ¹É•µ½Ñ•}ÁåÑ¡½¸°AÕÉ•A½Í¥áA…Ñ ¡…ÉÌ¹Ý½É­•È¤¤(€€€•±Í”è(€€€€€€€ÍÑ…ÑÕÍ}Á±…¸¡…ÉÌ¹Á±…¸¹É•Í½±Ù” ¤°…ÉÌ¹É•µ½Ñ•}ÁåÑ¡½¸°AÕÉ•A½Í¥áA…Ñ ¡…ÉÌ¹Ý½É­•È¤¤(()¥˜}}¹…µ•}|€ôô€‰}}µ…¥¹}|ˆè(€€€µ…¥¸ ¤(