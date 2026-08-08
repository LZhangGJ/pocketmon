# Codex start here — Windows controller, Linux execution

Do not guess the repository, branch, target Agent or execution host.

```text
repository:          LZhangGJ/pocketmon
remote:              https://github.com/LZhangGJ/pocketmon.git
source branch:       agent/experiment7-multideck-challengers-20260808
working branch:      codex/experiment7-multideck-challengers-20260808
main prompt:         experiment7/CODEX_TRAINING_PROMPT.md
machine plan:        experiment7/MULTIDECK_CHALLENGER_PLAN.json
bootstrap script:    experiment7/WINDOWS_REMOTE_BOOTSTRAP.ps1
archive status:      experiment7/SOURCE_ARCHIVE_STATUS.json

Codex host:          Windows workstation
Windows role:        edit code, Git, SSH orchestration, summary collection
Linux role:          replay processing, caches, GPU training, export, Arena
Linux repository:   /homes/lzhang/pocketmon
Linux Python:        /homes/lzhang/mypath/new/envs/trans/bin/python
ladder analysis:     /homes/lzhang/pocketmon/analysis/outputs/top_ladder_2026_08_07_20260808
replays:             /homes/lzhang/pocketmon/data/raw/replays/2026-08-06
servers:             doraemon02 doraemon03 doraemon15 doraemon16 doraemon19 doraemon20
primary target:      agents/lucario_rule
primary target deck: agents/lucario_rule/deck.csv
```

## Required source ZIP

Use the original verified ZIP supplied by the teammate/user:

```text
filename: experiment7_code_for_gpt_2026-08-08.zip
bytes:    94038
sha256:   9c0d24067eacee8abc38223dba28d893e5d1e4e9b75204a9ce92a03093558229
```

The GitHub binary copy is known to be truncated. Do not use it. Put the verified ZIP in the Windows Downloads directory or pass its path explicitly to the bootstrap script; the script copies it to a non-Git Linux path and verifies it again.

## Recommended Windows PowerShell 7 bootstrap

```powershell
$ErrorActionPreference = 'Stop'
$RepoUrl = 'https://github.com/LZhangGJ/pocketmon.git'
$SourceBranch = 'agent/experiment7-multideck-challengers-20260808'
$WorkBranch = 'codex/experiment7-multideck-challengers-20260808'
$LocalRepo = Join-Path $HOME 'source\pocketmon-experiment7'

if (-not (Test-Path (Join-Path $LocalRepo '.git'))) {
    New-Item -ItemType Directory -Force -Path (Split-Path $LocalRepo) | Out-Null
    git clone $RepoUrl $LocalRepo
}
Set-Location $LocalRepo
git remote set-url origin $RepoUrl
git fetch origin --prune
git switch --force-create $WorkBranch "origin/$SourceBranch"

.\experiment7\WINDOWS_REMOTE_BOOTSTRAP.ps1 `
  -LocalRepo $LocalRepo `
  -ArchivePath "$HOME\Downloads\experiment7_code_for_gpt_2026-08-08.zip"

Get-Content experiment7/CODEX_TRAINING_PROMPT.md
```

The script probes all configured servers with `ssh`, inventories GPUs, validates and transfers the ZIP, pushes the work branch, creates a remote Linux worktree from the immutable commit, unpacks the source and runs the first remote compile check.

Then execute the full prompt. Do not merely summarize it.

## Manual SSH inventory fallback

```powershell
foreach ($Server in @('doraemon02','doraemon03','doraemon15','doraemon16','doraemon19','doraemon20')) {
    Write-Host "===== $Server ====="
    ssh $Server "hostname; nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits; test -d /homes/lzhang/pocketmon && echo REPO_OK; test -x /homes/lzhang/mypath/new/envs/trans/bin/python && echo PYTHON_OK"
}
```

## Non-negotiable distinction

Commands containing any of the following belong on the remote Linux servers and must be invoked through `ssh`:

```text
/homes/
nvidia-smi
CUDA_VISIBLE_DEVICES
kaggle-environments official engine / cg
replay or deck-map processing
training checkpoints
Arena matches
Linux CPU latency measurement
```

Windows must not fabricate Linux results and must not claim a job started until an SSH command returns a concrete PID/job receipt.
