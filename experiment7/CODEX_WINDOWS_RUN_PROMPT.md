# Windows Codex run prompt — canonical source branch

Use this exact repository and branch. Do not search for similarly named branches.

```text
repository:       https://github.com/LZhangGJ/pocketmon.git
source branch:    agent/experiment7-training-ready-20260809
work branch:      codex/experiment7-multideck-run-20260809
Windows role:     Git, code edits, SSH orchestration, status collection
Linux role:       replay conversion, cache building, CUDA training, export, Arena
Linux repository: /homes/lzhang/pocketmon
Linux Python:     /homes/lzhang/mypath/new/envs/trans/bin/python
```

The teammate Experiment 7 code is already materialized as ordinary source files under:

```text
experiment7/reference/
```

This branch deliberately uses `experiment7/reference/`; do not require or create
`experiment7/vendor/reference/`. Do not look for, upload, unpack, or execute the old ZIP.

The Pocketmon integration code is already under:

```text
experiment7/integration/
```

Before starting remote work, verify these paths exist:

```text
experiment7/reference/training/deck_identity_model.py
experiment7/reference/training/train_multideck_identity.py
experiment7/reference/data_pipeline/build_token_cache.py
experiment7/reference/data_pipeline/build_sequence_cache.py
experiment7/reference/data_pipeline/build_deck_identity_cache.py
experiment7/integration/build_replay_catalog.py
experiment7/integration/build_from_pocketmon_replays.py
experiment7/integration/select_initial_decks.py
experiment7/integration/prepare_training_data.py
experiment7/integration/train_driver.py
experiment7/integration/multi_gpu_scheduler.py
experiment7/integration/remote_worker.py
experiment7/integration/export_and_package.py
experiment7/integration/arena.py
experiment7/configs/multideck_default.json
tests/test_experiment7_integration.py
```

## Windows setup

Run in PowerShell 7:

```powershell
$ErrorActionPreference = 'Stop'
$RepoUrl = 'https://github.com/LZhangGJ/pocketmon.git'
$SourceBranch = 'agent/experiment7-training-ready-20260809'
$WorkBranch = 'codex/experiment7-multideck-run-20260809'
$LocalRepo = Join-Path $HOME 'source\pocketmon-experiment7'

if (-not (Test-Path (Join-Path $LocalRepo '.git'))) {
    New-Item -ItemType Directory -Force -Path (Split-Path $LocalRepo) | Out-Null
    git clone $RepoUrl $LocalRepo
}
Set-Location $LocalRepo
git remote set-url origin $RepoUrl
git fetch origin --prune
git switch --force-create $WorkBranch "origin/$SourceBranch"
$SourceCommit = (git rev-parse HEAD).Trim()
git status --short --branch
Write-Host "SOURCE_COMMIT=$SourceCommit"

$Required = @(
  'experiment7/reference/training/deck_identity_model.py',
  'experiment7/reference/training/train_multideck_identity.py',
  'experiment7/integration/build_from_pocketmon_replays.py',
  'experiment7/integration/select_initial_decks.py',
  'experiment7/integration/prepare_training_data.py',
  'experiment7/integration/train_driver.py',
  'experiment7/integration/multi_gpu_scheduler.py',
  'experiment7/integration/remote_worker.py',
  'experiment7/integration/export_and_package.py',
  'experiment7/integration/arena.py',
  'experiment7/configs/multideck_default.json'
)
foreach ($Path in $Required) {
    if (-not (Test-Path $Path)) { throw "Required file missing: $Path" }
}
```

Then read and execute the full operational prompt:

```powershell
Get-Content -Raw experiment7\CODEX_WINDOWS_PROMPT.md
```

Do not merely summarize it.

## Fixed experiment inputs

```text
replays:       /homes/lzhang/pocketmon/data/raw/replays/2026-08-06
ladder report: /homes/lzhang/pocketmon/analysis/outputs/top_ladder_2026_08_07_20260808
target Agent:  agents/lucario_rule
servers:       doraemon02 doraemon03 doraemon15 doraemon16 doraemon19 doraemon20
```

Select 4–6 supported, distinct high-ladder exact decks; build one shared broad
pretrain and balanced multi-deck fine-tunes using the Experiment 7 reference
implementation; package one Agent per deck; and evaluate against the frozen
Lucario Agent with seat-balanced 20/100/200-game gates.

All data processing, training, export and Arena commands must run through SSH on
the Linux servers. Windows must only edit/push code and orchestrate immutable
commits. Do not modify `main`, do not submit Kaggle, and do not commit replays,
caches, checkpoints, portable weights, engine binaries, credentials, or large
per-game logs.
