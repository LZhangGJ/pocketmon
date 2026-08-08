# Windows Codex run prompt — canonical Experiment 7 source branch

Use this exact repository and branch. Do not search for similarly named branches.

```text
repository:        https://github.com/LZhangGJ/pocketmon.git
source branch:     agent/experiment7-training-ready-20260809
work branch:       codex/experiment7-multideck-run-20260809
Windows role:      code edits, Git, SSH orchestration, status collection
Linux role:        replay conversion, cache building, CUDA training, export, Arena
Linux repository:  /homes/lzhang/pocketmon
Linux Python:      /homes/lzhang/mypath/new/envs/trans/bin/python
replays:           /homes/lzhang/pocketmon/data/raw/replays/2026-08-06
ladder report:     /homes/lzhang/pocketmon/analysis/outputs/top_ladder_2026_08_07_20260808
target Agent:      agents/lucario_rule
servers:           doraemon02 doraemon03 doraemon15 doraemon16 doraemon19 doraemon20
```

## Critical source rule

`experiment7_code_for_gpt_2026-08-08.zip` is a source snapshot, not a training
entry point and not a runnable Agent. Never execute a Python module from inside
the ZIP and never use its historical result JSON as a local result.

The first Windows step is to validate the original ZIP, extract it into ordinary
files under `experiment7/reference/`, commit those ordinary files to the Codex
work branch, and push that immutable commit. Linux workers may start only after
that commit exists. The source directory name is deliberately
`experiment7/reference/`; do not require `experiment7/vendor/reference/`.

Correct ZIP identity:

```text
filename: experiment7_code_for_gpt_2026-08-08.zip
bytes:    94038
sha256:   9c0d24067eacee8abc38223dba28d893e5d1e4e9b75204a9ce92a03093558229
```

## 1. Windows Git setup

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
git status --short --branch
git rev-parse HEAD
```

Before materialization, verify the repository integration layer exists:

```powershell
$IntegrationRequired = @(
  'experiment7/integration/build_replay_catalog.py',
  'experiment7/integration/build_from_pocketmon_replays.py',
  'experiment7/integration/select_initial_decks.py',
  'experiment7/integration/prepare_training_data.py',
  'experiment7/integration/train_driver.py',
  'experiment7/integration/multi_gpu_scheduler.py',
  'experiment7/integration/remote_worker.py',
  'experiment7/integration/export_and_package.py',
  'experiment7/integration/arena.py',
  'experiment7/configs/multideck_default.json',
  'experiment7/MATERIALIZE_REFERENCE_WINDOWS.ps1'
)
foreach ($Path in $IntegrationRequired) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Required integration file missing: $Path" }
}
```

## 2. Materialize the teammate source as ordinary Git files

Locate the original ZIP on Windows. Prefer Downloads, but use the actual path if
it is elsewhere:

```powershell
$Archive = Join-Path $HOME 'Downloads\experiment7_code_for_gpt_2026-08-08.zip'
if (-not (Test-Path $Archive -PathType Leaf)) {
    throw "Original Experiment 7 ZIP not found: $Archive"
}

.\experiment7\MATERIALIZE_REFERENCE_WINDOWS.ps1 -ArchivePath $Archive
```

The script verifies the ZIP byte count, SHA-256, every entry in
`PACKAGE_MANIFEST.csv`, and the required training/runtime/validation files. It
then replaces any partial `experiment7/reference/` directory with the verified
ordinary source tree.

Verify and commit the ordinary source:

```powershell
$ReferenceRequired = @(
  'experiment7/reference/training/deck_identity_model.py',
  'experiment7/reference/training/train_multideck_identity.py',
  'experiment7/reference/data_pipeline/features.py',
  'experiment7/reference/data_pipeline/tokenizer.py',
  'experiment7/reference/data_pipeline/build_token_cache.py',
  'experiment7/reference/data_pipeline/build_sequence_cache.py',
  'experiment7/reference/data_pipeline/build_deck_identity_cache.py',
  'experiment7/reference/runtime_agent/main.py',
  'experiment7/reference/validation/arena_isolated.py',
  'experiment7/reference/docs/EXPERIMENT7_CLEANROOM_DESIGN.md',
  'experiment7/reference/IMPORT_RECEIPT.json'
)
foreach ($Path in $ReferenceRequired) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Materialized source missing: $Path" }
}

python -m compileall -q experiment7\reference experiment7\integration tests
git diff --check
git add experiment7/reference experiment7/integration experiment7/configs tests
git commit -m 'Materialize and integrate Experiment 7 multi-deck training source'
git push --set-upstream origin $WorkBranch
$TrainingCommit = (git rev-parse HEAD).Trim()
Write-Host "TRAINING_COMMIT=$TrainingCommit"
```

Do not distribute uncommitted source through SCP. Every Linux worker must check
out this exact pushed commit.

## 3. Read the full operational instructions

After the ordinary source commit is pushed, read and execute:

```powershell
Get-Content -Raw experiment7\CODEX_WINDOWS_PROMPT.md
```

Treat this file as the bootstrap correction if an older paragraph in the full
prompt says the reference source was already present. At this point it is now
present because you materialized and committed it.

## 4. Experiment objective

Use the existing integration layer and the verified Experiment 7 reference
implementation to:

1. inventory all doraemon GPUs through SSH without preempting existing jobs;
2. freeze `agents/lucario_rule` and its deck/engine receipt;
3. scan the ladder report and select 4–6 distinct exact 60-card decks with real
   replay support, excluding Lucario as a Challenger archetype;
4. construct leak-free chronological fit/calibration/holdout caches;
5. run the Experiment 7 shared broad pretrain and balanced multi-deck fine-tunes;
6. run at least the registered seeds `20260808`, `20260809`, `20260810` when
   resources and data gates permit;
7. export portable NumPy Agents, verify PyTorch/portable action-ranking parity,
   and package one Agent per deck;
8. evaluate against the frozen Lucario Agent using seat-balanced 20-game smoke,
   100-game screening, and 200-game confirmation gates.

The Experiment 7 reference model keeps the 320-dimensional state, 176-dimensional
legal-option features, 12-dimensional entity numerics, eight history slots,
exact own-deck multiset conditioning, visible-opponent evidence, a 128-wide
three-block Transformer, count/value heads, and the low-weight opponent-class
auxiliary loss. Do not substitute the old simplified RL-BC-004 implementation.

## 5. Execution boundary

All replay processing, cache creation, CUDA training, model export, CPU latency
measurement and Arena matches must execute through SSH on the Linux servers.
Windows is only the code/Git/control plane. A remote job is not considered
started until an SSH command returns a concrete PID and writes a job receipt.

Do not modify or force-push `main`. Do not submit Kaggle. Do not commit replays,
caches, checkpoints, portable weights, engine binaries, credentials, or large
per-game logs. Preserve all failures and do not select decks or epochs using the
sealed holdout or final target-match results.
