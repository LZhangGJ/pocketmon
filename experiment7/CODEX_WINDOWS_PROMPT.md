# Windows Codex prompt — run the integrated Experiment 7 pipeline on Linux GPUs

You are Codex running on my Windows workstation. Windows is only the Git and SSH
control plane. Every replay scan, cache build, CUDA job, model export, portable
verification and Arena match must execute on the Linux doraemon servers.

## Fixed context

```text
repository:       https://github.com/LZhangGJ/pocketmon.git
source branch:    agent/experiment7-multideck-ready-20260809
working branch:   codex/experiment7-multideck-run-20260809
Linux repository: /homes/lzhang/pocketmon
Linux Python:     /homes/lzhang/mypath/new/envs/trans/bin/python
replays:          /homes/lzhang/pocketmon/data/raw/replays/2026-08-06
ladder report:    /homes/lzhang/pocketmon/analysis/outputs/top_ladder_2026_08_07_20260808
target Agent:     agents/lucario_rule
servers:          doraemon02 doraemon03 doraemon15 doraemon16 doraemon19 doraemon20
```

The teammate Experiment 7 source is already present as ordinary files under
`experiment7/reference/`. Do not locate, upload, unpack or directly execute
`experiment7_code_for_gpt_2026-08-08.zip`. Do not substitute the old simplified
RL-BC-004 model. The code under `experiment7/integration/` is the repository
adapter you should execute and minimally repair if a real failure is found.

The objective is to select 4–6 distinct, supported high-ladder exact decks,
train one shared broad-pretrain model plus balanced multi-deck fine-tunes, package
one Agent per deck, and find multiple deck+policy combinations that beat the
frozen Lucario rule Agent.

## 1. Windows Git setup

Run in PowerShell 7:

```powershell
$ErrorActionPreference = 'Stop'
$RepoUrl = 'https://github.com/LZhangGJ/pocketmon.git'
$SourceBranch = 'agent/experiment7-multideck-ready-20260809'
$WorkBranch = 'codex/experiment7-multideck-run-20260809'
$LocalRepo = Join-Path $HOME 'source\pocketmon-experiment7'
$RemotePython = '/homes/lzhang/mypath/new/envs/trans/bin/python'
$RunRoot = '/homes/lzhang/pocketmon/runs/experiment7-multideck-20260809'

if (-not (Test-Path (Join-Path $LocalRepo '.git'))) {
    New-Item -ItemType Directory -Force -Path (Split-Path $LocalRepo) | Out-Null
    git clone $RepoUrl $LocalRepo
}
Set-Location $LocalRepo
git fetch origin --prune
git switch --force-create $WorkBranch "origin/$SourceBranch"
git status --short --branch
```

Read these files before running anything:

```text
experiment7/README.md
experiment7/reference/docs/EXPERIMENT7_CLEANROOM_DESIGN.md
experiment7/integration/prepare_training_data.py
experiment7/integration/train_driver.py
experiment7/integration/multi_gpu_scheduler.py
experiment7/integration/export_and_package.py
experiment7/integration/arena.py
```

The implementation is already present. Do not rewrite the model or invent a new
pipeline. Diagnose and patch only concrete failures.

## 2. Static gates

Run locally if Windows has the required Python dependencies; otherwise run the
same tests on `doraemon02` after bootstrapping:

```powershell
python -m unittest discover -s tests -p 'test_experiment7_*.py'
python -m unittest discover -s tests -p 'test_reference_model.py'
python -m compileall -q experiment7 tests
git diff --check
```

The reference package test must verify all 38 manifest-listed files with zero
size/SHA mismatches.

If no code change is needed, push the source branch state to the working branch.
If a fix is needed, test it first and then commit it:

```powershell
git add experiment7 tests
git commit -m "Integrate Experiment 7 multi-deck training"
git push --set-upstream origin $WorkBranch
$Commit = (git rev-parse HEAD).Trim()
```

Never modify, merge or force-push `main`.

## 3. Bootstrap an immutable shared Linux worktree

Use `doraemon02` as coordinator unless SSH proves it is unavailable. Create the
worktree directly from the immutable commit without switching the canonical
checkout:

```powershell
$Bootstrap = "cd /homes/lzhang/pocketmon && git fetch origin --prune && " +
             "git show '$Commit`:experiment7/integration/linux_bootstrap.sh' | " +
             "bash -s -- '$Commit'"
ssh doraemon02 $Bootstrap
$Worktree = "/homes/lzhang/worktrees/experiment7-$($Commit.Substring(0,12))"
```

Require the bootstrap receipt to show the exact commit, Python path and worktree.
All servers read this same shared worktree. Remote workers do not edit source.

## 4. Prepare replays, select decks and build all caches

Execute synchronously on the coordinator:

```powershell
$Prepare = @"
set -euo pipefail
cd '$Worktree'
export PYTHON='$RemotePython'
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
`$PYTHON experiment7/integration/prepare_training_data.py \
  --reference-root '$Worktree/experiment7/reference' \
  --raw-root '/homes/lzhang/pocketmon/data/raw/replays/2026-08-06' \
  --ladder-dir '/homes/lzhang/pocketmon/analysis/outputs/top_ladder_2026_08_07_20260808' \
  --cards '$Worktree/data/reference/official_cards.json' \
  --attacks '$Worktree/data/reference/official_attacks.json' \
  --target-deck '$Worktree/agents/lucario_rule/deck.csv' \
  --output-root '$RunRoot/prepared' \
  --python '$RemotePython' \
  --policy-source winners \
  --desired-decks 6 \
  --minimum-decks 4 \
  --min-actor-episodes 10 \
  --min-policy-decisions 500 \
  --near-duplicate-threshold 0.80 \
  --holdout-fraction 0.20 \
  --calibration-fraction 0.20 \
  --strict-catalog
"@
$Prepare | ssh doraemon02 'bash -s'
```

This stage must actually:

1. audit previous-observation/action alignment and legal actions;
2. deduplicate episode IDs and freeze replay/deck receipts;
3. recover canonical exact 60-card lists;
4. read the ladder report's `archetype_summary.csv` and
   `representative_decklists.csv`;
5. exclude the exact frozen target and labels containing Mega Lucario;
6. enforce exact replay support and remove near-duplicate Pokémon cores;
7. choose the newest compatible current module and adjacent pretrain window;
8. build broad-pretrain and per-deck chronological datasets;
9. build token, 8-step sequence, opponent-class and deck-identity caches.

The principal output is:

```text
$RunRoot/prepared/training_sources.json
```

Do not manually substitute unsupported decks. If fewer than four decks pass,
report `selection/deck_candidates.csv` and the exact support failure. A threshold
may be changed only after documenting why it is too strict for the real data.

## 5. Inventory all server GPUs

From Windows:

```powershell
python experiment7/integration/multi_gpu_scheduler.py inventory `
  --hosts doraemon02 doraemon03 doraemon15 doraemon16 doraemon19 doraemon20 `
  --output experiment7_gpu_inventory.json `
  --minimum-free-mib 12000 `
  --maximum-utilization 20
```

Do not terminate or preempt existing processes. One formal job per GPU. Refresh
inventory before each new training stage.

Set shared paths:

```powershell
$Sources = "$RunRoot/prepared/training_sources.json"
$Worker = "$Worktree/experiment7/integration/remote_worker.py"
$TrainingRoot = "$RunRoot/training"
```

## 6. Tiny-overfit gate

Create, launch and monitor one smoke job:

```powershell
python experiment7/integration/multi_gpu_scheduler.py make-training-plan `
  --inventory experiment7_gpu_inventory.json `
  --output experiment7_smoke_plan.json `
  --worktree $Worktree `
  --commit $Commit `
  --python $RemotePython `
  --sources $Sources `
  --run-root $TrainingRoot `
  --stage smoke

python experiment7/integration/multi_gpu_scheduler.py launch `
  --plan experiment7_smoke_plan.json `
  --remote-python $RemotePython `
  --worker $Worker

python experiment7/integration/multi_gpu_scheduler.py status `
  --plan experiment7_smoke_plan.json `
  --remote-python $RemotePython `
  --worker $Worker
```

Keep polling status and the remote log until the job is definitively succeeded or
failed. The smoke report must show finite training, strict checkpoint reload,
zero illegal predictions, and exact-semantic accuracy improving from
initialization. A failed smoke gate blocks long training.

## 7. Shared pretraining and formal fine-tuning

Refresh GPU inventory. Create and launch one `pretrain` plan. Wait until
`$TrainingRoot/pretrain/pretrain_model.pt` is complete. Refresh inventory again,
then create a `finetune` plan with:

```text
--pretrain-checkpoint $TrainingRoot/pretrain/pretrain_model.pt
--seeds 20260808 20260809 20260810
```

Require three genuinely idle GPUs. Preserve every failed seed and log. The
fine-tune driver balances all selected exact-deck arms and chooses each seed by
the unweighted calibration macro `exactSemantic`, while recording the worst
deck and deck standard deviation.

## 8. Freeze the best seed, open holdout once, export and package

After all formal seeds finish, run on `doraemon02`:

```bash
set -euo pipefail
cd "$Worktree"
PY=/homes/lzhang/mypath/new/envs/trans/bin/python
SRC="$RunRoot/prepared/training_sources.json"
SEL="$RunRoot/selection/best_seed.json"
FINAL="$RunRoot/final"
PKG="$RunRoot/packages"
mkdir -p "$RunRoot/selection" "$FINAL" "$PKG"

$PY experiment7/integration/export_and_package.py select-best --root "$TrainingRoot" --output "$SEL"
CHECKPOINT=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])' "$SEL")
$PY experiment7/integration/train_driver.py holdout --sources "$SRC" --checkpoint "$CHECKPOINT" --output "$FINAL/holdout.json"
$PY experiment7/integration/export_and_package.py export --checkpoint "$CHECKPOINT" --output "$FINAL/deck_identity_bc.npz"
$PY experiment7/integration/export_and_package.py verify --reference-root "$Worktree/experiment7/reference" --sources "$SRC" --checkpoint "$CHECKPOINT" --portable "$FINAL/deck_identity_bc.npz" --output "$FINAL/portable_parity.json" --python "$PY" --decisions-per-source 150
$PY experiment7/integration/export_and_package.py package --reference-root "$Worktree/experiment7/reference" --sources "$SRC" --portable "$FINAL/deck_identity_bc.npz" --output-root "$PKG"
```

The holdout is write-once. Portable verification must have zero stable ranking
mismatches; 150 decisions per source gives at least 600 checks for 4–6 decks.

## 9. Arena against the frozen Lucario target

Locate the already installed official `cg` directory; do not replace or redownload
it. Freeze the target with `target_receipt.py`. Generate schedules with
`arena.py make-schedule`, execute them with the existing
`scripts/run_league_schedule.py`, and summarize with `arena.py summarize`.

Stages:

1. smoke: 20 games/challenger;
2. screening: 100 games/challenger for smoke survivors;
3. confirmation: 200 games/challenger for the screening top three.

Confirmation thresholds:

```text
score rate >= 0.55
95% Wilson lower bound > 0.50
both seats score rate >= 0.45
zero failures, positive model calls, zero fallback calls
```

The target is at least two successful, distinct archetypes. Do not automatically
run 400-game confirmation or submit to Kaggle.

## 10. Git and artifact policy

Push source fixes, tests, configuration templates and small aggregate receipts to
`codex/experiment7-multideck-run-20260809`. Never commit replays, caches,
checkpoints, `.npz` files, engine binaries, bulk game logs, tokens or credentials.
Never submit to Kaggle in this task.

## 11. Final report

Report the repository, branch and immutable commit; selected exact decks and
support; module window; split counts; privacy audit; GPU/job allocation;
smoke/pretrain/all fine-tune seeds; selected checkpoint; one-shot holdout;
portable parity and latency; package receipts; 20/100/200-game results; every
error/fallback; and whether at least two distinct archetypes beat the target.

Do not use `reference/results/experiment7_summary.json` as evidence for this run.
