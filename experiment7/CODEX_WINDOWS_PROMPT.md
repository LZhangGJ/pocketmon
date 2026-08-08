# Windows Codex execution prompt: train several Experiment 7 deck agents on Linux GPUs

You are running on the user's **Windows workstation**. Windows is the control plane. All replay processing, cache construction, GPU training, portable-model verification, and Arena matches must run through SSH on the Linux doraemon servers.

## Fixed repository context

```text
GitHub repository:   LZhangGJ/pocketmon
remote:              https://github.com/LZhangGJ/pocketmon.git
source branch:       agent/experiment7-multideck-ready-20260808
working branch:      codex/experiment7-multideck-run-20260808

Linux canonical repo: /homes/lzhang/pocketmon
Linux Python:         /homes/lzhang/mypath/new/envs/trans/bin/python
replays:              /homes/lzhang/pocketmon/data/raw/replays/2026-08-06
ladder report:        /homes/lzhang/pocketmon/analysis/outputs/top_ladder_2026_08_07_20260808
servers:              doraemon02 doraemon03 doraemon15 doraemon16 doraemon19 doraemon20
primary target agent: agents/lucario_rule
```

The teammate's high-score Experiment 7 source has already been imported as normal repository files under:

```text
experiment7/reference_impl/
```

Do **not** look for, upload, extract, or run `experiment7_code_for_gpt_2026-08-08.zip`. The ZIP was only a source handoff. This branch already contains the integrated source and executable adapter code.

The repository adapters and orchestration code are under:

```text
experiment7/integration/
```

Use them. Do not replace the model with the older simplified RL-BC-004 implementation.

## Objective

1. Read the frozen ladder report on the Linux shared storage.
2. Reconstruct exact legal 60-card representative decks.
3. Match them against the audited replay/deck sidecar.
4. Select 4-6 distinct supported archetypes, default 6.
5. Train the accepted Experiment 7 deck-conditioned temporal Transformer on three independent seeds using currently idle GPUs across the doraemon servers.
6. Export the best calibration-selected checkpoint to NumPy portable inference.
7. Build one self-contained Agent package per selected deck.
8. Run seat-balanced local matches against the frozen `agents/lucario_rule` Agent.
9. Find at least two different archetypes that beat the target in fresh 200-game confirmation.

The comparison is a **deck + policy** comparison. Do not describe it as a pure model-architecture causal comparison.

## Model and training contract

Keep the imported high-score implementation:

```text
state_dim               320
option_dim              176
entity_numeric_dim       12
history_length            8
d_model                  128
attention_heads            4
transformer_blocks         3
ff_dim                    384
dropout                  0.05
opponent_aux_weight      0.05
pretrain_epochs            12
pretrain_batch            128
pretrain_lr              3e-4
finetune_batch_per_deck    48
finetune_lr              1e-4
optimizer                AdamW
weight_decay             1e-4
```

Runtime information boundary:

- own exact 60-card deck is allowed;
- only actor-visible opponent evidence is allowed;
- opponent hidden hand/deck/prize, future state, final result, player identity, directory labels, and replay-only identifiers are forbidden;
- opponent-class prediction is training-only auxiliary supervision and must never gate runtime policy.

## Step 1: initialize the Windows repository

Use PowerShell 7:

```powershell
$ErrorActionPreference = 'Stop'
$RepoUrl = 'https://github.com/LZhangGJ/pocketmon.git'
$SourceBranch = 'agent/experiment7-multideck-ready-20260808'
$WorkBranch = 'codex/experiment7-multideck-run-20260808'
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

Read before running:

```text
experiment7/README.md
experiment7/integration/README.md
experiment7/configs/multideck_default.json
docs/EXPERIMENT7_MULTIDECK_READY.md
```

Run Windows-safe static checks:

```powershell
python -m compileall -q experiment7 tests
python -m unittest discover -s tests -p 'test_experiment7_*.py'
git diff --check
```

If you modify code, commit and push before any remote run. Every Linux worker must use one immutable pushed commit SHA.

## Step 2: verify SSH and inventory GPUs

Do not assume a GPU is idle. Run:

```powershell
python experiment7/integration/windows_controller.py `
  --repo $LocalRepo `
  probe `
  --output experiment7_gpu_inventory.json
```

Reject unavailable hosts. Never kill or preempt an existing process. Formal training may use only GPUs with sufficient free memory and low utilization.

## Step 3: bootstrap the fixed commit on Linux

```powershell
python experiment7/integration/windows_controller.py `
  --repo $LocalRepo `
  bootstrap `
  --output experiment7_workers.json
```

The controller must push the current Windows branch, record `git rev-parse HEAD`, and create host-specific detached Linux worktrees from exactly that commit.

## Step 4: prepare data and choose initial decks on the coordinator

Run the preparation stage on `doraemon02` unless SSH/GPU/storage checks show a concrete reason to use another coordinator:

```powershell
python experiment7/integration/windows_controller.py `
  --repo $LocalRepo `
  prepare `
  --coordinator doraemon02 `
  --desired-decks 6 `
  --minimum-decks 4 `
  --min-actor-episodes 40 `
  --min-policy-decisions 800
```

This stage must actually execute on Linux and must use:

```text
scripts/convert_public_replays.py
scripts/build_replay_deck_map.py
experiment7/integration/select_initial_decks.py
experiment7/integration/build_from_pocketmon_canonical.py
experiment7/reference_impl/data_pipeline/build_token_cache.py
experiment7/reference_impl/data_pipeline/build_sequence_cache.py
experiment7/reference_impl/data_pipeline/build_opponent_deck_class_map.py
experiment7/reference_impl/data_pipeline/build_deck_identity_cache.py
```

Expected shared output:

```text
/homes/lzhang/pocketmon/runs/experiment7-multideck-20260808/data/
```

Do not hand-pick decks before inspecting the report and replay support. Selection must use ladder score, exact-deck legality, replay support, and archetype diversity. It must exclude the target's exact deck and must not use holdout or final Arena results.

If fewer than four decks pass the frozen support gates, stop and report the candidate table. Do not fill the set with unsupported handwritten decks.

Inspect these receipts before training:

```text
.../data/selection/candidate_table.csv
.../data/selection/selected_decks.json
.../data/datasets/dataset_manifest.json
.../data/ready_receipt.json
```

Require:

- invalid replay decisions = 0;
- conflicting episode IDs = 0;
- unknown submission status = 0;
- exact deck size = 60;
- at least four selected archetypes;
- chronological holdout and calibration are non-empty for every selected deck;
- hidden-opponent fields are absent from runtime features.

## Step 5: dispatch three formal seeds to idle GPUs

```powershell
python experiment7/integration/windows_controller.py `
  --repo $LocalRepo `
  train `
  --seeds 20260808 20260809 20260810 `
  --max-utilization 20 `
  --min-free-memory-mb 12000 `
  --output experiment7_jobs.json
```

The controller must return a concrete host, GPU, PID, worktree, commit, run directory, and receipt path for every seed. Do not claim a job started merely because `nohup` returned; verify the PID and `job_receipt.json`.

Monitor:

```powershell
python experiment7/integration/windows_controller.py `
  --repo $LocalRepo `
  status `
  --jobs experiment7_jobs.json
```

Continue monitoring until all seeds complete or fail. Preserve failed seed logs. Never report only the best seed.

## Step 6: finalize the best sealed-calibration seed on Linux

After all formal seeds finish, run on the coordinator worktree through SSH:

```bash
/homes/lzhang/mypath/new/envs/trans/bin/python \
  experiment7/integration/finalize_candidate.py \
  --dataset-manifest /homes/lzhang/pocketmon/runs/experiment7-multideck-20260808/data/datasets/dataset_manifest.json \
  --training-root /homes/lzhang/pocketmon/runs/experiment7-multideck-20260808/training \
  --selected-decks /homes/lzhang/pocketmon/runs/experiment7-multideck-20260808/data/selection/selected_decks.json \
  --engine-catalog /homes/lzhang/pocketmon/runs/experiment7-multideck-20260808/data/engine_catalog.json \
  --output-root /homes/lzhang/pocketmon/runs/experiment7-multideck-20260808/final
```

This selects by calibration only, exports portable weights, verifies at least 500 action rankings with zero mismatch, and builds one package per selected deck. The chronological holdout remains sealed during checkpoint selection.

Require:

- portable ranking mismatches = 0;
- every package imports successfully;
- every package contains `main.py`, `deck.csv`, `engine_catalog.json`, and `deck_identity_bc.npz`;
- no runtime fallback during smoke matches.

## Step 7: run the Arena gates on Linux

Use the package manifest produced under `.../final/packages/` and the frozen target `agents/lucario_rule`.

Generate the first schedule with:

```bash
python experiment7/integration/generate_arena_schedule.py \
  --package-manifest <package_manifest_json> \
  --target-agent <fixed_worktree>/agents/lucario_rule \
  --games-per-challenger 20 \
  --output-dir /homes/lzhang/pocketmon/runs/experiment7-multideck-20260808/arena/smoke
```

Run the schedule using the repository's isolated local-match/league runner and the same frozen official `cg` engine for every cell. Shard games across available doraemon servers; each shard writes its own CSV. Merge only after all shards finish.

Gate sequence:

1. 20 games per challenger, 10 each seat: runtime/legal smoke.
2. 100 fresh games per smoke-passing challenger, 50 each seat: screening.
3. Top three challengers: 200 fresh games each, 100 each seat: confirmation.

Use `experiment7/integration/summarize_challenger_results.py` for every stage.

A 200-game challenger passes only when:

- score rate >= 0.55;
- 95% Wilson lower bound > 0.50;
- each seat score rate >= 0.45;
- crash/timeout/illegal/load/inference/fallback counts are all zero.

The project objective is reached when at least two different archetypes pass the 200-game confirmation. Do not automatically run 400-game confirmation or submit to Kaggle.

## Git policy

Allowed to push:

- source code;
- tests;
- configuration templates;
- small aggregate audit/result receipts;
- documentation.

Never commit:

- replays;
- generated feature/token caches;
- checkpoints;
- `.pt`, `.pth`, `.ckpt`, `.npz`;
- `cg` engine binaries;
- credentials;
- large per-game logs.

Do not modify or merge `main`. Push implementation fixes only to:

```text
codex/experiment7-multideck-run-20260808
```

## Final report

Report verified facts only:

- Windows work branch and final commit;
- remote hosts/GPUs actually used;
- selected decks, archetypes, exact deck hashes, ladder metrics, and replay support;
- data-gate counts and chronological split sizes;
- all three seeds and per-epoch calibration metrics;
- selected seed/epoch and checkpoint hash;
- portable parity and Linux CPU latency;
- every challenger's 20/100/200 game W/D/L, score, Wilson interval, seat split, and failures;
- whether at least two different archetypes passed;
- any blockers and exact failing command.

Do not treat `experiment7/reference_impl/results/experiment7_summary.json` as a local result. It is historical reference evidence only.

Start executing now. Do not merely restate this prompt.
