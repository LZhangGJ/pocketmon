# Experiment 7 multi-deck Challenger handoff

This directory contains the teammate-provided high-scoring Experiment 7 source snapshot and the orchestration contract for training several deck-conditioned Challenger Agents.

## Exact repository context

```text
Repository:          LZhangGJ/pocketmon
Remote:              https://github.com/LZhangGJ/pocketmon.git
Source branch:       agent/experiment7-multideck-challengers-20260808
Codex work branch:   codex/experiment7-multideck-challengers-20260808
Linux repository:   /homes/lzhang/pocketmon
```

Codex runs on the user's Windows workstation. Windows is the control plane for code editing, Git and SSH. Replay processing, GPU training, portable export, official-engine evaluation and performance measurements run only on the Linux `doraemon` servers.

## Objective

Select several distinct high-performing exact 60-card decks from:

```text
/homes/lzhang/pocketmon/analysis/outputs/top_ladder_2026_08_07_20260808
```

Reuse the supplied Experiment 7 data/model/training/runtime implementation, make only the minimum repository adapters, and train multiple deck + policy Challengers against the frozen target:

```text
agents/lucario_rule
agents/lucario_rule/deck.csv
```

The detailed procedure is in `CODEX_TRAINING_PROMPT.md`; machine-readable constants are in `MULTIDECK_CHALLENGER_PLAN.json`.

## Source artifact

- Expected archive: `source/experiment7_code_for_gpt_2026-08-08.zip`
- Expected SHA-256: `9c0d24067eacee8abc38223dba28d893e5d1e4e9b75204a9ce92a03093558229`
- Entries: 39 total; 38 files listed in `PACKAGE_MANIFEST.csv`
- Expected static check: all 38 manifest entries match and all 29 Python files compile.

The archive intentionally excludes checkpoints, portable weights, engine/card catalogs, replays, generated caches, opponent packages, Arena logs and credentials. Those assets must be regenerated on the Linux servers and must not be committed.

## Windows start

Open PowerShell 7 and follow `CODEX_START_HERE.md`. The Windows host must not attempt to access `/homes/...` locally and must not perform training. It pushes an immutable commit, then starts server jobs through OpenSSH.

## Linux unpack

On a remote Linux worktree created from the fixed commit:

```bash
export PYTHON=/homes/lzhang/mypath/new/envs/trans/bin/python
bash experiment7/unpack_source.sh
```

The default destination is `runs/experiment7/source`, which is ignored by Git.

## Safety boundaries

- Do not modify or force-push `main`.
- Do not submit to Kaggle during this task.
- Do not commit replay data, caches, checkpoints, portable `.npz`, engines, credentials or large per-game logs.
- Do not replace the teammate's high-scoring implementation with the older simplified RL-BC-004 path.
- Do not use opponent hidden information or the opponent auxiliary classifier for runtime gating.
