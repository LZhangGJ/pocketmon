# Experiment 7: multi-deck challenger integration

This directory contains the teammate Experiment 7 implementation as ordinary
Git-tracked source plus the `pocketmon` adapters needed to select high-ladder
exact decks, prepare caches, train on several Linux GPUs, export portable
weights, package one Agent per deck, and evaluate them against the frozen
Lucario rule Agent.

## Source and execution boundaries

- `reference/` is the materialized source from the teammate code-review package.
  Its original `PACKAGE_MANIFEST.csv` verifies 38 listed files byte-for-byte.
- `integration/` contains the repository-specific adapters and orchestration.
- Windows is the Git/SSH control plane only.
- Replays, caches, CUDA jobs, model files and Arena games stay on the Linux
  doraemon servers and shared storage.
- The original ZIP is no longer an execution input.

## Important entry points

| File | Role |
|---|---|
| `build_replay_catalog.py` | Raw replay audit, exact-deck extraction and support counts |
| `select_initial_decks.py` | Ladder report + replay support → 4–6 distinct non-Lucario decks |
| `build_from_pocketmon_replays.py` | Pocketmon replay → Experiment 7 feature archive |
| `prepare_training_data.py` | Builds broad/current datasets and all token/sequence/identity caches |
| `train_driver.py` | Tiny-overfit, shared pretrain, balanced multi-deck fine-tune and one-shot holdout |
| `multi_gpu_scheduler.py` | Windows/Linux SSH GPU inventory, planning, launching and status |
| `remote_worker.py` | Detached Linux worker with one-job-per-GPU locking and receipts |
| `export_and_package.py` | Best-seed selection, NPZ export, parity verification and Agent packaging |
| `target_receipt.py` | Freezes Lucario Agent/deck/engine hashes |
| `arena.py` | Generates and summarizes seat-balanced 20/100/200-game stages |
| `linux_bootstrap.sh` | Creates an immutable shared Linux worktree from one commit |
| `configs/multideck_default.json` | Machine-readable server/model/training defaults |
| `CODEX_WINDOWS_PROMPT.md` | Final prompt for Codex running on the Windows workstation |

## Model contract retained from Experiment 7

- state 320, legal option 176, entity numeric 12;
- 8 causal history slots;
- exact own 60-card multiset token;
- visible-only opponent evidence token;
- `d_model=128`, 4 attention heads, 3 blocks, FFN 384, dropout 0.05;
- action, count, value and low-weight opponent-class auxiliary heads;
- one shared broad pretrain followed by balanced fine-tuning over all selected
  exact-deck sources.

The opponent classifier remains auxiliary and is never used for runtime routing.

## Branches

```text
repository:       LZhangGJ/pocketmon
source branch:    agent/experiment7-multideck-ready-20260809
Codex work branch: codex/experiment7-multideck-run-20260809
```

## Fixed server inputs

```text
replays: /homes/lzhang/pocketmon/data/raw/replays/2026-08-06
report:  /homes/lzhang/pocketmon/analysis/outputs/top_ladder_2026_08_07_20260808
python:  /homes/lzhang/mypath/new/envs/trans/bin/python
target:  agents/lucario_rule
servers: doraemon02 doraemon03 doraemon15 doraemon16 doraemon19 doraemon20
```

## Static gates

```bash
python -m unittest discover -s tests -p 'test_experiment7_*.py'
python -m unittest discover -s tests -p 'test_reference_model.py'
python -m compileall -q experiment7 tests
```

Run `train_driver.py smoke` before shared pretraining. Do not open the
chronological holdout until the best formal fine-tune seed is frozen.
