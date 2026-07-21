# Local Codex CLI handoff: public replay experiments

Last updated: 2026-07-20

## Mission and non-negotiable constraints

Continue on `exp/league-v1`. Do not modify or force-push `main`, do not open or merge a PR, and do not submit to Kaggle. The `main` champion remains `4762ba17fc01ca95b8f4f2207a0277782d9027f4` until server-side evaluation justifies a separate merge decision.

Run real data conversion and training on the project server. Synthetic or Codex Cloud checks verify code behavior only; they are not competition evidence.

## What “valid action rate” means

In DATA-001, an action is counted as a valid decision only when the recorded replay action can be explained by the candidate player observation:

- `action` is a list containing only integer option indices;
- the candidate observation and its `select` object exist;
- `select.option` is a list and its selection bounds are valid;
- the number of selected indices is between `minCount` and `maxCount`;
- indices are unique and each index is inside `select.option`.

A 60-item action with no `select` is treated as a deck-submission setup action. Setup actions are counted separately and excluded from both the numerator and denominator:

```text
valid_rate = valid_decisions / (valid_decisions + invalid_decisions)
```

This metric is more precisely a **replay-label/observation consistency rate**. It does not replay the complete card-game rules, prove that every deck is semantically valid, or measure the illegal-action rate of a trained agent.

DATA-001 compares two temporal alignments:

- `previous`: action stored at step `t` is paired with the same player's observation at step `t-1`;
- `same`: action stored at step `t` is paired with that player's observation at step `t`.

The synthetic shifted replay produced `previous = 4/4 = 100%` and `same = 1/4 = 25%`. This supports the implementation but does not yet prove the schema of real public replays. On real data, the strict DATA-001 gate defaults to `valid_rate >= 0.999` with no replay load errors.

## What “DATA-002 has zero invalid actions” means

With `--max-invalid-rate 0`, DATA-002 computes:

```text
invalid_rate = invalid_decisions / (canonical_rows + invalid_decisions)
```

Passing with zero invalid decisions means every converted ordinary replay decision is structurally consistent with its paired observation and selectable option list. It is a **dataset conversion gate**, not a claim that a learned policy has a 0% illegal-action rate in matches.

The output is promoted from its temporary file only when all of the following hold:

- at least one canonical decision row was produced;
- `invalid_rate <= --max-invalid-rate` (normally exactly `0`);
- replay load errors are `0`;
- conflicting duplicate episode IDs are `0`;
- episodes missing a terminal winner are `0`;
- terminal and top-level reward mismatches are `0`;
- non-initial actions with unknown submission status are `0`;
- with `--policy-source winners`, at least one winner policy row exists.

Exact duplicate episode IDs with the same SHA-256 are skipped. Top-level `observation.logs` is removed by default to reduce event-log leakage. Winner decisions receive policy weight; both winner and loser decisions with known outcomes receive value weight.

## Progress already on `exp/league-v1`

- DATA-001: `scripts/audit_replay_alignment.py` compares `previous` and `same`, uses the previous-step submission status, reports skipped placeholders, and supports a strict exit gate.
- Action-position diagnosis: `scripts/diagnose_replay_action_positions.py` traces empty required actions across configurable lags without changing the gate.
- DATA-002: `scripts/convert_public_replays.py` validates, deduplicates, converts, and atomically promotes canonical JSONL/JSONL.GZ trajectories.
- Shared replay logic: `rl/public_replay.py` implements temporal pairing, validation, terminal outcomes, policy/value weights, hashes, and log stripping.
- The majority baseline now uses previous-step labels. Legacy unvalidated majority models and unseen signatures are rejected instead of using an unsafe global default.
- RL proposal: `docs/PUBLIC_REPLAY_RL_PROPOSAL.md` recommends masked behavior cloning, then AWR or IQL with BC regularization, followed by recurrent PPO self-play and league evaluation.
- Experiment log: the first real DATA-001/002 pass is recorded; terminal-outcome hardening is marked `implemented_pending_server` until an independent snapshot passes.
- Local verification: 13 unit tests and compile checks pass. Tests cover reward win/loss, draw, result fallback only when reward is absent, ambiguous reward with a populated result, top-level reward mismatch, and failure to promote mismatched output.

Verified real-data history: the original 100-episode DATA-001 run failed (`previous=0.5040`, `same=0.3975`). After correcting the submission-status position, the same `2026-07-18` sample passed with `previous=1.0`, 16,111/16,111 valid decisions, 24 legal optional empty actions, and zero invalid decisions. DATA-002 then passed after resolving terminal winners from reward: 16,111 rows, 8,397 winner-policy rows, zero missing winners, and zero invalid decisions. Because all fixes were validated on the same sample, a different-date pass is still required.

The required independent pass is now complete on the different-date `2026-07-19` snapshot. The download produced 500 episode JSON files plus `manifest.csv`; no `2026-07-18` directory was copied or renamed. An initial attempt read the directory while the downloader was still writing and produced inconsistent 112/118/126-file reports plus one transient JSON load error. Those reports were rejected and overwritten only after the downloader exited and the directory stabilized at 500 JSON files.

At code commit `ce10e356a2135ec5cfc1b58f212f48e41d4acf51`, the stable 500-episode rerun passed all gates: action-position lag 1 was uniquely best at `1.0` (78,776/78,776, zero invalid, 569 optional empty actions, zero unresolved required-empty actions), and strict DATA-001 reported `previous=1.0`, `same=0.7853322569`, zero unknown statuses, and zero load errors. DATA-002 produced 78,776 schema-v2 rows, including 41,598 winner-policy rows and 78,776 value rows, with zero invalid decisions, missing/unresolved winners, reward mismatches, duplicate/conflicting IDs, unknown statuses, or load errors. All 500 outcomes came from terminal reward, all 500 agreed with top-level reward, and result fallback was never used. An independent streaming gzip read parsed all 78,776 rows, found no `observation.logs`, and confirmed all 569 empty actions had `minCount=0`; episode winners were player 0 in 276 episodes and player 1 in 224.

Current verification: 26 unit tests pass and compile checks pass. Timed stable runs were action-position 2:42.65 / 79,416 KB max RSS, DATA-001 0:58.66 / 86,624 KB, DATA-002 1:46.48 / 91,596 KB, tests 0:50.98 / 643,516 KB, and compileall 0:00.34 / 12,348 KB. Download timing and memory were not captured because the downloader had already been started before this continuation; the resulting race was detected rather than hidden.

## Completed server task and current handoff state

The second independent real DATA-001/002 validation is complete and evidence-backed on `exp/league-v1`. The reports under `results/` now describe the stable 500-episode rerun, not the rejected partial-directory attempt. RL-BC-001 is eligible as the next experiment, but was intentionally not started during this validation run.

For provenance, the commands used for the completed validation are retained below.

First synchronize and verify scope:

```bash
git fetch origin
git switch exp/league-v1
git pull --ff-only
git status --short --branch
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q rl scripts tests
```

Do not continue if the current branch is `main` or if unrelated local changes would be overwritten. Preserve all user changes.

Download a bounded real replay sample from a date other than `2026-07-18`, identify the resulting date directory, and run DATA-001. Prefer 100–500 episodes; do not reuse the prior files under a new directory name:

```bash
python scripts/download_ptcg_data.py --max-episodes 500
find data/raw/replays -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort

DATE=YYYY-MM-DD
python scripts/diagnose_replay_action_positions.py \
  --date "$DATE" \
  --max-files 500 \
  --min-lag -1 \
  --max-lag 4 \
  --expected-lag 1 \
  --output results/data001_action_positions.json

python scripts/audit_replay_alignment.py \
  --date "$DATE" \
  --max-files 500 \
  --min-valid-rate 0.999 \
  --strict \
  --output results/data001_replay_alignment.json
```

Inspect both reports before DATA-002. Expected evidence is that lag 1 and `previous` dominate, `gate_passed` is `true`, unknown submission statuses are zero, and load errors are zero. Review `empty_required_expected_actions`, `empty_required_unresolved`, and the trace examples even when the aggregate rate passes. If another lag wins, the valid rate is below `0.999`, or errors cluster by a reason, stop and inspect representative replay steps. Do not hide bad rows, relax the threshold, or start training.

Only after DATA-001 passes, convert all validated replays for that date:

```bash
python scripts/convert_public_replays.py \
  --date "$DATE" \
  --alignment previous \
  --policy-source winners \
  --max-invalid-rate 0 \
  --output data/processed/public_replay_v1.jsonl.gz \
  --report results/data002_public_replay_conversion.json

python - <<'PY'
import gzip, json
from pathlib import Path

path = Path("data/processed/public_replay_v1.jsonl.gz")
rows = 0
with gzip.open(path, "rt", encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        assert "logs" not in row["observation"]
        rows += 1
print({"readable_rows": rows, "bytes": path.stat().st_size})
PY
```

Before accepting DATA-002, require `episodes_missing_winner=0`, `episodes_unresolved_terminal_reward=0`, `reward_mismatches=0`, `unknown_submission_status_skipped=0`, `conflicting_episode_ids=0`, and `load_errors=0`. For current real replay schema, also expect `episodes_terminal_reward == episodes`, `top_level_reward_episodes == episodes`, and `episodes_result_fallback=0`; any deviation must be explained from representative raw episodes.

Record the exact input date/snapshot, replay and row counts, per-lag and both alignment rates, empty-action traces, skip/status counts, invalid-reason counts, duplicate/conflict counts, missing-winner counts, policy/value row counts, elapsed time, peak memory, failures, commands, and commit SHA. Update:

- `results/experiments.csv`;
- `docs/DECISION_LOG.md`;
- `docs/FAILURE_MODES.md` if any new schema or operational failure appears.

Clearly label facts, evidence-based inferences, unverified hypotheses, and the next experiment. Commit and push only to `exp/league-v1`; do not create a PR.

## Next modeling experiments after real DATA-001/002 pass

Only start these experiments after the second independent real DATA-001/002 pass.

1. **RL-BC-001: masked behavior cloning baseline.** Resolve selectable options semantically, encode categorical state, use a recurrent history encoder and variable-candidate pointer, and decode multi-select actions autoregressively with a stop action. Train policy loss on winner rows; train a value head on both sides. Always apply the legal mask and retain the rule fallback.
2. **RL-OFFLINE-001: conservative replay RL.** Compare AWR and IQL against RL-BC-001 using the same train/validation episode split. Keep a BC regularizer and cap advantage weights. Do not use unconstrained DQN on this heterogeneous logged dataset.
3. **RL-SELFPLAY-001: recurrent PPO plus league.** Initialize from the best validated offline checkpoint, then use legal masks, GAE, entropy regularization, rule fallback, historical agents, and exploiters.

Every comparison must use disjoint episode IDs, fixed seeds, common random seeds, and paired seat swaps. Report confidence intervals and operational failures, not only average return.

League promotion remains separate from DATA-001/002: use 10 learners, 400 games per public-opponent matchup by default, win rate at least 60%, 95% Wilson lower bound at least 55%, and zero crash/timeout/illegal actions. Every one of the 10 learners must pass every immutable date+hash public-opponent snapshot before evolution. Do not claim promotion from data-gate results.

## Definition of done for this handoff

Satisfied: the different-date server evidence covers a stable 500-episode snapshot, both real DATA-001 and DATA-002 gates pass without relaxed thresholds or filtered failures, and the reports plus logs are ready to commit on `exp/league-v1`. RL-BC-001 was not started in this run.

## RL-BC-001 update (2026-07-20)

The minimal stateless masked BC baseline and provenance/resume hardening are implemented. Code commit `af677c6cffb720839f7733cc446483f599e4d3b1` was pushed before a clean, from-scratch seed-20260720 run. See `results/rl_bc_001_metrics.json`, `results/rl_bc_001_split.json`, and `results/rl_bc_001_runs.csv`: the report records dirty=false, the exact code/input SHAs, 30 full epochs, complete resume state, and status `partial_formal` with missing seeds 17 and 42. Episode leakage, invalid decoding, unsupported rows, skipped rows, NaN, and final runtime failures are zero. Validation sequence/set exact are 0.443173/0.449077, multi-select accuracy is 0.494709, and optional-empty accuracy is 0.371795. The best checkpoint is gitignored and fingerprinted in the metrics report. Do not enter AWR/IQL; finish the missing formal seeds and continue BC/history work first.

## RL-BC-001 strict-config completion (authoritative 2026-07-20 update)

The paragraph above is retained as experiment history, but its batch512 run is not a formal seed: it mismatched the planned batch size 256. Its complete original evidence is preserved at `results/rl_bc_001_exploratory_seed20260720_bs512.json` with status `exploratory_config_mismatch`, `achieved_seeds=[]`, and `missing_seeds=[17,42,20260720]`.

Configuration/fingerprint enforcement was committed and pushed first as `0a5603f6e611d52cea7ad0eba2b456bb1d1b4c00`. The trainer now matches input SHA, split seed/fraction, epochs, batch size, learning rate, hidden dimension, patience, value-loss weight, gradient clipping, and architecture. Resume requires matching code commit, input SHA, and experiment fingerprint; aggregation only merges identical fingerprints.

From that clean code commit, seeds 17, 42, and 20260720 ran from scratch using exact `configs/rl_bc_001.json` settings (`batch_size=256`, 30 epochs). All share fingerprint `8a3a4f17f514f6aec6b5cc89b1f29e031e53ecb0a3225f752c36e411f1ded6a9`; all completed 30 full epochs with no resume and no dirty state. The final aggregate status is `completed_formal`, achieved seeds are `[17,42,20260720]`, and missing seeds are empty.

The fixed split is 400 train / 100 validation episodes, 63328 / 15448 rows, with no episode overlap. Across the three seeds, mean validation total/policy/value losses are 0.892082/0.699003/0.772319; mean sequence/set exact matches are 0.448626/0.454818; mean single/empty/multi-select accuracies are 0.449056/0.354701/0.473545. All three decode every one of 15448 validation rows legally; invalid actions, unsupported rows, skipped rows, and NaN/runtime failures are zero.

Detailed evidence is in `results/rl_bc_001_metrics.json`, per-seed raw reports `results/rl_bc_001_seed{17,42,20260720}_metrics.json`, the full episode manifest `results/rl_bc_001_split.json`, and `results/rl_bc_001_runs.csv`. Checkpoints remain gitignored under `checkpoints/rl_bc_001/0a5603f/`; their hashes and relative paths are recorded in the metrics. No game smoke was run because checkpoint-to-agent integration was not expanded. Do not enter AWR/IQL yet; the next recommendation is a BC-only history-encoder comparison, especially because optional-empty accuracy remains weak.

Final verification passed 47 unit tests plus `python -m compileall -q rl scripts tests`. A separate audit parsed every JSON/CSV, asserted all numeric values finite, checked the one completed formal experiment-log row points to code commit `0a5603f6`, and recomputed all three checkpoint SHA-256 values successfully.

## RL-BC-002 completed formal comparison (2026-07-21)

RL-BC-001 remains frozen. RL-BC-002 audit/config commit is `7780f9f`; implementation/test commit is `02782c3e881a3b7fa7e75c36e2a95fff76b873cf`. Formal runs started from that clean code commit with input SHA `2768c08ae2451f50b5382ad8eed0f3db2cc1facab53d8b25b19aeed39850b694`, the unchanged 400/100 episode split, batch size 256, and seeds 17/42/20260720. No run resumed RL-BC-001.

The history schema audit passed before implementation. All 78776 rows have explicit episode/player/action-step/observation-step keys, no duplicate group order key, lag one on every row, and zero player identity mismatch. The GRU arm groups by `(episode_id, player)`, sorts on explicit `action_step`, and uses only earlier rows. Its 16-token window produced 1124531 prior tokens for 77776 rows; file order and current/future steps were never used.

Arm A (`RL-BC-002-A`, fingerprint `b1fcdb91...`) completed 60 epochs for all seeds with best epochs 59/60/59. Mean total/policy/value loss is 0.855092/0.662517/0.770299. Mean sequence/set exact is 0.485773/0.492046; candidate precision/recall 0.520294/0.522584; single/empty/multi accuracy 0.489733/0.256410/0.468254. Overall exact metrics improve over RL-BC-001 in 3/3 seeds, but empty and multi-select improve in only 1/3. Best epochs at the upper boundary and decreasing late losses mean 60 epochs is not convincingly converged.

Arm B (`RL-BC-002-B`, fingerprint `1d69ddce...`) early-stopped at 31/25/29 epochs with best epochs 21/15/19. Mean total/policy/value loss is 0.894815/0.700314/0.778004. Mean sequence/set exact is 0.454080/0.460599; candidate precision/recall 0.489778/0.490901; single/empty/multi accuracy 0.453319/0.465812/0.481481. Compared with Arm A, overall loss and exact metrics regress in every seed, while empty and multi-select improve in every seed. This GRU is overfitting and is not the preferred overall checkpoint.

Both arms have legal rate 1.0 and zero invalid, unsupported, skipped, NaN, OOM, or runtime failures. Aggregate runtimes are 3116.96 seconds for A and 2035.06 seconds for B (B stopped early); peak RAM/VRAM are 1989.16/595.69 MB and 2211.76/609.70 MB. Detailed means, sample standard deviations, min/max, per-select type/context results, last-five-epoch trends, checkpoint paths, and hashes are in `results/rl_bc_002_stateless_long_metrics.json`, `results/rl_bc_002_history_gru_metrics.json`, `results/rl_bc_002_comparison.json`, the six raw seed reports, and `results/rl_bc_002_runs.csv`.

The offline prerequisites for a separate checkpoint-to-agent adapter smoke are satisfied. The official local `cg` engine was not found under the repository or a bounded `/homes/lzhang` search, so gameplay cannot be claimed until an engine path is supplied or restored. Prefer Arm A for that functional smoke, retain the legal mask and lucario rule fallback, and record gameplay separately. Do not begin AWR, IQL, PPO, self-play, league training, or opponent-pool updates.

Adapter code is now committed as `00d116bdb63549dfa8807ea0ef939761fcf327f7` under `agents/rl_bc_adapter/` and `rl/agent_adapter.py`. It defaults to the Arm A seed20260720 best checkpoint, preserves the masked decoder, uses lucario rule fallback on model failure, validates fallback structure, and reports model/fallback/error/illegal counters through `scripts/run_local_match.py`.

The clean-commit functional smoke loaded checkpoint SHA `2faac94de9e937dee77cd6d5d44036d7f45bb2dc4cc6491c1c97c0091f4fb216` and returned legal action `[0]` for a `minCount=maxCount=1` selection. Model actions were 1; fallback, load, inference, illegal-model, illegal-fallback, and emergency counts were all zero. The final full suite passes 62 tests and compileall.

Two preceding temporary harness attempts failed before model import: the first lost Python string quoting through SSH, and the second ran a `/tmp` script without the repository on `PYTHONPATH`. Both are retained in the adapter report; neither executed checkpoint inference. The explicit-PYTHONPATH rerun from clean adapter commit passed.

Actual gameplay remains explicitly blocked: no official `cg/api.py` was found in repository `tmp` or a bounded `/homes/lzhang` depth-six search. `results/rl_bc_002_adapter_smoke.json` records games=0 and no gameplay claim. Once a valid engine path is available, run a separate small, fixed-seed, seat-swapped smoke against `official_random` and `lucario_rule`, retaining fallback diagnostics. This does not authorize AWR, IQL, PPO, self-play, or league work.

## RL-BC-002 official engine recovery audit (2026-07-21)

The missing engine was recovered directly from the official Kaggle competition download. The 315883380-byte archive SHA is `09ad210b...c3282`; sample-submission `api.py/game.py/sim.py/libcg.so` hashes and the isolated `kaggle-environments==1.32.0` wheel/runtime hashes are recorded in `results/rl_bc_002_engine_audit.json`. No engine files or large archives are committed.

Do not start model matches yet. Official-random validation produced 8/8 normal terminations with zero crash/timeout/invalid/network use, but every one of four repeated Python-seed pairs produced a different full trace and step count, and two pairs changed winner. The supplied native source explains this: `BattleStart` accepts no seed and uses `std::random_device()` internally. In addition, the competition archive and CABT runtime `libcg.so` hashes differ, and CABT exposes no explicit termination reason even though rewards/status and turn are readable. `results/rl_bc_002_engine_blocker.json` is authoritative. Unblock only with a pinned official runtime plus an official seedable interface or explicitly approved unseeded protocol and termination definition. The frozen Arm A checkpoint remains unchanged and has not played a real game.

Final verification passed 62 unit tests (11.93 seconds, 696784 KB max RSS) and compileall (0.11 seconds, 12920 KB max RSS). No production game-runner code was added after the engine gate failed.
