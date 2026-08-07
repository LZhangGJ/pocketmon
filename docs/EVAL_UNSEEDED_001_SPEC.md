# EVAL-UNSEEDED-001: pinned official-engine protocol validation

Status: authorized for implementation and server validation only
Branch: `exp/league-v1`
Baseline commit: `de32f9901e7da71ea2284af43e9f5d56b38c6e1c`
Date: 2026-07-21

## 1. Objective

Determine whether the competition-supplied Pokémon TCG engine can support a reproducible **evaluation procedure** even though individual battles are not seedable or replay-identical.

This experiment does not change training, the frozen RL-BC-002 checkpoint, or the champion. It does not authorize AWR, IQL, PPO, self-play, league training, opponent-pool updates, a Pull Request, a merge to `main`, or a Kaggle submission.

## 2. Verified starting facts

- The official competition archive was downloaded through the Kaggle competition endpoint.
- Archive SHA-256: `09ad210b15476f5064c1509addb32a459c777d92d4e4e7db470f9d0c039c3282`.
- Competition sample-submission Linux `libcg.so` SHA-256: `feafd4046b2f688bdb33a4972c139b78e13e243ab5707ece52c43cf39a34b887`.
- `kaggle-environments==1.32.0` wheel SHA-256: `359226741a04fbe1dbbc10121aef140fd96ec4fa31bace2037d05e7ef2bbf4e8`.
- Its bundled CABT Linux `libcg.so` SHA-256 is different: `7acbfc7bc61d4f8233515c63debcfa454b8f804f138a6c395c599decc3dd17d0`.
- `BattleStart` exposes no seed and initializes native randomness through `std::random_device()`.
- Python seeding therefore controls policy-side sampling only, not the battle trajectory.
- The previous audit produced 8/8 normal official-random terminations, but 0/4 repeated-seed trace matches and 0/4 repeated-seed step-count matches.
- RL-BC-002-A seed-20260720 checkpoint remains frozen with SHA-256 `2faac94de9e937dee77cd6d5d44036d7f45bb2dc4cc6491c1c97c0091f4fb216`.

## 3. Core hypothesis

A statistically valid local evaluation procedure is possible without seedable battles if and only if:

1. the exact native engine used for every game is pinned and verified;
2. games are treated as independent stochastic trials, never paired by nominal Python seed;
3. schedule order is interleaved and seats are balanced;
4. terminal outcome is defined from official framework status and reward, not from the stale `current.result` field;
5. all crashes, timeouts, invalid actions, load failures, inference failures, and fallbacks are retained as failures rather than silently rerun;
6. confidence intervals and raw game records are reported.

## 4. Stage A: runtime identity and terminal-contract gate

### 4.1 Required implementation

Add the minimum necessary code and tests for a server-side evaluator that:

- accepts explicit paths to the unpacked competition archive and isolated `kaggle-environments==1.32.0` runtime;
- verifies every expected SHA-256 before importing the environment;
- records resolved Python module paths and the native library path actually loaded by the process;
- fails closed if the loaded native library cannot be proven to have SHA-256 `feafd4046b2f688bdb33a4972c139b78e13e243ab5707ece52c43cf39a34b887`;
- does not modify the archive, engine, shared conda environment, or system Python packages;
- disables network access during games;
- runs agents in the existing repository without copying secrets or checkpoints into Git;
- records final agent statuses, rewards, turn count, step/decision count, elapsed time, and diagnostics;
- never uses nominal Python seed as an engine-seed or pairing key.

Inspect the official archive's `sample_submission/cg/game.py` and `sim.py` before choosing the loop. Prefer the official competition wrapper. Do not patch `libcg.so`, reverse-engineer a seed hook, monkey-patch `std::random_device`, or claim equivalence between the two different native-library hashes.

### 4.2 Approved terminal definition

A game is a normal terminal episode only when all of the following are true:

- both final framework statuses are `DONE`;
- rewards are finite and exactly one of `[1, -1]`, `[-1, 1]`, or `[0, 0]`;
- the environment loop returned normally before `episodeSteps` and `runTimeout`;
- neither agent status is `ERROR`, `INVALID`, or `TIMEOUT`;
- no uncaught exception or process-level crash occurred.

The outcome is read from the reward vector. Record the generic terminal class as `status_reward_terminal`. Do not fabricate a card-game-specific termination reason. The stale final observation value `current.result == -1` is recorded but is not used to determine the winner.

### 4.3 Validation workload

Run 20 `official_random` versus `official_random` games:

- 10 with the first loaded agent in seat 0 and 10 in seat 1;
- alternate seat assignment every game;
- run in one immutable environment with the same verified archive/runtime hashes;
- retain every game, including operational failures;
- do not rerun or replace a failed game.

### 4.4 Stage A acceptance criteria

All criteria are mandatory:

- actual loaded native-library hash equals the competition archive hash;
- archive, Python wrapper, wheel, runtime, checkpoint, code commit, host, Python version, and commands are recorded;
- 20/20 games satisfy the approved terminal definition;
- crash = 0;
- timeout = 0;
- invalid action = 0;
- agent error = 0;
- network attempt = 0;
- no game is dropped, overwritten, or selectively rerun;
- unit tests and `python -m compileall -q rl scripts tests` pass;
- peak RSS and elapsed time are recorded.

If the exact archive native library cannot be loaded through an official wrapper, Stage A fails. Stop without model games and document the precise import/loading conflict.

## 5. Stage B: RL-BC-002-A integration smoke

Stage B is authorized only after Stage A passes from a clean commit.

Use the frozen Arm A seed-20260720 checkpoint and existing masked adapter with lucario-rule fallback. Run exactly four integration games:

1. RL-BC-002-A seat 0 vs `official_random` seat 1;
2. `official_random` seat 0 vs RL-BC-002-A seat 1;
3. RL-BC-002-A seat 0 vs `lucario_rule` seat 1;
4. `lucario_rule` seat 0 vs RL-BC-002-A seat 1.

This is an operational smoke, not a strength estimate. Do not rerun failed games.

### 5.1 Stage B acceptance criteria

- 4/4 games satisfy the approved terminal definition;
- model checkpoint hash matches the frozen SHA-256;
- model load errors = 0;
- inference errors = 0;
- illegal model actions = 0;
- illegal fallback actions = 0;
- emergency legal actions = 0;
- crash/timeout/invalid = 0;
- per-game model-action and fallback-action counts are recorded;
- p50/p95 decision latency, total elapsed time, peak RSS, and peak VRAM are recorded.

Any fallback action is evidence to investigate. It does not automatically invalidate the smoke, but Stage B cannot be described as pure-model gameplay and no strength claim is permitted.

## 6. Later statistical pilot, not yet authorized

Do not start the 40-game-per-opponent pilot in this task. After Stage B passes and its diff is reviewed, a separate authorization may allow an independent-trial, interleaved, seat-balanced pilot against `official_random` and `lucario_rule`.

Because native randomness is unseeded, that future pilot must use Wilson confidence intervals and must not call games paired or common-random-number comparisons. Forty games per opponent may establish operational behavior only; it is insufficient for champion replacement or league promotion.

## 7. Required files and records

Implementation should use the smallest necessary file set and add tests. On completion, update:

- `results/eval_unseeded_001_runtime_gate.json`;
- `results/eval_unseeded_001_games.jsonl`;
- `results/eval_unseeded_001_summary.json`;
- `results/experiments.csv`;
- `docs/DECISION_LOG.md`;
- `docs/FAILURE_MODES.md` only if a new failure mode is observed;
- `docs/CODEX_CLI_HANDOFF.md`.

Raw records must include sequential game ID, schedule position, seats, agent identities, statuses, rewards, outcome, terminal class, turn/steps, elapsed time, diagnostics, module/native paths, and all relevant hashes.

## 8. Required conclusion structure

Separate the final report into:

- verified facts;
- experiment-supported inferences;
- unverified hypotheses;
- recommended next experiment.

Explicitly state:

- whether Stage A passed;
- whether Stage B was allowed and passed;
- model game count;
- whether any gameplay-strength claim is justified;
- whether merge is recommended.

The expected merge recommendation for this protocol-validation task is `no`.