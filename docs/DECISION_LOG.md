# Decision log

## 2026-07-19 — League-v1 bootstrap

- Use 10 learners with independent seeds and hyperparameter/opponent-sampling diversity.
- Refresh public opponents daily but train and evaluate against immutable, hashed snapshots.
- Require every learner to pass every public opponent; use paired seats/common seeds and a Wilson lower-bound gate.
- Keep champion and historical agents in the league after public-pool mastery to reduce cycling and forgetting.
- Do not merge, submit, or claim training success before server gameplay evaluation.

## 2026-07-19 — DATA-001/002 public replay gate

- Treat replay action/observation alignment as an experiment, not an assumed schema. Compare previous-step and same-step candidates on real episodes and require at least 99.9% legal labels.
- Do not train from the committed legacy majority model; its label/action consistency audit failed.
- Standardize public replay decisions before modeling. Each row contains only the acting player's view, the legal option set, the selected action, outcome, manifest metadata, and source hash.
- Remove `observation.logs` from model input by default and never combine both players' views.
- Give policy loss weight to winning strong-teacher decisions; retain losing trajectories for value learning and failure analysis.
- Keep DATA-001/002 on `exp/league-v1`. Real-data results must be produced on the server before any merge recommendation.

## 2026-07-19 — DATA-001 real replay result: gate failed, do not proceed to DATA-002

- Downloaded 100 real episodes for `2026-07-18` via `scripts/download_ptcg_data.py --max-episodes 100` (`kaggle/pokemon-tcg-ai-battle-episodes-2026-07-18`) and ran the strict DATA-001 audit at commit `c3cef894decd0977f604c3ea0e5a312320d1ef87`.
- **Fact:** `previous` was better than `same` (0.5040 versus 0.3975), but both failed the required `>= 0.999` gate with zero load errors.
- **Fact:** DATA-002 was not run after DATA-001 failed.
- The first diagnostic filtered transitions using the status stored beside the action and raised `previous` to 0.8917, but still left many required selections paired with empty actions.

## 2026-07-19 — Correct action-submission status and add lag tracing

- **Verified code defect:** Kaggle Environments requests the action from the state before `Environment.step`, runs the interpreter, and then appends the resulting state with that action. Therefore `steps[t].action` was submitted under `steps[t-1].status`; `steps[t].status` is post-interpreter state. The first real-data diagnostic filtered the wrong status position.
- Change transition extraction, DATA-001, and DATA-002 to use `steps[action_step-1].status == "ACTIVE"`. Reset-step, inactive, and unknown-status actions are counted separately rather than silently discarded.
- Add `scripts/diagnose_replay_action_positions.py` to compare lag `action_step - observation_step` across a configurable window and trace every required-select empty action to any alternative valid lag. Alternative lags are diagnostic only and never relax DATA-001/002 automatically.
- Store `submission_status`, `observation_status`, and `action_status` separately in canonical schema version 2. DATA-002 now fails when any non-initial action has unknown submission status.
- **Synthetic verification only:** seven unit tests, compile checks, DATA-001, action-position diagnostics, and DATA-002 pass. A fresh run on the same 100 real episodes is required before deciding whether the empty-action problem is resolved.

## 2026-07-20 — Real rerun: DATA-001 passes; DATA-002 blocked, then fixed and passes

- Reused the same 100-episode `2026-07-18` local sample (`data/raw/replays/2026-07-18`, gitignored) so the before/after comparison is apples-to-apples, rather than downloading a new random sample.
- **Fact:** `scripts/diagnose_replay_action_positions.py --min-lag -1 --max-lag 4 --expected-lag 1` at commit `cdf5148dcffbc5a9f0c47ad67f1edabfb60f0b47` confirms lag 1 (== `previous` alignment) is uniquely correct: `valid_rate=1.0` (16111 valid, 0 invalid, 200 setup), while lags -1/0/2/3/4 all score 0.61-0.68. `empty_required_expected_actions=0` and `empty_required_unresolved=0` — no required selection anywhere lacks a resolvable action at lag 1.
- **Fact:** `scripts/audit_replay_alignment.py --strict --min-valid-rate 0.999` at the same commit passes: `previous` valid_rate=1.0 (16111/16111), `same`=0.7842, `unknown_submission_status_skipped=0`, `load_errors=0`, `gate_passed=true`. This directly confirms the status-position fix recorded above against real data, not just synthetic fixtures.
- **Fact:** `scripts/convert_public_replays.py --alignment previous --policy-source winners --max-invalid-rate 0` at commit `cdf5148` **failed** with `episodes_missing_winner=100` (all 100 episodes) despite `invalid_decisions=0`.
- **Verified code defect:** `terminal_winner()` read `observation.current.result`, which is `-1` in every one of 32246 occurrences across the 100-episode sample — it never carries a real outcome in this replay format. The real signal is each player's `reward` at the step where their `status` becomes `DONE`: it matches the episode's top-level `rewards` array with 0 mismatches across all 100 episodes (only clean `{1,-1}` pairs observed; no draws in this sample).
- **Decision:** fixed `terminal_winner()` at commit `6a1d619ed84993de2f1d5cf5fdf407fe4ab678f0` to prefer reward-based resolution at the terminal `DONE` step, falling back to the old `result`-based scan only when no reward is present (preserving the existing synthetic fixtures, which carry no `reward` key). It returns `None` — not a guess — for any reward pattern that isn't a clean one-winner/one-loser split or an all-zero draw, so DATA-002's missing-winner gate still catches genuinely ambiguous episodes. Added four regression tests (`tests/test_public_replay.py`) covering reward-resolved win/loss, draw, fallback-to-result, and the ambiguous/unresolved case.
- **Fact:** rerunning DATA-002 at commit `6a1d619` on the same 100-episode sample passes: 16111 rows, 8397 policy rows (winner decisions only), 16111 value rows, `invalid_decisions=0`, 0 missing-winner episodes, `gate_passed=true`. The promoted `data/processed/public_replay_v1.jsonl.gz` was independently verified: 16111 readable rows, `schema_version=2`, `observation.logs` absent from every row.
- Elapsed (single-run, not repeated for statistics): action-position diagnostic ~46.5s, DATA-001 ~31.2s, DATA-002 (passing run) ~166.5s, all over the same 100-episode/101-file local sample. Peak memory was not instrumented.
- **Next experiment:** DATA-001 and DATA-002 have now passed once on real data. Before starting RL-BC-001, get a second independent real-data pass — ideally a different date's snapshot — to confirm this isn't specific to the `2026-07-18` sample, since both fixes were validated against the same single 100-episode pull.

## 2026-07-20 — Harden terminal outcome audit before the independent rerun

- **Code-review finding:** `_terminal_winner_from_reward()` returned `None` for both “no terminal reward” and “terminal reward exists but is ambiguous”. `terminal_winner()` therefore fell back to `observation.current.result` in both cases, contradicting the intended rule that ambiguous reward evidence must remain unresolved rather than be overridden.
- **Decision:** terminal reward parsing now returns separate presence and validity states. Result fallback is allowed only when no terminal reward field exists. Numeric rewards must also be finite.
- **Decision:** DATA-002 now compares validated terminal per-player rewards with top-level `rewards` whenever both are present. Any missing/invalid mapping disagreement increments `reward_mismatches` and blocks output promotion.
- DATA-002 reports all audit counters explicitly, including zero values: missing winners, unresolved terminal rewards, reward mismatches, duplicate/conflicting episode IDs, unknown submission statuses, and load errors.
- **Synthetic verification:** 13 unit tests pass, including ambiguous reward plus populated result, top-level reward mismatch, and an end-to-end assertion that a mismatch fails the DATA-002 gate and does not promote its temporary output. Compile checks pass; a clean synthetic conversion still passes with four rows and zero invalid decisions.
- **Limitation:** the current environment does not contain a second independent real replay snapshot. These changes are `implemented_pending_server`; do not start RL-BC-001 until a different-date real run confirms DATA-001 and DATA-002 again with `reward_mismatches=0`.

## 2026-07-20 — Independent different-date server rerun passes DATA-001 and DATA-002

- Executed on server doraemon19:/homes/lzhang/pocketmon at commit ce10e356a2135ec5cfc1b58f212f48e41d4acf51 on branch exp/league-v1; no switch to main.
- Downloaded a different-date real snapshot using python scripts/download_ptcg_data.py --max-episodes 500; selected date 2026-07-19 with exactly 500 episode JSON files plus manifest.csv. No reuse or rename of 2026-07-18 files.
- Operational failure caught before acceptance: the downloader was still writing while an inherited validation attempt ran. Its reports selected only 112/118/126 files, and the action-position report saw one transient partial-JSON load error. Those reports were rejected; after confirming no downloader or validator remained and the directory was stable at 500 JSON files, every gate was rerun from scratch without deleting or filtering any episode.
- Fact: DATA-001 strict audit passes on the stable snapshot: previous valid_rate=1.0 (78776/78776), same=0.7853322569, unknown_submission_status_skipped=0, load_errors=0, gate_passed=true.
- Fact: Action-position diagnosis loads 500/500 files and confirms best_observed_lag=1. Lag rates for -1/0/1/2/3/4 are 0.6265543522 / 0.6729592860 / 1.0 / 0.6756794023 / 0.6641220417 / 0.6134927798; lag 1 has invalid_decisions=0 and empty_required_unresolved=0.
- Fact: DATA-002 strict conversion passes on all 500 episodes: rows=78776, policy_rows=41598, value_rows=78776, invalid_decisions=0, invalid_rate=0, episodes_missing_winner=0, episodes_unresolved_terminal_reward=0, reward_mismatches=0, unknown_submission_status_skipped=0, conflicting_episode_ids=0, load_errors=0, gate_passed=true.
- Fact: Reward-source requirements hold: episodes_terminal_reward == episodes == 500, top_level_reward_episodes == episodes, episodes_result_fallback=0. Independent output aggregation finds player 0 won 276 episodes and player 1 won 224; row outcomes are 41598 positive and 37178 negative.
- Fact: Optional empty actions remain legal evidence, not failures: DATA-001 lag-1 and DATA-002 both contain 569 empty actions, all independently confirmed to have minCount=0, with zero invalid or unresolved required-empty cases.
- Fact: Independent gzip integrity check passes: all 78776 rows parse, every row has schema_version=2, and observation.logs is absent from all rows.
- Elapsed and peak memory (single stable run): action-position 2:42.65, max RSS 79416 KB; DATA-001 audit 0:58.66, max RSS 86624 KB; DATA-002 conversion 1:46.48, max RSS 91596 KB; 26 unit tests 0:50.98, max RSS 643516 KB; compileall 0:00.34, max RSS 12348 KB. Download time/RSS were not captured because it had already been started before this continuation.
- Decision: The required second independent real DATA-001 and DATA-002 pass is complete and evidence-backed. Do not start RL-BC-001 in this same run; hand off with committed server evidence first.

## 2026-07-20 — RL-BC-001 stateless masked behavior-cloning baseline

- Fixed episode split uses seed 20260720: 400 train episodes / 63328 rows and 100 validation episodes / 15448 rows, with zero episode overlap and all 78776 rows represented. Input gzip SHA-256 is `2768c08ae2451f50b5382ad8eed0f3db2cc1facab53d8b25b19aeed39850b694` (19416172 bytes).
- Implemented a stateless state encoder, per-candidate encoder, autoregressive masked pointer with explicit STOP, and value head. Teacher forcing preserves replay order; STOP is masked below minCount, candidates are masked after selection, and maxCount forces STOP.
- Policy loss uses only winner rows (`policy_weight=1`); value MSE uses both players (`value_weight=1`). No label/post-action fields or logs enter model inputs. All 569 empty and 3656 multi-select rows are retained; unsupported/skipped rows are zero.
- Review hardening was committed first as `af677c6cffb720839f7733cc446483f599e4d3b1`. Formal training now refuses a dirty worktree, records code commit/dirty/input SHA, preserves complete best/history/stale/optimizer-step state on resume, excludes partial batches from completed epochs, and rejects evaluation input-SHA drift by default.
- Seed 20260720 was then retrained from random initialization on that clean code commit with no `--resume`: all 30 planned epochs completed, best epoch=30, optimizer steps=3720, partial history=0, dirty=false. Status is correctly `partial_formal`, not completed, because planned seeds 17 and 42 remain missing.
- Validation loss=0.903794840 (policy=0.710068977, value=0.774903455), sequence exact=0.443173432, set exact=0.449077491, empty accuracy=0.371794872, multi-select accuracy=0.494708995, candidate precision/recall=0.479496361/0.481163788.
- Hard offline gate passes: all 15448 validation rows decode legally, legal rate=1.0, invalid actions=0, unsupported/skipped rows=0. Independent checkpoint reload reproduced the metrics with matching input SHA.
- Best checkpoint is gitignored at `checkpoints/rl_bc_001/seed_20260720_best.pt`, SHA-256 `13a1d3bf471e6ce6d4d1c36faa07d26b74f10b0c379eb8ddf01eb72682971829`. Training took 523.31s internally (8:58.88 wall), peak RAM 1989.53 MB and peak VRAM 1052.16 MB on doraemon03 RTX 3090.
- Failures retained from initial implementation: CUDA peak-memory statistics required setting the current device first; mixed action lengths initially produced all-masked inactive logits and NaN; doraemon02/04 GPUs are sm_61 and incompatible with the installed PyTorch. All were diagnosed without filtering data.
- Decision: provenance/resume review gates and offline legality pass, but the formal plan is incomplete until seeds 17 and 42 run. Do not enter AWR/IQL; continue improving and completing BC evidence first. No game smoke was attempted because checkpoint-to-agent integration remains separately scoped.
