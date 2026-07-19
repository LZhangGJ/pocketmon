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
