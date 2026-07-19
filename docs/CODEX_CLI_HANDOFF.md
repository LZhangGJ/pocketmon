# Local Codex CLI handoff: public replay experiments

Last updated: 2026-07-19

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

## Immediate server task for local Codex CLI

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

The local Codex CLI should finish by pushing a different-date server-evidence commit on `exp/league-v1` that contains real DATA-001 and DATA-002 reports plus updated logs. If either gate fails, push the failure evidence and diagnosis instead of producing or training on a silently filtered dataset. Do not start RL-BC-001 in the same run unless both independent gates pass.
