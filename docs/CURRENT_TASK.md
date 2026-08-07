# Current task: EVAL-UNSEEDED-001

Read and execute `docs/EVAL_UNSEEDED_001_SPEC.md` exactly.

## Scope

- Branch: `exp/league-v1`
- Start from commit: `726b14994a8635d2fdb6c0364a4b1c3ebde4a0ed`
- First implement and validate Stage A only.
- Stage B is permitted in the same server run only if every Stage A gate passes from a clean committed implementation.
- Do not start the later 40-game-per-opponent pilot.
- Do not modify or force-push `main`.
- Do not create or merge a Pull Request.
- Do not submit to Kaggle.
- Do not start AWR, IQL, PPO, self-play, league training, or opponent-pool updates.

## Required initial commands

```bash
git fetch origin
git switch exp/league-v1
git pull --ff-only
git status --short --branch
git rev-parse HEAD
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q rl scripts tests
```

Stop before editing if the branch is `main`, the worktree contains unrelated changes, or HEAD is not at least `726b14994a8635d2fdb6c0364a4b1c3ebde4a0ed`.

## Deliverables

Implement the minimum evaluator/runtime-verification code and tests required by the spec. Run the server validation, retain all failures, and update the required result and decision files. Commit and push only to `exp/league-v1`.

The final response must report:

- implementation commit;
- server-result commit;
- exact commands;
- Stage A pass/fail and each gate;
- whether Stage B was authorized and run;
- model game count;
- tests and compile status;
- elapsed time, peak RSS, and peak VRAM;
- all operational failures;
- merge recommendation;
- verified facts, experiment-supported inferences, unverified hypotheses, and next experiment.
