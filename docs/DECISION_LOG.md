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

- Downloaded 100 real episodes for `2026-07-18` via `scripts/download_ptcg_data.py --max-episodes 100` (`kaggle/pokemon-tcg-ai-battle-episodes-2026-07-18`) and ran `scripts/audit_replay_alignment.py --date 2026-07-18 --max-files 100 --min-valid-rate 0.999 --strict --output results/data001_replay_alignment.json` at commit `c3cef894decd0977f604c3ea0e5a312320d1ef87`.
- **Fact:** `previous` remains the better alignment (0.5040 valid rate, 16340/32422 decisions) versus `same` (0.3975), matching the synthetic test's direction. **Fact:** the gate did not pass — both alignments are far below the required `>= 0.999`, and `gate_passed=false` with `load_errors=0`.
- **Evidence-based finding:** `scripts/audit_replay_alignment.py` / `rl/public_replay.iter_transitions` do not filter by the per-step `status` field. A diagnostic (non-shipped) rerun that restricts to `status == "ACTIVE"` transitions only raises the `previous` valid rate to 0.8917 — still failing, but showing that roughly half of the counted "decisions" are `INACTIVE` placeholder entries (the non-turn player's carried-over/empty action for that step), not real decisions. This is real signal, not proof of a fix.
- **Evidence-based finding:** even restricted to `ACTIVE` status, the dominant remaining failure reason is `selection_count_out_of_bounds` (1456 of 1756 `previous`+`ACTIVE` invalid rows), and in every one of those the recorded action is an empty list against a `select` that requires 1 or more picks. About 150 of those match a single-option `select.option[0].type == 14` pattern (looks like a forced/auto-acknowledged choice), but roughly 900 involve ordinary `type == 3` board/card selects with option counts ranging 1–13+, and those are unexplained by any hypothesis tested so far.
- **Unverified hypothesis, not applied:** possible causes include (a) a further temporal lag beyond the previous/same pairing DATA-001 currently tests (e.g., the real action for a `type: 3` select may be recorded one or more steps after the `ACTIVE` marker rather than at it), or (b) some decisions are legitimately auto-resolved server-side with no player action, similar to the `type: 14` case, across more option types than currently known. Neither hypothesis has been tested against real data; do not implement either without further evidence.
- **Decision:** did not modify `rl/public_replay.py` or the audit gate logic to chase the failure — that would risk quietly redefining what counts as a "decision" to force a pass, which the handoff explicitly forbids. Recorded the failure and diagnosis instead.
- **Decision:** DATA-002 conversion was not run. The handoff requires DATA-001 to pass first; running it anyway on unaligned data would produce a `public_replay_v1.jsonl.gz` conflated with the same alignment bug (or a `--max-invalid-rate` too permissive to trust).
- Next step before any retry: manually trace 2-3 full episodes step-by-step against whatever reference client/spec exists for `kaggle/pokemon-tcg-ai-battle-episodes-*` to establish the true action/observation pairing rule for `type: 3` selects, rather than guessing from aggregate statistics.
