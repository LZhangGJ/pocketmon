# RL-BC local adapter

This is a local functional-smoke adapter, not a submission artifact or a league-promotion claim. It defaults to the RL-BC-002-A seed-20260720 best checkpoint and can be redirected with `POCKETMON_RL_CHECKPOINT`. Checkpoints remain gitignored.

Every model decision uses the masked autoregressive decoder. Model load/inference failure or an illegal model result invokes the existing lucario rule agent. If that fallback is structurally invalid, the adapter emits a minimal deterministic legal selection and records the event in `diagnostics()`.

Run through `scripts/run_local_match.py` with an official `cg` package directory. The current server audit did not find that engine, so no gameplay result is bundled with this adapter.
