# Experiment 7 final archive (2026-08-17)

This directory is the recoverable final snapshot retained after stopping the competition workloads.

## Contents

- `manifest.json`: maps every frozen league/pool model identity to a SHA-256-addressed weight blob.
- `weights/`: 56 unique, deduplicated checkpoint/portable weight files (1,640,495,729 bytes).
- `training_code/production_worktree/`: exact training and packaging code copied from the production worktree after shutdown.
- `training_code/local_control/`: local Experiment 7 configuration and control scripts needed to reproduce the orchestration.
- `training-code-manifest.json`: SHA-256 and size for each retained source/configuration file.
- `state/`: final frozen league and pool indexes; these contain metadata only, not replay or rollout data.
- `VALIDATION.json`: archive completeness result. `valid: true` means every weight exists and matches its recorded SHA-256 and byte size.

## Restore a model

1. Find the model or snapshot in `manifest.json` under `models`.
2. Read its `archivePath` and `weightSha256`.
3. Copy the referenced blob from `weights/` to the desired runtime location.
4. Verify it with `sha256sum` (Linux) or `Get-FileHash -Algorithm SHA256` (Windows).

Weights are stored through Git LFS. Clone with Git LFS installed and run `git lfs pull` before restoring models.

Replay files, rollout buffers, training datasets, caches, logs, and historical non-current generation directories are intentionally excluded.
