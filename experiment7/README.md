# Experiment 7 teammate code import

This directory preserves the teammate-provided Experiment 7 source archive as an immutable input artifact. The team merge was reported as complete before this import.

## Source artifact

- File: `source/experiment7_code_for_gpt_2026-08-08.zip`
- SHA-256: `9c0d24067eacee8abc38223dba28d893e5d1e4e9b75204a9ce92a03093558229`
- Files inside: 39 total entries; 38 entries listed in `PACKAGE_MANIFEST.csv`
- Static checks before import: package-manifest SHA/size verification passed for all 38 listed files; all 29 Python files passed `py_compile`.

The archive intentionally excludes checkpoints, portable weights, engine/card catalogs, replays, generated caches, opponent packages, Arena logs and credentials. It is therefore a source snapshot, not a standalone runnable submission.

## Unpack

From the repository root:

```bash
bash experiment7/unpack_source.sh
```

The default destination is `runs/experiment7/source`, which is ignored by the repository. The script verifies the archive SHA before extraction and validates the package manifest afterward.

## Training handoff

Use `CODEX_TRAINING_PROMPT.md` as the exact task prompt for the local Codex agent. It requires a clean integration branch, audited conversion from the repository's real replay format, sealed chronological holdouts, portable-inference parity and a 20-game local runtime gate. It explicitly forbids Kaggle submission and committing data, credentials or model artifacts.
