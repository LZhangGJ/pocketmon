# Experiment 7 multideck training branch

This branch contains the teammate high-score Experiment 7 implementation as ordinary, reviewable source files plus the pocketmon adapters required to select ladder decks, build audited caches, train on several Linux GPUs, export portable agents, and schedule challenger matches.

- High-score reference implementation: `experiment7/reference_impl/`
- Pocketmon integration: `experiment7/integration/`
- Windows Codex prompt: `experiment7/CODEX_WINDOWS_PROMPT.md`
- Default contract: `experiment7/configs/multideck_default.json`

No ZIP extraction is required. Training artifacts remain outside Git under `/homes/lzhang/pocketmon/runs/`.
