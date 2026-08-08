# Codex start here — corrected

The previous handoff incorrectly treated the teammate ZIP as if it were the runnable experiment artifact. Do **not** follow that workflow.

Use this prompt instead:

```text
experiment7/CODEX_WINDOWS_PROMPT.md
```

The correct execution model is:

1. Codex runs on the Windows workstation.
2. The ZIP is a code-review source snapshot only; it is not a runnable training package or Agent.
3. Windows Codex first validates and extracts the source, imports it into the Git work branch as ordinary files, reviews it, and implements the pocketmon adapters/tests.
4. Windows commits and pushes an immutable SHA.
5. Linux doraemon servers run replay processing, cache construction, GPU training, export and Arena from that committed SHA through SSH.
6. `runtime_agent/main.py` must not be run until weights and engine catalog have been regenerated.

Fixed repository context:

```text
repository:          LZhangGJ/pocketmon
remote:              https://github.com/LZhangGJ/pocketmon.git
source branch:       agent/experiment7-multideck-challengers-20260808
working branch:      codex/experiment7-multideck-challengers-20260808
correct prompt:      experiment7/CODEX_WINDOWS_PROMPT.md
Linux repository:   /homes/lzhang/pocketmon
Linux Python:        /homes/lzhang/mypath/new/envs/trans/bin/python
ladder analysis:     /homes/lzhang/pocketmon/analysis/outputs/top_ladder_2026_08_07_20260808
replays:             /homes/lzhang/pocketmon/data/raw/replays/2026-08-06
servers:             doraemon02 doraemon03 doraemon15 doraemon16 doraemon19 doraemon20
primary target:      agents/lucario_rule
```
