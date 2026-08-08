# Experiment 7 multi-deck integration receipt

- Repository: `LZhangGJ/pocketmon`
- Branch: `agent/experiment7-multideck-ready-20260809`
- Reference input: teammate Experiment 7 ZIP, 94,038 bytes
- Reference SHA-256: `9c0d24067eacee8abc38223dba28d893e5d1e4e9b75204a9ce92a03093558229`
- Ordinary materialized source: `experiment7/reference/`
- Pocketmon integration: `experiment7/integration/`
- Windows Codex prompt: `experiment7/CODEX_WINDOWS_PROMPT.md`
- Default experiment config: `experiment7/configs/multideck_default.json`
- Frozen primary target: `agents/lucario_rule`

The branch contains replay-to-cache conversion, exact-deck ladder selection, data preparation, two-stage Experiment 7 training drivers, Linux multi-GPU scheduling/worker code, portable export and packaging, and seat-balanced Arena scheduling/aggregation.

Local staging verification completed before push:

```text
python -m unittest discover -s tests -p 'test_experiment7_*.py' -v
python -m unittest discover -s tests -p 'test_reference_model.py' -v
python -m compileall -q experiment7 tests
```

No replay, cache, checkpoint, portable weights, engine binary, credentials, or Kaggle submission is committed. Linux training and Arena execution remain to be run by the Windows-hosted Codex through SSH.
