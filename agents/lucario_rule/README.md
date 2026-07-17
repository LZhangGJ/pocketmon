# Mega Lucario rule agent

This is the repository version of the public Mega Lucario rule baseline from
`makthanithin/pokemon-tcg-ai-battle-1084-5-baseline` (reported LB 1084.5).

The policy scores legal simulator options using board state, prize value,
energy readiness and matchup-specific rules. In particular, it keeps Hariyama
as a non-ex route into the Crustle wall matchup.

Build a Kaggle archive from the repository root:

```powershell
python scripts/build_submission.py --agent agents/lucario_rule --cg-dir PATH/TO/cg
```

The generated archive is written to `dist/lucario_rule_submission.tar.gz`.
