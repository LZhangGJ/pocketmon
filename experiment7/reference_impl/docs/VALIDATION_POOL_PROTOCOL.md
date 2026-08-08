# Frozen local promotion protocol — 2026-08-07

## Boundary

- Local evaluation only. No Kaggle submission is authorized by this protocol.
- Frozen control: pure Transformer `h8/pretrain-12`.
- Frozen engine and opponent identities come only from `validation_pool_manifest.json`.
- Public scores are descriptive snapshots, never local acceptance targets.

## Pools

- Promotion pool: Tetsutani, biohack44 and romanrozen. They must resolve to three
  distinct deck SHA256 values.
- Legality stress only: official random sample. Its win rate does not enter the
  promotion aggregate.

## Progression for every experiment arm

1. **20 games per opponent** (10 per seat): legality and catastrophic-regression screen.
2. **200 games per promotion opponent** (100 per seat): directional comparison
   against the frozen control matrix.
3. **400 fresh games per promotion opponent** (200 per seat): independent
   replication for the best candidate only.

All arenas use process-isolated agents, four independent shards, one OpenBLAS
thread per agent, the frozen engine root, exact package receipts, seat balance,
and zero tolerated engine errors or candidate fallbacks.

## Promotion rule

- No error or fallback in any accepted run.
- No promotion-pool opponent may lose more than 5 percentage points versus the
  corresponding frozen-control 200-game cell.
- The unweighted promotion-pool macro score must improve by at least 3 percentage
  points at 200 games per opponent.
- The same requirements must hold in a fresh 400-game-per-opponent replication.
- Offline accuracy, a single matchup, or a single small screen cannot promote a model.

The formula and opponent set are frozen before candidate results are observed.
