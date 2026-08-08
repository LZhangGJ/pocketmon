# Experiment 7 multi-deck challenger integration

The teammate implementation is vendored as source under `experiment7/reference_impl`. The integration layer deliberately reuses pocketmon's validated replay alignment and deck sidecar instead of the teammate's private ZIP-catalog path.

The pipeline is:

```text
pocketmon raw replay
  -> canonical audited decisions + exact own-deck sidecar
  -> ladder representative selection with replay-support gates
  -> Experiment 7 state/action/entity/history/deck caches
  -> K-deck balanced fine-tuning on three independent GPUs/seeds
  -> calibration-only checkpoint selection
  -> NumPy portable export and 500-decision ranking parity
  -> one Agent package per selected exact deck
  -> seat-balanced challenger-vs-Lucario Arena
```

The original model dimensions, token organization, heads and losses are unchanged. The compatibility changes in the vendored trainer are limited to: (1) an unseen module-version string receives weight 1.0 instead of raising a `KeyError`, while known historical module weights remain unchanged; and (2) auxiliary opponent classes with zero visible-evidence examples receive zero loss weight instead of aborting the main policy run. Neither change exposes hidden information or alters runtime gating.

The training holdout is represented by `validation == 1` in each current-deck feature cache. The trainer uses only the chronological tail of the remaining training episodes for calibration and does not open holdout during checkpoint selection.
