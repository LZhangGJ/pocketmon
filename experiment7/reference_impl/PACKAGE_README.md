# Imported Experiment 7 implementation

> This source has two documented pocketmon compatibility edits. See `ADAPTATION_NOTES.md`.

# Experiment 7 code-review package

This is a code-only snapshot of the accepted local Experiment 7 candidate:
deck-conditioned temporal Transformer behavior cloning with history length 8.

## Suggested reading order

1. `REVIEW_PROMPT.md`
2. `docs/EXPERIMENT7_CLEANROOM_DESIGN.md`
3. `data_pipeline/features.py` and `data_pipeline/tokenizer.py`
4. `training/deck_identity_model.py`
5. `training/train_multideck_identity.py`
6. `runtime_agent/main.py` and `runtime_agent/deck_identity_portable.py`
7. `validation/arena_isolated.py` and the validation documents

## Included

- Raw replay-to-feature, token-cache, sequence-cache and deck-identity-cache code.
- The exact Transformer model and training/evaluation source used by Experiment 7.
- NumPy-only portable inference and the Kaggle agent entry point.
- The agent deck list, aggregate result summary, validation code and design/audit notes.

`training/train_sequence.py` imports two earlier model variants unconditionally, so
their small model definitions are included only to keep the reviewed import graph
complete. They are not Experiment 7 branches.

## Intentionally excluded

- Model weights/checkpoints (`.npz`, `.pt`, `.pth`, `.onnx`).
- Engine/card catalog, replay files, feature caches and class-map artifacts.
- Opponent agents/decks, Arena per-game logs, Kaggle submission ZIPs and credentials.
- Local absolute paths and the unsanitized training report.

Consequently, this archive is suitable for static review and syntax inspection,
but is not a standalone runnable submission. `runtime_agent/main.py` expects the
excluded portable weights and engine catalog at runtime. Training requires the
public replay data and generated caches described by the data-pipeline scripts.

## Scope and sharing

This package was prepared for the project owner's private GPT code review. It is
not intended for redistribution to another competition team before a rules-safe
team merge or public release.

