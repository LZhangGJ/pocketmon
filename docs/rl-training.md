# RL training design

The first training stage is deliberately an offline, candidate-scoring actor-critic rather than end-to-end DQN. Option indices are temporary: their meaning changes at every engine selection. The model therefore combines a public-state encoding with features of every currently legal option.

## Public replay gate

Before using downloaded episodes, run DATA-001 and DATA-002:

```powershell
python scripts/audit_replay_alignment.py --date YYYY-MM-DD --max-files 100 --strict
python scripts/convert_public_replays.py --date YYYY-MM-DD --alignment previous --policy-source winners
```

The converter blocks promotion of its temporary output when any invalid decision, load error, or conflicting episode hash remains under the default zero-tolerance gate. It stores only the acting player's observation and removes `observation.logs` by default. See [`PUBLIC_REPLAY_RL_PROPOSAL.md`](PUBLIC_REPLAY_RL_PROPOSAL.md) for the staged BC, offline-RL, PPO, and league proposal.

## Existing local-simulation pipeline

1. Collect teacher games against the configured opponent pool.
2. Warm-start the actor with behavior cloning and train the value head on final outcomes.
3. Evaluate every checkpoint against fixed seeds and all opponents, especially Archaludon and library-out.
4. Only after the warm start is stable, add on-policy sampling and PPO updates. Keep legal-action generation and a low-confidence fallback in the rule agent.

```powershell
python -m pip install -r requirements-train.txt
python scripts/collect_rl_trajectories.py --cg-dir tmp/official_cg --episodes 10000
python scripts/train_rl_policy.py --epochs 20 --device auto --validation-fraction 0.1
```

`auto` selects CUDA when available. Episodes, rather than individual decisions, are assigned to the validation split so decisions from the same game cannot leak into both sets. The run writes both the last checkpoint and a `.best.pt` checkpoint selected by validation loss; logs include policy accuracy and value loss.

Evaluate a checkpoint against a separate trajectory file:

```powershell
python scripts/evaluate_rl_checkpoint.py --checkpoint artifacts/rl/candidate_actor_critic.best.pt --input data/rl/held_out.jsonl --device auto
```

The current local collector alternates seats and samples opponents from `configs/opponent_pool.json`. It is separate from the public replay converter and still requires teacher/outcome filtering before being used as policy supervision.

## Current scope and next stage

The current warm start trains decisions where exactly one option is selected. Multi-option decisions are collected but excluded until a sequential masked decoder is implemented. PPO should use the same encoder, clipped policy objective, GAE, entropy regularization, and a frozen-opponent mixture. A checkpoint must beat the rule baseline over several hundred paired-seat games before it is wired into the submission.
