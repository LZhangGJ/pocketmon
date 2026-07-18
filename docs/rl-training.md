# RL training design

The first training stage is deliberately an offline, candidate-scoring actor-critic rather than end-to-end DQN. Option indices are temporary: their meaning changes at every engine selection. The model therefore combines a public-state encoding with features of every currently legal option.

## Pipeline

1. Collect teacher games against the configured opponent pool.
2. Warm-start the actor with behavior cloning and train the value head on final outcomes.
3. Evaluate every checkpoint against fixed seeds and all opponents, especially Archaludon and library-out.
4. Only after the warm start is stable, add on-policy sampling and PPO updates. Keep legal-action generation and a low-confidence fallback in the rule agent.

```powershell
python -m pip install -r requirements-train.txt
python scripts/collect_rl_trajectories.py --cg-dir tmp/official_cg --episodes 10000
python scripts/train_rl_policy.py --epochs 20
```

The collector alternates seats and samples opponents from `configs/opponent_pool.json`. JSONL is used initially so trajectories are inspectable; large runs should later be sharded into compressed Parquet files.

## Current scope and next stage

The warm start trains decisions where exactly one option is selected. Multi-option decisions are collected but excluded until a sequential masked decoder is implemented. PPO should use the same encoder, clipped policy objective, GAE, entropy regularization, and a frozen-opponent mixture. A checkpoint must beat the rule baseline over several hundred paired-seat games before it is wired into the submission.
