# RL-BC-004：固定热门牌组 Transformer-8 BC 快速实验流程

状态：可执行  
目标分支：`agent/rl-bc-004-transformer8-fast`  
基线：`RL-BC-003` structured BC  
主实验变量：在现有结构化卡牌/攻击/盘面/己方牌组编码上增加最近 8 个决策的因果 Transformer

## 1. 实验目标

以最短路径验证以下假设：

> 对一个近期高频且公开回放充足的固定牌组，使用当前盘面卡牌表示和最近 8 个己方决策的时序 Transformer，进行 12 轮 warm-start BC，能够比无时序 structured BC 产生更好的本地对局策略。

第一轮只验证时序模块本身，不同时加入 opponent belief、Action-Q、PPO 或牌组遗传搜索。原因是同时改变多个变量会失去归因能力。

### 唯一主变量

- 对照：`RL-BC-003` structured current-state model；
- 候选：同一 structured encoder + 8-step Transformer；
- 固定：target deck、replay snapshot、deck-map、episode split、warm-start checkpoint、opponent pool、seat distribution 和 evaluation protocol。

## 2. 最小新增实现

稳定的 Gold V3.1 训练与运行链不直接修改。新增代码位于：

- `rl/temporal_model.py`
- `rl/temporal_agent_adapter.py`
- `scripts/train_rl_bc_004_transformer8.py`
- `scripts/write_temporal_specialist_config.py`
- `scripts/select_modal_deck.py`
- `scripts/materialize_temporal_specialist_agent.py`
- `scripts/run_rl_bc_004_fast.sh`
- `agents/rl_bc_temporal_specialist/main.py`
- `tests/test_temporal_transformer.py`

若该方向失败，删除这些新增文件即可完整回滚；原 `train_rl_policy.py`、`rl/model.py` 和 `rl/agent_adapter.py` 不受影响。

## 3. 固定牌组选择

快速默认方案使用近期公开的 Grimmsnarl Agent 作为 archetype reference：

```text
research/kaggle_intelligence/2026-08-08/notebooks/source/
07__tetsutani__grimmsnarl-ex-damage-transfer-control.ipynb
```

脚本不会直接把 Notebook 中的精确 60 卡列表作为最终训练 deck，而是：

1. 从 Notebook 提取 reference deck；
2. 只取其 Pokémon core；
3. 在当前 replay deck-map 中筛选相同 Pokémon core；
4. 统计每个精确 60 卡列表的频率；
5. 选择出现次数最多的 exact modal deck；
6. 冻结该 deck 进行训练、打包和评测。

产物：

```text
<RUN_ROOT>/data/target_deck.csv
<RUN_ROOT>/data/modal_deck_audit.json
```

审计文件必须包含 matched deck entries、exact variant 数量、modal frequency、modal share 和 target deck SHA256。

## 4. 数据构建

默认使用已完成 DATA-001 / DATA-002 对齐审计的 processed replay：

```text
/homes/lzhang/pocketmon/data/processed/public_replay_2026-08-05.jsonl.gz
/homes/lzhang/pocketmon/data/processed/replay_decks_2026-08-05.jsonl.gz
```

固定牌组 specialist 数据规则：

- exact target deck 轨迹全部保留；
- 加入 15% generic policy rows，降低过拟合和缺失 context 风险；
- episode/player 为历史分组；
- 按 `action_step` 排序；
- 每个决策最多看到之前 8 个己方决策；
- 当前或未来步骤使用数必须为 0；
- opponent hand/prize identity 和完整未来 logs 禁止进入特征。

产物：

```text
<RUN_ROOT>/data/replay.jsonl.gz
<RUN_ROOT>/data/decks.jsonl.gz
<RUN_ROOT>/data/dataset_audit.json
```

## 5. 模型

### 当前状态分支

直接复用 RL-BC-003：Card ID/metadata、Attack ID/metadata、visible entities、zone embedding、DeepSets mean/max pooling、acting-player submitted deck、masked autoregressive pointer、explicit STOP 和 value head。

### 时序分支

最近 8 个 completed decisions，每步使用已有的 causal history token：

```text
prior pre-action state
+ prior selected-option summary
+ selection count
+ empty-action indicator
```

Transformer 配置：

```yaml
history_length: 8
hidden_dim: 192
layers: 2
heads: 4
ffn_dim: 768
dropout: 0.10
```

当前 structured state 与 temporal state 通过 learnable sigmoid gate 融合。

### Warm start

默认从以下 checkpoint 初始化 base modules：

```text
/homes/lzhang/rl_bc_003_outputs/seed_17/checkpoints/seed_17_best.pt
```

加载规则：

- 原 structured 参数必须全部匹配；
- 只允许新的 temporal module 参数缺失；
- 任意 base weight 缺失或 unexpected weight 都立即失败。

学习率：

```yaml
base_learning_rate: 1.0e-4
temporal_learning_rate: 3.0e-4
```

## 6. 最快执行入口

### Checkout

每台服务器使用独立 worktree，避免共享 checkout 冲突：

```bash
cd /homes/lzhang/pocketmon
git fetch origin
git worktree add \
  /homes/lzhang/worktrees/rl-bc-004-$(hostname) \
  origin/agent/rl-bc-004-transformer8-fast

cd /homes/lzhang/worktrees/rl-bc-004-$(hostname)
```

### 环境

```bash
export PYTHON=/homes/lzhang/mypath/new/envs/trans/bin/python
export RUN_ID=rl-bc-004-tx8-$(hostname)-seed17
export CUDA_DEVICE=0
```

### 一键执行

```bash
bash scripts/run_rl_bc_004_fast.sh
```

该脚本依次执行：

1. Python 编译；
2. temporal unit test；
3. 提取 Grimmsnarl reference deck；
4. 选择当前 deck-map 内 modal exact deck；
5. 构建 specialist replay；
6. 生成不可变 planned config；
7. 2-batch smoke；
8. 12-epoch formal training；
9. 构建 immutable local Agent package；
10. 在提供 `CG_DIR` 时运行本地 opponent-pool gate。

带本地对局 gate：

```bash
export CG_DIR=/path/to/official/cg
export OPPONENT_MANIFEST=configs/opponent_pool_frontier_server.json
bash scripts/run_rl_bc_004_fast.sh
```

只跑 smoke：

```bash
RUN_FORMAL=0 \
RUN_ID=rl-bc-004-smoke-$(hostname) \
bash scripts/run_rl_bc_004_fast.sh
```

## 7. 并行服务器分工

### doraemon02：数据与 baseline

- 执行 modal deck selection；
- 固定 dataset/config hashes；
- 跑同一 deck 的无时序 structured baseline；
- 保管共享只读 dataset artifact。

### doraemon03：主候选 seed 17

- 跑本分支一键脚本；
- 12 epochs；
- 生成第一个 temporal package；
- 完成 20–24 局 diagnostic gate。

### doraemon14：复现 seed 42

仅在 seed 17 满足以下条件后启动：legal decode 100%、checkpoint 能加载、model actions > 0、0 crash/timeout/illegal，且 offline validation 未明显低于固定牌组 baseline。

### doraemon15：对局 shard

- candidate vs frozen opponent pool；
- 双席位；
- 200 局筛选；
- 不参与训练和参数选择。

### doraemon16：独立 holdout

- 使用未用于调参的 opponent subset；
- 对入围 candidate 进行独立复现；
- 检查 seat gap、matchup collapse 和 runtime。

## 8. 分阶段门禁

### Gate A：静态检查

必须全部通过：

- `py_compile`；
- `tests/test_temporal_transformer.py`；
- planned config 与实际参数一致；
- target deck 60 张；
- target deck、dataset、deck-map、warm-start 均记录 SHA256；
- formal run 使用 clean worktree。

### Gate B：2-batch smoke

必须满足：

- train/validation forward 成功；
- loss finite；
- decode legal rate = 1.0；
- invalid actions = 0；
- checkpoint 可重新加载；
- empty-history row 不产生 NaN。

失败时不运行 12 epochs。

### Gate C：12-epoch offline

主要诊断指标：sequence exact match、set exact match、single/empty/multi-select accuracy、per-context accuracy、value loss、peak RAM/VRAM、best epoch 和 late validation trend。

最低要求：

- decode legal rate = 1.0；
- invalid actions = 0；
- validation loss 不出现持续发散；
- best checkpoint 不得来自无法复现的 dirty run。

离线 accuracy 只用于淘汰，不能直接晋级。

### Gate D：20–24 局运行诊断

必须满足：normal terminal、model action count > 0、0 loader error、0 inference error、0 illegal action、0 timeout、0 unexplained fallback、双席位实际运行，且 CPU runtime 可接受。

### Gate E：200 局筛选

比较对象：同一 target deck 的无时序 structured baseline、temporal candidate、rule/fallback 和 frozen multi-archetype opponent snapshot。

报告：W/D/L、score rate、Wilson interval、per-matchup result、seat gap、crash/timeout/illegal 以及 p50/p95/max latency。

进入下一阶段的建议门槛：

- candidate 对 baseline 直接 score rate >= 0.55；
- overall opponent-pool score rate 不低于 baseline；
- 关键 matchup 不下降超过 10 个百分点；
- failure rate = 0。

### Gate F：400 + 独立 400

只有通过 200 局筛选的 immutable package 进入 400 局正式 gate 和另一服务器独立 400 局 holdout；中途不允许修改 checkpoint、deck、config 或 opponent snapshot。

## 9. 对照实验

第一批只保留三个 arm：

| Arm | Current structured board | History | Epochs |
|---|---:|---:|---:|
| A | 是 | 无 | 12 |
| B | 是 | GRU-8 | 12 |
| C | 是 | Transformer-8 | 12 |

优先运行 A 和 C。只有 C 相比 A 有信号后再跑 B，避免浪费机器。

## 10. Opponent belief 并行支线

Opponent probability 不阻塞 Transformer MVP。主候选完成 Gate D 后，新建实验 `RL-BC-005-BELIEF`。

输入只使用当时已经公开的 opponent active、bench、discard、attached energy/tool/pre-evolution、opponent-owned stadium 和可见日志前缀中的 reveal/move events。

第一版输出：archetype posterior，以及 64–128 张 key-card 的 expected remaining count。

禁止使用完整真实 opponent deck 作为在线输入、从 replay 未来步骤回填，或把 belief 与 Transformer 同时作为 RL-BC-004 的未登记变量。

## 11. 自博弈支线

纯 PPO 不作为第一步。顺序为：

1. fixed-deck Transformer BC；
2. local gate；
3. 收集低置信度、规则冲突和关键终局状态；
4. bounded counterfactual rollout；
5. DAgger 式追加 hard-state labels；
6. 再考虑 KL-constrained PPO。

开始 PPO 前必须已有一个通过 200 局筛选的 BC checkpoint。

## 12. 结果目录

默认：

```text
/homes/lzhang/pocketmon/runs/<RUN_ID>/
├── reference/
├── data/
│   ├── target_deck.csv
│   ├── modal_deck_audit.json
│   ├── replay.jsonl.gz
│   ├── decks.jsonl.gz
│   └── dataset_audit.json
├── config/planned.json
├── smoke/
├── formal/
│   ├── checkpoints/
│   ├── metrics.json
│   └── split.json
├── packages/candidate/
└── gate/
```

任何正式比较必须引用 Git commit、planned config hash、checkpoint hash、target deck hash、opponent manifest hash 和 per-game raw results。

## 13. 停止条件

立即停止某一 arm：

- 数据 alignment/audit 失败；
- warm-start base weights 不完整；
- smoke 出现 NaN；
- legal decode < 100%；
- model action count = 0；
- 任意 crash/timeout/illegal；
- 训练明显发散；
- CPU 推理无法满足第一轮资源约束；
- 200 局无时序 baseline 明显优于 Transformer。

若 Transformer 无增益，下一步不是增加层数，而是检查 fixed deck 数据量、最近轨迹质量、history token 是否表达关键动作、opponent matchup 分布，以及是否需要逐步盘面实体历史而不是当前 compact history token。
