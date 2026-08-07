# Pocketmon 项目状态（2026-08-07）

## 一句话结论

项目已经从规则 Agent 和 replay 数据校验，推进到结构化行为克隆、牌组专项、PPO/Action-Q 原型与联赛自动化。当前最强的**离线**结果是 RL-BC-003：三随机种子平均 sequence exact match 为 62.50%，解码合法率为 100%。但正式的隔离环境 Stage B 尚未完成有效模型决策，因此没有证据表明它能提升真实对战胜率；`main` champion 暂不应替换。

## 当前思路

整体路线分为四层，后一层只有在前一层通过硬门禁后才有资格晋级：

1. **可信数据层**：验证 public replay 的时序、动作合法性、终局奖励和去重规则，生成无日志泄漏的单玩家视角轨迹。
2. **离线策略层**：用 action mask 保证候选动作合法，先做 BC，再逐步加入结构化卡牌/攻击特征、历史信息、Action-Q 或离线强化学习信号。
3. **在线适配层**：将 checkpoint 封装为比赛 Agent，在固定 runtime、无网络、逐局隔离的环境中验证加载、推理、超时和非法动作。
4. **联赛与晋级层**：通过固定 opponent snapshot、换边配对、Wilson 下界和失败率门禁比较候选；牌组搜索、专项模型和 PPO 只能在可复现门禁通过后晋级。

当前工程侧的重点不是继续堆训练轮数，而是打通“模型确实参与决策”的隔离对战闭环，并把离线指标与真实对局证据连接起来。

正式训练、native runtime 验证和比赛均在 `doraemon` 系列服务器上执行。本机检查只用于语法、单元行为和跨平台开发便利性，不能替代服务器对局证据。

## 已完成进度与证据

### 1. Replay 数据链路

- 已确认 replay action 与前一步同玩家 observation 对齐；独立日期 `2026-07-19` 的 500 局验证达到 78,776 / 78,776 合法决策，非法决策、未知提交状态、加载错误和奖励不一致均为 0。
- `2026-08-05` 数据扩展到 4,740 局、760,912 行，其中 393,043 行带 policy 权重、760,912 行带 value 权重；`observation.logs` 出现次数为 0。
- 数据转换使用 schema v2，检查 episode 去重、终局 reward、top-level reward 一致性、winner policy 权重和原子输出晋级。
- 原始 replay 与生成的大型轨迹位于 `data/raw/` 和 `data/processed/`，由 `.gitignore` 排除，不进入 Git。

### 2. 行为克隆

| 实验 | 数据规模 | 主要变化 | 三种子平均 sequence exact | 合法解码 |
| --- | ---: | --- | ---: | ---: |
| RL-BC-001 | 500 局 / 78,776 行 | Stateless masked pointer，30 epochs | 44.86% | 100% |
| RL-BC-002 stateless-long | 同上 | 60 epochs | 48.58% | 100% |
| RL-BC-002 history-GRU | 同上 | 因果历史 GRU | 45.41% | 100% |
| RL-BC-003 | 4,740 局 / 760,912 行 | 结构化 observation、卡牌和攻击元数据 | 62.50% | 100% |

RL-BC-003 的三种子平均 set exact 为 63.04%，candidate precision / recall 为 65.53% / 65.76%，所有正式种子均完成且没有 NaN、非法解码或缺失种子。历史 GRU 对空动作和多选动作有帮助，但整体指标落后于同预算 stateless-long，因此当前不作为主干替代。

### 3. Runtime 与隔离评测

- EVAL-ISOLATION-001 的无对局预检通过：checkpoint、native runtime、网络隔离、`torch`、`os.urandom` 和资源记录门禁均通过。
- EVAL-UNSEEDED-001 Stage A 完成 20 局 runtime smoke，异常退出、超时、非法动作和网络尝试均为 0；但 `model_games=0`，它只证明 runtime/fallback 路径可运行。
- Stage B 安排了 4 次模型尝试，但 checkpoint 加载门禁失败、正常终局为 0、模型决策为 0。历史字段中的 `model_games=4` 只表示计划尝试，修正报告确认完成的模型局数为 0。
- 因此目前没有可信的在线模型胜率，也不能从 BC 离线 exact-match 推断比赛强度。

### 4. 牌组搜索与联赛

- 已实现固定对手池、多 learner 调度、断点续跑、换边配对、汇总排名、Wilson 下界和 promotion gate。
- bootstrap 联赛完成 260 局；10 个 learner 中 `learner_02` 为 17 胜 9 负，但每个 learner 只有 26 局且引擎 seed 不可控，只能用于筛选，不能作为正式晋级证据。
- 三个 G0 自有牌组完成 156 局：Lopunny/Froslass 19/52、Dragapult/Munkidori 15/52、Raging Bolt 6/52。
- G1 hard gate 完成 120 局：Lopunny/Froslass 19/60，Dragapult/Munkidori 17/60；均不足以支持晋级。
- 已加入 deck mutation、专项数据集、专项 Agent 打包和连续 RL pipeline，但这些仍属于待严格复验的工程能力。

## 代码地图

### 核心库

- `rl/public_replay.py`：replay 时序配对、动作验证、终局结果、规范化训练行。
- `rl/features.py`：基础与结构化 observation/action 特征、卡牌/攻击元数据、历史特征。
- `rl/model.py`：masked candidate-pointer actor-critic 与结构化模型。
- `rl/bc.py`：轨迹数据集、batch loss、合法 greedy decode 和离线评估。
- `rl/agent_adapter.py`：checkpoint 到比赛 Agent 的加载与决策适配。
- `rl/ppo.py`：rollout 行、动作采样、GAE、PPO loss。
- `rl/action_q.py`：候选动作 Q 模型与监督目标。
- `rl/counterfactual.py`：隐藏区采样和反事实动作价值。
- `rl/promotion.py`：换边配对、Wilson 下界与晋级判定。

### 主要命令入口

- 数据：`scripts/download_ptcg_data.py`、`scripts/audit_rl_trajectory.py`、`scripts/convert_public_replays.py`、`scripts/build_replay_deck_map.py`。
- 训练：`scripts/train_rl_policy.py`、`scripts/train_action_q.py`、`scripts/collect_ppo_rollouts.py`、`scripts/train_masked_ppo.py`。
- Agent：`scripts/evaluate_rl_checkpoint.py`、`scripts/materialize_rl_specialist_agent.py`、`scripts/build_submission.py`。
- 对局：`scripts/run_local_match.py`、`scripts/run_league_schedule.py`、`scripts/summarize_league_round.py`。
- 牌组：`scripts/mutate_legal_decks.py`、`scripts/build_deck_specialist_dataset.py`、`scripts/extract_specialist_deck.py`。
- 自动化：`scripts/continuous_rl_pipeline.py`，负责 rollout、训练、打包、gate 和牌组进化的阶段化运行。
- Kaggle 情报：`scripts/download_kaggle_intelligence.py`、`scripts/analyze_kaggle_intelligence.py`、`scripts/build_gold_strategy_artifact.py`。

### 配置与证据

- `configs/rl_bc_*.json`：BC 各阶段正式配置。
- `configs/*learners*.json`、`configs/*opponent*.json`：联赛 learner 与对手池。
- `configs/decks/`：当前专项牌组。
- `results/rl_bc_003/summary.json`：当前最强离线 BC 汇总。
- `results/*/summary/`：G0、bootstrap、G1 联赛汇总与排名。
- `results/experiments.csv`、`docs/DECISION_LOG.md`、`docs/FAILURE_MODES.md`：实验台账、决策和失败记录。
- `tests/`：覆盖 replay、BC/PPO、Agent 打包、联赛、连续 pipeline 和牌组变异。

## 当前风险

1. **在线证据缺失**：Stage B 尚无有效模型动作，任何“模型更强”的结论都仅限离线。
2. **引擎不可控随机性**：现有联赛 `engine_seed_controlled=false`，小样本排名方差较大。
3. **数据分布与对战分布偏移**：BC 学到的是公开 replay 行为，不保证覆盖当前 opponent pool。
4. **结构化模型资源上升**：RL-BC-003 峰值 RAM 约 20.5–21.5 GB、VRAM 约 2.1–2.7 GB，需要在提交/runtime 限制下复核。
5. **连续训练尚未正式封板**：PPO、Action-Q、专项模型和 deck evolution 已有代码，但还缺少清洁 commit、固定输入 hash、三种子或成对对局证据。

## 下一步建议

1. 修复 Stage B checkpoint/runtime 兼容性，要求至少完成一组“checkpoint 已加载、模型动作数大于 0、正常终局、0 非法动作、0 fallback”的隔离对局。
2. 用同一 opponent snapshot 对 rule fallback、RL-BC-002 stateless-long 和 RL-BC-003 做换边配对；先做小规模可诊断 gate，再扩大样本。
3. 如果 RL-BC-003 在线有效，再将其作为 PPO/Action-Q 的固定 actor 起点；否则先处理 observation/schema 或 cardinality/STOP 建模问题。
4. 对 Lopunny/Froslass 和 Dragapult/Munkidori 保留专项路线，但只有在公共对手与 hard gate 均满足 Wilson 下界门槛时才晋级。
5. `main` champion 保持不变，所有实验继续在 `exp/league-v1` 上进行，禁止 force-push 和未经验证的 Kaggle 提交。

服务器端每次正式复验都应记录实际主机、代码 commit、输入与 checkpoint SHA-256、完整命令、Python/runtime 版本、逐局结果、模型动作计数、fallback 计数、耗时、峰值 RSS/VRAM 和失败日志。
