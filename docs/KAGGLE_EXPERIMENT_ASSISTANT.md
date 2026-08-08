# Pokémon TCG AI Battle：情报与实验助手运行手册

最后更新：2026-08-08（JST）

## 1. 目标与边界

本助手负责两条相互独立、但可追溯关联的工作流：

1. **情报工作流**：每天获取 Kaggle 公开 Discussion 的完整快照，识别新增、编辑和删除的消息，并把高价值信息归类为可验证假设。
2. **实验工作流**：把假设转化为有对照、有门禁、有失败记录的实验；只有满足在线运行、合法动作、稳定性和统计证据要求的候选，才允许进入下一阶段。

明确边界：

- 只读取公开页面和公开 API；不把 Kaggle 凭证、Cookie 或个人信息写入仓库、日志或 artifact。
- 自动化只负责下载、差分、校验和归档，**不自动提交 Kaggle Agent**。
- Discussion 中的排名、Elo、方法自述和经验数字默认属于线索，不属于已验证结论。
- `main` champion 只能由预先定义的 promotion gate 替换，不能因为单次 Leaderboard 上升、离线 accuracy 或公开帖子而替换。

## 2. 每日情报同步契约

### 2.1 调度

- GitHub Actions：每天 **09:17 JST**（00:17 UTC）运行。
- 支持 `workflow_dispatch` 手动补跑。
- 定时任务只有在 workflow 文件存在于 GitHub 默认分支时才会自动触发。

### 2.2 每次运行的产物

完整快照包括：

- `discussions/raw/<topic_id>.json`：主题及完整回复树；
- `discussions/markdown/<topic_id>.md`：便于检索与阅读的 Markdown；
- `discussions/index.csv`：主题级索引；
- `download_manifest.json`：主题数、消息数、抓取时间和完整性统计。

完整快照作为 GitHub Actions artifact 保存 14 天。为避免每天重复提交全部历史消息，Git 中长期保存：

- `research/kaggle_intelligence/state.json`：消息指纹游标；
- `research/kaggle_intelligence/daily/YYYY-MM-DD/<run>/`：新增、编辑、删除消息的差分；
- `research/kaggle_intelligence/LATEST.md`：最近一次差分摘要；
- `research/kaggle_intelligence/latest_summary.json`：机器可读摘要；
- `research/kaggle_intelligence/runs.jsonl`：运行历史。

持续数据写入独立分支 `data/kaggle-intelligence`，不污染实验分支提交历史。

### 2.3 完整性门禁

一次同步只有同时满足以下条件才算成功：

- Kaggle 返回的主题总数与唯一下载主题数一致；
- manifest 中主题数和消息数均大于 0；
- 差分脚本统计的快照主题数、消息数与 manifest 一致；
- 消息 ID 缺失数为 0；
- 差分单元测试和 Python 编译检查通过；
- 状态游标和差分数据成功推送，完整快照 artifact 成功上传。

同步失败时，不得用不完整数据更新研究结论。

## 3. 情报证据等级

每条信息必须标注证据等级，避免把社区猜测当成事实。

| 等级 | 定义 | 可直接影响什么 |
|---|---|---|
| A | 主办方公告、规则、官方引擎代码或可复现的官方数据 | 合法性约束、提交规则、评测设计、runtime 设计 |
| B | 可运行代码、公开 replay、明确数据版本和可复现实验 | 候选实现和正式实验设计 |
| C | 多个独立参与者的一致经验，或带样本量的社区分析 | 假设优先级、对手池和压力测试设计 |
| D | 单个参与者自述、截图、排名猜测、无代码结论 | 仅进入候选假设列表，不得直接晋级 |

同一主题中，官方回复与普通回复必须分别记录；“帖子获赞多”不能替代证据等级。

## 4. 每日分析分类

新增 Discussion 按以下六类打标签：

1. **rules-engine**：规则差异、合法动作、引擎 bug、卡牌文本和 simulator 行为；
2. **deck-meta**：牌组流行度、matchup、反制、卡池变化和 deck construction；
3. **policy-search**：规则策略、有限深度搜索、rollout、belief sampling、MCTS/PUCT；
4. **state-learning**：状态编码、BC/IL、value/Q、offline RL、PPO、自博弈；
5. **evaluation-rating**：Elo/μ/σ、匹配频率、重复提交方差、线上线下偏移；
6. **runtime-data**：时间、内存、GPU、日志、replay、训练数据质量和泄漏风险。

对每条高价值信息输出：

- 原始事实及证据等级；
- 对当前代码或实验的影响；
- 可证伪假设；
- 最小实验；
- 成功/失败门槛；
- 是否改变当前优先级。

## 5. 截至 2026-08-08 的已确认约束

以下内容来自已归档的官方 Discussion，属于 A 级证据：

- 第一阶段截止日为 **2026-08-16**，此后约两周继续运行对局以收敛最终 Simulation Leaderboard。
- 同时只有两个 active submissions，且是最近提交的两个版本；每天最多提交五次。重复提交相同代码也会作为新 submission 重新建立评分不确定性。
- simulator 的行为是比赛中的权威规则；Agent 应只从 observation 暴露的合法 options 中选择动作。
- Strategy Category 决定进入第二轮的八支队伍，但不改变最终 Simulation Leaderboard。
- 第二轮为东京线下 BO3；同一场 BO3 内必须使用相同牌组和代码，但后续局可以读取前序局日志。
- 第二轮资源公告为 H100 80 GB、256 GiB RAM、16 vCPU，每局总思考时间 30 分钟；卡池会扩大。

对应归档：

- `research/kaggle_intelligence/2026-08-08/discussions/markdown/714189.md`
- `research/kaggle_intelligence/2026-08-08/discussions/markdown/708586.md`
- `research/kaggle_intelligence/2026-08-08/discussions/markdown/732331.md`

## 6. 当前技术判断

截至 2026-08-08，公开证据不支持“纯 RL 已普遍压倒规则 Agent”的结论。更合理的工作假设是：

- **牌组和 matchup 选择是一级变量**；可见 meta 非平稳，照搬当前热门牌组通常存在滞后。
- 强方案常见形态是**规则护栏 + 牌组专项策略 + 有限搜索或 BC**，而不是完全无约束的端到端策略。
- 72%–80% 的 imitation policy accuracy 可能对应很弱或不稳定的线上表现；该指标只能用于模型诊断。
- Leaderboard 的早期匹配和评分不确定性会使相同 Agent 的不同 submission 出现显著差异，因此单次线上排名不能作为因果证据。
- 隐藏信息搜索应使用可行的对手牌组/手牌 belief，而不是把不可见状态当成已知状态。

这些判断主要是 B/C/D 级混合证据，必须通过本项目自己的对局和运行时证据验证。

## 7. 标准实验生命周期

### Stage 0：假设登记

每个实验必须先登记：

- 实验 ID、动机和来源 Discussion；
- 代码 commit；
- 自有牌组 hash、对手池版本/hash；
- 数据、特征 schema、checkpoint hash；
- 唯一变化项；
- 主指标、guardrails、晋级阈值和停止条件。

禁止在看到结果后修改主指标或晋级门槛；确需修改时必须创建新实验 ID。

### Stage 1：静态与合法性检查

最低要求：

- 单元测试、编译检查和 submission 打包通过；
- 牌组合法、动作 mask 正确、空选择和多选动作覆盖；
- 无网络访问；
- checkpoint 可加载；
- 固定 fallback 行为可审计。

失败即停止，不进入对局比较。

### Stage 2：20 局诊断 smoke

目的不是估计胜率，而是证明候选真正参与决策。

硬门禁：

- 正常终局；
- `model_action_count > 0`；
- 0 非法动作；
- 0 timeout；
- 0 网络尝试；
- 0 未解释 fallback；
- 记录 p50/p95/max 决策时间、峰值 RSS/VRAM。

当前项目最优先事项是完成这一闭环。离线 BC 分数不能替代该门禁。

### Stage 3：200 局筛选

- 与固定 opponent snapshot 比较；
- 双方换边，尽量使用成对 seed；
- 与当前 champion、rule fallback 和最近强候选同时比较；
- 报告 W/D/L、分 matchup 胜率、Wilson 置信下界、失败率和耗时。

只有达到预登记阈值且所有 guardrail 通过的候选，才进入 Stage 4。

### Stage 4：400 局正式 gate

- 冻结代码、牌组、对手池、配置和 checkpoint；
- 不允许边跑边调参；
- 运行完整配对比较；
- 必须保留逐局结果、日志、资源记录和失败样本。

### Stage 5：独立 400 局 holdout

使用未参与调参的对手版本、replay 日期或 seed 范围。候选必须在 holdout 上维持优势，并且没有特定 matchup 灾难性回退。

### Stage 6：晋级与 Kaggle 决策

晋级条件至少包括：

- 主指标和 Wilson 下界达到阈值；
- 非法动作、timeout、fallback 和网络 guardrail 全部通过；
- runtime 资源满足比赛限制；
- 结果可由固定 commit/config/hash 重现；
- 没有发现数据泄漏；
- challenger 相对 champion 的改动和风险明确。

Kaggle 两个 active slots 的默认策略：

- 一个经过完整 gate 的稳定 champion；
- 一个单变量 challenger。

由于“最近两个 submission 才 active”且重提会重建评分过程，实验版本不得随意占用线上 slot。

## 8. 评测设计

### 8.1 对手池

至少维护三个池：

- `frozen_core`：固定版本的代表性强对手，用于跨实验可比性；
- `rolling_meta`：最近公开 replay/高分方法中观察到的流行牌组与策略；
- `counter_pool`：专门攻击候选弱点的反制 Agent 和压力测试牌组。

不得只对一个己方最擅长的对手调参。

### 8.2 必报指标

- 总体与分 matchup 的 W/D/L；
- win rate 及 Wilson 区间/下界；
- 双座位结果和座位差；
- 决策次数、合法率、fallback 数、timeout 数；
- p50/p95/max 决策时延；
- 峰值 RSS/VRAM；
- 模型加载成功率和实际模型动作数；
- 训练/验证/holdout 数据日期及去重统计。

BC exact match、top-k accuracy、value loss、policy entropy 等仅作为诊断指标，不作为线上晋级的替代品。

### 8.3 泄漏控制

- 训练、调参和独立 holdout 使用不重叠的 episode/date/team 分区；
- 去除 `observation.logs` 等不应在当前决策时可见的信息；
- 只使用当前玩家视角可观察信息；
- 隐藏状态只能通过采样 belief 生成，不能从 replay 真值回填到线上特征；
- public notebook/replay 代码先静态检查，不执行不可信嵌入 payload。

## 9. 当前实验优先队列

### P0：完成模型在线闭环

修复 Stage B checkpoint/runtime 兼容性。目标是至少完成一组模型真实决策局，满足 `model_action_count > 0`、正常终局、0 非法、0 timeout、0 未解释 fallback。

### P1：建立稳定规则基线与专项牌组矩阵

把当前 champion、rule fallback、公开高分方法家族和主要 archetype 纳入固定对手矩阵。先确认现有牌组/策略真正的 matchup 结构，再训练更复杂模型。

### P2：RL-BC-003 对照实验

在同一 opponent snapshot 下，对 rule fallback、RL-BC-002、RL-BC-003 做换边配对。只有在线证据优于基线，才允许作为 PPO、Action-Q 或 search policy/value 的初始化。

### P3：有限 belief search

在强规则/BC prior 上增加受预算约束的 rollout/search；对不可见牌组、手牌和奖品牌进行多次可行采样，比较平均价值及不确定性。先验证搜索收益是否超过耗时和模型误差。

### P4：rolling meta 与反制实验

每日情报只更新 `rolling_meta` 候选，不直接改 champion。对出现的新牌组、引擎行为或反制策略，先生成专项对手，再经过 20→200→400→holdout 门禁。

### P5：第二轮 BO3 适应

仅在第一轮主线稳定后研究跨局记忆：从 Game 1/2 日志更新对手策略 belief，但同一 BO3 中保持牌组和代码不变。该能力与第一轮 leaderboard Agent 分开评测。

## 10. 每日情报报告模板

```markdown
# Kaggle 情报日报 YYYY-MM-DD

## 同步状态
- topics / messages / new / edited / deleted
- workflow run / artifact / state cursor
- 完整性门禁

## A 级规则与引擎变更
- 原文事实
- 对现有实现的影响
- 必须执行的回归测试

## B/C 级技术情报
- deck/meta
- policy/search
- state/learning
- evaluation/runtime

## 新实验候选
| ID | 假设 | 唯一变量 | 最小实验 | 晋级门槛 | 优先级 |

## 噪声或暂不采纳
- 信息及不采纳原因

## 当前推荐动作
- 当天最多一个最高优先级动作，不能同时改 deck、state、model 和 search。
```

## 11. 手动运行

完整 Discussion 快照：

```bash
python scripts/download_kaggle_intelligence.py \
  --skip-notebooks \
  --output /tmp/kaggle-intelligence-snapshot \
  --state research/kaggle_intelligence/state.json
```

提取相对旧游标的差分：

```bash
python scripts/extract_kaggle_discussion_delta.py \
  --snapshot /tmp/kaggle-intelligence-snapshot \
  --previous-state /tmp/kaggle-intelligence-state-before.json \
  --output research/kaggle_intelligence/daily/YYYY-MM-DD/manual-run
```

校验：

```bash
python -m unittest discover -s tests -p 'test_extract_kaggle_discussion_delta.py' -v
python -m compileall -q scripts/extract_kaggle_discussion_delta.py
```

## 12. 自动化启用条件

`.github/workflows/kaggle-discussions-daily.yml` 合并到 GitHub 默认分支后，定时任务才会按日自动触发。在此之前可以通过该 workflow 的 `workflow_dispatch` 进行人工运行，但分支上的 workflow 文件本身不代表日程已经生效。
