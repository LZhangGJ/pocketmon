# Agent 55328694 全量 Replay 败局审计

快照日期：2026-08-09

Kaggle submission：[`55328694`](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/submissions?submissionId=55328694)

提交说明：Experiment7 deck-conditioned Transformer，package `fb448ada`

## Executive Summary

- 完整范围是 **108 场 episode**，不是网页首屏的 4 场。其中 107 场是公开对局，结果为 **64 胜 43 负，胜率 59.8%**；`90751124` 是同一 submission 的 validation 自对局。
- 43 场公开败局中，**22 场（51.2%）至少出现一次高置信度的单步资源浪费**：检索选空、可贴能量却结束回合、或可进化却结束回合。三类合计 37 个事件。
- 同样的三类信号在胜局中覆盖 20/64（31.2%）。它们与败局相关，但不代表每一次都直接导致输牌；仍有 21 场败局没有发现这类明确的单步失误。
- 用原始 `submission.tar.gz` 对全部 108 份 replay 重新推理，**10,546 个动作与记录完全一致，0 mismatch、0 fallback**。因此这些问题是模型真实决策偏差，不是异常被吞掉或非法动作造成。
- 最弱的两个小样本对局是 **Mega Lopunny：2-5（28.6%）** 和 **Grass / Ogerpon：3-4（42.9%）**。应作为下一轮定向回归集，而不是仅凭 7 场样本立即换牌。

## 数据范围与验证

| 项目 | 结果 |
|---|---:|
| Kaggle episode | 108 |
| 公开对局 | 107 |
| 公开结果 | 64 胜 / 43 负 |
| 公开胜率 | 59.8% |
| Validation 自对局 | 1 场、2 个 player episode |
| 合法动作检查 | 0 invalid |
| 原提交动作复现 | 10,546 / 10,546 一致 |
| 原提交 fallback | 0 |
| 实体截断 | 6 次，全部出现在胜局 |

动作使用 Kaggle 的标准时序对齐：`steps[t].action` 对应 `steps[t-1].observation`。公开胜率不包含 validation 自对局；但 validation 败方动作仍单独审计。

## 三类重复失误

| 失误模式 | 败局事件 | 涉及败局 | 败局覆盖 | 胜局覆盖 | 差异 |
|---|---:|---:|---:|---:|---:|
| 可选检索直接选空 | 15 | 11 | 25.6% | 10.9% | +14.6 pp |
| 可贴能量却结束回合 | 10 | 7 | 16.3% | 9.4% | +6.9 pp |
| 可进化却结束回合 | 12 | 7 | 16.3% | 12.5% | +3.8 pp |

三类信号有重叠，合计覆盖 22 场公开败局。

### 1. 已支付检索成本，却在合法目标前返回空选择

15 次事件分布在 11 场败局。典型流程是 agent 已经打出 Buddy-Buddy Poffin，环境给出 Snorunt 或 Impidimp，但 optional-count head 选择数量 0。

提交策略的 `choose()` 先独立对可选数量做 argmax，再按 option logits 取 top-k。即使目标排序正确，只要数量头预测 0，最终动作仍为空。这是最清晰的结构性入口。

高频样本：

- `90907191`：第 2、6、8 回合共三次检索选空。
- `90756224`：第 5 回合连续两次检索选空。
- `90768268`：第 6、10 回合两次检索选空。
- `91325674`：第 2 回合第二张 Poffin 在三个合法目标前选空。

### 2. 可贴能量却直接结束回合

10 次事件分布在 7 场败局。最严重的是首回合空过贴能，直接损失一个不可追回的手贴节奏：

- `90761788`：第 1、7、9 回合三次跳过贴能。
- `90771647`、`91103678`、`91221601`、`91307030`：第 1 回合跳过贴能。
- `90780965`：第 3、5 回合两次跳过贴能。
- `91092609`：第 3 回合跳过贴能。

模型没有在 END 之前执行最低限度的资源推进检查，因此可以在 `energyAttached = false` 且合法 ATTACH 存在时直接结束回合。

### 3. 可进化却直接结束回合

12 次事件分布在 7 场败局。主要后果是推迟 Grimmsnarl ex 的血量、攻击和后续资源窗口。

- `91325674`：第 6、8、10、12 回合连续四次可进化 Grimmsnarl ex 却 END。
- `90764263`、`91160774`：各连续两回合跳过 Grimmsnarl ex 进化。
- `90752841`、`90826802`、`91135935`、`91168560`：各一次。

## 91325674：最典型的连续失误

这场对局不是一次偶然排序错误，而是一整段主阶段价值排序失灵：

1. 第 2 回合先用第一张 Buddy-Buddy Poffin 正常检索两个 Impidimp。
2. 同回合打出第二张 Poffin，面对 Snorunt、Impidimp、Snorunt 三个合法目标返回空选择。
3. 第 6、8、10、12 回合，手牌中都有 Grimmsnarl ex、场上有可进化 Morgrem，却连续四次选择 END。
4. 直到第 14 回合才通过 Rare Candy 建成第一只 Grimmsnarl ex。

这说明需要的是策略层 guardrail，而不只是增加少量同类训练样本。

## 逐对局的高置信度失误

| Episode | 严重度 | 对手类型 | 事件 |
|---:|:---:|---|---|
| 91325674 | 高 | Grimmsnarl 镜像 | 检索选空 ×1（T2）；可进化却 END ×4（T6/8/10/12） |
| 90761788 | 高 | Grimmsnarl 镜像 | 可贴能量却 END ×3（T1/7/9） |
| 90907191 | 高 | Grimmsnarl 镜像 | 检索选空 ×3（T2/6/8） |
| 90756224 | 高 | Mega Starmie | 检索选空 ×2（T5/T5） |
| 90764263 | 高 | Mega Lucario | 可进化却 END ×2（T6/8） |
| 90768268 | 高 | Grimmsnarl 镜像 | 检索选空 ×2（T6/10） |
| 90780965 | 高 | Alakazam | 可贴能量却 END ×2（T3/5） |
| 91160774 | 高 | N's Zoroark | 可进化却 END ×2（T6/8） |
| 91168560 | 高 | Grimmsnarl 镜像 | 可进化却 END ×1（T3）；检索选空 ×1（T7） |
| 91221601 | 高 | Grimmsnarl 镜像 | 可贴能量却 END ×1（T1）；检索选空 ×1（T13） |
| 90771647 | 高 | Dragapult | 可贴能量却 END ×1（T1） |
| 91103678 | 高 | Mega Lopunny | 可贴能量却 END ×1（T1） |
| 91307030 | 高 | Mega Lucario | 可贴能量却 END ×1（T1） |
| 90752841 | 中 | Alakazam | 可进化却 END ×1（T15） |
| 90758524 | 中 | Grimmsnarl 镜像 | 检索选空 ×1（T8） |
| 90817453 | 中 | Mega Lopunny | 检索选空 ×1（T12） |
| 90826802 | 中 | Alakazam | 可进化却 END ×1（T6） |
| 91050216 | 中 | Grimmsnarl 镜像 | 检索选空 ×1（T5） |
| 91061018 | 中 | Grimmsnarl 镜像 | 检索选空 ×1（T7） |
| 91092609 | 中 | Grimmsnarl 镜像 | 可贴能量却 END ×1（T3） |
| 91135935 | 中 | Grimmsnarl 镜像 | 可进化却 END ×1（T4） |
| 91178158 | 中 | Grimmsnarl 镜像 | 检索选空 ×1（T15） |

其余 21 场公开败局没有发现这三类明确的单步资源浪费。它们仍可能存在路线、换位、攻击目标或资源保留的累计次优，但仅凭线性 replay 不能可靠做反事实归因。完整 episode 表见 [`output/episodes.csv`](output/episodes.csv)，紧凑败局表见 [`output/loss_review.csv`](output/loss_review.csv)。

## 对手牌组表现

| 对手类型 | 场次 | 胜负 | 胜率 | 败局占比 |
|---|---:|---:|---:|---:|
| Grimmsnarl 镜像 | 44 | 27-17 | 61.4% | 39.5% |
| Alakazam | 22 | 13-9 | 59.1% | 20.9% |
| Mega Lucario | 9 | 7-2 | 77.8% | 4.7% |
| Mega Kangaskhan | 8 | 6-2 | 75.0% | 4.7% |
| Grass / Ogerpon | 7 | 3-4 | 42.9% | 9.3% |
| Mega Lopunny | 7 | 2-5 | 28.6% | 11.6% |
| Dragapult | 4 | 2-2 | 50.0% | 4.7% |
| 其他 | 4 | 3-1 | 75.0% | 2.3% |
| Mega Starmie | 2 | 1-1 | 50.0% | 2.3% |

Grimmsnarl 镜像和 Alakazam 合计贡献 26/43 败局，但它们也占 66/107 场公开对局，胜率接近或高于总体，属于高暴露量而不是最弱 matchup。Mega Lopunny 与 Grass / Ogerpon 才是当前最值得定向补样本的两组。

## 哪些“可疑动作”没有被算作失误

攻击时仍有 PLAY、ABILITY、EVOLVE 或 ATTACH 可用，不一定是错误：保留资源、避免下回合手牌被干扰、或直接完成击倒都可能更优。更重要的是，这些信号在胜局中反而更常见：

- 攻击时仍有能力：败局 16.3%，胜局 60.9%。
- 攻击时仍有手牌可打：败局 58.1%，胜局 93.8%。
- 攻击时仍可进化：败局 34.9%，胜局 46.9%。
- 攻击前仍可贴能量：败局 32.6%，胜局 46.9%。

因此它们只保留为候选信号，没有进入高置信度失误计数。

## Validation `90751124`

用户提供的三个文件对应一场 validation 自对局和双方日志。两份日志的 stdout/stderr 均为空；replay 的败方出现三次高置信度事件：

- 第 4 回合可贴暗能量却结束回合。
- 第 10 回合两次面对 Snorunt 可选目标返回空选择。

这场不进入公开胜率，但和公开败局中的同类模型偏差一致。

## 根因判断

1. **数量头与目标头解耦。** 模型先预测选几个，再按目标分数取 top-k；数量头选 0 会覆盖正确的目标排序。
2. **主阶段没有 END guardrail。** 模型可以在合法 ATTACH / EVOLVE 存在时直接 END，没有最低资源推进约束。
3. **异常兜底不是本次原因。** 精确重放为 0 fallback，广泛的 `except Exception` 虽然应改为显式记录，但没有解释这些 replay 中的失误。
4. **对手牌组辅助信号没有形成强策略路由。** 9,559 次模型调用中，牌组头只有 101 次置信度超过 60%，并且当前输出只进入统计。它可能解释 matchup 适配不足，但不能从 replay 直接证明因果。

## 推荐修改顺序

1. **先加低风险 guardrail。** 模型选择 END 时，若本回合尚未贴能且存在合法 ATTACH，优先贴给可成型的 Grimmsnarl 线；若存在合法 Grimmsnarl ex 进化，除显式反制规则外禁止直接 END。
2. **约束 Poffin 的 optional count。** 已支付 Poffin 成本、板凳有空位且存在合法目标时，将选择下限改为至少 1，并按盘面需要排序 Snorunt / Impidimp。
3. **对 22 场败局做 hard-negative 微调。** 为 END 对合法 ATTACH / EVOLVE、空检索对非空检索构造成对样本；新增三项 replay 事件率作为验收指标，而不只看 action accuracy。
4. **建立定向 matchup 回归。** Mega Lopunny 和 Grass / Ogerpon 各补到至少 30 场，再决定是否需要牌组条件化策略或换牌。
5. **增加反事实模拟。** 将 37 个动作逐一替换为 guardrail 动作，从对应状态续跑，测量真实可挽回场次；这是把“相关失误”升级为“因果损失”的必要步骤。

## 可复现产物

- [`analyze_replays.py`](analyze_replays.py)：逐 replay 对齐、合法性检查、牌组识别与动作信号提取。
- [`summarize_findings.py`](summarize_findings.py)：胜负对照、牌组聚合与逐败局汇总。
- [`replay_submitted_agent.py`](replay_submitted_agent.py)：使用原提交包精确重放动作并读取 fallback 统计。
- [`build_report_artifact.py`](build_report_artifact.py)：构建并校验 Data Analytics 报告快照。
- [`output/review_summary.json`](output/review_summary.json)：报告核心结构化指标。
- [`output/agent_replay_check.json`](output/agent_replay_check.json)：10,546 次动作复现结果。
- [`output/artifact.json`](output/artifact.json)：已通过原生报告校验器的完整 manifest/snapshot。

## 限制

- “高置信度失误”表示当时存在合法且通常增值的动作，agent 却选择空检索或 END；它不保证替换该动作一定改变最终胜负。
- 对手隐藏信息以 agent 当时可见 observation 为准；完整 replay 只用于核对后续结果，不应倒灌到动作评价。
- 小样本 matchup 只用于排序测试优先级，不用于确定换牌或策略强度。
- 快照反映 2026-08-09 时 submission `55328694` 的 episode 列表；Kaggle 后续若追加对局，需要重新运行下载和汇总流程。
