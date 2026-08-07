# PTCG AI Battle 金牌冲刺情报与行动方案

**情报快照：2026-08-06。** 已下载并审计 Kaggle 公开 Discussion 202 篇、消息 1,011 条，以及按 Code 页 Public Score 排序观察到的前 20 个 Notebook。Notebook 分数范围为 774.0–947.5；该分数是动态快照，不等于最终稳定实力。

## Executive Summary

1. **不要在最后十天单押纯 RL。** 前 20 个公开高分 Notebook 中，没有一个可证明是纯 RL/纯 BC；唯一明显含学习模型的第 3 名，实际是“学习型 ensemble + 多专家路由 + 大量手写 guard”的厚混合系统。公开强者的共识是：牌组专精、动作排序、对局路由、合法性/残局护栏，再用受控搜索或小模型做增量。
2. **我们的主线应从“Lucario 单体规则 Agent + 尚未超越教师的 BC”升级为双槽混合组合。** 槽 A 追求对当前主流的最高加权胜率；槽 B 使用不同牌组/失败模式做对冲。两个最终 Agent 都必须有完整规则 fallback；模型只能在高置信、经过离线提升验证的决策上覆盖规则策略。
3. **Public Score 不能当优化目标。** Discussion 中同一字节 Agent 两次提交曾相差约 400 rating；官方采用带不确定性的动态匹配评分，新 Agent 又有加速对局。候选晋级必须以本地成对换座、固定对手快照和 Wilson 区间为主，线上评分只做外部校准。
4. **最优短期技术赌注是：强规则教师 + 置信度门控 BC + 仅限关键节点的搜索。** BC 当前三种子 sequence exact 仅 0.448626，仍不适合全权控制；但它非常适合做候选排序、置信度/异常检测和规则残差。搜索只有在叶价值可靠、调用隔离且收益超过噪声时才开启。

## 下载与审计范围

- Discussion：202/202 个主题，1,011 条主帖/回复，保存为原始 JSON 和可搜索 Markdown。
- Notebook：前 20 个全部保存源码与元数据；静态审计包含代码单元、函数/类、搜索/模型信号、内嵌压缩资产和源码相似度。
- 去重：20 个条目折叠为约 10 个方法谱系。排名 4/8/14 的 Alakazam 源码完全相同；排名 10/16 的策略相似度 95.7%；排名 1/12 的 meta builder 相似度 94.2%。所以不能把 20 个条目视为 20 票独立证据。
- 注意：排名 1 的 “Meta Snapshot” 实际使用 2026-06-29 左右的数据，并明确标记 `NEED_MORE_DATA`；它的 947.5 高分不能证明其 Archaludon/Alakazam 组合仍适合 8 月元环境。

## 高分 Notebook 的方法地图

| 观察排名 | 分数 | 方法 | 我们应该吸收什么 | 主要风险 |
|---:|---:|---|---|---|
| 1 | 947.5 | 元环境统计 + 两个规则 Agent 的组合构建器 | payoff matrix、组合互补、holdout/Wilson 门槛 | 数据截至 6 月底，严重过期 |
| 2 | 933.8 | Lucario 强启发式 + deterministic beam/rollout | 用规则缩小候选，再做有限前瞻 | 叶价值错误会覆盖正确节奏 |
| 3 | 909.3 | Grimmsnarl 学习 ensemble + expert router + 大量 guard | 这是最强的混合模板：模型建议、专家投票、窄 guard、合法 fallback | 系统庞大；模型本身不是胜因的独立证明 |
| 4/8/14 | 892.4/840.3/790.1 | 同一份 Alakazam 启发式 + 隐藏信息采样 + 2-ply minimax | 关键决策才搜索；可见对手线启用 matchup tech | 完全相同代码仍相差 102.3 分，直接展示榜分噪声 |
| 5/7/17/20 | 864.6/846.8/779.7/774.0 | Archaludon Metal specialist、matchup routing、严格回退 | 牌组专精、对局检测、负结果保留、固定 deck/policy | 部分结论来自早期 meta；搜索实验甚至被作者否决 |
| 6/9/10/16 | 851.5/828.4/817.6/784.8 | Lucario 规则、Crustle-aware、rollout/expectimax 变体 | 可直接成为我们的 Lucario 对照组和教师家族 | 同源变体多，Public Score 不支持细粒度优劣 |
| 11/18 | 814.1/779.3 | 极简 Crustle 规则，靠牌组稳定性取胜 | “让牌组降低策略难度”；setup 后 attack last | 日一策略，已被后续 meta 针对 |
| 13/19 | 798.3/778.2 | Alakazam 可见对手 belief/template + 有门槛搜索 | 只用公开可见信息激活对局模板；搜索必须有 margin gate | 隐藏状态采样偏差、运行时与 RNG 污染 |
| 15 | 789.9 | Dragapult specialist 规则 Agent | 作为 Grimmsnarl 的潜在反制轴和不同失败模式 | Hammer/Pult 策略复杂，规则实现难度高 |

### 真正的共同结构

公开高分方法不是“规则 vs RL”的二选一，而是以下流水线：

`牌组降低决策难度 → 专用规则/专家产生可靠基线 → 对局路由 → 关键节点受控搜索或模型建议 → 窄 guard 修复已知灾难 → 任何异常回退到合法规则动作`

排名 3 的压缩资产尤其说明这一点：其中既有 `policy_ensemble.bin.gz`，也有 mirror/tempo/coalition experts、strategic/tactical/development/robustness guards，以及多代 handwritten policy。它更像“学习模型嵌入专家系统”，不是端到端 RL。

## Discussion 的关键结论

### 1. 搜索不是免费午餐

- 一篇系统复盘中，beam search 给弱 Alakazam 启发式带来 +11.3pp，却让强 Starmie 策略下降约 15pp。粗糙叶价值会用“即时奖品”覆盖正确的铺场/节奏。
- 另一个高分 builder 实现了 Search API attack oracle，但在隔离评测后没有显著增益，因此没有 ship。
- 不完全信息搜索需要对对手牌组/手牌做合理采样；如果 value head 不能可靠比较局面，增加深度只会更自信地犯错。

**含义：** 我们只在高价值、低分支、规则与模型意见冲突且价值头已校准的节点搜索；每次 search override 必须要求最小价值 margin，并保留超时/异常回退。

### 2. BC 高准确率仍可能输给教师

- 社区复盘中，DAgger/BC 即使与教师动作分歧低于 8%，on-policy 仍低于镜像 50%；小错误把 Agent 推入教师数据很少覆盖的状态。
- 最新讨论中，有人约 72.8% validation accuracy 只得到约 600 ELO；也有人约 75% accuracy 曾到 928，随后跌至约 700。
- 公开纯自博弈经验指出，达到银牌需要大量游戏；较强 Agent 训练约一天、3–5 million games。先用强公开 replay 做 BC 验证架构，成本约为直接 RL 的一小部分。

**含义：** 我们的 BC 三种子 sequence exact 0.448626、set exact 0.454818、optional-empty 0.354701、multi-select 0.473545，且虽然解码动作全合法，但尚未达到替换规则教师的证据门槛。短期角色应是 residual/candidate ranker，而不是全权 pilot。

### 3. 评测噪声足以制造假进步

- 8 月最新 ELO 讨论记录了同一 Agent 两次提交约 400 分差；另有同一 Agent 一份低于 600、另一份一度约 1,000。
- 系统复盘曾把同一配置在 2,000 局中测出 1.37pp 差异；若 hill-climb 接受阈值仅 0.5pp，就会持续学习噪声。
- 官方评分从 μ=600 初始化，按相近 rating 匹配，并随游戏降低不确定性；新 Agent 的对局频率更高，只有最近两个提交保持活跃。

**含义：** 不从单次线上分数晋级；线上只回答“是否明显崩溃/是否进入预期 rating band”，本地严谨联赛决定代码是否替换。

### 4. 当前 meta 是循环，不是单一最强牌组

- 7 月 26 日公开 meta 追踪显示 Grimmsnarl 约占 51.3%，并描述 `Grimmsnarl > Alakazam > Garchomp > Grimmsnarl` 的循环；该数据已经有时效风险。
- 8 月讨论仍认为 Grimmsnarl 因对 Alakazam/Crustle 的良好路线和大量 replay/BC 数据而流行，但有经验玩家指出 Dragapult + Crushing Hammer 是可行反制，只是 pilot 更难写。
- 早期 15,225 局样例牌组结果中，Lucario 60.4%、Dragapult 55.6%、Iono 43.8%、Abomasnow 40.2%；这反映“样例牌组 + 样例 pilot”，不能代表 8 月牌组上限。

**含义：** 最终两个槽不能是同一策略的小改版。应从 Grimmsnarl/Alakazam/Garchomp/Dragapult-Hammer/当前 Lucario 中按最新 replay payoff matrix 选择一主一反制；牌组决定必须每日刷新，而不是照抄 6 月底高票 Notebook。

## 我们的金牌方案

### 槽 A：Meta-weighted champion

目标是最大化对预期高 rating 对手分布的加权胜率，而不是全场平均。候选至少包含：

1. **Grimmsnarl hybrid**：以公开第 3 名结构为模板，保留我们的可审计规则核心，加入 BC candidate ranking/置信度和窄 guard；
2. **Alakazam bounded-search specialist**：只在可稳定 determinize 的关键决策启用 2-ply/beam，叶价值由已校准价值头 + 明确奖品/KO/节奏规则共同决定；
3. **现有 Lucario champion**：作为稳定基线与特定 matchups 的候选，但不默认它仍是 8 月最优 deck。

晋级条件：对最新对手池的加权胜率、最差 matchup 和 failure rate 同时优于当前 champion；不能只看均值。

### 槽 B：Decorrelated counter

槽 B 与槽 A 必须在牌组、资源曲线和主要克制关系上不同。若 A 是 Grimmsnarl/Alakazam，则优先审查：

- **Dragapult + Crushing Hammer**：潜在压制 Grimmsnarl，失败模式与 A 不同；
- **Garchomp**：针对 Alakazam，同时与 Grimmsnarl形成循环对冲；
- 若 specialist pilot 无法达到可靠性门槛，则用成熟的 Alakazam/Metal 规则 Agent，而不是提交未验证的复杂反制。

组合选择使用二人零和 payoff matrix：最大化两槽对预期 top-band meta 的组合覆盖，同时最小化共同 hard counter。不要简单选离线分数最高的两个同源 Agent。

### 模型与搜索的职责边界

- **规则策略拥有最终合法性和灾难 guard。** 包括：立即 KO、奖品牌序、deck-out、过量能量、错误 retreat、无 Active、重复 ability、超时。
- **BC 只做三件事：** 候选动作排序、规则不确定性识别、对规则动作的窄 residual。若 top-1 置信度低、动作 OOD 或与 guard 冲突，立即回退。
- **搜索只在门控后运行：** MAIN context、选项数受限、时间预算约 0.2–0.8 秒、多个 hidden-state determinization、override margin 足够大。所有 search 调用必须进程隔离或证明不污染真实 RNG。
- **RL 暂不作为 deadline 前的主晋级路径。** 继续作为 deadline 后/报告创新线；只有当自博弈策略在冻结联赛上稳定超越规则/BC hybrid，才有资格进入提交候选。

## 未来 10 天执行计划

| 时间 | 必须交付 | 退出/晋级门槛 |
|---|---|---|
| 8/6–8/7 | 把 10 个公开方法谱系转换为可运行、哈希冻结的 opponent pool；抽取 Grim/Alakazam/Metal/Lucario/Dragapult 代表 | 每个对手可重复运行；0 crash/timeout/illegal |
| 8/7–8/8 | 用最新公开 replay 重建 deck clusters、top-band 使用率与 payoff matrix | 不使用 6 月快照决定牌组；标注样本量/Wilson 区间 |
| 8/8–8/10 | 在现有 Lucario 上加入 BC confidence gate；并行实现一个 Grim hybrid 或 Alakazam bounded-search challenger | 模型覆盖率可观测；每次 override 可回放；fallback 100% 可用 |
| 8/10–8/12 | 快筛：每 matchup 200–400 个 paired-seat games；只保留明显胜者 | 先看失败率、最差 matchup、配对差值区间；不做 0.5pp hill-climb |
| 8/12–8/14 | 确认赛：前 2–3 个候选扩大到约 1,000+ 局/关键 matchup，冻结代码和对手 hash | 对 champion 有可解释且超出噪声的提升；0 功能失败 |
| 8/14–8/15 | 保留一个线上 champion，另一个槽逐个测试 challenger；至少观察一个稳定窗口 | Public Score 仅作异常/量级校准，不因短时 spike 晋级 |
| 8/15–8/16 | 上传最终两个互补、已冻结 Agent，停止高风险改动 | 两槽代码/牌组/依赖/大小/自对局 validation 全通过 |

## 评测协议

1. 每个候选与对手用相同随机条件做双方换座；报告 paired win difference 与 Wilson/Bootstrap 区间。
2. 对手池按 `archetype × pilot × snapshot hash` 固定，至少覆盖公开强规则 Agent、搜索 Agent、Grim hybrid 和简单 exploiters。
3. 主要指标：top-band meta 加权胜率、最差 matchup、10% CVaR、第一/第二座差、平均耗时/p95、crash/timeout/illegal、规则 fallback 率、模型/search override 率及其反事实收益。
4. 分阶段预算：200–400 局筛选，约 1,000+ 局确认；差异越小，所需样本越大。任何宣称小于约 2pp 的提升都要先做功效分析。
5. 保留负结果；相同策略不得因换 random seed 或重新提交就被当成新方法。

## Recommended next steps

1. 立即把已下载前 20 Notebook 中的 10 个方法谱系变成冻结 opponent fixtures；优先第 3 名 Grim hybrid、第 2 名 Lucario search、第 4/8/14 名 Sol Eclipse、第 5/7 名 Metal。
2. 在最新 replay 上重算 top-band meta payoff matrix，再决定槽 A/B 的牌组；这是当前最大的信息缺口。
3. 将 RL-BC-002 的目标从“替换规则 Agent”调整为“confidence-gated residual/candidate ranker”，新增覆盖率、override precision 和 on-policy matchup 指标。
4. 选一个高杠杆 search challenger，但先做 placebo：search 结果不使用时，真实游戏轨迹必须与无 search 完全一致；否则评测被 RNG 污染。
5. 8 月 12 日前冻结最终确认赛协议，8 月 15 日冻结两份最终包；8 月 16 日只做已验证上传。

## Further questions

- 最新官方 replay 中，rating top band 的 Grimmsnarl/Alakazam/Garchomp/Dragapult 实际份额与互胜矩阵是什么？
- 我们能否从第 3 名 Grim hybrid 的专家/guard 结构中抽取可兼容模块，而不引入其庞大依赖和不可解释覆盖？
- 当前 BC 在“高置信覆盖 10%/25%/50% 决策”下，override 的净胜率提升分别是多少？
- Search API 在最终 Kaggle runtime 中是否稳定可用；进程隔离后是否仍有正收益？
- 两个最终槽的共同 hard counter 是什么，是否需要牺牲少量平均胜率换更低尾部风险？

## Caveats and Assumptions

- Notebook Public Score、票数和排序均为 2026-08-06 观察快照，会随 ladder 继续变化。
- 公开 Notebook 可能是提交 builder、复制/改版或过期 meta 报告；高分不能单独证明方法因果有效。
- Discussion 中的 rating、胜率和方法归类有自报与推断成分；本报告仅把它们作为与源码/本地结果交叉验证后的方向性证据。
- 7 月 26 日 meta 数据和 6 月 29 日高分 Notebook 已过期；最终牌组建议必须由最新 replay 重新验证。
- 本报告没有使用或保存聊天中暴露的 Kaggle API Token。

## Sources

- Kaggle competition overview/evaluation: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview/description
- Discussion archive index: `discussions/index.csv`
- Notebook ranking snapshot: `notebooks/index.csv`
- Static notebook audit: `analysis/notebook_audit.csv` and `analysis/notebook_audit.json`
- Key discussions: 713608, 717697, 724187, 724362, 729644, 729926, 731739, 732905, 733083
- Repository baselines: `README.md`, `docs/CODEX_CLI_HANDOFF.md`, `docs/RL_BC_002_PLAN.md`, `docs/FAILURE_MODES.md`
