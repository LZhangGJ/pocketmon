# 本地 Codex 执行任务：整合并训练 Experiment 7

你正在 `LZhangGJ/pocketmon` 仓库中工作。目标是把已正式合队后取得的 Experiment 7 队友代码接入当前数据与本地引擎，完成可审计的训练闭环。不要把历史汇总数字当作本次运行结果；所有结论必须来自本机/服务器重新生成的证据。

## 固定边界

- 源分支：`agent/experiment7-team-code-20260808`
- 代码包：`experiment7/source/experiment7_code_for_gpt_2026-08-08.zip`
- 代码包 SHA-256：`9c0d24067eacee8abc38223dba28d893e5d1e4e9b75204a9ce92a03093558229`
- 基线分支：`exp/league-v1`
- 禁止修改、合并或 force-push `main`。
- 不提交 Kaggle；本任务只做本地数据、训练、便携推理与本地 Arena 门禁。
- 不得提交 replay、缓存、权重、`.npz/.pt`、引擎、凭证或逐局大日志。只提交源代码、配置、审计摘要和小型结果收据。
- `KAGGLE_API_TOKEN` 只从已有环境变量读取，禁止打印、写入文件或提交。
- 对手隐藏手牌、牌库、奖赏卡、未来状态、最终胜负、目录标签、玩家身份和对手完整牌表不得进入运行时特征。

## 可用基础设施

- 仓库：`/homes/lzhang/pocketmon`
- 公开 replay：`/homes/lzhang/pocketmon/data/raw/replays/2026-08-06`
- Python：`/homes/lzhang/mypath/new/envs/trans/bin/python`
- 可用服务器：`doraemon02 doraemon03 doraemon15 doraemon16 doraemon19 doraemon20`
- 服务器共享存储；每台服务器必须使用独立 git worktree 和独立运行目录，严禁多个进程写同一缓存或 checkpoint。

## 总体要求

先审计、再适配、再 smoke、最后正式训练。不要为了“跑起来”静默删样本、放松合法性门禁、打开留出集或改动实验 7 的核心语义。发现源包缺失脚本或接口时，补最小、安全、可测试的桥接实现，并在报告中明确标记为“本仓库适配代码”，不要伪称为队友原始代码。

## 1. 建立隔离工作区

```bash
cd /homes/lzhang/pocketmon
git fetch origin
git worktree add /homes/lzhang/worktrees/experiment7-$(hostname) \
  origin/agent/experiment7-team-code-20260808
cd /homes/lzhang/worktrees/experiment7-$(hostname)
git switch -c agent/experiment7-train-integration-$(hostname)
export PYTHON=/homes/lzhang/mypath/new/envs/trans/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

确认：

```bash
git status --short --branch
git rev-parse HEAD
sha256sum experiment7/source/experiment7_code_for_gpt_2026-08-08.zip
bash experiment7/unpack_source.sh
```

解压目录固定为：

```text
runs/experiment7/source
```

先阅读：

1. `runs/experiment7/source/PACKAGE_README.md`
2. `runs/experiment7/source/REVIEW_PROMPT.md`
3. `runs/experiment7/source/docs/EXPERIMENT7_CLEANROOM_DESIGN.md`
4. `runs/experiment7/source/data_pipeline/features.py`
5. `runs/experiment7/source/data_pipeline/tokenizer.py`
6. `runs/experiment7/source/training/deck_identity_model.py`
7. `runs/experiment7/source/training/train_multideck_identity.py`
8. `runs/experiment7/source/runtime_agent/main.py`

## 2. 静态完整性与依赖审计

执行并保存输出：

```bash
$PYTHON -m compileall -q runs/experiment7/source
$PYTHON - <<'PY'
import csv, hashlib
from pathlib import Path
root = Path('runs/experiment7/source')
rows = list(csv.DictReader((root/'PACKAGE_MANIFEST.csv').open(encoding='utf-8-sig')))
errors = []
for row in rows:
    p = root / row['path']
    if not p.is_file():
        errors.append(f"missing:{row['path']}")
        continue
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual.lower() != row['sha256'].lower():
        errors.append(f"sha:{row['path']}")
if errors:
    raise SystemExit('\n'.join(errors))
print({'manifest_files': len(rows), 'errors': 0})
PY
```

生成 `runs/experiment7/audit/source_audit.json`，至少记录：源 commit、ZIP SHA、manifest 文件数、编译结果、Python/Torch/NumPy/CUDA 版本、主机、GPU、检查时间。

同时确认并记录源包明确缺失的资产：训练权重、portable `.npz`、engine catalog、replay/cache、opponent class map、对手 Agent、逐局 Arena 日志。不得伪造或从汇总 JSON 反推这些资产。

## 3. 先做接口差异分析，不要直接开训

比较两边数据接口：

- 当前仓库：`rl/public_replay.py`、`rl/features.py`、`scripts/convert_public_replays.py`、`scripts/build_replay_deck_map.py`
- Experiment 7：`data_pipeline/build_expert_v1_from_zips.py`、`build_token_cache.py`、`build_sequence_cache.py`、`build_deck_identity_cache.py`

输出 `runs/experiment7/audit/integration_plan.md`，逐项说明：

- replay action/observation 对齐；
- episode 时间戳来源；
- module version；
- actor 自身完整 60 张牌表来源；
- 对手牌表 hash 仅作为 fit-only 辅助标签的来源；
- engine card/attack catalog 来源；
- 当前版本两个 exact-deck arm 的选择方法；
- pretrain/current/calibration/holdout 的划分；
- 现有源包中缺少 portable 权重导出脚本的问题及最小修复。

如果原 `build_expert_v1_from_zips.py` 依赖的 frozen ZIP catalog 在服务器不存在，不要伪造 catalog。新增桥接脚本，例如：

```text
experiment7/integration/build_from_pocketmon_replays.py
```

桥接脚本必须直接读取当前真实 replay/deck-map，并输出与 Experiment 7 cache 构建器兼容的 `decisions.jsonl.gz`、`features.npz`、`decisions.csv` 和审计收据。保留源包目录不变。

## 4. 数据门禁

以完整 episode 为最小单位，先扫描 `2026-08-06` replay，并根据真实数据决定：

- 当前 module version；
- 可兼容的相邻旧 module version 预训练窗口；
- 当前版本中样本量最大的 exact own-deck 变体；
- 至少两个 exact-deck arm。不要使用源包中被清洗掉的历史内部路径作为真实资产位置。

选择两个 arm 时输出完整但不泄露隐藏信息的审计：episode 数、actor-episode 数、非强制决策数、时间范围、精确牌表 SHA、牌表频率、重复 episode 数、无效动作数。不得以最终胜率或留出集表现选择牌表。

强制门禁：

- action 与前一时刻 actor observation 的合法率必须为 100%；
- invalid、unknown status、conflicting episode、duplicate episode、future-history use 均为 0；
- 同一 episode 不得跨 fit/calibration/holdout；
- chronological calibration 与 holdout 必须使用后发生的 episode；
- holdout 在 checkpoint 选择前保持封闭；
- 强制动作不进入策略训练；
- opponent hidden fields 的置乱/删除不能改变运行时特征；
- own-deck 顺序置乱不能改变 deck token，但同名牌数量改变必须改变 token。

若任一门禁失败，停止正式训练，保留失败报告并修复数据或代码。

## 5. 构建 Experiment 7 缓存

为以下三类数据分别生成 feature/token/sequence/identity cache：

1. 广覆盖 pretrain；
2. 当前版本 exact deck A；
3. 当前版本 exact deck B。

使用源包的语义：

- state 320 维；
- option 176 维；
- entity numeric 12 维；
- history length 8；
- own deck 60 张多重集合；
- opponent token 只聚合 actor-visible 对手实体；
- opponent class map 只用 fit episode 构造，calibration/holdout 不参与类别选择。

每个缓存保存 manifest、输入 SHA、行数、episode 数、时间范围、shape、dtype、非法/跳过计数。缓存写入：

```text
/homes/lzhang/pocketmon/runs/experiment7-data-20260806/
```

只允许一个主机生成共享缓存。其他主机只读使用。

## 6. 单元测试与最小过拟合

在正式训练前新增并运行测试，至少覆盖：

- exact own-deck permutation invariance；
- deck multiplicity sensitivity；
- hidden-opponent-field permutation invariance；
- history 只含过去 8 个 completed decisions；
- empty history finite；
- legal option mask；
- count head 在 `[minCount,maxCount]` 内裁剪；
- set-semantic 与 ordered-semantic 两类评价；
- stable tie ordering；
- episode reset；
- PyTorch 与 NumPy portable 前向/排序一致。

先用极小数据运行：

- 2 batch forward/backward；
- 32–128 个决策过拟合；
- finite loss；
- legal prediction 100%；
- checkpoint 保存/加载一致；
- 不打开正式 holdout。

将 smoke 结果写到 `runs/experiment7/smoke/`。smoke 失败不得启动正式训练。

## 7. 正式训练

使用 `training/train_multideck_identity.py` 的实验 7规格：

- `d_model=128`
- `layers=3`
- `heads=4`
- `ff_dim=384`
- `history_length=8`
- `dropout=0.05`
- `pretrain_epochs=12`
- `pretrain_batch=128`
- `pretrain_lr=3e-4`
- `finetune_batch_per_deck=48`
- `finetune_lr=1e-4`
- `opponent_loss_weight=0.05`
- AdamW，weight decay 1e-4
- 选模指标：两个 current exact-deck calibration 的无权 `exactSemantic` 宏平均

第一轮在 `doraemon03` 跑 seed `20260807`。训练时保持 holdout 封闭，不传 `--evaluate-holdout`。若 calibration 最佳 epoch 已确定，冻结 checkpoint、配置和输入 hash 后，再单独执行一次 holdout evaluator。

建议运行目录：

```text
/homes/lzhang/pocketmon/runs/experiment7-train-20260806-seed20260807/
```

完整保存逐 epoch：policy/count/value/opponent loss、两个 arm 的 calibration、macro exactSemantic、single/multi/count accuracy、peak RSS/VRAM、耗时、异常和 checkpoint SHA。

只有 seed 20260807 全部门禁通过后，才在 `doraemon15` 与 `doraemon16` 启动两个独立 seed。不要三个主机同时重建缓存。

## 8. 一次性 holdout 与基线比较

冻结候选后：

- 选定 checkpoint 不再修改；
- 使用同一两个 exact-deck holdout；
- 只打开一次；
- 与冻结的无 deck-identity 序列 Transformer baseline 比较；
- 报告每个 arm 和无权宏平均的 exactSemantic 及 delta；
- 不根据 holdout 重新调 epoch、特征、牌表或类别映射。

如果当前仓库不存在兼容 baseline checkpoint，先按相同数据/split 训练 baseline，不能拿不同数据分布的 RL-BC-003 数字直接比较。

## 9. Portable 导出与一致性

源包包含 NumPy portable inference，但未包含明确的 checkpoint→`.npz` 导出 CLI。实现最小导出脚本：

```text
experiment7/integration/export_deck_identity_portable.py
```

要求：

- 逐个保存 `state_dict` float32 数组；
- 保存 `config_json`；
- 记录 checkpoint 与 portable SHA；
- 不包含 optimizer；
- 500 个抽样决策中 stable action ranking mismatch 必须为 0；
- CPU 单线程测 mean/p95/max latency；
- 多线程与单线程动作必须一致。

验证成功后，构建一个完整 runtime Agent 目录，包含：

- `main.py`
- `features.py`
- `tokenizer.py`
- `portable.py`
- `deck_identity_portable.py`
- `deck_identity_bc.npz`
- `engine_catalog.json`
- `deck.csv`

任何缺文件、加载异常、非法动作或 fallback 都视为失败。

## 10. 本地对局门禁

训练和 portable 验证通过后，先只跑每个固定对手 20 局、双方各 10 个座位。要求：

- model action count > 0；
- normal terminal；
- engine/load/inference/illegal/timeout/fallback 全部为 0；
- 记录逐局原始结果和 Agent diagnostics。

不要自动启动 200/400 局，除非 20 局门禁通过且用户明确要求继续。不得提交 Kaggle。

## 11. 代码与证据提交

提交到当前 integration 分支的内容仅限：

- `experiment7/integration/` 新增桥接、导出、打包脚本；
- 新增测试；
- 小型配置模板；
- `docs/EXPERIMENT7_LOCAL_REPRODUCTION.md`；
- 小型审计/结果摘要，不能包含内部绝对资产路径以外的凭证或敏感值。

不要提交：replay、cache、checkpoint、portable 权重、引擎、逐局大 CSV、Kaggle token。

每次提交前运行：

```bash
$PYTHON -m unittest discover -s tests -p 'test_*.py'
$PYTHON -m compileall -q rl scripts tests experiment7/integration

git status --short
git diff --check
```

## 12. 最终报告格式

最终回复必须分成四类：

1. **已验证事实**：commit、主机、命令、输入/checkpoint SHA、数据规模、split、测试、训练结果、资源、portable parity、20 局门禁；
2. **实验支持的推断**：哪些提升可能来自 exact deck、history、visible-opponent token；
3. **未验证假设**：Arena 200/400、Kaggle Elo、Private 表现；
4. **下一步实验**：严格依据失败点或门禁结果。

不得把源包中的历史 `experiment7_summary.json` 当成本次复现结果，也不得声称已达到其 78.19% 或 65.33% 参考数字，除非本次运行独立得到相应证据。
