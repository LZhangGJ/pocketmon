# Windows Codex Prompt — 先整合队友源码，再在 Linux 服务器训练

你运行在用户的 Windows 主机上。Windows 只作为代码与调度控制面；所有 replay 处理、缓存构建、GPU 训练、模型导出和 Arena 对局都必须通过 SSH 在 Linux 服务器执行。

## 0. 先纠正一个关键误区

`experiment7_code_for_gpt_2026-08-08.zip` 是队友提供的 **code-review source snapshot**，不是已经适配本仓库的最终代码，也不是可直接运行的训练包或 Agent。

它明确缺少：

- checkpoint / portable `.npz`；
- engine/card catalog；
- replay 与生成缓存；
- opponent class map；
- 对手 Agent 与 Arena 原始日志。

因此：

1. **禁止把 ZIP 当成训练入口直接运行。**
2. **禁止直接运行 ZIP 内的 `runtime_agent/main.py`。** 它依赖缺失的权重与 catalog。
3. ZIP 只用于导入、审查和改写源码；正式训练必须运行 Git 分支中已经整合、测试、commit 的代码。
4. 第一阶段任务是把 ZIP 源码实体化为普通仓库文件并完成适配；不是立即开训。

## 1. 固定仓库与执行拓扑

```text
GitHub repository:  LZhangGJ/pocketmon
remote:              https://github.com/LZhangGJ/pocketmon.git
source branch:       agent/experiment7-multideck-challengers-20260808
work branch:         codex/experiment7-multideck-challengers-20260808

Windows role:        编辑代码、测试、commit、push、SSH 调度、汇总结果
Linux role:          replay、cache、GPU 训练、portable 导出、官方引擎 Arena
Linux repository:   /homes/lzhang/pocketmon
Linux Python:        /homes/lzhang/mypath/new/envs/trans/bin/python
replays:             /homes/lzhang/pocketmon/data/raw/replays/2026-08-06
ladder analysis:     /homes/lzhang/pocketmon/analysis/outputs/top_ladder_2026_08_07_20260808
servers:             doraemon02 doraemon03 doraemon15 doraemon16 doraemon19 doraemon20
primary target:      agents/lucario_rule
primary target deck: agents/lucario_rule/deck.csv
```

禁止修改、合并或 force-push `main`。禁止提交 Kaggle。

## 2. Windows 仓库初始化

在 PowerShell 7 中：

```powershell
$ErrorActionPreference = 'Stop'
$RepoUrl = 'https://github.com/LZhangGJ/pocketmon.git'
$SourceBranch = 'agent/experiment7-multideck-challengers-20260808'
$WorkBranch = 'codex/experiment7-multideck-challengers-20260808'
$LocalRepo = Join-Path $HOME 'source\pocketmon-experiment7'

if (-not (Test-Path (Join-Path $LocalRepo '.git'))) {
    New-Item -ItemType Directory -Force -Path (Split-Path $LocalRepo) | Out-Null
    git clone $RepoUrl $LocalRepo
}

Set-Location $LocalRepo
git remote set-url origin $RepoUrl
git fetch origin --prune
git switch --force-create $WorkBranch "origin/$SourceBranch"
git status --short --branch
git remote -v
```

确认 Windows 有 `git`、PowerShell 7、`ssh`、`scp`。如果 `ssh doraemon02 hostname` 失败，停止并报告连接问题。

## 3. 导入队友源码：只解压和版本化，不执行训练

原始 ZIP 应位于 Windows，例如：

```text
$HOME\Downloads\experiment7_code_for_gpt_2026-08-08.zip
```

验证：

```text
bytes: 94038
SHA-256: 9c0d24067eacee8abc38223dba28d893e5d1e4e9b75204a9ce92a03093558229
```

PowerShell：

```powershell
$Archive = Join-Path $HOME 'Downloads\experiment7_code_for_gpt_2026-08-08.zip'
$ExpectedSha = '9c0d24067eacee8abc38223dba28d893e5d1e4e9b75204a9ce92a03093558229'
$ExpectedBytes = 94038

if (-not (Test-Path $Archive)) { throw "ZIP not found: $Archive" }
if ((Get-Item $Archive).Length -ne $ExpectedBytes) { throw 'ZIP byte-size mismatch' }
if ((Get-FileHash -Algorithm SHA256 $Archive).Hash.ToLowerInvariant() -ne $ExpectedSha) {
    throw 'ZIP SHA-256 mismatch'
}
```

将源码解压到临时目录，检查 `PACKAGE_README.md`、`REVIEW_PROMPT.md` 和 `PACKAGE_MANIFEST.csv`，验证 manifest 中全部文件。然后把源码作为普通文件导入工作分支：

```text
experiment7/high_score_solution/
```

必须保留原目录结构：

```text
experiment7/high_score_solution/data_pipeline/
experiment7/high_score_solution/training/
experiment7/high_score_solution/runtime_agent/
experiment7/high_score_solution/validation/
experiment7/high_score_solution/docs/
```

导入后：

```powershell
python -m compileall -q experiment7\high_score_solution
git add experiment7/high_score_solution
git commit -m "Import teammate Experiment 7 source tree"
git push --set-upstream origin $WorkBranch
```

这一步只是把源代码版本化。不要在这一步声称代码已适配、模型已训练或 Agent 可运行。

## 4. 源码审查与本仓库适配

完整阅读：

```text
experiment7/high_score_solution/PACKAGE_README.md
experiment7/high_score_solution/REVIEW_PROMPT.md
experiment7/high_score_solution/docs/EXPERIMENT7_CLEANROOM_DESIGN.md
experiment7/high_score_solution/data_pipeline/features.py
experiment7/high_score_solution/data_pipeline/tokenizer.py
experiment7/high_score_solution/training/deck_identity_model.py
experiment7/high_score_solution/training/train_multideck_identity.py
experiment7/high_score_solution/runtime_agent/main.py
experiment7/high_score_solution/validation/*
```

然后比较当前仓库：

```text
rl/public_replay.py
rl/features.py
scripts/convert_public_replays.py
scripts/build_replay_deck_map.py
scripts/run_local_match.py
scripts/run_league_schedule.py
agents/lucario_rule/
```

原始队友源码作为高分方案主实现。不要用旧简化 `RL-BC-004 transformer8` 代替，也不要重新设计模型。新增的本仓库桥接与调度代码写入：

```text
experiment7/integration/
```

至少实现：

```text
build_from_pocketmon_replays.py
select_ladder_decks.py
build_target_receipt.py
build_multideck_manifest.py
export_deck_identity_portable.py
materialize_challenger_agent.py
inventory_remote_gpus.py
build_remote_job_matrix.py
summarize_challenger_arena.py
remote_coordinator.sh
remote_worker.sh
remote_train.sh
remote_arena.sh
```

先审查 `train_multideck_identity.py` 的真实接口。它使用可重复的 `--current-source`，因此如果代码确实支持任意数量 source，就直接沿用原生 K-deck balanced fine-tune；不要无理由改回两个牌组。如果发现隐藏的两牌组假设，只做最小、安全、带测试的修复。

## 5. 保持高分方案的模型与训练规格

```text
state_dim                 320
option_dim                176
entity_numeric_dim         12
history_length              8
d_model                    128
attention_heads              4
transformer_blocks           3
ff_dim                      384
dropout                    0.05
opponent_aux_loss_weight   0.05

pretrain_epochs             12
pretrain_batch             128
pretrain_lr               3e-4
finetune_batch_per_deck     48
finetune_lr               1e-4
optimizer                 AdamW
weight_decay              1e-4
```

运行时只允许：自身精确 60 卡牌表、当前合法动作、自己实际经历的最近 8 个历史槽、当前玩家可见的对手证据。对手隐藏手牌、牌库、奖赏卡、未来状态、最终结果、玩家名、目录标签和对手完整牌表严禁进入运行时策略。对手类别头只作低权重训练辅助，禁止运行时门控。

## 6. Windows 修改代码，Linux 只运行固定 commit

纪律：

1. Windows 修改代码与测试；
2. Windows commit、push；
3. 记录 immutable commit SHA；
4. 所有 Linux worker 从该 SHA 建立 detached worktree；
5. worker 禁止自行修改代码；
6. 服务器发现 bug 后回 Windows 修复并 push 新 SHA；
7. 不用 SCP 分发未提交源码。

每轮 push 前：

```powershell
python -m compileall -q experiment7 scripts rl tests
git diff --check
git status --short --branch
```

## 7. 通过 SSH 做服务器探测

```powershell
$Servers = @('doraemon02','doraemon03','doraemon15','doraemon16','doraemon19','doraemon20')
foreach ($Server in $Servers) {
    Write-Host "===== $Server ====="
    ssh $Server "hostname; nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits; test -d /homes/lzhang/pocketmon && echo REPO_OK; test -x /homes/lzhang/mypath/new/envs/trans/bin/python && echo PYTHON_OK"
}
```

不抢占、不终止现有 GPU 任务。每块 GPU 同时最多一个正式训练进程。共享 cache 单写者，其他机器只读。每个 job 使用独立 worktree、run dir、日志、PID 和 receipt。

## 8. 在服务器上先做适配验证，不直接长训练

Windows push 固定 commit 后，远程建立 worktree并运行：

- source compile；
- replay 接口审计；
- action/observation 对齐审计；
- source → current cache 格式桥接；
- 2-batch forward/backward；
- 32–128 decision tiny overfit；
- checkpoint save/reload；
- legal prediction 100%；
- hidden-opponent-field invariance；
- own-deck permutation invariance 与 multiplicity sensitivity；
- chronological split 与 episode isolation；
- PyTorch/NumPy portable parity。

任何一项失败，禁止长训练。

## 9. 从天梯分析筛选多个牌组

唯一牌组候选来源：

```text
/homes/lzhang/pocketmon/analysis/outputs/top_ladder_2026_08_07_20260808
```

递归识别文件，不假设文件名。默认选 6 个、允许 4–6 个、至少覆盖 4 个不同 archetype。每个正式牌组必须：

- 可恢复为合法精确 60 卡牌表；
- 记录来源、天梯排名/分数、deck SHA；
- exact-deck actor episodes >= 40；
- 非强制 policy decisions >= 800；
- chronological fit/calibration/holdout 均非空；
- 不以 sealed holdout 或最终 Lucario 对局结果作为入选信号。

不足 4 个时停止正式训练并报告，不用手写或低分牌组凑数。

## 10. 训练

### Shared broad pretrain

- 所有兼容高手 replay 的非强制决策；
- 12 epochs；
- batch 128；
- lr 3e-4；
- AdamW，weight decay 1e-4；
- 只训练一次并冻结输入/配置/checkpoint SHA；
- 不读取正式 holdout。

### K-deck balanced fine-tune

- 复用高分代码原生多 source 训练；
- 每 deck 每批参考 48；
- lr 1e-4；
- seeds: 20260808, 20260809, 20260810；
- 每 epoch 报告每 deck calibration exactSemantic；
- checkpoint 按所有 deck calibration exactSemantic 的无权宏平均选择；
- checkpoint 冻结后，holdout 每 seed 只评估一次。

不要启动 PPO、AWR、IQL、Action-Q、自博弈或遗传牌组搜索。

## 11. 导出与 Agent 打包

只有训练完成后才构造 `runtime_agent` 所需缺失资产。实现 checkpoint → `.npz` 导出和 engine catalog 适配。每个 Challenger package 必须含：

```text
main.py
features.py
tokenizer.py
portable.py
deck_identity_portable.py
deck_identity_bc.npz
engine_catalog.json
deck.csv
receipt.json
```

抽查至少 500 个真实决策，PyTorch 与 portable stable action ranking mismatch 必须为 0。Linux CPU 测单线程 mean/p95/max latency。缺文件、load error、inference error、illegal 或 fallback 均失败。

## 12. Arena

目标固定为冻结的：

```text
agents/lucario_rule
agents/lucario_rule/deck.csv
```

训练前冻结 target source、deck、package、engine SHA。所有 Challenger 使用同一 target 与 engine。

分阶段：

1. 20 局/Challenger smoke，双方座位各 10；
2. 100 局/通过 smoke 的 Challenger 预筛；
3. 前 3 名各 200 局正式确认，双方座位各 100；
4. 400 局独立确认只有用户再次授权后才运行。

200 局成功标准：score rate >= 0.55，95% Wilson lower bound > 0.50，任一座位 >= 0.45，全部错误与 fallback 为 0。目标是至少两个不同 archetype 通过。

## 13. Git 与最终交付

提交到：

```text
codex/experiment7-multideck-challengers-20260808
```

允许提交源代码、测试、配置和小型摘要。禁止提交 replay、cache、checkpoint、`.npz`、engine、凭证和大型逐局日志。禁止提交 Kaggle。

最终报告必须明确区分：已验证事实、实验支持的推断、未验证假设。不得把源包的历史 `results/experiment7_summary.json` 当成本次复现结果。

现在开始时，第一步是 **导入并版本化 ZIP 内源码，完成仓库适配和测试**；不是直接运行 ZIP 或直接启动训练。
