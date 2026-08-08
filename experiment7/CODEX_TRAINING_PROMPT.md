# Windows Codex 主任务：在远程 Linux 服务器训练 Experiment 7 多牌组 Challenger

## 0. 不可误解的执行拓扑

你（Codex）运行在用户的 **Windows 主机**上。Windows 是控制面，只负责：

- 编辑、审查和测试仓库代码；
- 创建 Git 分支、commit、push；
- 通过 Windows OpenSSH 调用远程 Linux 服务器；
- 生成远程任务脚本、监控状态、汇总小型结果收据。

**所有 replay 扫描、缓存构建、GPU 训练、模型导出、官方引擎对局和性能测量都必须在远程 Linux 服务器执行。不要在 Windows 本机寻找 `/homes/...`，不要在 Windows 上训练，也不要把 Linux 路径转换为 Windows 路径。**

固定上下文：

```text
GitHub repository:  LZhangGJ/pocketmon
Git remote:         https://github.com/LZhangGJ/pocketmon.git
source branch:      agent/experiment7-multideck-challengers-20260808
Codex work branch:  codex/experiment7-multideck-challengers-20260808

Windows role:       code + git + SSH orchestration only
Linux repo:         /homes/lzhang/pocketmon
Linux Python:       /homes/lzhang/mypath/new/envs/trans/bin/python
Linux replays:      /homes/lzhang/pocketmon/data/raw/replays/2026-08-06
ladder analysis:    /homes/lzhang/pocketmon/analysis/outputs/top_ladder_2026_08_07_20260808
remote servers:     doraemon02 doraemon03 doraemon15 doraemon16 doraemon19 doraemon20
primary target:     agents/lucario_rule
primary target deck: agents/lucario_rule/deck.csv
```

若 Windows 上没有 Git、PowerShell 7 或 OpenSSH Client，或者 `ssh doraemon02 hostname` 无法成功，先停止并报告环境阻塞，不得伪称已在服务器执行。

## 1. 唯一目标

从高分天梯分析中筛选若干个**不同 archetype 的精确合法 60 卡牌组**，沿用队友提供的高分 Experiment 7 训练实现，构建多个“牌组 + 策略” Challenger，并通过换座位本地对局筛选出能稳定击败当前冻结 Mega Lucario Agent 的方案。

成功目标：至少两个不同 archetype 在全新 200 局、座位平衡的直接对局中达到：

- score rate `>= 0.55`；
- 95% Wilson 下界 `> 0.50`；
- 任一座位 score rate `>= 0.45`；
- crash、timeout、illegal、load error、inference error、fallback 均为 0。

这里比较的是“牌组 + 策略组合”，不得把结果误写成纯模型架构因果增益。

## 2. 以队友高分方案为主实现，禁止另起炉灶

解压后的队友代码位于远程 worktree 的：

```text
runs/experiment7/source
```

优先直接复用：

```text
data_pipeline/features.py
data_pipeline/tokenizer.py
data_pipeline/build_token_cache.py
data_pipeline/build_sequence_cache.py
data_pipeline/build_deck_identity_cache.py
training/deck_identity_model.py
training/train_multideck_identity.py
runtime_agent/deck_identity_portable.py
runtime_agent/portable.py
runtime_agent/main.py
validation/*
```

核心原则：

1. 不用旧的简化 `RL-BC-004 transformer8` 代替队友实现；
2. 不重新设计模型，不随意改变特征维度、token 组织、损失或便携推理；
3. 只补最薄的仓库适配层：当前 replay → 队友数据格式、牌组清单、远程调度、导出、打包和对局汇总；
4. 原始队友代码保持原样，适配代码放在 `experiment7/integration/`；
5. 若发现源实现只支持两个 deck arm：
   - 先判断能否通过很小的 manifest/sampler 修改原生支持任意 deck list；
   - 若需要大改训练器，则不要重写模型，改为共享一次 broad pretrain，然后把候选牌组按多样性分组，运行多组原生双牌组 balanced fine-tune；
   - 每个候选最终都要生成独立 `deck.csv` Agent；
6. 不启动 PPO、AWR、IQL、自博弈或 Action-Q，直到高分 BC 流程和 Arena 门禁完成。

保持高分方案的参考规格：

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
pretrain_epochs              12
pretrain_batch              128
pretrain_lr                3e-4
finetune_batch_per_deck      48
finetune_lr                1e-4
optimizer                  AdamW
weight_decay               1e-4
```

保持信息边界：自身精确 60 卡牌表可以进入模型；对手只使用当时可见证据；对手类别只作低权重训练辅助，不得运行时门控。fileciteturn46file0

## 3. Windows 本地仓库初始化

在 Windows PowerShell 7 中执行。若已有仓库，复用现有 clone；否则克隆到 `$HOME\source\pocketmon-experiment7`。

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

Windows 只执行无需 Linux 引擎/数据/GPU 的静态检查。任何依赖 Linux 路径、CUDA、官方 `cg` 或 replay 的命令必须通过 SSH 远程执行。

## 4. Git/远程运行纪律

Codex 在 Windows 编辑代码后必须：

1. 运行可在 Windows 完成的静态测试；
2. commit 并 push 到 `codex/experiment7-multideck-challengers-20260808`；
3. 记录 commit SHA；
4. 远程所有 worker 从这个**固定 SHA**建立 worktree；
5. worker 不允许自行修改代码；需要修复时回到 Windows 修改、push 新 SHA，然后重启相应阶段。

禁止通过 SCP 把未提交源码散发到不同服务器，避免服务器运行不同代码。小型任务配置和 shell 脚本也应提交到 Git 后再运行。

## 5. Windows 通过 SSH 探测服务器

从 PowerShell 执行：

```powershell
$Servers = @('doraemon02','doraemon03','doraemon15','doraemon16','doraemon19','doraemon20')
foreach ($Server in $Servers) {
    Write-Host "===== $Server ====="
    ssh $Server "hostname; nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits; test -d /homes/lzhang/pocketmon && echo REPO_OK; test -x /homes/lzhang/mypath/new/envs/trans/bin/python && echo PYTHON_OK"
}
```

生成并提交代码模板，但运行产生的清单写到共享存储：

```text
/homes/lzhang/pocketmon/runs/experiment7-multideck/audit/gpu_inventory.json
/homes/lzhang/pocketmon/runs/experiment7-multideck/audit/job_matrix.csv
```

调度规则：

- 不终止或抢占现有进程；
- 优先利用率低于 20% 且显存足够的 GPU；
- 每个 GPU 同时只跑一个正式训练任务；
- 每台服务器使用不同 worktree；
- 共享缓存仅一个 coordinator 写，其他 worker 只读；
- 每个任务使用独立 run dir 和日志；
- 每个远程命令记录主机、GPU、commit、开始/结束时间和 exit code。

## 6. 远程 worktree 启动方式

Windows push 完成后，选择 `doraemon02` 作为默认 coordinator；如其资源不合适，可根据 GPU inventory 改选，但必须记录原因。

以下命令由 Windows 通过 SSH 调用，不能在 Windows 本地直接执行：

```powershell
$Commit = (git rev-parse HEAD).Trim()
$RemoteBranch = 'codex/experiment7-multideck-challengers-20260808'
$RemoteCommand = @"
set -euo pipefail
cd /homes/lzhang/pocketmon
git fetch origin --prune
git cat-file -e $Commit^{commit}
WT=/homes/lzhang/worktrees/experiment7-coordinator-$Commit
if [ ! -d \"$WT\" ]; then
  git worktree add --detach \"$WT\" $Commit
fi
cd \"$WT\"
export PYTHON=/homes/lzhang/mypath/new/envs/trans/bin/python
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
bash experiment7/unpack_source.sh
\$PYTHON -m compileall -q runs/experiment7/source experiment7/integration scripts rl tests
"@
ssh doraemon02 $RemoteCommand
```

注意：PowerShell 字符串转义可按实际环境调整。若复杂命令转义不稳定，优先把 Linux 操作写成已提交的 `experiment7/integration/remote_*.sh`，Windows 只执行：

```powershell
ssh doraemon02 "bash /homes/lzhang/worktrees/<fixed-worktree>/experiment7/integration/remote_coordinator.sh <commit>"
```

## 7. 冻结当前目标 Agent

在训练前，远程生成：

```text
runs/experiment7-multideck/audit/target_agent_receipt.json
```

至少记录：

- `agents/lucario_rule/main.py` SHA-256；
- `agents/lucario_rule/deck.csv` SHA-256；
- 整个目标 Agent 的确定性目录哈希；
- 官方引擎和 native library SHA-256；
- Python/kaggle-environments 版本；
- 构建后的 immutable target package SHA-256。

所有 Challenger 对同一个目标包评测。不得边看比分边修改靶子。

## 8. 从高分天梯中筛选多个初始牌组

远程递归扫描：

```text
/homes/lzhang/pocketmon/analysis/outputs/top_ladder_2026_08_07_20260808
```

先生成输入清单和 SHA，不假设文件名：

```text
runs/experiment7-multideck/audit/ladder_input_inventory.json
runs/experiment7-multideck/audit/deck_candidate_table.csv
runs/experiment7-multideck/audit/deck_selection_receipt.json
```

目标 4–6 个正式候选，尽量覆盖至少 4 个不同 archetype。每个候选必须：

- 来自该天梯分析；
- 能恢复为合法精确 60 卡多重集合；
- 记录排名、分数/胜率、来源和 deck SHA；
- 有足够 replay/deck-map 支持；
- 不使用 sealed holdout 或对 Lucario 的最终比分来入选；
- 去除精确重复和近重复微调版本。

默认支持门槛：

```text
exact-deck actor episodes >= 40
non-forced policy decisions >= 800
fit/calibration/holdout 均非空
chronological holdout 非空
```

不足的高分牌组保留在审计表中，标记 `insufficient_support`，不得用低分随机牌组凑数。

## 9. 数据适配与完整性门禁

复用当前仓库已经审计的 replay action/observation 对齐事实。新增桥接代码，例如：

```text
experiment7/integration/build_from_pocketmon_replays.py
```

输出队友代码可直接消费的 decisions/features/token/sequence/deck-identity cache。不得修改原始队友源码来迁就错误数据。

强制门禁：

- 完整 episode 为最小 split 单位；
- action 对 actor 前一时刻 observation 的合法率 100%；
- duplicate/conflicting episode、unknown status、future-history use 均为 0；
- 强制动作不进入 policy loss；
- chronological fit/calibration/holdout；
- holdout 在 checkpoint 冻结前保持封闭；
- own-deck 顺序置乱不改变 token；
- 卡牌 multiplicity 改变必须改变 token；
- 对手隐藏字段删除/置乱不改变主策略输出；
- 运行时禁止使用对手完整牌表、隐藏手牌、牌库、奖赏卡、未来状态、最终胜负或玩家身份。

## 10. 训练计划：优先原生高分代码路径

### 10.1 Shared broad pretrain

只构建一次，所有候选共享：

```text
12 epochs
batch 128
lr 3e-4
AdamW, weight_decay 1e-4
```

### 10.2 多牌组 fine-tune

先检查 `train_multideck_identity.py` 是否原生接受任意数量 deck manifest：

- 若支持：对 4–6 个候选做等权 balanced fine-tune；选模为各 deck calibration exactSemantic 的无权宏平均；
- 若只支持两个 arm：按 archetype 多样性把候选组成若干双牌组任务，共享同一个 pretrain checkpoint；不要大改模型。确保每个候选至少进入一个平衡任务。

使用正式 seeds：

```text
20260808
20260809
20260810
```

不同 seed 分配到不同空闲 GPU。每个 seed 的 checkpoint 用 calibration 选择，冻结后只打开一次 holdout。

### 10.3 可选 specialist

先完成 universal/paired 模型的 20 局 smoke。之后最多对三个最有希望的牌组从冻结 checkpoint 做 1–3 epoch specialist fine-tune。不能用 200 局最终确认结果反复调 epoch。

## 11. 正式训练前 smoke

远程新增并运行至少以下测试：

- source archive/manifest 完整性；
- compileall；
- deck permutation invariance；
- multiplicity sensitivity；
- hidden-opponent invariance；
- history 只含过去 8 个 completed decisions；
- empty history finite；
- legal option mask；
- count head 范围裁剪；
- semantic-set 与 ordered-semantic；
- stable tie ordering；
- episode reset；
- balanced sampler（原生 K 或双牌组路径）；
- checkpoint save/reload；
- PyTorch 与 NumPy portable parity。

先运行 2-batch forward/backward 和 32–128 决策 tiny overfit。任一 smoke 失败，不得启动长训练。

## 12. Agent 导出

每个候选牌组至少构建一个自包含 Agent：

```text
/homes/lzhang/pocketmon/runs/experiment7-multideck/packages/<deck_id>/<model_id>/
```

包含：

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

对 500 个真实决策验证 PyTorch 与 NumPy stable action ranking mismatch = 0，并在 Linux CPU 单线程测 mean/p95/max latency。多线程与单线程动作必须一致。

## 13. 直接对当前 Agent 的分阶段 Arena

Primary matchup：

```text
Experiment 7 Challenger + 其自身候选牌组
vs
冻结 agents/lucario_rule + 原 deck.csv
```

### Gate A：20 局 smoke/Challenger

- 每个座位 10 局；
- 全新 game IDs；
- model actions > 0；
- 所有运行失败与 fallback 为 0。

### Gate B：100 局筛选/通过 smoke 的 Challenger

- 每个座位 50 局；
- 不复用 smoke；
- 报告 W/D/L、score、Wilson、seat gap。

### Gate C：前 3 名各 200 局确认

- 每个座位 100 局；
- 不复用前序对局；
- 成功门槛见第 1 节；
- 至少两个不同 archetype 通过即完成本轮目标。

不要自动运行 400 局或提交 Kaggle；等待用户再次授权。

## 14. Windows 监控与收据同步

Windows Codex 可以通过 SSH 查看日志、`nvidia-smi` 和状态 JSON，但不要持续抓取巨大日志到 Windows。远程 worker 每个阶段写小型 `status.json` 和 `summary.json`；Windows 仅拉取这些摘要用于判断下一步。

不要在回答中声称任务在后台继续。只有实际发起 SSH 命令并取得 PID/job receipt 后，才可报告远程任务已启动。

## 15. 提交边界

Windows Codex 最终将**代码、测试、配置模板和小型审计摘要** commit/push 到：

```text
codex/experiment7-multideck-challengers-20260808
```

不得提交：

- replay/cache；
- checkpoint 或 `.pt/.pth/.ckpt/.npz`；
- 引擎；
- Kaggle token；
- 大型逐局日志；
- 服务器私密凭证。

每次 push 前：

```powershell
git diff --check
git status --short --branch
```

并通过 SSH 在至少一台 Linux 服务器执行：

```bash
$PYTHON -m unittest discover -s tests -p 'test_*.py'
$PYTHON -m compileall -q rl scripts tests experiment7/integration
```

## 16. 最终报告

必须报告：

1. Windows 本地 clone 路径、source/work branch、最终 commit；
2. 实际 SSH 连通的服务器和 GPU inventory；
3. 每个远程 worker 使用的固定 commit/worktree/run dir；
4. 目标 Agent/deck/engine 收据；
5. 天梯候选表、最终牌组、archetype、deck SHA 和 replay 支持；
6. 数据合法性、时间 split 和泄漏审计；
7. tiny overfit；
8. shared pretrain 和各 fine-tune seed 的逐 epoch指标；
9. 每牌组 calibration/一次性 holdout；
10. portable 500 决策 parity 与 Linux CPU 延迟；
11. 每个 Challenger 的 20/100/200 局结果、Wilson、seat gap 和错误计数；
12. 是否至少两个不同 archetype 击败冻结目标；
13. 已验证事实、实验支持的推断、未验证假设和下一步。

不得把队友包内的 `experiment7_summary.json` 当作本次复现结果，也不得把未实际执行的远程步骤写成已完成。
