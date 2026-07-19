# PTCG AI Battle Challenge Simulation 本地环境

## 可提交的规则 Agent

仓库包含一个牌组感知的 Mega Lucario 规则 Agent：`agents/lucario_rule/`。
它根据场面、奖赏卡价值、能量进度和对局类型为合法选项打分，并保留 Hariyama
作为对抗 Crustle wall 的非 ex 路线。

取得官方 `cg` 目录后构建提交包：

```powershell
python scripts/build_submission.py --cg-dir path/to/sample_submission/cg
```

输出为 `dist/lucario_rule_submission.tar.gz`。

使用官方引擎进行一场本地自对弈：

```powershell
python scripts/run_local_match.py --cg-dir path/to/sample_submission/cg
```

使用本地公开对手池进行批量评测（默认每个对手 20 局并交替先后手）：

```powershell
python scripts/run_opponent_pool.py --cg-dir path/to/sample_submission/cg --games 20
```

## RL warm start

The initial RL pipeline records teacher/self-play trajectories and trains a variable-candidate actor-critic. See [`docs/rl-training.md`](docs/rl-training.md) for the design and commands. Training stays local; a model is only added to the submission after it beats the rule fallback across the opponent pool.

已配置内容：
- Python 3.12 环境
- 依赖安装（`kaggle`、`pandas` 等）
- 数据下载脚本：`scripts/download_ptcg_data.py`
- 数据目录：`data/raw/replays/`

## 1) 配置 Kaggle 凭证

比赛数据通过 Kaggle API 获取。先创建 `.env`：

```powershell
Copy-Item .env.example .env
```

然后在 `.env` 填写：

- `KAGGLE_USERNAME`
- `KAGGLE_KEY`

也支持你当前的变量名：`KAGGLE_API_TOKEN`（脚本会自动映射为 `KAGGLE_KEY`）。

## 2) 下载数据

下载最新日期的前 50 条 replay：

```powershell
python scripts/download_ptcg_data.py --max-episodes 50
```

下载指定日期的前 200 条 replay：

```powershell
python scripts/download_ptcg_data.py --date 2026-07-16 --max-episodes 200
```

## 3) DATA-001：验证 replay 动作对齐

禁止直接假设同一 step 内的 `action` 和 `observation` 属于同一个决策。先比较
`action[t] → observation[t-1]` 与 `action[t] → observation[t]`：

```powershell
python scripts/audit_replay_alignment.py --date 2026-07-16 --max-files 100 --strict
```

真实 replay 必须达到合法动作率至少 99.9%，否则不能进入训练。

## 4) DATA-002：生成标准离线 RL 轨迹

```powershell
python scripts/convert_public_replays.py --date 2026-07-16 --alignment previous --policy-source winners
```

输出默认写入 `data/processed/public_replay_v1.jsonl.gz`。每条样本仅保存行动玩家当时
可见的 observation、完整合法选项、已选动作、最终胜负、manifest 元数据和源文件哈希；
默认移除 `observation.logs`。胜方轨迹具有 policy 权重，双方合法轨迹都可用于 value 学习。

## 5) 诊断性 majority baseline

```powershell
python scripts/prepare_baseline_dataset.py --date 2026-07-16
python scripts/train_majority_baseline.py
```

仓库中旧的 `models/majority_baseline.json` 来自未校验的同-step 标签，不能用于 Agent。
预测脚本会拒绝该旧模型；重新训练后的 majority 表也仅用于数据诊断，未见状态必须回退到规则 Agent。

## 6) 目录结构

```text
pocketmon/
  data/raw/replays/
    _index/manifest.csv
    YYYY-MM-DD/
      manifest.csv
      <episode_id>.json
  data/processed/public_replay_v1.jsonl.gz
  results/data001_replay_alignment.json
  results/data002_public_replay_conversion.json
```
