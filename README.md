# PTCG AI Battle Challenge Simulation 本地环境

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
C:/Users/DHU_Z/AppData/Local/Programs/Python/Python312/python.exe scripts/download_ptcg_data.py --max-episodes 50
```

下载指定日期（例如 2026-07-05）的前 200 条 replay：

```powershell
C:/Users/DHU_Z/AppData/Local/Programs/Python/Python312/python.exe scripts/download_ptcg_data.py --date 2026-07-05 --max-episodes 200
```

## 3) 目录结构

```text
pocketmon/
  data/raw/replays/
    _index/manifest.csv
    YYYY-MM-DD/
      manifest.csv
      <episode_id>.json
  scripts/download_ptcg_data.py
```

## 4) 训练一个最小 baseline（多数投票）

先把 replay 提取成训练表：

```powershell
C:/Users/DHU_Z/AppData/Local/Programs/Python/Python312/python.exe scripts/prepare_baseline_dataset.py --date 2026-07-15
```

训练 baseline 模型：

```powershell
C:/Users/DHU_Z/AppData/Local/Programs/Python/Python312/python.exe scripts/train_majority_baseline.py
```

对单条 `select` 做预测（推荐文件输入，避免 shell 转义问题）：

```powershell
C:/Users/DHU_Z/AppData/Local/Programs/Python/Python312/python.exe scripts/predict_with_majority_baseline.py --model models/majority_baseline.json --select-file data/processed/sample_select.json
```
