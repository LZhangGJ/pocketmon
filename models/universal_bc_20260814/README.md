# Universal BC checkpoints (2026-08-14)

这里发布当前完成验证的两档通用 Behavior Cloning（BC）模型，以及它们的卡组覆盖清单。

## 模型

| 文件 | 定位 | 参数量 | 最佳 epoch | validation exact-semantic | SHA256 |
|---|---:|---:|---:|---:|---|
| `large_256x6_best_model.pt` | 大容量 | 6,442,756 | 4 | 0.766390 | `3a81fc7d…cc3af` |
| `standard_1m_best_model.pt` | 标准 | 1,053,188 | 4 | 0.763876 | `bb6864f1…7876` |

两者都在 2026-08-02 至 2026-08-11 的 replay 切分上完成验证，共 347,998 个验证决策；最佳 checkpoint 均没有 illegal prediction。大容量版按同档已完成候选中的验证分数选择，标准版同理。

完整架构、验证指标、文件大小和哈希见 `model_manifest.json`。

## 卡组覆盖

- `compatible_archetypes.csv`：34 个卡组 archetype 的简表。
- `compatible_decks.csv`：133 个精确卡组变体，包含选择顺序、证据等级、训练采样权重、原始牌表文件名和 canonical deck SHA256（对排序后的卡牌 multiset JSON 计算，不是 CSV 文件字节哈希）。
- “compatible”在这里表示该卡组进入了本次通用 BC 的观察/训练支持池。模型结构可以编码任意合法 60 张牌表，但不应把未列出的新卡组视为已验证支持。

清单中特别包含 Grimmsnarl / Froslass、Festival Lead Dipplin、Mega Lucario、Raging Bolt / Ogerpon、Mega Lopunny 系、Dragapult 等当前重点卡组。

## 加载 checkpoint

checkpoint 使用 schema version 2，包含 `config`、`state_dict` 和训练 metadata。仓库内对应实现是 `experiment7/integration/universal_deck_model.py`。

```python
from pathlib import Path
import sys
import torch

sys.path.insert(0, "experiment7/integration")
from universal_deck_model import (
    UniversalDeckModelConfig,
    UniversalDeckTransformerPolicy,
)

checkpoint = torch.load(
    Path("models/universal_bc_20260814/standard_1m_best_model.pt"),
    map_location="cpu",
    weights_only=False,
)
config = UniversalDeckModelConfig(**checkpoint["config"])
model = UniversalDeckTransformerPolicy(config)
model.load_state_dict(checkpoint["state_dict"], strict=True)
model.eval()
```

训练/导出入口见 `experiment7/integration/train_universal_bc.py` 和 `experiment7/integration/export_and_package.py`。推理时必须使用与训练一致的特征、card vocabulary 和 action option schema。
