from __future__ import annotations

import json

import numpy as np
import torch


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    left = torch.ones((256, 256), dtype=torch.float16, device=device)
    result = left @ left
    print(json.dumps({
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.cuda.is_available(),
        "device": str(device),
        "sum": float(result.sum().item()),
    }), flush=True)


if __name__ == "__main__":
    main()
