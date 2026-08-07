from __future__ import annotations

import os
import random

import torch


def seed_deterministically(seed: int) -> None:
    """Seed training and reject known nondeterministic CUDA kernels."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cuda = torch.backends.cuda
        if hasattr(cuda, "enable_flash_sdp"):
            cuda.enable_flash_sdp(False)
        if hasattr(cuda, "enable_mem_efficient_sdp"):
            cuda.enable_mem_efficient_sdp(False)
        if hasattr(cuda, "enable_math_sdp"):
            cuda.enable_math_sdp(True)
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except TypeError:
        torch.use_deterministic_algorithms(True)
