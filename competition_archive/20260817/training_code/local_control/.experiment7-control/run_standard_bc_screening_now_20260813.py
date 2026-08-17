from __future__ import annotations

import importlib.util
from pathlib import Path


CONTROLLER = Path("/homes/lzhang/run_bc_replacement_screening_20260813.py")
GUARDED_RUNNER = Path("/homes/lzhang/run_load_guarded_arena_shard.sh")


def main() -> None:
    spec = importlib.util.spec_from_file_location("bc_replacement", CONTROLLER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {CONTROLLER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RUN_SHARD = GUARDED_RUNNER
    portable = module.ROOT / "standard_1m/universal_bc.npz"
    parity = module.ROOT / "standard_1m/parity.json"
    if not portable.is_file() or not parity.is_file():
        raise FileNotFoundError("standard BC portable/parity is not ready")
    # Start the candidate immediately. Baseline and large-profile stages are
    # independently scheduled, so neither blocks this evidence collection.
    module.run_stage("standard_1m-smoke", portable, 2, 16)
    module.run_stage("standard_1m-frozen40", portable, 40, 45)


if __name__ == "__main__":
    main()
