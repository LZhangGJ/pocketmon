"""v5 entry point for the image-whitelist evaluation.

v5 intentionally keeps the validated v4 schedule and retry semantics.  The
separate design version prevents checkpoint-refresh rounds from being mixed
with v4 artifacts.
"""

from __future__ import annotations

import run_hourly_cache_pool_eval_v4_image_whitelist as implementation


implementation.EVALUATION_DESIGN_VERSION = 5


if __name__ == "__main__":
    raise SystemExit(implementation.main())
