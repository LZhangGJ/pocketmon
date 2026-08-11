from pathlib import Path
import sys


integration = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(integration))

from run_async_ppo_learner import select_minimum_batch  # noqa: E402


paths = [Path("one"), Path("two")]
summaries = [{"decisions": 100}, {"decisions": 200}]
selected, _, decisions = select_minimum_batch(paths, summaries, 400)
assert selected == [] and decisions == 300
selected, _, decisions = select_minimum_batch(paths, summaries, 250)
assert selected == paths and decisions == 300
print("MINIMUM_BATCH_TEST_OK")
