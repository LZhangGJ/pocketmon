#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811")


def main() -> None:
    league = json.loads((ROOT / "state/league.json").read_text())
    for chain, item in league["chains"].items():
        current = item["current"]
        checkpoint = Path(current["checkpoint"])
        metrics_path = checkpoint.with_name("metrics.json")
        metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
        epoch = (metrics.get("epochs") or [{}])[-1]
        chain_root = ROOT / "learners" / chain
        failed = sum(1 for _ in chain_root.glob("generation-*/FAILED.json"))
        successful = sum(
            1
            for path in chain_root.glob("generation-*/metrics.json")
            if not (path.parent / "FAILED.json").exists()
        )
        print(
            f"{chain}\tgeneration={current['generation']}"
            f"\tpublishedAt={current.get('publishedAt', '-')}"
            f"\tfailedArtifacts={failed}\tsuccessMetrics={successful}"
            f"\tkl={epoch.get('approximateKl', '-')}"
            f"\tclip={epoch.get('clipFraction', '-')}"
        )


if __name__ == "__main__":
    main()
