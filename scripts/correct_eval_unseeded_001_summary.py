from __future__ import annotations

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl.unseeded_eval import sha256_file
from scripts.evaluate_unseeded_runtime import write_json


def main() -> int:
    p = argparse.ArgumentParser(description="non-destructive companion correction for EVAL-UNSEEDED-001")
    p.add_argument("--stage-a-summary", type=Path, required=True)
    p.add_argument("--stage-a-games", type=Path, required=True)
    p.add_argument("--stage-b-summary", type=Path, required=True)
    p.add_argument("--stage-b-games", type=Path, required=True)
    p.add_argument("--external-stage-a-peak-rss-kb", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise SystemExit(f"refusing to overwrite correction evidence: {args.output}")
    stage_a = json.loads(args.stage_a_summary.read_text())
    rows_a = [json.loads(line) for line in args.stage_a_games.read_text().splitlines()]
    stage_b = json.loads(args.stage_b_summary.read_text())
    rows_b = [json.loads(line) for line in args.stage_b_games.read_text().splitlines()]
    parent = int(stage_a["peak_rss_kb"])
    max_child = max(int(row.get("peak_rss_kb", 0)) for row in rows_a)
    correction = {"experiment_id":"EVAL-UNSEEDED-001","correction_only":True,
        "historical_files_unchanged":True,
        "historical_sha256":{str(path):sha256_file(path) for path in
            (args.stage_a_summary,args.stage_a_games,args.stage_b_summary,args.stage_b_games)},
        "stage_a":{"parent_peak_rss_kb":parent,"max_child_peak_rss_kb":max_child,
            "max_process_tree_peak_rss_kb":max(max_child,args.external_stage_a_peak_rss_kb),
            "overall_peak_rss_kb":max(parent,max_child,args.external_stage_a_peak_rss_kb),
            "overall_peak_rss_definition":"maximum of historical parent ru_maxrss, per-child ru_maxrss, and external GNU time maximum; historical run did not sample descendant RSS sums",
            "external_gnu_time_peak_rss_kb":args.external_stage_a_peak_rss_kb},
        "stage_b":{"legacy_model_games":stage_b.get("model_games"),
            "legacy_model_games_meaning":"scheduled attempts, not completed model games",
            "scheduled_model_attempts":len(rows_b),
            "started_model_processes":sum(row.get("child_evidence_present") is True for row in rows_b),
            "checkpoint_loaded_games":sum(row.get("checkpoint_hash_verified") is True for row in rows_b),
            "model_action_games":sum(int(row.get("candidate_diagnostics",{}).get("model_actions",0))>0 for row in rows_b),
            "completed_model_games":sum(row.get("normal_terminal") is True for row in rows_b),
            "model_decisions":stage_b.get("model_decisions",0)}}
    write_json(args.output,correction)
    return 0

if __name__ == "__main__": raise SystemExit(main())
