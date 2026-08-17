import gzip
import json

from experiment7.integration.summarize_compact import chain_is_active, generation_external


def _write_rollout(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def test_generation_external_excludes_entire_conflicting_episode(tmp_path):
    rollout = tmp_path / "rollout.jsonl.gz"
    _write_rollout(
        rollout,
        [
            {"episode_id": "clean-win", "player": 0, "self_play": False, "outcome": 1.0},
            {"episode_id": "clean-win", "player": 0, "self_play": False, "outcome": 1.0},
            {"episode_id": "clean-loss", "player": 1, "self_play": False, "outcome": -1.0},
            {"episode_id": "self-play", "player": 0, "self_play": True, "outcome": 1.0},
            {"episode_id": "conflict", "player": 0, "self_play": False, "outcome": 1.0},
            {"episode_id": "conflict", "player": 0, "self_play": False, "outcome": -1.0},
        ],
    )
    result = generation_external(
        [
            {
                "output": {"path": str(rollout), "sha256": "abc"},
                "wins": 1,
                "losses": 1,
                "draws": 0,
                "createdAt": "2026-08-15T00:00:00Z",
            }
        ],
        generation=7,
        snapshot_id="chain-g000007-deadbeef",
    )

    assert result["externalEpisodes"] == 2
    assert (result["wins"], result["losses"], result["draws"]) == (1, 1, 0)
    assert result["seat"]["0"]["episodes"] == 1
    assert result["seat"]["1"]["episodes"] == 1
    assert result["selfPlayEpisodes"] == 1
    assert result["episodeConflicts"] == 1
    assert result["excludedConflictEpisodes"] == 1
    assert "never cumulative" in result["aggregationScope"]


def test_chain_is_active_uses_runtime_controls_and_excludes_retired():
    active = {
        "trainingControl": {
            "rollout": {"enabled": True},
            "learner": {},
        }
    }
    assert chain_is_active("a08_maxbelt_large_g9", active)
    assert not chain_is_active(
        "a08_rabsca",
        {
            **active,
            "retirement": {"status": "retired_disabled"},
        },
    )
    assert not chain_is_active(
        "universal_ppo_standard_1m",
        {"trainingControl": {"rollout": {"enabled": False}, "learner": {}}},
    )
    assert chain_is_active(
        "universal_ppo_large_256x6",
        {"trainingControl": {"learner": {"enabled": True}}},
    )
