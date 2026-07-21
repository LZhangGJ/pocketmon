from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import math
import socket
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str, label: str) -> str:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


class OfficialCabtModuleFinder(importlib.abc.MetaPathFinder):
    """Resolve CABT's game/sim modules to unmodified competition wrappers."""

    PREFIX = "kaggle_environments.envs.cabt.cg"

    def __init__(self, archive_cg: Path) -> None:
        self.archive_cg = Path(archive_cg).resolve()
        self.targets = {
            f"{self.PREFIX}.game": self.archive_cg / "game.py",
            f"{self.PREFIX}.sim": self.archive_cg / "sim.py",
        }

    def find_spec(self, fullname: str, path=None, target=None):
        source = self.targets.get(fullname)
        if source is None:
            return None
        if not source.is_file():
            raise ImportError(f"official CABT wrapper missing: {source}")
        return importlib.util.spec_from_file_location(fullname, source)


class NetworkBlocker:
    """Fail closed and count socket connection attempts during games."""

    def __init__(self) -> None:
        self.attempts = 0
        self._originals: dict[str, Any] = {}

    def _deny(self, *args, **kwargs):
        self.attempts += 1
        raise RuntimeError("network access disabled during EVAL-UNSEEDED-001")

    def __enter__(self):
        self._originals = {
            "create_connection": socket.create_connection,
            "connect": socket.socket.connect,
            "connect_ex": socket.socket.connect_ex,
        }
        socket.create_connection = self._deny
        socket.socket.connect = self._deny
        socket.socket.connect_ex = self._deny
        return self

    def __exit__(self, exc_type, exc, traceback):
        socket.create_connection = self._originals["create_connection"]
        socket.socket.connect = self._originals["connect"]
        socket.socket.connect_ex = self._originals["connect_ex"]


def alternating_schedule(games: int) -> list[dict[str, Any]]:
    if games <= 0 or games % 2:
        raise ValueError("games must be a positive even number")
    schedule = []
    for position in range(games):
        first_in_seat_zero = position % 2 == 0
        schedule.append({
            "game_id": position + 1,
            "schedule_position": position,
            "seat0": "official_random_first" if first_in_seat_zero else "official_random_second",
            "seat1": "official_random_second" if first_in_seat_zero else "official_random_first",
        })
    return schedule


def approved_terminal(statuses: list[Any], rewards: list[Any], returned_normally: bool) -> bool:
    if not returned_normally or statuses != ["DONE", "DONE"]:
        return False
    if len(rewards) != 2 or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in rewards
    ):
        return False
    return list(rewards) in ([1, -1], [-1, 1], [0, 0])


def outcome_from_rewards(rewards: list[int | float]) -> str | None:
    if rewards == [1, -1]:
        return "seat0_win"
    if rewards == [-1, 1]:
        return "seat1_win"
    if rewards == [0, 0]:
        return "draw"
    return None


def summarize_stage_a(records: list[dict[str, Any]], expected_games: int, network_attempts: int) -> dict[str, Any]:
    normal = sum(bool(row.get("normal_terminal")) for row in records)
    crashes = sum(bool(row.get("process_crash")) for row in records)
    timeouts = sum("TIMEOUT" in row.get("statuses", []) for row in records)
    invalid = sum("INVALID" in row.get("statuses", []) for row in records)
    agent_errors = sum("ERROR" in row.get("statuses", []) for row in records)
    exceptions = sum(bool(row.get("exception")) for row in records)
    gates = {
        "loaded_native_hash_matches": all(row.get("native_hash_verified") is True for row in records),
        "game_count_complete": len(records) == expected_games,
        "normal_terminals": normal == expected_games,
        "crash_zero": crashes == 0,
        "timeout_zero": timeouts == 0,
        "invalid_action_zero": invalid == 0,
        "agent_error_zero": agent_errors == 0,
        "exception_zero": exceptions == 0,
        "network_attempt_zero": network_attempts == 0,
        "no_games_dropped": [row.get("game_id") for row in records] == list(range(1, expected_games + 1)),
    }
    return {
        "games": len(records),
        "normal_terminals": normal,
        "crashes": crashes,
        "timeouts": timeouts,
        "invalid_actions": invalid,
        "agent_errors": agent_errors,
        "exceptions": exceptions,
        "network_attempts": network_attempts,
        "gates": gates,
        "gate_passed": all(gates.values()),
    }


def loaded_native_libraries() -> list[Path]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        raise RuntimeError("/proc/self/maps unavailable; cannot prove loaded native library")
    paths = set()
    for line in maps.read_text(encoding="utf-8").splitlines():
        candidate = line.rsplit(maxsplit=1)[-1]
        if candidate.startswith("/") and Path(candidate).name == "libcg.so":
            paths.add(Path(candidate).resolve())
    return sorted(paths)
