from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import math
import socket
import subprocess
import sys
import types
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


def install_agent_cg_alias(archive_cg: Path, loaded_sim: Any) -> types.ModuleType:
    """Expose the verified CABT sim to sample agents without reinitializing libcg."""

    archive_cg = Path(archive_cg).resolve()
    package = types.ModuleType("cg")
    package.__path__ = [str(archive_cg)]
    package.__package__ = "cg"
    package.__file__ = str(archive_cg / "__init__.py")
    sys.modules["cg"] = package
    sys.modules["cg.sim"] = loaded_sim
    return package


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
    hard_timeouts = sum(bool(row.get("hard_timeout")) for row in records)
    abnormal_exits = sum(bool(row.get("abnormal_exit")) for row in records)
    timeouts = sum("TIMEOUT" in row.get("statuses", []) for row in records)
    invalid = sum("INVALID" in row.get("statuses", []) for row in records)
    agent_errors = sum("ERROR" in row.get("statuses", []) for row in records)
    exceptions = sum(bool(row.get("exception")) for row in records)
    gates = {
        "loaded_native_hash_matches": all(row.get("native_hash_verified") is True for row in records),
        "game_count_complete": len(records) == expected_games,
        "normal_terminals": normal == expected_games,
        "crash_zero": crashes == 0,
        "hard_timeout_zero": hard_timeouts == 0,
        "abnormal_exit_zero": abnormal_exits == 0,
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
        "hard_timeouts": hard_timeouts,
        "abnormal_exits": abnormal_exits,
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


def run_game_subprocess(
    command: list[str], result_path: Path, hard_timeout_seconds: float
) -> dict[str, Any]:
    """Run exactly one game and retain its process-level outcome without retrying."""

    started = __import__("time").perf_counter()
    process = subprocess.Popen(command)
    timed_out = False
    try:
        exit_code = process.wait(timeout=hard_timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        exit_code = process.wait()
    elapsed = __import__("time").perf_counter() - started
    signal_number = -exit_code if exit_code < 0 else None
    payload: dict[str, Any] = {}
    if result_path.is_file():
        try:
            import json

            value = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                payload = value
        except Exception as exc:
            payload = {"exception": f"invalid child evidence: {type(exc).__name__}: {exc}"}
    elif not timed_out and exit_code == 0:
        payload = {"exception": "child exited successfully without evidence"}
    payload.update({
        "child_command": command,
        "child_exit_code": exit_code,
        "child_signal": signal_number,
        "hard_timeout": timed_out,
        "process_crash": signal_number is not None,
        "abnormal_exit": exit_code != 0 and not timed_out,
        "parent_elapsed_seconds": elapsed,
        "child_evidence_present": result_path.is_file(),
        "retry_count": 0,
    })
    if timed_out:
        payload["returned_normally"] = False
        payload["normal_terminal"] = False
        payload["exception"] = f"hard timeout after {hard_timeout_seconds} seconds"
    elif exit_code != 0:
        payload["returned_normally"] = False
        payload["normal_terminal"] = False
        payload.setdefault("exception", f"child process exited with code {exit_code}")
    return payload
