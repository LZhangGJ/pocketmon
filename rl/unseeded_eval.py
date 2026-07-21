from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import math
import socket
import subprocess
import sys
import time
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


def stage_b_schedule() -> list[dict[str, Any]]:
    matchups = [
        ("rl_bc_002_a", "official_random"),
        ("official_random", "rl_bc_002_a"),
        ("rl_bc_002_a", "lucario_rule"),
        ("lucario_rule", "rl_bc_002_a"),
    ]
    return [{"game_id": 21 + i, "schedule_position": i, "seat0": a, "seat1": b}
            for i, (a, b) in enumerate(matchups)]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (index - low)


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
        "games": len(records), "normal_terminals": normal, "crashes": crashes,
        "hard_timeouts": hard_timeouts, "abnormal_exits": abnormal_exits,
        "timeouts": timeouts, "invalid_actions": invalid, "agent_errors": agent_errors,
        "exceptions": exceptions, "network_attempts": network_attempts,
        "gates": gates, "gate_passed": all(gates.values()),
    }


def summarize_stage_b(records: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics_keys = ("model_actions", "fallback_actions", "load_errors", "inference_errors",
                        "illegal_model_actions", "illegal_fallback_actions", "emergency_legal_actions")
    totals = {key: sum(int(row.get("candidate_diagnostics", {}).get(key, 0)) for row in records)
              for key in diagnostics_keys}
    latencies = [float(value) for row in records for value in row.get("model_decision_latency_ms", [])]
    gates = {
        "game_count_complete": len(records) == 4,
        "normal_terminals": sum(bool(row.get("normal_terminal")) for row in records) == 4,
        "checkpoint_hash_matches": all(row.get("checkpoint_hash_verified") is True for row in records),
        "model_load_error_zero": totals["load_errors"] == 0,
        "inference_error_zero": totals["inference_errors"] == 0,
        "illegal_model_action_zero": totals["illegal_model_actions"] == 0,
        "illegal_fallback_action_zero": totals["illegal_fallback_actions"] == 0,
        "emergency_legal_action_zero": totals["emergency_legal_actions"] == 0,
        "crash_zero": not any(row.get("process_crash") for row in records),
        "hard_timeout_zero": not any(row.get("hard_timeout") for row in records),
        "abnormal_exit_zero": not any(row.get("abnormal_exit") for row in records),
        "framework_failure_zero": not any(set(row.get("statuses", [])) & {"TIMEOUT", "INVALID", "ERROR"} for row in records),
        "exception_zero": not any(row.get("exception") for row in records),
        "network_attempt_zero": sum(int(row.get("network_attempts", 0)) for row in records) == 0,
        "no_games_dropped": [row.get("game_id") for row in records] == [21, 22, 23, 24],
    }
    return {"games": len(records), "scheduled_model_attempts": len(records),
            "started_model_processes": sum(row.get("child_evidence_present") is True for row in records),
            "checkpoint_loaded_games": sum(row.get("checkpoint_hash_verified") is True for row in records),
            "model_action_games": sum(int(row.get("candidate_diagnostics", {}).get("model_actions", 0)) > 0 for row in records),
            "completed_model_games": sum(bool(r.get("normal_terminal")) for r in records),
            "normal_terminals": sum(bool(r.get("normal_terminal")) for r in records),
            "candidate_diagnostics": totals, "model_decisions": len(latencies),
            "decision_latency_ms": {"p50": percentile(latencies, .5), "p95": percentile(latencies, .95),
                                    "max": max(latencies) if latencies else None},
            "gates": gates, "gate_passed": all(gates.values())}
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
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=hard_timeout_seconds)
        exit_code = process.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        stdout, stderr = process.communicate()
        exit_code = process.returncode
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
    if timed_out:
        process_outcome = "hard_timeout"
    elif signal_number is not None:
        process_outcome = "signal_crash"
    elif exit_code != 0:
        process_outcome = "abnormal_exit"
    else:
        process_outcome = "normal_exit"
    payload.update({
        "child_command": command,
        "child_exit_code": exit_code,
        "child_signal": signal_number,
        "child_stdout": stdout.decode("utf-8", errors="replace"),
        "child_stderr": stderr.decode("utf-8", errors="replace"),
        "process_outcome": process_outcome,
        "hard_timeout": process_outcome == "hard_timeout",
        "process_crash": process_outcome == "signal_crash",
        "abnormal_exit": process_outcome == "abnormal_exit",
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


def require_isolation_prefix(tokens: list[str]) -> None:
    if "--unshare-net" not in tokens:
        raise ValueError("OS isolation must include --unshare-net")
    try:
        dev_index = tokens.index("--dev-bind")
    except ValueError as exc:
        raise ValueError("OS isolation must explicitly dev-bind /dev") from exc
    if tokens[dev_index + 1:dev_index + 3] != ["/dev", "/dev"]:
        raise ValueError("--dev-bind must map /dev to /dev")


def resource_peak_fields(parent_peak_rss_kb: int, child_peaks: list[int], tree_peaks: list[int] | None = None) -> dict[str, Any]:
    max_child = max(child_peaks, default=0)
    max_tree = max(tree_peaks if tree_peaks is not None else child_peaks, default=0)
    return {"parent_peak_rss_kb": parent_peak_rss_kb, "max_child_peak_rss_kb": max_child,
            "max_process_tree_peak_rss_kb": max_tree, "overall_peak_rss_kb": max(parent_peak_rss_kb, max_tree),
            "overall_peak_rss_definition": "maximum of parent peak RSS and process-tree peak RSS; sequential attempts are not summed"}


def preflight_gates(child: dict[str, Any], process: dict[str, Any]) -> dict[str, bool]:
    true_fields = ("urandom_readable", "os_urandom_successful", "torch_import_successful",
        "torch_cpu_tensor_successful", "checkpoint_loaded", "checkpoint_hash_verified", "native_loaded",
        "native_hash_verified", "eth0_absent", "tcp_unavailable", "dns_unavailable")
    gates = {field: child.get(field) is True for field in true_fields}
    gates.update({"exit_code_zero": process.get("exit_code") == 0, "no_signal": process.get("signal") is None,
                  "no_timeout": process.get("hard_timeout") is False, "no_exception": child.get("exception") is None})
    return gates


def _rss_kb(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return 0


def _descendant_pids(root_pid: int) -> set[int]:
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "stat").read_text(encoding="utf-8")
            after_name = text[text.rfind(")") + 2:].split()
            parents[int(entry.name)] = int(after_name[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return descendants


def run_monitored_subprocess(command: list[str], timeout_seconds: float) -> dict[str, Any]:
    """Run once while sampling child and process-tree RSS from procfs."""

    started = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    max_child = 0
    max_tree = 0
    timed_out = False
    while process.poll() is None:
        pids = _descendant_pids(process.pid)
        rss = {pid: _rss_kb(pid) for pid in pids}
        max_child = max(max_child, rss.get(process.pid, 0))
        max_tree = max(max_tree, sum(rss.values()))
        if time.perf_counter() - started > timeout_seconds:
            timed_out = True
            process.kill()
            break
        time.sleep(0.01)
    stdout, stderr = process.communicate()
    exit_code = process.returncode
    signal_number = -exit_code if exit_code < 0 else None
    return {
        "command": command, "exit_code": exit_code, "signal": signal_number,
        "hard_timeout": timed_out, "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace"), "elapsed_seconds": time.perf_counter() - started,
        "max_child_peak_rss_kb": max_child, "max_process_tree_peak_rss_kb": max_tree,
        "retry_count": 0,
    }
