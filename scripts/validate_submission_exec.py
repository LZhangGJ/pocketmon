from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path


REQUIRED_ZERO_DIAGNOSTICS = (
    "load_errors",
    "inference_errors",
    "illegal_model_actions",
    "fallback_actions",
    "illegal_fallback_actions",
    "emergency_legal_actions",
    "q_load_errors",
)


def _safe_extract(archive_path: Path, output: Path) -> None:
    output = output.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (output / member.name).resolve()
            if output != target and output not in target.parents:
                raise ValueError(f"archive member escapes package root: {member.name}")
        archive.extractall(output)


def _validate_model_diagnostics(diagnostics: object) -> dict[str, object]:
    if not isinstance(diagnostics, dict):
        raise TypeError("agent diagnostics must be a dictionary")
    if diagnostics.get("checkpoint_exists") is not True:
        raise RuntimeError("checkpoint was not found and loaded during preflight")
    if int(diagnostics.get("model_actions", 0)) < 1:
        raise RuntimeError("preflight did not execute a model decision")
    nonzero = {
        key: diagnostics.get(key)
        for key in REQUIRED_ZERO_DIAGNOSTICS
        if diagnostics.get(key, 0) not in (0, None)
    }
    if nonzero:
        raise RuntimeError(f"model preflight diagnostic counters are nonzero: {nonzero}")
    return diagnostics


def _worker(package_root: Path) -> dict[str, object]:
    package_root = package_root.resolve()
    os.chdir(package_root)
    source = (package_root / "main.py").read_text(encoding="utf-8")
    namespace: dict[str, object] = {"__name__": "submitted_agent"}
    started = time.perf_counter()
    # Kaggle's get_last_callable compiles and execs main.py without defining
    # __file__. Keep this deliberately different from importlib-based matches.
    exec(compile(source, "/kaggle_simulations/agent/main.py", "exec"), namespace)
    import_seconds = time.perf_counter() - started
    named_agent = namespace.get("agent")
    if not callable(named_agent):
        raise TypeError("main.py did not expose a callable agent")
    selected_agent = [value for value in namespace.values() if callable(value)][-1]
    if selected_agent is not named_agent:
        raise TypeError("the last callable selected by Kaggle is not main.agent")
    setup_action = selected_agent({"select": None})
    if not isinstance(setup_action, list) or len(setup_action) != 60:
        raise ValueError("setup action must be a 60-card deck")
    decision_action = selected_agent({
        "current": {"turn": 1},
        "select": {"option": [{}], "minCount": 1, "maxCount": 1},
    })
    if decision_action != [0]:
        raise ValueError(f"single-option decision was not legal: {decision_action!r}")
    diagnostics_fn = namespace.get("diagnostics")
    diagnostics = diagnostics_fn() if callable(diagnostics_fn) else {}
    diagnostics = _validate_model_diagnostics(diagnostics)
    return {
        "gate_passed": True,
        "package_root": str(package_root),
        "import_seconds": import_seconds,
        "setup_cards": len(setup_action),
        "decision_action": decision_action,
        "diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an archive with Kaggle's no-__file__ exec semantics")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--worker-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_root is not None:
        print(json.dumps(_worker(args.worker_root), ensure_ascii=False))
        return
    if args.archive is None:
        parser.error("--archive is required")
    archive = args.archive.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="pocketmon-kaggle-exec-") as directory:
        package_root = Path(directory)
        _safe_extract(archive, package_root)
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker-root", str(package_root)],
            cwd=package_root,
            text=True,
            capture_output=True,
            timeout=120,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr or completed.stdout)
        report = json.loads(completed.stdout)
        report["archive"] = str(archive)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
