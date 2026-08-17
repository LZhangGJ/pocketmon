from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


@contextmanager
def isolated_agent_workdir(path: Path) -> Iterator[Path]:
    """Give an external agent a disposable cwd without copying large weights.

    Competition agents commonly resolve imports and model assets relative to
    their package directory, but some also write ``deck.csv`` at import time.
    The package itself is a frozen input, so expose its top-level assets via
    links and give ``deck.csv`` a private copy. Windows unit
    tests use copies because creating symlinks may require extra privileges;
    production rollout collection runs on Linux and links large model assets.
    """

    with tempfile.TemporaryDirectory(prefix="universal-ppo-opponent-") as temporary:
        workdir = Path(temporary)
        for source in path.iterdir():
            if source.name == "__pycache__":
                continue
            target = workdir / source.name
            if source.name == "deck.csv" or os.name == "nt":
                if source.is_dir():
                    shutil.copytree(source, target)
                else:
                    shutil.copy2(source, target)
            else:
                target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
        yield workdir


def load_agent(path: Path, name: str, workdir: Path | None = None) -> Any:
    local_names = {source.stem for source in path.glob("*.py") if source.name != "main.py"}
    for local_name in local_names:
        sys.modules.pop(local_name, None)
    spec = importlib.util.spec_from_file_location(name, path / "main.py")
    if spec is None or spec.loader is None:
        raise ImportError(path)
    previous = Path.cwd()
    import_cwd = path if workdir is None else workdir
    inserted = str(path) not in sys.path
    try:
        os.chdir(import_cwd)
        if inserted:
            sys.path.insert(0, str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted and str(path) in sys.path:
            sys.path.remove(str(path))
        os.chdir(previous)


def call_agent(module: Any, observation: dict[str, Any], workdir: Path) -> Any:
    previous = Path.cwd()
    try:
        os.chdir(workdir)
        return module.agent(observation)
    finally:
        os.chdir(previous)
