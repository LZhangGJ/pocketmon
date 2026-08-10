from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import shutil
import stat
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from common import (
    Experiment7Error,
    canonical_deck_sha256,
    directory_sha256,
    read_deck,
    read_json,
    sha256_file,
    stable_runtime_files,
    utc_now,
    write_json,
)


PUBLIC_ARCHIVE_SHA256 = "097f5d6c283839843c2502d65620a9790ebfc17736594f1544b4338b28eeb450"

TEAM_SUBMISSIONS = {
    "team_grim_model_a": {
        "archiveSha256": "9c6760b9c9f4aba9e8aa12a57c389ad3cd0fa778a06e8c1517b1c98f8aff6e5e",
        "modelSha256": "55cb54f27761ef1c5539232d94c192e4a1874d750f2e0aa2ba019d67cee182dd",
    },
    "team_grim_model_b": {
        "archiveSha256": "63509ef1d541d173edbc835aaa2c7cca4ac2315fa2bf323fbc7958dab978e35f",
        "modelSha256": "baa25d6dc7296665fad1d29dcd40cd3ef2ac581bb10328e5b7507dfc6be98b62",
    },
}

TEAM_MAIN_SHA256 = "62f3ed6df7c96a0ea41410393bcadf5b4f0d0747b52bbbe706579b9da7c4612d"
TEAM_DECK_FILE_SHA256 = "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d"
TEAM_DECK_CANONICAL_SHA256 = "cafa7652a6349be806d8ac2b9abfdb6c72ca3821f368e0d912e2d989f3b54cdd"

# Only runtime files required by the selected public agents are copied.  In
# particular, every archive-provided ``cg`` directory and native library is
# excluded; official Arena supplies the verified engine.
PUBLIC_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "name": "public_archaludon_meta",
        "archetype": "archaludon_cinderace",
        "prefix": "public_baselines/biohack44_meta_snapshot_07_july/selected_agent_build",
        "files": {
            "main.py": "a4c53101be301c181bd477204a72c0e5cba65fddd34d8cd0ec4d36e4b41c9518",
            "deck.csv": "fbe6ab59992260b0d6774abed19469be315521b5ed0546de8c20f329607693e6",
        },
        "deckCanonicalSha256": "133a2b1304995d44d0cc46f43faef915a81f19ee0202b6d0afebba18c018a5c8",
    },
    {
        "name": "public_lucario_search",
        "archetype": "mega_lucario",
        "prefix": "public_baselines/daniilkrasnovvv_conservative_probabilistic_agent/arena_agent",
        "files": {
            "main.py": "370b5c5725b395f82437cc5c3cb8c5771ea76238b87579559331d86b174e794f",
            "deck.csv": "780e9eb63b0736dc90c70b39b9b58d3cf7eea59db11dbc146fafc08af5e2292f",
        },
        "deckCanonicalSha256": "ed840b99364baa5b5cc03a3120e9d3c982d7c905e2ed8bea2b9e9d2017fa19b7",
        "runtimeWarnings": ["writes deck.csv in the per-match working directory at import"],
    },
    {
        "name": "public_garchomp_v28",
        "archetype": "cynthia_garchomp",
        "prefix": "public_baselines/jazivxt_garchomp_gpu_v28/arena_agent",
        "files": {
            "main.py": "e99465c757231679cd038a9ebb401b3c1e9228bf58e3efcd386ab681b5ca6fbc",
            "deck.csv": "6a4da49026b58ecb3ea608afbb9222ba4b2e55bce8350519c226eec60a96fcf7",
            "gpu_submission_inference_v28.py": "1ebe4769f79abe39770d8c87d350e3d356556f7a9ac0c58dae9c547fb23a25fa",
            "selector_weights_v28.json": "a1dfe314f894bb2c3ef98bdf399d8a2262ba0ff3d203e6750f5a7885cb688b36",
        },
        "deckCanonicalSha256": "39fb18fd9ff204e86299a92ac22092fdd41fb6111a48f8ebb82aabd2039d01ef",
    },
    {
        "name": "public_alakazam_search_v9",
        "archetype": "alakazam",
        "prefix": "public_baselines/prvsiyan_search_audited_alakazam_v9/arena_agent",
        "files": {
            "main.py": "7f82cfe51329263d46b34d71405876db881fb840e97258fe6f52d6b37876162f",
            "deck.csv": "a8c9177354b92abe5fb877f46b792b86f8ec9c4bc3551d5d16d4a89128f00976",
        },
        "deckCanonicalSha256": "eda192deda5cc7c5ec09e5a4d64db4af1769a763ce74d3a6ef23d58d1b8955b0",
    },
    {
        "name": "public_alakazam_visible_v21",
        "archetype": "alakazam",
        "prefix": "public_baselines/prvsiyan_visible_grim_belief_alakazam_v21/arena_agent",
        "files": {
            "main.py": "26430640c7670d6de1bf5d8e0818d18ce04c3c510402634539a2c555478242cc",
            "deck.csv": "0598646548d081832ec311c15fdc369b32c6f5e63175b0cfd1904d21fd082451",
        },
        "deckCanonicalSha256": "606a775392ffe25e058b19c17801d58a4bf30f7cd8c62782388d3de7e7eb5283",
    },
    {
        "name": "public_alakazam_roman_v10",
        "archetype": "alakazam",
        "prefix": "public_baselines/romanrozen_strong_start_v10",
        "files": {
            "main.py": "f31eba2e819ee2b3d46765b4195ea7dab8f32d0b5d09cafd39b3823661f6b5aa",
            "deck.csv": "8eccc69c3bf7d499f38c6116c33c5fac837050bf0ec71a5a1883f0f20f41ddbc",
        },
        "deckCanonicalSha256": "e656740ab5d19a958fe1a2d05ca05d49bea09b273a5cb593de5e1d4d9cbb8340",
    },
)

NOTEBOOK_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "name": "notebook_lucario_crustle_aware",
        "archetype": "mega_lucario",
        "filename": "06__yaroslavkholmirzayev__mega-lucario-v2-crustle-aware-best-submit.ipynb",
        "sourceSha256": "ec34151a6e4070925d001241a7bd5e870ae47b505160a949107b6ef5fb67c1fe",
        "mainSha256": "5e648f63a6e1cec594285a6bf1ad22bf4c8d8232a642a82fc7f9ed299c1705f3",
        "deckFileSha256": "406e2e9bd6ae82b8008b16ee64ffcbb58e4a50cd6bc36e33ae655456c6b9afee",
        "deckCanonicalSha256": "282bbb43e78cd05d63c1bf2e680202537bdc5ad680966ead77e8dc8400f65cce",
    },
    {
        "name": "notebook_dragapult_rules",
        "archetype": "dragapult",
        "filename": "13__skarin__phantom-dive-or-go-home-a-dragapult-ex-deck.ipynb",
        "sourceSha256": "9e3b503e3e410343e5478c536f35281b496f331f0b8399a32561553dbbf64b97",
        "mainSha256": "4f479bdc6d00c1efd120ade40c245bdce881f1d59e9ca18a8825ce17d5d95a22",
        "deckFileSha256": "2d61ac7f752b70e07c1d6be1bfd178870e1fbd2ff1f5e7d012b120773b7e7d29",
        "deckCanonicalSha256": "17cd742b94386fa5e003e8e43a7ea5b4bbb2f39443f75b0fe7ba81d15f3a1f35",
    },
    {
        "name": "notebook_library_out_1208",
        "archetype": "great_tusk_crustle_library_out",
        "filename": "28__soutasakurai__max-elo-1208-libraryout-w-crustle-great-tusk.ipynb",
        "sourceSha256": "aa6fc63c0c113787afa6e5d1f194f493bda719f3924df6122c7d49014dfe77a7",
        "mainSha256": "3c96c2e01c10ef9fda75a2b374056f1f59fa0cec21df42ee74755177f64bcbc1",
        "deckFileSha256": "5d40a8182cc6661379c97eeb68e80513f8d0927a7db573d240324c33012a5a73",
        "deckCanonicalSha256": "ff107989f334ddf6d62186b3791bbb5846fc5045796ae020bb2059cf436eedeb",
    },
    {
        "name": "notebook_dragapult_ucb1",
        "archetype": "dragapult",
        "filename": "37__faheemgurkani__dragapult-ucb1-search-baseline-establishment.ipynb",
        "sourceSha256": "d009087ffd20f3b45e391f2a3898db84a995ec6c8a6a42206970ccce91f155fb",
        "mainSha256": "3eed2fd959aea779d114bbb37da2e48880fddd534c496cbafe33a53b85d3e1a5",
        "deckFileSha256": "30c8c7365c75f38fd6e7e1d8543c42ce7055ed6fd1c6e9eb244e44484b78e724",
        "deckCanonicalSha256": "17cd742b94386fa5e003e8e43a7ea5b4bbb2f39443f75b0fe7ba81d15f3a1f35",
    },
    {
        "name": "notebook_library_out_control_v11",
        "archetype": "great_tusk_crustle_library_out",
        "filename": "39__prvsiyan__ptcg-ai-battle-control-v11-meta-portfolio.ipynb",
        "sourceSha256": "e8dd716d559e3b2821b92a072d0bf434af68d4b1db3b7f537f7ec59630714952",
        "mainSha256": "82bb92ad77cbd395373047a54d8bda30ddbf60f8d15cd5be9ae85b620ae7c2ed",
        "deckFileSha256": "5d40a8182cc6661379c97eeb68e80513f8d0927a7db573d240324c33012a5a73",
        "deckCanonicalSha256": "ff107989f334ddf6d62186b3791bbb5846fc5045796ae020bb2059cf436eedeb",
    },
    {
        "name": "notebook_crustle_wall",
        "archetype": "crustle_wall",
        "filename": "49__biohack44__beating-the-day-2-new.ipynb",
        "sourceSha256": "79c6d59f2421fc0564aa26591f47682b9de502010a01e41f2a18864f655e5ac6",
        "mainSha256": "07389afcb8790a101f102f6dea8a9d3056abfe42c32f09c70f7170a7d6d701cf",
        "deckFileSha256": "0c76a8b38467cf2dd9afaca291685926aeab97afece13ea30f3e75a94ddabf0f",
        "deckCanonicalSha256": "2a498f19095c419807da7bad84a4218406c14ff3c5ceca5d9a7e91008e63b36e",
    },
)

QUARANTINED_CANDIDATES: tuple[dict[str, str], ...] = (
    {
        "name": "public_tetsutani_grim_root",
        "reason": "pickle.load runtime asset and prior official-engine SIGABRT; do not materialize",
    },
    {
        "name": "public_tetsutani_grim_v32",
        "reason": "relative-package entrypoint is not standalone; requires a separately reviewed adapter",
    },
    {
        "name": "notebook_archaludon_v28",
        "reason": "safe materializer omits required gpu_submission_inference_v28.py companion",
    },
    {
        "name": "notebook_crustle_counter_v29",
        "reason": "safe materializer omits required gated_submission_inference_v29.py companion",
    },
    {
        "name": "notebook_battlecore_48",
        "reason": "current static materializer cannot extract one unambiguous agent",
    },
    {
        "name": "notebook_alakazam_43_47",
        "reason": "notebook-export/build scripts are not standalone frozen agent directories",
    },
)

TEAM_ALLOWED_FILES = {
    "main.py",
    "tokenizer.py",
    "portable.py",
    "features.py",
    "deck_identity_portable.py",
    "deck_identity_bc.npz",
    "engine_catalog.json",
    "opponent_deck_classes.json",
    "deck.csv",
}

FORBIDDEN_RUNTIME_SUFFIXES = {
    ".dll",
    ".dylib",
    ".exe",
    ".pkl",
    ".pickle",
    ".ps1",
    ".sh",
    ".so",
}

MAX_EXTERNAL_MEMBER_BYTES = 64 * 1024 * 1024
MAX_EXTERNAL_ARCHIVE_BYTES = 256 * 1024 * 1024
EXTERNAL_ALLOWED_SUFFIXES = {".csv", ".json", ".npz", ".py"}


def _archive_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise Experiment7Error(f"unsafe archive member: {name!r}")
    return path


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise Experiment7Error(f"{label} SHA256 mismatch: expected={expected} actual={actual}")


def _write_member(target: Path, content: bytes, expected_sha256: str | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    if expected_sha256 is not None:
        _require_hash(target, expected_sha256, target.name)


def copy_public_candidate(archive_path: Path, candidate: dict[str, Any], target: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        infos = {str(_archive_path(info.filename)): info for info in archive.infolist()}
        for relative, expected in candidate["files"].items():
            member = f"{candidate['prefix']}/{relative}"
            info = infos.get(member)
            if info is None or info.is_dir():
                raise Experiment7Error(f"public candidate member missing: {member}")
            unix_mode = (info.external_attr >> 16) & 0o170000
            if unix_mode == 0o120000:
                raise Experiment7Error(f"symlink archive member rejected: {member}")
            _write_member(target / relative, archive.read(info), expected)


def copy_team_submission(archive_path: Path, target: Path, expected_model_sha256: str) -> None:
    copied: set[str] = set()
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive.getmembers():
            path = _archive_path(member.name)
            if len(path.parts) != 1 or path.name not in TEAM_ALLOWED_FILES:
                continue
            if not member.isfile():
                raise Experiment7Error(f"non-file team submission member rejected: {member.name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise Experiment7Error(f"cannot read team submission member: {member.name}")
            _write_member(target / path.name, handle.read())
            copied.add(path.name)
    missing = TEAM_ALLOWED_FILES - copied
    if missing:
        raise Experiment7Error(f"team submission missing files: {sorted(missing)}")
    _require_hash(target / "main.py", TEAM_MAIN_SHA256, "team main.py")
    _require_hash(target / "deck.csv", TEAM_DECK_FILE_SHA256, "team deck.csv")
    _require_hash(target / "deck_identity_bc.npz", expected_model_sha256, "team model")
    validate_safe_npz(target / "deck_identity_bc.npz")


def copy_external_submission(archive_path: Path, target: Path) -> dict[str, Any]:
    """Materialize one untrusted top-level agent archive without executing it."""
    _ensure_new_staging_root(target)
    copied: list[str] = []
    total_bytes = 0
    seen: set[str] = set()
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive.getmembers():
            path = _archive_path(member.name)
            if len(path.parts) != 1:
                raise Experiment7Error(
                    f"external submission must contain only top-level files: {member.name}"
                )
            name = path.name
            if name in seen:
                raise Experiment7Error(f"duplicate external submission member: {name}")
            seen.add(name)
            if not member.isfile():
                raise Experiment7Error(f"non-regular external submission member rejected: {name}")
            suffix = Path(name).suffix.lower()
            if suffix in FORBIDDEN_RUNTIME_SUFFIXES:
                raise Experiment7Error(f"forbidden external runtime asset: {name}")
            if suffix not in EXTERNAL_ALLOWED_SUFFIXES:
                raise Experiment7Error(f"unsupported external runtime asset: {name}")
            if member.size < 0 or member.size > MAX_EXTERNAL_MEMBER_BYTES:
                raise Experiment7Error(f"external submission member is too large: {name}")
            total_bytes += member.size
            if total_bytes > MAX_EXTERNAL_ARCHIVE_BYTES:
                raise Experiment7Error("external submission exceeds the materialization size limit")
            handle = archive.extractfile(member)
            if handle is None:
                raise Experiment7Error(f"cannot read external submission member: {name}")
            content = handle.read(MAX_EXTERNAL_MEMBER_BYTES + 1)
            if len(content) != member.size:
                raise Experiment7Error(f"external submission member size mismatch: {name}")
            _write_member(target / name, content)
            copied.append(name)
    missing = {"main.py", "deck.csv"} - set(copied)
    if missing:
        raise Experiment7Error(f"external submission missing files: {sorted(missing)}")
    for path in sorted(target.glob("*.npz")):
        validate_safe_npz(path)
    return {
        "files": sorted(copied),
        "fileCount": len(copied),
        "uncompressedBytes": total_bytes,
    }


def stage_external_submission(args: argparse.Namespace) -> dict[str, Any]:
    archive = args.archive.resolve()
    expected_sha256 = str(args.expected_sha256).lower()
    _require_hash(archive, expected_sha256, "external submission")
    name = _require_safe_agent_name(args.name)
    staging_root = args.staging_root.resolve()
    _ensure_new_staging_root(staging_root)
    target = staging_root / "agents" / name
    copy_receipt = copy_external_submission(archive, target)
    scan = static_scan_agent(target)
    row = {
        **_staged_row(
            {"name": name, "archetype": str(args.archetype)},
            target,
            scan,
            "external_submission_static_materializer",
        ),
        "sourceArchiveSha256": expected_sha256,
        "materialization": copy_receipt,
    }
    packages_path = staging_root / "packages.json"
    write_json(
        packages_path,
        {
            "schemaVersion": 1,
            "packages": [
                {
                    "name": name,
                    "agentDir": str(target.resolve()),
                    "status": "staging",
                    "archetype": str(args.archetype),
                    "deckCanonicalSha256": row["deckCanonicalSha256"],
                    "directorySha256": row["directorySha256"],
                }
            ],
        },
    )
    payload = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "status": "static_staging_only_not_arena_admitted",
        "source": {"path": str(archive), "sha256": expected_sha256},
        "stagingRoot": str(staging_root),
        "agents": [row],
        "packages": {"path": str(packages_path), "sha256": sha256_file(packages_path)},
        "externalAgentCodeExecuted": False,
    }
    write_json(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def validate_safe_npz(path: Path) -> None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not archive.files:
                raise Experiment7Error(f"empty NPZ archive: {path}")
            for name in archive.files:
                array = archive[name]
                if array.dtype.hasobject:
                    raise Experiment7Error(f"object dtype rejected in {path}: {name}")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise Experiment7Error(f"unsafe or invalid NPZ archive: {path}: {exc}") from exc


def _load_notebook_materializer():
    repo_root = Path(__file__).resolve().parents[2]
    source = repo_root / "scripts" / "materialize_notebook_agent.py"
    spec = importlib.util.spec_from_file_location("experiment7_static_notebook_materializer", source)
    if spec is None or spec.loader is None:
        raise Experiment7Error(f"cannot load notebook materializer: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.materialize


def materialize_notebook_candidate(source: Path, candidate: dict[str, Any], target: Path) -> None:
    _require_hash(source, candidate["sourceSha256"], source.name)
    materialize = _load_notebook_materializer()
    materialize(source, target)
    _remove_generated_bytecode(target)
    _require_hash(target / "main.py", candidate["mainSha256"], f"{candidate['name']} main.py")
    _require_hash(target / "deck.csv", candidate["deckFileSha256"], f"{candidate['name']} deck.csv")


def static_scan_agent(agent_dir: Path) -> dict[str, Any]:
    if (agent_dir / "cg").exists():
        raise Experiment7Error(f"archive-provided cg directory rejected: {agent_dir}")
    if not (agent_dir / "main.py").is_file() or not (agent_dir / "deck.csv").is_file():
        raise Experiment7Error(f"agent is missing main.py or deck.csv: {agent_dir}")
    imports: set[str] = set()
    warnings: set[str] = set()
    python_files = []
    for path in stable_runtime_files(agent_dir):
        relative = path.relative_to(agent_dir)
        if path.suffix.lower() in FORBIDDEN_RUNTIME_SUFFIXES:
            raise Experiment7Error(f"forbidden runtime asset: {relative}")
        if "cg" in relative.parts:
            raise Experiment7Error(f"archive-provided cg asset rejected: {relative}")
        if path.suffix != ".py":
            continue
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(relative))
        except SyntaxError as exc:
            raise Experiment7Error(f"Python syntax error in {relative}: {exc}") from exc
        python_files.append(relative.as_posix())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name in {"eval", "exec", "os.system", "subprocess.run", "subprocess.Popen"}:
                    raise Experiment7Error(f"dynamic/process execution rejected in {relative}: {name}")
                if name in {"pickle.load", "pickle.loads"}:
                    raise Experiment7Error(f"pickle deserialization rejected in {relative}: {name}")
                if name in {"Path.write_text", "Path.write_bytes", "write_text", "write_bytes"} or name.endswith(
                    (".write_text", ".write_bytes")
                ):
                    warnings.add("runtime file write detected; require per-match writable temp cwd")
    if imports & {"requests", "socket", "urllib", "urllib3"}:
        warnings.add("network module import detected; require network-disabled runtime")
    cards = read_deck(agent_dir / "deck.csv")
    canonical_hash = canonical_deck_sha256(cards)
    return {
        "pythonFiles": sorted(python_files),
        "imports": sorted(imports),
        "warnings": sorted(warnings),
        "deckCards": len(cards),
        "deckCanonicalSha256": canonical_hash,
        "mainSha256": sha256_file(agent_dir / "main.py"),
        "deckFileSha256": sha256_file(agent_dir / "deck.csv"),
        "directorySha256": directory_sha256(agent_dir),
        "archiveProvidedCg": False,
        "executedAgentCode": False,
    }


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _remove_generated_bytecode(root: Path) -> None:
    for path in sorted(root.rglob("*.pyc")):
        if "__pycache__" in path.relative_to(root).parts:
            path.unlink()
    for path in sorted(root.rglob("__pycache__"), key=lambda value: len(value.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def _ensure_new_staging_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise Experiment7Error(f"staging root must be absent or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def make_plan(args: argparse.Namespace) -> dict[str, Any]:
    public_archive = args.public_archive.resolve()
    team_a = args.team_submission_a.resolve()
    team_b = args.team_submission_b.resolve()
    notebook_root = args.notebook_root.resolve()
    _require_hash(public_archive, PUBLIC_ARCHIVE_SHA256, "public agent archive")
    _require_hash(team_a, TEAM_SUBMISSIONS["team_grim_model_a"]["archiveSha256"], "team submission A")
    _require_hash(team_b, TEAM_SUBMISSIONS["team_grim_model_b"]["archiveSha256"], "team submission B")
    for candidate in NOTEBOOK_CANDIDATES:
        source = notebook_root / candidate["filename"]
        _require_hash(source, candidate["sourceSha256"], source.name)
    plan_path = args.output.resolve()
    payload = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "purpose": "static-only materialization of frozen Experiment 7 opponent candidates",
        "sources": {
            "publicArchive": {"path": str(public_archive), "sha256": PUBLIC_ARCHIVE_SHA256},
            "teamSubmissionA": {
                "path": str(team_a),
                "sha256": TEAM_SUBMISSIONS["team_grim_model_a"]["archiveSha256"],
            },
            "teamSubmissionB": {
                "path": str(team_b),
                "sha256": TEAM_SUBMISSIONS["team_grim_model_b"]["archiveSha256"],
            },
            "notebookRoot": str(notebook_root),
        },
        "stagingRoot": str(args.staging_root.resolve()),
        "readyCandidates": [
            *[row["name"] for row in PUBLIC_CANDIDATES],
            *TEAM_SUBMISSIONS.keys(),
            *[row["name"] for row in NOTEBOOK_CANDIDATES],
        ],
        "quarantinedCandidates": list(QUARANTINED_CANDIDATES),
        "materializeCommand": [
            os.fspath(Path(sys.executable).resolve()),
            os.fspath(Path(__file__).resolve()),
            "materialize",
            "--plan",
            str(plan_path),
        ],
        "contracts": {
            "executeExternalAgentCode": False,
            "copyArchiveCg": False,
            "allowPickle": False,
            "arenaEngine": "trusted official engine supplied outside each frozen agent directory",
        },
    }
    write_json(plan_path, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def _verify_plan_sources(plan: dict[str, Any]) -> None:
    for key in ("publicArchive", "teamSubmissionA", "teamSubmissionB"):
        row = plan["sources"][key]
        _require_hash(Path(row["path"]), row["sha256"], key)


def materialize_plan(plan_path: Path) -> dict[str, Any]:
    plan = read_json(plan_path)
    if plan.get("schemaVersion") != 1:
        raise Experiment7Error(f"unsupported opponent staging plan: {plan_path}")
    _verify_plan_sources(plan)
    staging_root = Path(plan["stagingRoot"])
    _ensure_new_staging_root(staging_root)
    public_archive = Path(plan["sources"]["publicArchive"]["path"])
    notebook_root = Path(plan["sources"]["notebookRoot"])
    staged: list[dict[str, Any]] = []

    for candidate in PUBLIC_CANDIDATES:
        target = staging_root / "agents" / candidate["name"]
        copy_public_candidate(public_archive, candidate, target)
        scan = static_scan_agent(target)
        _require_deck_identity(candidate, scan)
        staged.append(_staged_row(candidate, target, scan, "public_archive"))

    for index, name in enumerate(TEAM_SUBMISSIONS, start=1):
        target = staging_root / "agents" / name
        archive = Path(plan["sources"][f"teamSubmission{'A' if index == 1 else 'B'}"]["path"])
        copy_team_submission(archive, target, TEAM_SUBMISSIONS[name]["modelSha256"])
        scan = static_scan_agent(target)
        if scan["deckCanonicalSha256"] != TEAM_DECK_CANONICAL_SHA256:
            raise Experiment7Error(f"team deck canonical SHA256 mismatch: {name}")
        staged.append(
            {
                **_staged_row(
                    {"name": name, "archetype": "grimmsnarl_froslass_munkidori"},
                    target,
                    scan,
                    "team_submission",
                ),
                "modelSha256": TEAM_SUBMISSIONS[name]["modelSha256"],
            }
        )

    for candidate in NOTEBOOK_CANDIDATES:
        target = staging_root / "agents" / candidate["name"]
        source = notebook_root / candidate["filename"]
        materialize_notebook_candidate(source, candidate, target)
        scan = static_scan_agent(target)
        _require_deck_identity(candidate, scan)
        staged.append(_staged_row(candidate, target, scan, "notebook_static_materializer"))

    packages = {
        "schemaVersion": 1,
        "packages": [
            {
                "name": row["name"],
                "agentDir": row["agentDir"],
                "status": "staging",
                "archetype": row["archetype"],
                "deckCanonicalSha256": row["deckCanonicalSha256"],
                "directorySha256": row["directorySha256"],
            }
            for row in staged
        ],
    }
    packages_path = staging_root / "packages.json"
    write_json(packages_path, packages)
    manifest = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "plan": {"path": str(plan_path.resolve()), "sha256": sha256_file(plan_path)},
        "stagingRoot": str(staging_root.resolve()),
        "status": "static_staging_only_not_arena_admitted",
        "agents": staged,
        "quarantinedCandidates": list(QUARANTINED_CANDIDATES),
        "packages": {"path": str(packages_path.resolve()), "sha256": sha256_file(packages_path)},
        "nextCommands": {
            "rebasePackagesAfterCopyToDataT0": [
                "<linux-python>",
                "<linux-repository>/experiment7/integration/stage_opponent_pool.py",
                "rebase-packages",
                "--staging-root",
                "<dataT0-copied-staging-root>",
                "--output",
                "<dataT0-copied-staging-root>/packages.json",
            ],
            "prepareWritableArenaRuntime": [
                "<linux-python>",
                "<linux-repository>/experiment7/integration/stage_opponent_pool.py",
                "prepare-arena-runtime",
                "--packages",
                "<dataT0-copied-staging-root>/packages.json",
                "--opponents",
                "<frozen-opponents-manifest-with-directory-hashes>",
                "--arena-stage",
                "<writable-arena-stage>",
                "--shard-count",
                "<positive-shard-count>",
            ],
            "makeOfficialArenaSchedule": [
                os.fspath(Path(sys.executable).resolve()),
                os.fspath(Path(__file__).with_name("arena.py").resolve()),
                "make-schedule",
                "--packages",
                str(packages_path.resolve()),
                "--target-agent",
                "<trusted-frozen-lucario-agent-dir>",
                "--output-dir",
                "<arena-output-dir>",
                "--games-per-challenger",
                "<positive-even-number>",
                "--stage",
                "opponent-pool-smoke",
            ]
        },
        "contracts": plan["contracts"],
    }
    manifest_path = staging_root / "staging_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return manifest


def rebase_packages(staging_root: Path, output: Path) -> dict[str, Any]:
    root = staging_root.resolve()
    manifest_path = root / "staging_manifest.json"
    manifest = read_json(manifest_path)
    packages = []
    for row in manifest.get("agents", []):
        name = str(row["name"])
        if Path(name).name != name or name in {".", ".."}:
            raise Experiment7Error(f"unsafe staged agent name: {name!r}")
        agent_dir = root / "agents" / name
        actual = directory_sha256(agent_dir)
        expected = str(row["directorySha256"])
        if actual != expected:
            raise Experiment7Error(
                f"relocated agent hash mismatch for {name}: expected={expected} actual={actual}"
            )
        packages.append(
            {
                "name": name,
                "agentDir": str(agent_dir),
                "status": "staging",
                "archetype": row["archetype"],
                "deckCanonicalSha256": row["deckCanonicalSha256"],
                "directorySha256": expected,
            }
        )
    if not packages:
        raise Experiment7Error(f"no staged agents in {manifest_path}")
    payload = {"schemaVersion": 1, "packages": packages}
    write_json(output, payload)
    receipt = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "stagingRoot": str(root),
        "sourceManifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "packages": {"path": str(output.resolve()), "sha256": sha256_file(output)},
        "agents": len(packages),
        "externalAgentCodeExecuted": False,
    }
    write_json(output.with_name(output.stem + "_rebase_receipt.json"), receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return receipt


def _is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _require_safe_agent_name(value: Any) -> str:
    name = str(value)
    path = PurePosixPath(name.replace("\\", "/"))
    if not name or path.is_absolute() or len(path.parts) != 1 or path.name in {".", ".."}:
        raise Experiment7Error(f"unsafe arena agent name: {name!r}")
    return name


def _require_regular_source_tree(root: Path, label: str) -> None:
    if not root.exists() or not root.is_dir():
        raise Experiment7Error(f"{label} source directory is missing: {root}")
    if _is_link(root):
        raise Experiment7Error(f"{label} source directory link rejected: {root}")
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for child_name in [*directory_names, *file_names]:
            child = current_path / child_name
            if _is_link(child):
                raise Experiment7Error(f"{label} source link rejected: {child}")
        for child_name in file_names:
            child = current_path / child_name
            if not child.is_file():
                raise Experiment7Error(f"{label} non-regular source file rejected: {child}")
    missing = [name for name in ("main.py", "deck.csv") if not (root / name).is_file()]
    if missing:
        raise Experiment7Error(f"{label} source missing {', '.join(missing)}: {root}")


def _manifest_rows(payload: Any, collection: str, manifest_path: Path) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get(collection), list):
        raise Experiment7Error(f"invalid {collection} manifest: {manifest_path}")
    rows = payload[collection]
    if not rows:
        raise Experiment7Error(f"empty {collection} manifest: {manifest_path}")
    return rows


def _expected_directory_hash(row: dict[str, Any], label: str) -> str:
    for key in ("directorySha256", "directory_sha256", "sourceDirectorySha256"):
        value = row.get(key)
        if isinstance(value, str) and len(value) == 64:
            return value.lower()
    raise Experiment7Error(f"{label} has no frozen directory SHA256")


def _source_agent_dir(row: dict[str, Any], label: str) -> Path:
    value = row.get("agentDir") or row.get("agent_dir") or row.get("path")
    if not value:
        raise Experiment7Error(f"{label} has no source directory")
    # ``Path.resolve`` would follow a symlink and hide it from the link gate.
    # Normalize dot segments while preserving the final filesystem object.
    return Path(os.path.abspath(os.fspath(Path(str(value)).expanduser())))


def _require_unlinked_output_path(path: Path) -> None:
    current = path.parent
    while True:
        if current.exists() and _is_link(current):
            raise Experiment7Error(f"arena stage parent link rejected: {current}")
        parent = current.parent
        if parent == current:
            break
        current = parent


def _validated_runtime_sources(
    manifest_path: Path,
    collection: str,
    role: str,
) -> list[dict[str, Any]]:
    payload = read_json(manifest_path)
    rows = _manifest_rows(payload, collection, manifest_path)
    validated = []
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise Experiment7Error(f"invalid {role} row in {manifest_path}: {row!r}")
        if role == "opponents" and row.get("status", "accepted") != "accepted":
            continue
        name = _require_safe_agent_name(row.get("name", ""))
        if name in names:
            raise Experiment7Error(f"duplicate {role} name: {name}")
        names.add(name)
        label = f"{role} {name}"
        source = _source_agent_dir(row, label)
        expected = _expected_directory_hash(row, label)
        _require_regular_source_tree(source, label)
        actual = directory_sha256(source)
        if actual.lower() != expected:
            raise Experiment7Error(
                f"{label} source hash mismatch: expected={expected} actual={actual}"
            )
        validated.append(
            {
                "name": name,
                "sourceAgentDir": str(source),
                "sourceDirectorySha256": expected,
            }
        )
    if not validated:
        raise Experiment7Error(f"no accepted {role} in {manifest_path}")
    return validated


def _copy_writable_runtime_tree(source: Path, target: Path) -> None:
    if target.exists():
        raise Experiment7Error(f"arena runtime destination already exists: {target}")
    target.mkdir(parents=True)
    for source_file in stable_runtime_files(source):
        relative = source_file.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination)
    for path in sorted(target.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        mode = path.stat().st_mode
        path.chmod(mode | (stat.S_IRUSR | stat.S_IWUSR) | (stat.S_IXUSR if path.is_dir() else 0))
    target.chmod(target.stat().st_mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)


def prepare_arena_runtime(
    packages_manifest: Path,
    opponents_manifest: Path,
    arena_stage: Path,
    shard_count: int,
) -> dict[str, Any]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    packages_path = packages_manifest.resolve()
    opponents_path = opponents_manifest.resolve()
    stage = Path(os.path.abspath(os.fspath(arena_stage.expanduser())))
    if stage.exists():
        raise Experiment7Error(f"arena stage already exists; refusing overwrite: {stage}")
    _require_unlinked_output_path(stage)

    # Validate every immutable source before creating any writable output.
    learners = _validated_runtime_sources(packages_path, "packages", "learners")
    opponents = _validated_runtime_sources(opponents_path, "agents", "opponents")
    stage.mkdir(parents=True)
    shard_receipts = []
    for shard_index in range(shard_count):
        shard_root = stage / f"runtime-shard-{shard_index}"
        runtime_rows: dict[str, list[dict[str, Any]]] = {"learners": [], "opponents": []}
        for role, sources in (("learners", learners), ("opponents", opponents)):
            for source_row in sources:
                name = source_row["name"]
                destination = shard_root / role / name
                _copy_writable_runtime_tree(Path(source_row["sourceAgentDir"]), destination)
                runtime_hash = directory_sha256(destination)
                expected = source_row["sourceDirectorySha256"]
                if runtime_hash != expected:
                    raise Experiment7Error(
                        f"{role} {name} runtime copy hash mismatch: expected={expected} actual={runtime_hash}"
                    )
                runtime_rows[role].append(
                    {
                        "name": name,
                        "agent_dir": str(destination.resolve()),
                        "status": "accepted",
                        "source_agent_dir": source_row["sourceAgentDir"],
                        "source_directory_sha256": expected,
                        "runtime_initial_directory_sha256": runtime_hash,
                    }
                )
        learners_path = shard_root / "learners.json"
        opponents_path_for_shard = shard_root / "opponents.json"
        write_json(learners_path, {"schemaVersion": 1, "agents": runtime_rows["learners"]})
        write_json(opponents_path_for_shard, {"schemaVersion": 1, "agents": runtime_rows["opponents"]})
        shard_receipts.append(
            {
                "shardIndex": shard_index,
                "runtimeRoot": str(shard_root.resolve()),
                "learners": {
                    "path": str(learners_path.resolve()),
                    "sha256": sha256_file(learners_path),
                    "agents": len(runtime_rows["learners"]),
                },
                "opponents": {
                    "path": str(opponents_path_for_shard.resolve()),
                    "sha256": sha256_file(opponents_path_for_shard),
                    "agents": len(runtime_rows["opponents"]),
                },
            }
        )
    receipt = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "arenaStage": str(stage),
        "shardCount": shard_count,
        "sources": {
            "packages": {"path": str(packages_path), "sha256": sha256_file(packages_path)},
            "opponents": {"path": str(opponents_path), "sha256": sha256_file(opponents_path)},
            "learners": learners,
            "opponentAgents": opponents,
        },
        "shards": shard_receipts,
        "frozenSourcesModified": False,
        "externalAgentCodeExecuted": False,
    }
    receipt_path = stage / "prepare_arena_runtime_receipt.json"
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return receipt


def _require_deck_identity(candidate: dict[str, Any], scan: dict[str, Any]) -> None:
    expected = candidate["deckCanonicalSha256"].lower()
    actual = scan["deckCanonicalSha256"].lower()
    if actual != expected:
        raise Experiment7Error(
            f"{candidate['name']} canonical deck SHA256 mismatch: expected={expected} actual={actual}"
        )


def _staged_row(
    candidate: dict[str, Any], target: Path, scan: dict[str, Any], source_kind: str
) -> dict[str, Any]:
    warnings = sorted(set(scan["warnings"]) | set(candidate.get("runtimeWarnings", [])))
    requires_temporary_working_directory = any(
        "file write" in warning or "writes deck.csv" in warning for warning in warnings
    )
    return {
        "name": candidate["name"],
        "status": "staging",
        "sourceKind": source_kind,
        "archetype": candidate["archetype"],
        "agentDir": str(target.resolve()),
        "mainSha256": scan["mainSha256"],
        "deckFileSha256": scan["deckFileSha256"],
        "deckCanonicalSha256": scan["deckCanonicalSha256"],
        "directorySha256": scan["directorySha256"],
        "imports": scan["imports"],
        "staticWarnings": warnings,
        "archiveProvidedCg": False,
        "externalAgentCodeExecuted": False,
        "requiresPerMatchTemporaryWorkingDirectory": requires_temporary_working_directory,
        "arenaStatus": "not_run",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or statically materialize the audited Experiment 7 frozen opponent pool"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--public-archive", type=Path, required=True)
    plan.add_argument("--team-submission-a", type=Path, required=True)
    plan.add_argument("--team-submission-b", type=Path, required=True)
    plan.add_argument("--notebook-root", type=Path, required=True)
    plan.add_argument("--staging-root", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--plan", type=Path, required=True)

    rebase = subparsers.add_parser("rebase-packages")
    rebase.add_argument("--staging-root", type=Path, required=True)
    rebase.add_argument("--output", type=Path, required=True)

    prepare = subparsers.add_parser("prepare-arena-runtime")
    prepare.add_argument("--packages", type=Path, required=True)
    prepare.add_argument("--opponents", type=Path, required=True)
    prepare.add_argument("--arena-stage", type=Path, required=True)
    prepare.add_argument("--shard-count", type=int, required=True)

    stage_external = subparsers.add_parser("stage-external")
    stage_external.add_argument("--archive", type=Path, required=True)
    stage_external.add_argument("--expected-sha256", required=True)
    stage_external.add_argument("--name", required=True)
    stage_external.add_argument("--archetype", required=True)
    stage_external.add_argument("--staging-root", type=Path, required=True)
    stage_external.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "plan":
        make_plan(args)
    elif args.command == "materialize":
        materialize_plan(args.plan.resolve())
    elif args.command == "rebase-packages":
        rebase_packages(args.staging_root, args.output)
    elif args.command == "prepare-arena-runtime":
        prepare_arena_runtime(args.packages, args.opponents, args.arena_stage, args.shard_count)
    else:
        stage_external_submission(args)


if __name__ == "__main__":
    main()
