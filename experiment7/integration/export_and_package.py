from __future__ import annotations

import argparse
import compileall
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import (
    Experiment7Error,
    directory_sha256,
    read_json,
    run_checked,
    sha256_file,
    utc_now,
    write_json,
)


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def export_checkpoint(checkpoint_path: Path, output: Path) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path)
    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise Experiment7Error("checkpoint lacks config/state_dict")
    arrays: dict[str, np.ndarray] = {}
    for name, value in checkpoint["state_dict"].items():
        if not isinstance(value, torch.Tensor):
            raise Experiment7Error(f"state_dict value is not a tensor: {name}")
        arrays[name] = value.detach().cpu().numpy().astype(np.float32, copy=False)
    arrays["config_json"] = np.asarray(
        [json.dumps(checkpoint["config"], separators=(",", ":"), sort_keys=True)]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)
    payload = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": sha256_file(checkpoint_path)},
        "portable": {"path": str(output.resolve()), "sha256": sha256_file(output), "bytes": output.stat().st_size},
        "config": checkpoint["config"],
        "arrayCount": len(arrays) - 1,
        "optimizerIncluded": False,
    }
    write_json(output.with_suffix(".receipt.json"), payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def verify_portable(reference_root: Path, sources_path: Path, checkpoint: Path, portable: Path, output: Path, python: str, decisions: int) -> dict[str, Any]:
    manifest = read_json(sources_path)
    arguments = [
        python,
        str(reference_root / "training" / "verify_deck_identity_portable.py"),
        "--checkpoint",
        str(checkpoint),
        "--portable",
        str(portable),
    ]
    for row in manifest["currentSources"]:
        arguments.extend(
            [
                "--source",
                row["name"],
                row["features"],
                row["tokenCache"],
                row["sequenceCache"],
                row["identityCache"],
            ]
        )
    arguments.extend(
        [
            "--decisions-per-source",
            str(decisions),
            "--output",
            str(output),
        ]
    )
    completed = run_checked(
        arguments,
        cwd=reference_root / "training",
        env={
            "PYTHONPATH": os_path_join(reference_root / "training", reference_root / "data_pipeline"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        log_path=output.with_suffix(".log"),
    )
    return read_json(output)


def os_path_join(*paths: Path) -> str:
    import os

    return os.pathsep.join(str(path) for path in paths)


def select_best(root: Path, output: Path) -> dict[str, Any]:
    candidates = []
    for report_path in sorted(root.rglob("finetune_report.json")):
        report = read_json(report_path)
        checkpoint = Path(report["checkpoint"]["path"])
        if not checkpoint.is_file():
            checkpoint = report_path.parent / "best_model.pt"
        if not checkpoint.is_file():
            continue
        selected = report.get("selectedCalibration", {})
        per_deck = [float(value["exactSemantic"]) for value in selected.values()]
        candidates.append(
            {
                "report": str(report_path.resolve()),
                "checkpoint": str(checkpoint.resolve()),
                "checkpointSha256": sha256_file(checkpoint),
                "seed": report.get("seed"),
                "macro": float(report["selectedCalibrationMacroExactSemantic"]),
                "worst": min(per_deck) if per_deck else -1.0,
                "std": float(np.std(per_deck)) if per_deck else float("inf"),
            }
        )
    if not candidates:
        raise Experiment7Error(f"no usable finetune reports under {root}")
    candidates.sort(key=lambda row: (-row["macro"], -row["worst"], row["std"], str(row["seed"])))
    payload = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "selectionMetric": "calibration macro exactSemantic, then worst deck, then lower deck std",
        "selected": candidates[0],
        "candidates": candidates,
        "holdoutUsed": False,
    }
    write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def select_universal(root: Path, output: Path, top_n: int) -> dict[str, Any]:
    candidates = []
    for report_path in sorted(root.rglob("training_report.json")):
        report = read_json(report_path)
        if report.get("stage") != "universal_bc" or not report.get("best"):
            continue
        checkpoint = Path(report["best"]["path"])
        if not checkpoint.is_file():
            checkpoint = report_path.parent / "best_model.pt"
        if not checkpoint.is_file():
            continue
        best_epoch = int(report["best"]["epoch"])
        epoch = next(
            (row for row in report.get("epochs", []) if int(row["epoch"]) == best_epoch),
            None,
        )
        if epoch is None:
            raise Experiment7Error(
                f"best epoch {best_epoch} missing from {report_path}"
            )
        validation = epoch["validation"]
        candidates.append(
            {
                "report": str(report_path.resolve()),
                "checkpoint": str(checkpoint.resolve()),
                "checkpointSha256": sha256_file(checkpoint),
                "seed": int(report["seed"]),
                "epoch": best_epoch,
                "exactSemantic": float(validation["exactSemantic"]),
                "exactIndex": float(validation["exactIndex"]),
                "countAccuracy": float(validation["countAccuracy"]),
                "illegalPredictionCount": int(validation["illegalPredictionCount"]),
            }
        )
    if not candidates:
        raise Experiment7Error(f"no completed Universal BC reports under {root}")
    if any(row["illegalPredictionCount"] for row in candidates):
        raise Experiment7Error("a Universal BC candidate has illegal predictions")
    candidates.sort(
        key=lambda row: (
            -row["exactSemantic"],
            -row["exactIndex"],
            -row["countAccuracy"],
            row["seed"],
        )
    )
    payload = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "architecture": "experiment7_universal_deck8_autoregressive_stop",
        "selectionMetric": (
            "validation exactSemantic, then exactIndex, then countAccuracy; "
            "final teacher remains gated by official-engine Arena"
        ),
        "holdoutUsed": False,
        "selected": candidates[0],
        "shortlist": candidates[: max(1, top_n)],
        "candidates": candidates,
    }
    write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def verify_universal_portable(
    reference_root: Path,
    sources_path: Path,
    checkpoint: Path,
    portable: Path,
    output: Path,
    python: str,
    decisions: int,
) -> dict[str, Any]:
    verifier = Path(__file__).resolve().parent / "verify_universal_portable.py"
    run_checked(
        [
            python,
            str(verifier),
            "--reference-root",
            str(reference_root),
            "--sources",
            str(sources_path),
            "--checkpoint",
            str(checkpoint),
            "--portable",
            str(portable),
            "--output",
            str(output),
            "--decisions-per-source",
            str(decisions),
        ],
        cwd=Path(__file__).resolve().parent,
        env={
            "PYTHONPATH": os_path_join(
                Path(__file__).resolve().parent,
                reference_root / "training",
                reference_root / "data_pipeline",
            ),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        log_path=output.with_suffix(".log"),
    )
    return read_json(output)


def package_universal_agents(
    reference_root: Path,
    sources_path: Path,
    decks_path: Path,
    portable: Path,
    output_root: Path,
    name_prefix: str,
) -> dict[str, Any]:
    sources = read_json(sources_path)
    if sources.get("kind") != "experiment7_universal_bc":
        raise Experiment7Error("expected an experiment7_universal_bc source manifest")
    selected = read_json(decks_path).get("selected", [])
    if not selected:
        raise Experiment7Error("selected deck receipt has no decks")
    engine_catalog = Path(sources["engineCatalog"]["path"])
    runtime = reference_root / "runtime_agent"
    universal_portable = Path(__file__).resolve().parent / "universal_deck_portable.py"
    safe_features = Path(__file__).resolve().parent / "safe_features.py"
    required = [
        runtime / "main.py",
        runtime / "portable.py",
        runtime / "tokenizer.py",
        reference_root / "data_pipeline" / "features.py",
        universal_portable,
        safe_features,
        portable,
        engine_catalog,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    packages = []
    for row in selected:
        deck_path = Path(row["deckPath"])
        if not deck_path.is_file():
            raise FileNotFoundError(deck_path)
        package_name = f"{name_prefix}__{row['name']}" if name_prefix else row["name"]
        package = output_root / package_name
        if package.exists():
            shutil.rmtree(package)
        package.mkdir(parents=True)
        shutil.copy2(runtime / "main.py", package / "main.py")
        with (package / "main.py").open("a", encoding="utf-8") as handle:
            handle.write(
                "\n\ndef diagnostics():\n"
                "    return {\"bc\": bc_advisor.get_stats()}\n"
            )
        shutil.copy2(runtime / "portable.py", package / "portable.py")
        shutil.copy2(universal_portable, package / "deck_identity_portable.py")
        shutil.copy2(runtime / "tokenizer.py", package / "tokenizer.py")
        shutil.copy2(
            reference_root / "data_pipeline" / "features.py",
            package / "features_vendor.py",
        )
        shutil.copy2(safe_features, package / "features.py")
        shutil.copy2(portable, package / "deck_identity_bc.npz")
        shutil.copy2(engine_catalog, package / "engine_catalog.json")
        shutil.copy2(deck_path, package / "deck.csv")
        if not compileall.compile_dir(str(package), quiet=1, force=True):
            raise Experiment7Error(f"package compileall failed: {package}")
        for cache_dir in package.rglob("__pycache__"):
            shutil.rmtree(cache_dir)
        files = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(package.iterdir())
            if path.is_file()
        }
        receipt = {
            "schemaVersion": 1,
            "createdAt": utc_now(),
            "architecture": "experiment7_universal_deck8_autoregressive_stop",
            "name": package_name,
            "archetypeId": row.get("archetypeId"),
            "archetypeLabel": row.get("archetypeLabel"),
            "deckSha256": row["deckSha256"],
            "sourceDeck": str(deck_path.resolve()),
            "portableSource": {
                "path": str(portable.resolve()),
                "sha256": sha256_file(portable),
            },
            "engineCatalogSource": {
                "path": str(engine_catalog.resolve()),
                "sha256": sha256_file(engine_catalog),
            },
            "files": files,
        }
        write_json(package / "receipt.json", receipt)
        packages.append(
            {
                "name": package_name,
                "agentDir": str(package.resolve()),
                "deckSha256": row["deckSha256"],
                "archetypeId": row.get("archetypeId"),
                "archetypeLabel": row.get("archetypeLabel"),
                "directorySha256": directory_sha256(package),
            }
        )
    payload = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "architecture": "experiment7_universal_deck8_autoregressive_stop",
        "sources": {"path": str(sources_path.resolve()), "sha256": sha256_file(sources_path)},
        "decks": {"path": str(decks_path.resolve()), "sha256": sha256_file(decks_path)},
        "portable": {"path": str(portable.resolve()), "sha256": sha256_file(portable)},
        "packages": packages,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "packages.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def package_agents(reference_root: Path, sources_path: Path, portable: Path, output_root: Path) -> dict[str, Any]:
    manifest = read_json(sources_path)
    engine_catalog = Path(manifest["engineCatalog"]["path"])
    runtime = reference_root / "runtime_agent"
    required = [
        runtime / "main.py",
        runtime / "portable.py",
        runtime / "deck_identity_portable.py",
        runtime / "tokenizer.py",
        reference_root / "data_pipeline" / "features.py",
        portable,
        engine_catalog,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    safe_features = Path(__file__).resolve().parent / "safe_features.py"
    packages = []
    for row in manifest["currentSources"]:
        deck_path = Path(row["deckPath"])
        package = output_root / row["name"]
        if package.exists():
            shutil.rmtree(package)
        package.mkdir(parents=True)
        shutil.copy2(runtime / "main.py", package / "main.py")
        main_path = package / "main.py"
        with main_path.open("a", encoding="utf-8") as handle:
            handle.write("\n\ndef diagnostics():\n    return {\"bc\": bc_advisor.get_stats()}\n")
        shutil.copy2(runtime / "portable.py", package / "portable.py")
        shutil.copy2(runtime / "deck_identity_portable.py", package / "deck_identity_portable.py")
        shutil.copy2(runtime / "tokenizer.py", package / "tokenizer.py")
        shutil.copy2(reference_root / "data_pipeline" / "features.py", package / "features_vendor.py")
        shutil.copy2(safe_features, package / "features.py")
        shutil.copy2(portable, package / "deck_identity_bc.npz")
        shutil.copy2(engine_catalog, package / "engine_catalog.json")
        shutil.copy2(deck_path, package / "deck.csv")
        if not compileall.compile_dir(str(package), quiet=1, force=True):
            raise Experiment7Error(f"package compileall failed: {package}")
        for cache_dir in package.rglob("__pycache__"):
            shutil.rmtree(cache_dir)
        files = {}
        for path in sorted(package.iterdir()):
            if path.is_file():
                files[path.name] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        receipt = {
            "schemaVersion": 1,
            "createdAt": utc_now(),
            "name": row["name"],
            "deckSha256": row["deckSha256"],
            "sourceDeck": str(deck_path.resolve()),
            "portableSource": {"path": str(portable.resolve()), "sha256": sha256_file(portable)},
            "engineCatalogSource": {"path": str(engine_catalog.resolve()), "sha256": sha256_file(engine_catalog)},
            "files": files,
        }
        write_json(package / "receipt.json", receipt)
        packages.append({"name": row["name"], "agentDir": str(package.resolve()), "deckSha256": row["deckSha256"]})
    payload = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "sources": {"path": str(sources_path.resolve()), "sha256": sha256_file(sources_path)},
        "portable": {"path": str(portable.resolve()), "sha256": sha256_file(portable)},
        "packages": packages,
    }
    write_json(output_root / "packages.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and package Experiment 7 multi-deck Agents")
    sub = parser.add_subparsers(dest="command", required=True)

    export_parser = sub.add_parser("export")
    export_parser.add_argument("--checkpoint", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)

    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--reference-root", type=Path, required=True)
    verify_parser.add_argument("--sources", type=Path, required=True)
    verify_parser.add_argument("--checkpoint", type=Path, required=True)
    verify_parser.add_argument("--portable", type=Path, required=True)
    verify_parser.add_argument("--output", type=Path, required=True)
    verify_parser.add_argument("--python", default=sys.executable)
    verify_parser.add_argument("--decisions-per-source", type=int, default=100)

    select_parser = sub.add_parser("select-best")
    select_parser.add_argument("--root", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)

    package_parser = sub.add_parser("package")
    package_parser.add_argument("--reference-root", type=Path, required=True)
    package_parser.add_argument("--sources", type=Path, required=True)
    package_parser.add_argument("--portable", type=Path, required=True)
    package_parser.add_argument("--output-root", type=Path, required=True)

    universal_select = sub.add_parser("select-universal")
    universal_select.add_argument("--root", type=Path, required=True)
    universal_select.add_argument("--output", type=Path, required=True)
    universal_select.add_argument("--top-n", type=int, default=2)

    universal_verify = sub.add_parser("verify-universal")
    universal_verify.add_argument("--reference-root", type=Path, required=True)
    universal_verify.add_argument("--sources", type=Path, required=True)
    universal_verify.add_argument("--checkpoint", type=Path, required=True)
    universal_verify.add_argument("--portable", type=Path, required=True)
    universal_verify.add_argument("--output", type=Path, required=True)
    universal_verify.add_argument("--python", default=sys.executable)
    universal_verify.add_argument("--decisions-per-source", type=int, default=150)

    universal_package = sub.add_parser("package-universal")
    universal_package.add_argument("--reference-root", type=Path, required=True)
    universal_package.add_argument("--sources", type=Path, required=True)
    universal_package.add_argument("--decks", type=Path, required=True)
    universal_package.add_argument("--portable", type=Path, required=True)
    universal_package.add_argument("--output-root", type=Path, required=True)
    universal_package.add_argument("--name-prefix", default="")

    args = parser.parse_args()
    if args.command == "export":
        export_checkpoint(args.checkpoint.resolve(), args.output.resolve())
    elif args.command == "verify":
        verify_portable(
            args.reference_root.resolve(),
            args.sources.resolve(),
            args.checkpoint.resolve(),
            args.portable.resolve(),
            args.output.resolve(),
            args.python,
            args.decisions_per_source,
        )
    elif args.command == "select-best":
        select_best(args.root.resolve(), args.output.resolve())
    elif args.command == "package":
        package_agents(
            args.reference_root.resolve(),
            args.sources.resolve(),
            args.portable.resolve(),
            args.output_root.resolve(),
        )
    elif args.command == "select-universal":
        select_universal(args.root.resolve(), args.output.resolve(), args.top_n)
    elif args.command == "verify-universal":
        verify_universal_portable(
            args.reference_root.resolve(),
            args.sources.resolve(),
            args.checkpoint.resolve(),
            args.portable.resolve(),
            args.output.resolve(),
            args.python,
            args.decisions_per_source,
        )
    elif args.command == "package-universal":
        package_universal_agents(
            args.reference_root.resolve(),
            args.sources.resolve(),
            args.decks.resolve(),
            args.portable.resolve(),
            args.output_root.resolve(),
            args.name_prefix,
        )


if __name__ == "__main__":
    main()
