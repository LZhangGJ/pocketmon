#!/usr/bin/env python3
"""Static audit of downloaded Kaggle notebooks and selected discussions.

The output is deliberately evidence-first: it records source-level signals and
near-duplicate pairs, while leaving strategic interpretation to the report.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import io
import json
import re
import tarfile
from collections import Counter
from pathlib import Path


SIGNALS = {
    "reinforcement_learning": (r"\bppo\b", r"reinforcement learning", r"\bpolicy[_ -]?network\b", r"actor.critic"),
    "behavior_cloning": (r"behavior cloning", r"behaviour cloning", r"imitation learning", r"\bbc\b"),
    "search": (r"search_begin", r"search_step", r"\bmcts\b", r"minimax", r"rollout", r"lookahead"),
    "probabilistic": (r"probab", r"expected value", r"monte carlo", r"random\.choices", r"random\.shuffle"),
    "heuristic": (r"heuristic", r"rule[- ]?based", r"score_option", r"option_scores", r"priority", r"weights?\s*="),
    "opponent_model": (r"opponent", r"belief", r"template_sig", r"visible_opponent", r"matchup"),
    "fallback": (r"fallback", r"except exception", r"safe_agent", r"default_action"),
    "meta_analysis": (r"meta snapshot", r"usage_share", r"wilson", r"archetype", r"matchup_weighted"),
}

ARCHETYPES = {
    "Grimmsnarl": ("grimmsnarl", "grimm"),
    "Alakazam": ("alakazam", "abra", "kadabra"),
    "Lucario": ("lucario", "riolu"),
    "Dragapult": ("dragapult", "dreepy", "drakloak"),
    "Archaludon/Metal": ("archaludon", "duraludon", "metal"),
    "Crustle": ("crustle", "dwebble"),
    "Great Tusk": ("great_tusk", "great tusk"),
}

# Human-reviewed labels based on notebook narrative, callable structure, and
# (for rank 3) static inspection of the embedded tar payload.  These override
# noisy keyword classification but preserve the raw signal counts for audit.
REVIEWED = {
    1: ("meta analysis + rule-agent portfolio builder", "meta_snapshot", "June 29 data; stale for August decisions"),
    2: ("Lucario heuristic + deterministic beam/rollout search", "lucario_search", "strictly heuristic; engine search"),
    3: ("Grimmsnarl learned ensemble + expert router + extensive rule guards", "grim_hybrid", "compressed payload statically inspected; not pure RL"),
    4: ("Alakazam heuristic + determinization + bounded 2-ply minimax", "sol_eclipse", "exact-source family"),
    5: ("Archaludon deterministic heuristic + matchup routing", "metal_tempo", "stable rule submission"),
    6: ("Lucario heuristic + optional forward rollout", "lucario_search", "rule fallback on search failure"),
    7: ("Archaludon deterministic heuristic + matchup routing", "battlecore", "search tested but rejected from shipped agent"),
    8: ("Alakazam heuristic + determinization + bounded 2-ply minimax", "sol_eclipse", "exact duplicate of ranks 4/14"),
    9: ("Lucario deterministic heuristic with Crustle-specific rules", "lucario_rules", "no search/ML"),
    10: ("Lucario heuristic + probabilistic expectimax/rollout", "lucario_search", "near-copy family with rank 16"),
    11: ("Crustle counter deck + minimal deterministic rules", "crustle_simple", "deck construction is the main lever"),
    12: ("meta analysis + rule-agent portfolio builder", "meta_snapshot", "near-copy of rank 1; June 29 data"),
    13: ("Alakazam heuristic + visible-opponent belief + bounded search", "alakazam_belief", "public-state opponent templates"),
    14: ("Alakazam heuristic + determinization + bounded 2-ply minimax", "sol_eclipse", "exact duplicate of ranks 4/8"),
    15: ("Dragapult deterministic specialist heuristic", "dragapult_rules", "single-shot rule agent"),
    16: ("Lucario heuristic + conservative probabilistic rollout", "lucario_search", "95.7% similar to rank 10"),
    17: ("Archaludon deterministic heuristic + matchup routing", "battlecore", "same public builder lineage as rank 7"),
    18: ("Crustle minimal deterministic rules + safety fallbacks", "crustle_simple", "fixed package of day-one policy"),
    19: ("Alakazam heuristic + audited bounded search", "alakazam_belief", "search gated by margin and fallbacks"),
    20: ("Archaludon specialist assembled from deterministic policy modules", "metal_tempo", "fixed deck beat a failed three-deck ensemble"),
}


def read_notebook_text(path: Path) -> tuple[str, int, int, int, str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        obj = json.loads(raw, strict=False)
    except json.JSONDecodeError:
        return raw, 0, 0, 0, "raw_text"
    if not isinstance(obj, dict) or "cells" not in obj:
        return raw, 0, 0, 0, "json_without_cells"
    cells = obj.get("cells") or []
    pieces: list[str] = []
    code_cells = markdown_cells = 0
    for cell in cells:
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        pieces.append(str(source))
        if cell.get("cell_type") == "code":
            code_cells += 1
        elif cell.get("cell_type") == "markdown":
            markdown_cells += 1
    return "\n\n".join(pieces), len(cells), code_cells, markdown_cells, "notebook_json"


def inspect_embedded_archives(text: str) -> tuple[str, list[str]]:
    """Read text members from inline base64 tar assets without executing code."""
    appended: list[str] = []
    member_names: list[str] = []
    for match in re.finditer(r"(?m)^\s*[A-Z][A-Z0-9_]*B64\s*=\s*['\"]([A-Za-z0-9+/=]{256,})['\"]", text):
        try:
            payload = base64.b64decode(match.group(1), validate=True)
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    member_names.append(member.name)
                    if member.size > 5_000_000 or not member.name.lower().endswith((".py", ".md", ".txt", ".csv", ".json")):
                        continue
                    handle = archive.extractfile(member)
                    if handle is not None:
                        appended.append(handle.read().decode("utf-8", errors="replace"))
        except (ValueError, tarfile.TarError, OSError):
            continue
    return "\n\n".join(appended), sorted(set(member_names))


def normalized_policy_text(text: str) -> str:
    """Normalize likely policy code enough to detect public copy families."""
    chunks = []
    for match in re.finditer(r"(?s)(?:MAIN_SOURCE|main_source)\s*=\s*r?([\"']{3})(.*?)\1", text):
        chunks.append(match.group(2))
    candidate = "\n".join(chunks) if chunks else text
    candidate = re.sub(r"(?m)^\s*#.*$", "", candidate)
    candidate = re.sub(r"\s+", " ", candidate).lower().strip()
    return candidate


def policy_fingerprint(text: str, max_items: int = 25_000) -> set[int]:
    """Return a bounded token-shingle fingerprint for fast copy-family checks."""
    tokens = re.findall(r"[a-z_]\w*|\d+|==|!=|<=|>=|[-+*/%]", text)
    values = {
        int.from_bytes(hashlib.blake2b("\x1f".join(tokens[i : i + 7]).encode(), digest_size=8).digest(), "big")
        for i in range(max(0, len(tokens) - 6))
    }
    if len(values) > max_items:
        return set(sorted(values)[:max_items])
    return values


def extract_definitions(text: str) -> tuple[list[str], list[str]]:
    classes = sorted(set(re.findall(r"(?m)^\s*class\s+([A-Za-z_]\w*)", text)))
    functions = sorted(set(re.findall(r"(?m)^\s*def\s+([A-Za-z_]\w*)", text)))
    return classes, functions


def classify(signals: dict[str, int], title: str) -> str:
    low = title.lower()
    if signals["reinforcement_learning"] or signals["behavior_cloning"]:
        if signals["search"] or signals["heuristic"]:
            return "ML/RL + heuristic/search"
        return "ML/RL"
    if signals["search"]:
        return "heuristic + bounded search"
    if signals["meta_analysis"] and not re.search(r"agent|submit|bot", low):
        return "meta analysis + agent builder"
    if signals["probabilistic"]:
        return "probabilistic heuristic"
    return "deterministic heuristic/rules"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    notebook_dir = root / "notebooks"
    output_dir = root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (notebook_dir / "index.csv").open(encoding="utf-8-sig", newline="") as handle:
        index = list(csv.DictReader(handle))
    use_legacy_reviewed_labels = (
        len(index) == 20
        and index[0].get("ref") == "biohack44/pok-mon-tcg-ai-battle-meta-snapshot-07-july"
    )

    rows: list[dict[str, object]] = []
    normalized: dict[int, str] = {}
    fingerprints: dict[int, set[int]] = {}
    for item in index:
        rank = int(item["rank_observed"])
        path = root / item["source_file"]
        text, cells, code_cells, markdown_cells, source_format = read_notebook_text(path)
        embedded_text, embedded_members = inspect_embedded_archives(text)
        if embedded_text:
            text = text + "\n\n" + embedded_text
        low = text.lower()
        signal_counts = {
            name: sum(len(re.findall(pattern, low, flags=re.I)) for pattern in patterns)
            for name, patterns in SIGNALS.items()
        }
        archetypes = [name for name, terms in ARCHETYPES.items() if any(term in low for term in terms)]
        classes, functions = extract_definitions(text)
        policy = normalized_policy_text(text)
        normalized[rank] = policy
        fingerprints[rank] = policy_fingerprint(policy)
        automatic_method = classify(signal_counts, item["title"])
        reviewed_method, lineage, review_note = (
            REVIEWED[rank]
            if use_legacy_reviewed_labels
            else (automatic_method, "unreviewed_frontier", "automatic source classification; human review pending")
        )
        score = (
            float(item["public_score_observed"])
            if item.get("public_score_observed") not in (None, "")
            else None
        )
        votes = int(item["votes_observed"]) if item.get("votes_observed") not in (None, "") else None
        rows.append(
            {
                "rank": rank,
                "ref": item["ref"],
                "title": item["title"],
                "score": score,
                "votes": votes,
                "source_file": item["source_file"],
                "source_format": source_format,
                "source_bytes": path.stat().st_size,
                "cells": cells,
                "code_cells": code_cells,
                "markdown_cells": markdown_cells,
                "embedded_archive_members": embedded_members,
                "line_count": text.count("\n") + 1,
                "method_family": automatic_method,
                "reviewed_method": reviewed_method,
                "lineage": lineage,
                "review_note": review_note,
                "archetypes": archetypes,
                "signals": signal_counts,
                "classes": classes,
                "functions": functions,
                "policy_sha256": hashlib.sha256(policy.encode()).hexdigest(),
                "url": item["url"],
            }
        )

    similarities: list[dict[str, object]] = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            a = fingerprints[int(rows[left]["rank"])]
            b = fingerprints[int(rows[right]["rank"])]
            ratio = len(a & b) / len(a | b) if a or b else 1.0
            if ratio >= 0.72:
                similarities.append(
                    {
                        "rank_a": rows[left]["rank"],
                        "rank_b": rows[right]["rank"],
                        "similarity": round(ratio, 4),
                        "ref_a": rows[left]["ref"],
                        "ref_b": rows[right]["ref"],
                    }
                )
    similarities.sort(key=lambda row: float(row["similarity"]), reverse=True)

    family_counts = Counter(str(row["method_family"]) for row in rows)
    lineage_counts = Counter(str(row["lineage"]) for row in rows)
    scores = [float(row["score"]) for row in rows if row["score"] is not None]
    summary = {
        "notebook_count": len(rows),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "method_family_counts": dict(family_counts),
        "reviewed_lineage_counts": dict(lineage_counts),
        "reviewed_unique_lineages": len(lineage_counts),
        "rl_or_bc_source_hits": sum(
            1
            for row in rows
            if row["signals"]["reinforcement_learning"] or row["signals"]["behavior_cloning"]
        ),
        "search_source_hits": sum(1 for row in rows if row["signals"]["search"]),
        "near_duplicate_pairs_at_0_72": len(similarities),
    }
    (output_dir / "notebook_audit.json").write_text(
        json.dumps({"summary": summary, "notebooks": rows, "similarities": similarities}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "notebook_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "rank", "ref", "title", "score", "votes", "reviewed_method", "lineage", "review_note", "method_family", "archetypes",
            "source_format", "source_bytes", "cells", "code_cells", "line_count",
            "rl_hits", "bc_hits", "search_hits", "probabilistic_hits", "heuristic_hits", "url",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row[key] for key in fields if key in row},
                    "archetypes": "; ".join(row["archetypes"]),
                    "rl_hits": row["signals"]["reinforcement_learning"],
                    "bc_hits": row["signals"]["behavior_cloning"],
                    "search_hits": row["signals"]["search"],
                    "probabilistic_hits": row["signals"]["probabilistic"],
                    "heuristic_hits": row["signals"]["heuristic"],
                }
            )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
