from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import py_compile
import re
import tarfile
from pathlib import Path


ALLOWED_OUTPUTS = {"main.py", "deck.csv", "requirements.txt"}
MAX_EMBEDDED_FILE_BYTES = 8 * 1024 * 1024


def extract_literal_deck(payload: dict) -> list[int] | None:
    """Find a literal 60-card list/string without executing notebook code."""
    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.startswith("%%"):
            source = "\n".join(source.splitlines()[1:])
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            try:
                literal = ast.literal_eval(value)
            except (ValueError, TypeError, SyntaxError):
                continue
            if isinstance(literal, (list, tuple)) and len(literal) == 60:
                if all(isinstance(card, int) for card in literal):
                    return list(literal)
            if isinstance(literal, str):
                try:
                    cards = [int(line.strip()) for line in literal.splitlines() if line.strip()]
                except ValueError:
                    continue
                if len(cards) == 60:
                    return cards
    return None


def extract_commented_deck(payload: dict) -> list[int] | None:
    """Extract `CARD = id  # x4` style deck declarations used by sample agents."""
    pattern = re.compile(r"^\s*[A-Za-z_]\w*\s*=\s*(\d+)\s*#.*?[x×]\s*(\d+)\s*$", re.IGNORECASE)
    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        cards: list[int] = []
        for line in "".join(cell.get("source", [])).splitlines():
            match = pattern.match(line)
            if match:
                cards.extend([int(match.group(1))] * int(match.group(2)))
        if len(cards) == 60:
            return cards
    return None


def notebook_code(payload: dict) -> str:
    return "\n\n".join(
        "".join(cell.get("source", []))
        for cell in payload.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def extract_literal_agent_sources(payload: dict) -> dict[str, bytes] | None:
    """Find literal Python/deck payload strings without executing the notebook."""
    values: dict[str, str] = {}
    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        try:
            tree = ast.parse("".join(cell.get("source", [])))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            if isinstance(value, str):
                values[targets[0].id.lower()] = value
    main_values = [value for name, value in values.items() if "main" in name and ("source" in name or "py" in name)]
    deck_values = [value for name, value in values.items() if "deck" in name and ("csv" in name or "source" in name)]
    for main in main_values:
        for deck in deck_values:
            cards = [line.strip() for line in deck.splitlines() if line.strip()]
            if len(cards) == 60 and all(card.isdigit() for card in cards) and "def agent" in main:
                return {"main.py": main.encode("utf-8"), "deck.csv": ("\n".join(cards) + "\n").encode("utf-8")}
    return None


def extract_embedded_agent(payload: dict) -> dict[str, bytes] | None:
    """Inspect base64 tar assets and return one unambiguous packaged agent."""
    text = notebook_code(payload)
    archive_pattern = re.compile(
        r"(?m)^\s*[A-Z_a-z][A-Z_a-z0-9]*B64\s*=\s*['\"]([A-Za-z0-9+/=]{64,})['\"]"
    )
    candidates: list[dict[str, bytes]] = []
    for match in archive_pattern.finditer(text):
        try:
            decoded = base64.b64decode(match.group(1), validate=True)
            with tarfile.open(fileobj=io.BytesIO(decoded), mode="r:*") as archive:
                grouped: dict[str, dict[str, bytes]] = {}
                for member in archive.getmembers():
                    if not member.isfile() or member.size > MAX_EMBEDDED_FILE_BYTES:
                        continue
                    member_path = Path(member.name)
                    if member_path.is_absolute() or ".." in member_path.parts:
                        continue
                    name = member_path.name
                    if name not in ALLOWED_OUTPUTS:
                        continue
                    handle = archive.extractfile(member)
                    if handle is not None:
                        grouped.setdefault(member_path.parent.as_posix(), {})[name] = handle.read()
                candidates.extend(files for files in grouped.values() if {"main.py", "deck.csv"} <= files.keys())
        except (ValueError, tarfile.TarError, OSError):
            continue
    unique: dict[tuple[str, str], dict[str, bytes]] = {}
    for files in candidates:
        key = (
            base64.b16encode(files["main.py"][:64]).decode("ascii"),
            base64.b16encode(files["deck.csv"][:64]).decode("ascii"),
        )
        unique[key] = files
    return next(iter(unique.values())) if len(unique) == 1 else None


def materialize(notebook: Path, output: Path, fallback_deck: Path | None = None) -> list[Path]:
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        lines = source.splitlines(keepends=True)
        if not lines or not lines[0].startswith("%%writefile "):
            continue
        name = lines[0].removeprefix("%%writefile ").strip()
        if name not in ALLOWED_OUTPUTS:
            continue
        target = output / name
        target.write_text("".join(lines[1:]), encoding="utf-8")
        written.append(target)
    names = {path.name for path in written}
    if not names >= {"main.py", "deck.csv"}:
        packaged = extract_embedded_agent(payload) or extract_literal_agent_sources(payload)
        if packaged is not None:
            for name, content in packaged.items():
                target = output / name
                target.write_bytes(content)
                if target not in written:
                    written.append(target)
            names = {path.name for path in written}
    if "deck.csv" not in names:
        cards = extract_literal_deck(payload) or extract_commented_deck(payload)
        if cards is not None:
            target = output / "deck.csv"
            target.write_text("\n".join(map(str, cards)) + "\n", encoding="utf-8")
            written.append(target)
            names.add("deck.csv")
    if "deck.csv" not in names and fallback_deck is not None:
        target = output / "deck.csv"
        target.write_text(fallback_deck.read_text(encoding="utf-8"), encoding="utf-8")
        written.append(target)
        names.add("deck.csv")
    if not names >= {"main.py", "deck.csv"}:
        raise ValueError(f"Notebook does not contain main.py and deck.csv writefile cells: {notebook}")
    py_compile.compile(str(output / "main.py"), doraise=True)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract public notebook agent sources for local evaluation")
    parser.add_argument("notebook")
    parser.add_argument("output")
    parser.add_argument("--fallback-deck", type=Path)
    args = parser.parse_args()
    for path in materialize(Path(args.notebook), Path(args.output), args.fallback_deck):
        print(path)


if __name__ == "__main__":
    main()
