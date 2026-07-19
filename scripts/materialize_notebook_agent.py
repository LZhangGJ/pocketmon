from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


ALLOWED_OUTPUTS = {"main.py", "deck.csv", "requirements.txt"}


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
