"""Small, dependency-free Markdown tables for the Task 1 paper artifacts."""

from collections.abc import Iterable, Sequence
from pathlib import Path


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Return a readable Markdown table from column labels and ordered rows."""

    def cell(value: object) -> str:
        """Escape Markdown column separators in one display value."""

        return str(value).replace("|", r"\|").replace("\n", " ")

    width = len(headers)
    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        if len(row) != width:
            raise ValueError(f"Markdown row has {len(row)} cells; expected {width}")
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown_companion(artifact: Path, title: str, body: str) -> Path:
    """Write one same-stem Markdown companion next to a TeX or PDF artifact."""

    path = artifact.with_suffix(".md")
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")
    return path
