"""Loading and chunking the corpus, and loading the eval set."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from .models import Chunk, EvalCase

DOC_SUFFIXES = (".md", ".markdown", ".txt")
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")


def load_corpus(path: str | Path, max_chars: int = 900, overlap: int = 120) -> list[Chunk]:
    """Read every markdown/text file under `path` and chunk it."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"corpus path does not exist: {root}")

    files = [root] if root.is_file() else sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in DOC_SUFFIXES
    )

    chunks: list[Chunk] = []
    for file in files:
        doc_id = file.name if root.is_file() else file.relative_to(root).as_posix()
        chunks.extend(
            chunk_document(doc_id, file.read_text(encoding="utf-8"), max_chars, overlap)
        )
    return sorted(chunks, key=lambda c: (c.doc_id, c.index))


def chunk_document(
    doc_id: str, text: str, max_chars: int = 900, overlap: int = 120
) -> list[Chunk]:
    """Split at markdown headings first, then at paragraphs, then hard-split.

    Splitting on headings keeps a chunk's topic intact, which matters more for
    retrieval quality than hitting an exact chunk size.
    """
    chunks: list[Chunk] = []
    for heading, body in _sections(text):
        for piece in _split_body(body, max_chars, overlap):
            index = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{doc_id}#{index}",
                    doc_id=doc_id,
                    index=index,
                    heading=heading,
                    text=piece,
                )
            )
    return chunks


def load_evalset(path: str | Path) -> list[EvalCase]:
    """Read the YAML eval set. Accepts a bare list or a mapping with a `cases` key."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("cases")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a non-empty list of cases")

    cases: list[EvalCase] = []
    seen: set[str] = set()
    for position, entry in enumerate(raw):
        where = f"case #{position + 1}"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: expected a mapping, got {type(entry).__name__}")

        values = {}
        for field in ("id", "question", "gold"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{where}: '{field}' is required and must be a non-empty string")
            values[field] = value.strip()

        if values["id"] in seen:
            raise ValueError(f"duplicate case id {values['id']!r}")
        seen.add(values["id"])

        answer = entry.get("answer")
        if answer is not None and not isinstance(answer, str):
            raise ValueError(f"case {values['id']}: 'answer' must be a string if present")

        cases.append(
            EvalCase(
                id=values["id"],
                question=values["question"],
                gold=values["gold"],
                answer=answer,
                retrieved_ids=entry.get("retrieved_ids"),
                tags=list(entry.get("tags") or []),
            )
        )
    return cases


def _sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            sections.append((match.group(2).strip(), []))
        else:
            sections[-1][1].append(line)
    return [(heading, "\n".join(lines).strip()) for heading, lines in sections]


def _split_body(body: str, max_chars: int, overlap: int) -> list[str]:
    if not body.strip():
        return []
    if len(body) <= max_chars:
        return [body]

    pieces: list[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", body):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_hard_split(paragraph, max_chars, overlap))
        elif not current:
            current = paragraph
        elif len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}"
        else:
            pieces.append(current)
            current = paragraph
    if current:
        pieces.append(current)
    return pieces


def _hard_split(text: str, max_chars: int, overlap: int) -> list[str]:
    step = max(1, max_chars - overlap)
    pieces = []
    start = 0
    while start < len(text):
        pieces.append(text[start : start + max_chars])
        start += step
    return pieces
