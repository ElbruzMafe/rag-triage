"""Rendering triage results as text, markdown or JSON."""

from __future__ import annotations

from .models import VERDICT_FIX, CaseResult, Verdict
from .triage import summarize

_ORDER = [
    Verdict.MISSING_FROM_CORPUS,
    Verdict.RETRIEVAL_MISS,
    Verdict.GENERATION_MISS,
    Verdict.UNGROUNDED,
    Verdict.NO_ANSWER,
    Verdict.OK,
]


def render_text(results: list[CaseResult], *, chunks: int, judge: str, k: int) -> str:
    lines = [
        f"rag-triage  {len(results)} cases  |  {chunks} chunks  |  judge {judge}  |  k={k}",
        "",
    ]

    width = max((len(r.case.id) for r in results), default=4)
    lines.append(f"  {'case'.ljust(width)}  {'verdict'.ljust(19)}  support rank")
    lines.append(f"  {'-' * width}  {'-' * 19}  ------------")
    for result in sorted(results, key=lambda r: _ORDER.index(r.verdict)):
        rank = "-" if result.best_support_rank is None else f"#{result.best_support_rank}"
        lines.append(f"  {result.case.id.ljust(width)}  {result.verdict.value.ljust(19)}  {rank}")

    counts = summarize(results)
    lines += ["", "summary"]
    for verdict in _ORDER:
        if counts[verdict.value]:
            lines.append(f"  {verdict.value.ljust(19)} {counts[verdict.value]}")

    failing = [v for v in _ORDER if v is not Verdict.OK and counts[v.value]]
    if failing:
        lines += ["", "what to fix first"]
        for verdict in sorted(failing, key=lambda v: -counts[v.value]):
            lines.append(f"  {counts[verdict.value]}x {verdict.value} -> {VERDICT_FIX[verdict]}")
    else:
        lines += ["", "every case passed"]
    return "\n".join(lines)


def render_markdown(results: list[CaseResult]) -> str:
    counts = summarize(results)
    lines = ["# RAG triage report", "", "| verdict | cases |", "| --- | --- |"]
    for verdict in _ORDER:
        if counts[verdict.value]:
            lines.append(f"| `{verdict.value}` | {counts[verdict.value]} |")

    lines += ["", "## Cases", ""]
    for result in sorted(results, key=lambda r: _ORDER.index(r.verdict)):
        lines.append(f"### {result.case.id} - `{result.verdict.value}`")
        lines.append("")
        lines.append(f"**Question:** {result.case.question}")
        lines.append("")
        lines.append(f"**Gold:** {result.case.gold}")
        if result.case.answer:
            lines.append("")
            lines.append(f"**Answer:** {result.case.answer}")
        lines.append("")
        lines.append(f"**Fix:** {result.fix_hint}")
        if result.support:
            evidence = ", ".join(f"`{hit.chunk.id}`" for hit in result.support)
            lines.append("")
            lines.append(f"**Evidence lives in:** {evidence}")
        for note in result.notes:
            lines.append("")
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)


def to_dict(results: list[CaseResult]) -> dict:
    return {
        "summary": summarize(results),
        "cases": [
            {
                "id": result.case.id,
                "question": result.case.question,
                "gold": result.case.gold,
                "answer": result.case.answer,
                "verdict": result.verdict.value,
                "fix": result.fix_hint,
                "support_chunks": [hit.chunk.id for hit in result.support],
                "retrieved_chunks": [hit.chunk.id for hit in result.retrieved],
                "best_support_rank": result.best_support_rank,
                "notes": result.notes,
                "tags": result.case.tags,
            }
            for result in results
        ],
    }
