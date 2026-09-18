"""Rendering triage results as text, markdown, HTML or JSON."""

from __future__ import annotations

from html import escape

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

_STAGE = {
    Verdict.MISSING_FROM_CORPUS: "corpus",
    Verdict.RETRIEVAL_MISS: "retrieval",
    Verdict.GENERATION_MISS: "generation",
    Verdict.UNGROUNDED: "generation",
    Verdict.NO_ANSWER: "-",
    Verdict.OK: "-",
}


def by_verdict(results: list[CaseResult]) -> list[CaseResult]:
    """Worst first, then alphabetical, so two runs of the same set print identically."""
    return sorted(results, key=lambda r: (_ORDER.index(r.verdict), r.case.id))


def render_text(
    results: list[CaseResult], *, chunks: int, judge: str, k: int, retriever: str = "bm25"
) -> str:
    lines = [
        f"rag-triage  {len(results)} cases  |  {chunks} chunks  |  "
        f"retriever {retriever}  |  judge {judge}  |  k={k}",
        "",
    ]

    width = max((len(r.case.id) for r in results), default=4)
    lines.append(f"  {'case'.ljust(width)}  {'verdict'.ljust(19)}  support rank")
    lines.append(f"  {'-' * width}  {'-' * 19}  ------------")
    for result in by_verdict(results):
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


def render_case(result: CaseResult) -> str:
    """The full picture for a single case: what was asked, what came back, and why."""
    case = result.case
    lines = [
        f"case     {case.id}",
        f"verdict  {result.verdict.value}  (stage: {_STAGE[result.verdict]})",
        f"fix      {result.fix_hint}",
        "",
        "question",
        f"  {case.question}",
        "gold",
        f"  {case.gold}",
    ]
    if case.answer:
        lines += ["answer", f"  {case.answer}"]

    support_ids = {hit.chunk.id for hit in result.support}
    lines += ["", "evidence for the gold answer"]
    if result.support:
        lines += [f"  {_hit_line(hit)}" for hit in result.support]
    else:
        lines.append("  none found in the corpus")

    count = len(result.retrieved)
    lines += ["", f"retrieved context ({count} chunk{'' if count == 1 else 's'})"]
    if result.retrieved:
        for hit in result.retrieved:
            marker = "  <- supporting" if hit.chunk.id in support_ids else ""
            lines.append(f"  {_hit_line(hit)}{marker}")
    else:
        lines.append("  nothing retrieved")

    if result.claims:
        lines += ["", "answer sentences"]
        for check in result.claims:
            mark = "ok" if check.supported else "!!"
            where = check.chunk_id if check.supported else "unsupported"
            lines.append(f"  {mark}  [{where}] {check.text}")

    if result.notes:
        lines += ["", "notes"]
        lines += [f"  - {note}" for note in result.notes]
    return "\n".join(lines)


def render_markdown(results: list[CaseResult]) -> str:
    counts = summarize(results)
    lines = ["# RAG triage report", "", "| verdict | cases |", "| --- | --- |"]
    for verdict in _ORDER:
        if counts[verdict.value]:
            lines.append(f"| `{verdict.value}` | {counts[verdict.value]} |")

    lines += ["", "## Cases", ""]
    for result in by_verdict(results):
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
        loose = [check for check in result.claims if not check.supported]
        if loose:
            lines.append("")
            lines.append("**Sentences no retrieved chunk supports:**")
            lines.append("")
            lines += [f"- {check.text}" for check in loose]
        for note in result.notes:
            lines.append("")
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)


def render_html(
    results: list[CaseResult], *, chunks: int, judge: str, k: int, retriever: str = "bm25"
) -> str:
    counts = summarize(results)
    failed = sum(counts[v.value] for v in _ORDER if v is not Verdict.OK)
    present = [v for v in _ORDER if counts[v.value]]

    # The tiles double as filters: a hidden radio per verdict, and CSS that hides the
    # cards of every other verdict. Keeps the report one file with no JavaScript.
    radios = '<input type="radio" name="filter" id="f-all" checked>' + "".join(
        f'<input type="radio" name="filter" id="f-{v.value}">' for v in present
    )
    tiles = f'<label class="tile all" for="f-all"><b>{len(results)}</b><span>all</span></label>'
    tiles += "".join(
        f'<label class="tile {v.value}" for="f-{v.value}">'
        f"<b>{counts[v.value]}</b><span>{v.value}</span></label>"
        for v in present
    )
    filter_css = "\n".join(
        f"#f-{v.value}:checked ~ .cards .case:not(.v-{v.value}) {{ display:none }}\n"
        f'#f-{v.value}:checked ~ .tiles label[for="f-{v.value}"] {{ background:#000 }}'
        for v in present
    )

    cards = []
    for result in by_verdict(results):
        support_ids = {hit.chunk.id for hit in result.support}
        rows = "".join(
            f'<tr class="{"hit" if hit.chunk.id in support_ids else ""}">'
            f"<td>#{hit.rank}</td><td><code>{escape(hit.chunk.id)}</code></td>"
            f"<td>{escape(hit.chunk.heading or '-')}</td>"
            f"<td>{escape(_preview(hit.chunk.text))}</td></tr>"
            for hit in result.retrieved
        )
        claims = "".join(
            f'<li class="{"ok" if c.supported else "bad"}">{escape(c.text)}'
            f'<span>{escape(c.chunk_id or "unsupported")}</span></li>'
            for c in result.claims
        )
        notes = "".join(f"<li>{escape(note)}</li>" for note in result.notes)
        evidence = (
            ", ".join(f"<code>{escape(hit.chunk.id)}</code>" for hit in result.support)
            or "nothing in the corpus"
        )
        tags = "".join(f'<span class="tag">{escape(tag)}</span>' for tag in result.case.tags)
        cards.append(
            f"""<details class="case v-{result.verdict.value}">
<summary><span class="chip {result.verdict.value}">{result.verdict.value}</span>
<b>{escape(result.case.id)}</b><em>{escape(_preview(result.case.question, 90))}</em></summary>
<p class="fix">{escape(result.fix_hint)}</p>
<dl><dt>gold</dt><dd>{escape(result.case.gold)}</dd>
<dt>answer</dt><dd>{escape(result.case.answer or "(none recorded)")}</dd>
<dt>evidence</dt><dd>{evidence}</dd>
{f"<dt>tags</dt><dd>{tags}</dd>" if tags else ""}</dl>
{f'<h4>answer sentences</h4><ul class="claims">{claims}</ul>' if claims else ""}
<h4>retrieved context</h4>
{f"<table>{rows}</table>" if rows else "<p>nothing retrieved</p>"}
{f"<h4>notes</h4><ul>{notes}</ul>" if notes else ""}
</details>"""
        )

    return _HTML.format(
        cases=len(results),
        failed=failed,
        chunks=chunks,
        judge=escape(judge),
        retriever=escape(retriever),
        k=k,
        radios=radios,
        tiles=tiles,
        filter_css=filter_css,
        cards="\n".join(cards),
    )


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
                "stage": _STAGE[result.verdict],
                "fix": result.fix_hint,
                "support_chunks": [hit.chunk.id for hit in result.support],
                "retrieved_chunks": [hit.chunk.id for hit in result.retrieved],
                "best_support_rank": result.best_support_rank,
                "claims": [
                    {
                        "text": check.text,
                        "supported": check.supported,
                        "confidence": check.confidence,
                        "chunk_id": check.chunk_id,
                    }
                    for check in result.claims
                ],
                "notes": result.notes,
                "tags": result.case.tags,
            }
            for result in results
        ],
    }


def _hit_line(hit) -> str:
    heading = hit.chunk.heading or hit.chunk.doc_id
    return f"#{hit.rank}  {hit.chunk.id.ljust(18)}  score {hit.score:<8}  {heading}"


def _preview(text: str, limit: int = 160) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rag-triage report</title>
<style>
:root {{
  --bg:#11131a; --card:#191c26; --line:#2b3040; --text:#dfe3ee; --dim:#8d94a8;
  --corpus:#e0654f; --retrieval:#e0a23c; --generation:#7c8ae0; --neutral:#616a85; --ok:#4fae7c;
}}
* {{ box-sizing:border-box }}
body {{ margin:0; padding:2rem 1rem; background:var(--bg); color:var(--text);
  font:15px/1.55 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif }}
main {{ max-width:900px; margin:0 auto }}
h1 {{ font-size:1.5rem; margin:0 0 .25rem }}
.meta {{ color:var(--dim); margin:0 0 1.5rem }}
.tiles {{ display:flex; flex-wrap:wrap; gap:.6rem; margin-bottom:1.5rem }}
.tile {{ flex:1 1 120px; background:var(--card); border:1px solid var(--line);
  border-left-width:4px; border-radius:8px; padding:.7rem .9rem; cursor:pointer;
  user-select:none }}
.tile:hover {{ border-color:var(--dim) }}
.tile b {{ display:block; font-size:1.6rem; line-height:1.1 }}
.tile span {{ color:var(--dim); font-size:.8rem }}
.tile.all {{ border-left-color:var(--dim) }}
input[name="filter"] {{ display:none }}
#f-all:checked ~ .tiles label[for="f-all"] {{ background:#000 }}
{filter_css}
.missing_from_corpus {{ border-left-color:var(--corpus) }}
.retrieval_miss {{ border-left-color:var(--retrieval) }}
.generation_miss, .ungrounded {{ border-left-color:var(--generation) }}
.no_answer {{ border-left-color:var(--neutral) }}
.ok {{ border-left-color:var(--ok) }}
.case {{ background:var(--card); border:1px solid var(--line); border-radius:8px;
  margin-bottom:.6rem; padding:.7rem .9rem }}
summary {{ cursor:pointer; display:flex; align-items:center; gap:.6rem }}
summary em {{ color:var(--dim); font-style:normal; overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap }}
.chip {{ font-size:.72rem; padding:.15rem .5rem; border-radius:99px; white-space:nowrap;
  background:#000a; border:1px solid var(--line); border-left-width:3px }}
.fix {{ color:var(--dim); margin:.8rem 0 }}
dl {{ display:grid; grid-template-columns:6rem 1fr; gap:.35rem .8rem; margin:0 0 1rem }}
dt {{ color:var(--dim); font-size:.82rem }}
dd {{ margin:0 }}
h4 {{ margin:1rem 0 .4rem; font-size:.8rem; text-transform:uppercase; color:var(--dim);
  letter-spacing:.06em }}
table {{ width:100%; border-collapse:collapse; font-size:.85rem }}
td {{ padding:.35rem .5rem; border-top:1px solid var(--line); vertical-align:top }}
tr.hit td {{ background:#4fae7c1a }}
td:nth-child(4) {{ color:var(--dim) }}
code {{ font-size:.85em; color:#9fb4e8 }}
ul {{ margin:.3rem 0; padding-left:1.1rem; color:var(--dim) }}
.claims {{ list-style:none; padding:0 }}
.claims li {{ padding:.35rem .6rem; border-left:3px solid var(--ok); margin-bottom:.3rem;
  background:#0006; color:var(--text) }}
.claims li.bad {{ border-left-color:var(--corpus) }}
.claims span {{ display:block; color:var(--dim); font-size:.75rem }}
.tag {{ display:inline-block; font-size:.72rem; color:var(--dim); border:1px solid var(--line);
  border-radius:99px; padding:.05rem .5rem; margin-right:.3rem }}
</style></head><body><main>
<h1>rag-triage report</h1>
<p class="meta">{cases} cases &middot; {failed} not ok &middot; {chunks} chunks &middot;
retriever {retriever} &middot; judge {judge} &middot; k={k}</p>
{radios}
<div class="tiles">{tiles}</div>
<div class="cards">{cards}</div>
</main></body></html>
"""
