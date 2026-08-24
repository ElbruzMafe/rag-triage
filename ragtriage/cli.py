"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .corpus import load_corpus, load_evalset
from .judge import LexicalJudge
from .models import Verdict
from .report import render_case, render_html, render_markdown, render_text, to_dict
from .retriever import BM25Retriever
from .triage import TriageConfig, triage_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-triage",
        description="Say whether a wrong RAG answer is a corpus, retrieval or generation problem.",
    )
    parser.add_argument("--corpus", required=True, help="directory of .md/.txt documents")
    parser.add_argument("--evalset", required=True, help="YAML file of eval cases")
    parser.add_argument("-k", type=int, default=5, help="top-k the system under test retrieves")
    parser.add_argument("--judge", choices=("lexical", "claude"), default="lexical")
    parser.add_argument("--model", default=None, help="model id for --judge claude")
    parser.add_argument("--max-chars", type=int, default=900, help="chunk size in characters")
    parser.add_argument("--overlap", type=int, default=120, help="chunk overlap in characters")
    parser.add_argument(
        "--support-scan",
        type=int,
        default=25,
        help="how many candidate chunks per case get judged when looking for the evidence",
    )
    parser.add_argument(
        "--explain",
        metavar="CASE_ID",
        help="print the full trace for one case instead of the summary table",
    )
    parser.add_argument(
        "--no-claim-detail",
        action="store_true",
        help="skip the per-sentence grounding check (fewer judge calls)",
    )
    parser.add_argument("--json", dest="json_out", help="write the full result as JSON")
    parser.add_argument("--markdown", dest="md_out", help="write a markdown report")
    parser.add_argument("--html", dest="html_out", help="write a self-contained HTML report")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any case is not ok")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.k < 1:
        print("rag-triage: -k must be at least 1", file=sys.stderr)
        return 2

    try:
        chunks = load_corpus(args.corpus, args.max_chars, args.overlap)
        cases = load_evalset(args.evalset)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"rag-triage: {exc}", file=sys.stderr)
        return 2

    if not chunks:
        print(f"rag-triage: no .md or .txt documents under {args.corpus}", file=sys.stderr)
        return 2

    if args.explain and not any(case.id == args.explain for case in cases):
        known = ", ".join(case.id for case in cases)
        print(f"rag-triage: no case {args.explain!r} in the eval set. Known ids: {known}",
              file=sys.stderr)
        return 2

    if args.judge == "claude":
        from .claude_judge import ClaudeJudge

        judge = ClaudeJudge(model=args.model) if args.model else ClaudeJudge()
    else:
        judge = LexicalJudge()

    # --explain is a drill-down, so only the case being explained is worth triaging.
    if args.explain:
        cases = [case for case in cases if case.id == args.explain]

    retriever = BM25Retriever(chunks)
    config = TriageConfig(
        k=args.k, support_scan=args.support_scan, claim_detail=not args.no_claim_detail
    )
    try:
        results = triage_all(cases, retriever, judge, config)
    except ValueError as exc:
        print(f"rag-triage: {exc}", file=sys.stderr)
        return 2

    if args.explain:
        print(render_case(results[0]))
    else:
        print(render_text(results, chunks=len(chunks), judge=judge.name, k=args.k))

    usage = getattr(judge, "usage", None)
    if usage is not None and usage.calls:
        print(f"\njudge usage\n  {usage.summary()}")

    try:
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(to_dict(results), indent=2), encoding="utf-8")
        if args.md_out:
            Path(args.md_out).write_text(render_markdown(results), encoding="utf-8")
        if args.html_out:
            Path(args.html_out).write_text(
                render_html(results, chunks=len(chunks), judge=judge.name, k=args.k),
                encoding="utf-8",
            )
    except OSError as exc:
        print(f"rag-triage: could not write report: {exc}", file=sys.stderr)
        return 2

    failed = any(result.verdict is not Verdict.OK for result in results)
    return 1 if (args.strict and failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
