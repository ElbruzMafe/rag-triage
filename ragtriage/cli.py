"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .compare import Arm, compare, summarize_comparison
from .corpus import load_corpus, load_evalset
from .judge import build_judge
from .models import Verdict
from .report import (
    comparison_to_dict,
    render_case,
    render_comparison,
    render_html,
    render_markdown,
    render_text,
    to_dict,
)
from .retriever import BM25Retriever
from .triage import TriageConfig, triage_all
from .vectors import VectorRetriever, embed_inputs, load_vectors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-triage",
        description="Say whether a wrong RAG answer is a corpus, retrieval or generation problem.",
    )
    parser.add_argument("--corpus", required=True, help="directory of .md/.txt documents")
    parser.add_argument("--evalset", required=True, help="YAML file of eval cases")
    parser.add_argument("-k", type=int, default=5, help="top-k the system under test retrieves")
    parser.add_argument(
        "--judge",
        default="lexical",
        metavar="SPEC",
        help="lexical, claude, or lexical@0.8 to move the support threshold",
    )
    parser.add_argument(
        "--judge-b",
        dest="judge_b",
        metavar="SPEC",
        help="second judge for --compare judge, same spec form as --judge",
    )
    parser.add_argument("--model", default=None, help="model id for the claude judge")
    parser.add_argument("--max-chars", type=int, default=900, help="chunk size in characters")
    parser.add_argument("--overlap", type=int, default=120, help="chunk overlap in characters")
    parser.add_argument(
        "--support-scan",
        type=int,
        default=25,
        help="how many candidate chunks per case get judged when looking for the evidence",
    )
    parser.add_argument(
        "--vectors",
        metavar="FILE",
        help="JSON file of chunk and query vectors; rank by cosine instead of BM25",
    )
    parser.add_argument(
        "--embed-inputs",
        dest="embed_inputs",
        metavar="FILE",
        help="write the chunk and query texts that a --vectors file must cover, then exit",
    )
    parser.add_argument(
        "--compare",
        nargs="?",
        const="retriever",
        choices=("retriever", "judge"),
        metavar="AXIS",
        help="compare two retrievers (needs --vectors) or two judges (needs --judge-b); "
        "defaults to retriever",
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
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 if any case is not ok; with --compare, if B loses ground A held",
    )
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

    if args.embed_inputs:
        payload = embed_inputs(chunks, cases)
        try:
            Path(args.embed_inputs).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"rag-triage: could not write {args.embed_inputs}: {exc}", file=sys.stderr)
            return 2
        print(
            f"wrote {len(payload['chunks'])} chunk texts and "
            f"{len(payload['queries'])} query texts to {args.embed_inputs}"
        )
        return 0

    if args.compare == "retriever" and not args.vectors:
        print(
            "rag-triage: --compare retriever needs --vectors: "
            "it compares the bm25 baseline against a vector retriever",
            file=sys.stderr,
        )
        return 2

    if args.compare == "judge" and not args.judge_b:
        print(
            "rag-triage: --compare judge needs --judge-b, the second judge to grade with",
            file=sys.stderr,
        )
        return 2

    if args.judge_b and args.compare != "judge":
        print("rag-triage: --judge-b only does something with --compare judge", file=sys.stderr)
        return 2

    if args.compare == "judge" and args.judge_b == args.judge:
        print(
            f"rag-triage: --judge and --judge-b are both {args.judge!r}; "
            "comparing a judge with itself measures nothing",
            file=sys.stderr,
        )
        return 2

    if args.compare:
        if args.explain:
            print("rag-triage: --compare and --explain cannot be used together", file=sys.stderr)
            return 2
        if args.md_out or args.html_out:
            print("rag-triage: --compare writes text and --json only", file=sys.stderr)
            return 2

    if args.explain and not any(case.id == args.explain for case in cases):
        known = ", ".join(case.id for case in cases)
        print(f"rag-triage: no case {args.explain!r} in the eval set. Known ids: {known}",
              file=sys.stderr)
        return 2

    try:
        judge = build_judge(args.judge, args.model)
        judge_b = build_judge(args.judge_b, args.model) if args.judge_b else None
    except ValueError as exc:
        print(f"rag-triage: {exc}", file=sys.stderr)
        return 2
    except ImportError:
        print(
            "rag-triage: the claude judge needs the anthropic SDK: "
            'pip install "rag-triage[claude]"',
            file=sys.stderr,
        )
        return 2

    config = TriageConfig(
        k=args.k, support_scan=args.support_scan, claim_detail=not args.no_claim_detail
    )

    if args.compare:
        try:
            if args.compare == "judge":
                # One retriever instance for both arms: the ranking is the constant here.
                retriever = _build_retriever(chunks, args.vectors)
                arm_a = Arm(judge.name, retriever, judge)
                arm_b = Arm(judge_b.name, retriever, judge_b)
                fixed = retriever.name
            else:
                # One judge instance for both arms, which is what lets compare() locate
                # the evidence once and hand the same answer to each retriever.
                baseline = BM25Retriever(chunks)
                candidate = _build_retriever(chunks, args.vectors)
                arm_a = Arm(baseline.name, baseline, judge)
                arm_b = Arm(candidate.name, candidate, judge)
                fixed = judge.name
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"rag-triage: {exc}", file=sys.stderr)
            return 2

        try:
            comparisons = compare(cases, arm_a, arm_b, config)
        except (ValueError, LookupError) as exc:
            print(f"rag-triage: {exc}", file=sys.stderr)
            return 2

        summary = summarize_comparison(comparisons)
        print(
            render_comparison(
                comparisons,
                summary,
                axis=args.compare,
                chunks=len(chunks),
                k=args.k,
                name_a=arm_a.label,
                name_b=arm_b.label,
                fixed=fixed,
            )
        )
        _print_judge_usage(judge, judge_b)

        if args.json_out:
            payload = comparison_to_dict(
                comparisons,
                summary,
                axis=args.compare,
                name_a=arm_a.label,
                name_b=arm_b.label,
            )
            try:
                Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except OSError as exc:
                print(f"rag-triage: could not write report: {exc}", file=sys.stderr)
                return 2

        losses = summary.masked if args.compare == "judge" else summary.regressions
        return 1 if (args.strict and losses) else 0

    # --explain is a drill-down, so only the case being explained is worth triaging.
    if args.explain:
        cases = [case for case in cases if case.id == args.explain]

    try:
        retriever = _build_retriever(chunks, args.vectors)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"rag-triage: {exc}", file=sys.stderr)
        return 2

    try:
        results = triage_all(cases, retriever, judge, config)
    except (ValueError, LookupError) as exc:
        print(f"rag-triage: {exc}", file=sys.stderr)
        return 2

    if args.explain:
        print(render_case(results[0]))
    else:
        print(
            render_text(
                results,
                chunks=len(chunks),
                judge=judge.name,
                k=args.k,
                retriever=retriever.name,
            )
        )

    _print_judge_usage(judge)

    try:
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(to_dict(results), indent=2), encoding="utf-8")
        if args.md_out:
            Path(args.md_out).write_text(render_markdown(results), encoding="utf-8")
        if args.html_out:
            Path(args.html_out).write_text(
                render_html(
                    results,
                    chunks=len(chunks),
                    judge=judge.name,
                    k=args.k,
                    retriever=retriever.name,
                ),
                encoding="utf-8",
            )
    except OSError as exc:
        print(f"rag-triage: could not write report: {exc}", file=sys.stderr)
        return 2

    failed = any(result.verdict is not Verdict.OK for result in results)
    return 1 if (args.strict and failed) else 0


def _print_judge_usage(*judges) -> None:
    """Judge calls are the cost centre, so any run that spent some says how many."""
    spent = []
    for judge in judges:
        usage = getattr(judge, "usage", None)
        if usage is None or not usage.calls or any(judge is other for other, _ in spent):
            continue
        spent.append((judge, usage))
    if not spent:
        return
    lines = ["", "judge usage"]
    for judge, usage in spent:
        prefix = f"{judge.name}  " if len(spent) > 1 else ""
        lines.append(f"  {prefix}{usage.summary()}")
    print("\n".join(lines))


def _build_retriever(chunks, vectors_path):
    if not vectors_path:
        return BM25Retriever(chunks)
    chunk_vectors, query_vectors, model = load_vectors(vectors_path)
    return VectorRetriever(chunks, chunk_vectors, query_vectors, model=model)


if __name__ == "__main__":
    raise SystemExit(main())
