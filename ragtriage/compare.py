"""Side-by-side comparison of two retrievers across the same eval set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .models import CaseResult, EvalCase, Verdict
from .triage import TriageConfig, find_support, triage_case

if TYPE_CHECKING:
    from .retriever import Retriever

RETRIEVED_EVIDENCE_FAILS = frozenset({Verdict.MISSING_FROM_CORPUS, Verdict.RETRIEVAL_MISS})


def retrieved_evidence(result: CaseResult) -> bool:
    """Whether evidence exists in the corpus and reached the model."""
    return result.verdict not in RETRIEVED_EVIDENCE_FAILS


@dataclass(frozen=True)
class CaseComparison:
    case_id: str
    a: CaseResult
    b: CaseResult

    @property
    def verdict_changed(self) -> bool:
        return self.a.verdict is not self.b.verdict

    @property
    def rank_delta(self) -> int | None:
        if self.a.best_support_rank is None or self.b.best_support_rank is None:
            return None
        return self.b.best_support_rank - self.a.best_support_rank


@dataclass(frozen=True)
class ComparisonSummary:
    changed: int
    unchanged: int
    a_retrieved: int
    b_retrieved: int
    gains: list[CaseComparison]
    regressions: list[CaseComparison]
    rank_gains: list[CaseComparison]
    rank_regressions: list[CaseComparison]


def summarize_comparison(comparisons: list[CaseComparison]) -> ComparisonSummary:
    changed = sum(1 for c in comparisons if c.verdict_changed)
    unchanged = len(comparisons) - changed
    a_retrieved = sum(1 for c in comparisons if retrieved_evidence(c.a))
    b_retrieved = sum(1 for c in comparisons if retrieved_evidence(c.b))
    gains = [c for c in comparisons if not retrieved_evidence(c.a) and retrieved_evidence(c.b)]
    regressions = [
        c for c in comparisons if retrieved_evidence(c.a) and not retrieved_evidence(c.b)
    ]
    rank_gains = sorted(
        [c for c in comparisons if c.rank_delta is not None and c.rank_delta < 0],
        key=lambda c: c.rank_delta,
    )
    rank_regressions = sorted(
        [c for c in comparisons if c.rank_delta is not None and c.rank_delta > 0],
        key=lambda c: -c.rank_delta,
    )
    return ComparisonSummary(
        changed=changed,
        unchanged=unchanged,
        a_retrieved=a_retrieved,
        b_retrieved=b_retrieved,
        gains=gains,
        regressions=regressions,
        rank_gains=rank_gains,
        rank_regressions=rank_regressions,
    )


def compare(
    cases: list[EvalCase],
    retriever_a: Retriever,
    retriever_b: Retriever,
    judge,
    config: TriageConfig | None = None,
) -> list[CaseComparison]:
    """Run the same eval set through two retrievers and pair the results up by case.

    Evidence is located once per case, with retriever A, and handed to both sides.
    Letting each retriever find its own would make `missing_from_corpus` - a statement
    about the corpus - depend on which retriever is under test, so a weak embedding
    would report an ingestion bug instead of its own recall problem. Locating it once
    also halves the judge calls a comparison costs.
    """
    config = config or TriageConfig()
    comparisons = []
    for case in cases:
        evidence = find_support(retriever_a, judge, case.gold, config)
        comparisons.append(
            CaseComparison(
                case_id=case.id,
                a=triage_case(case, retriever_a, judge, config, evidence),
                b=triage_case(case, retriever_b, judge, config, evidence),
            )
        )
    return comparisons
