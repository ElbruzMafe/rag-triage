"""Side-by-side comparison of two triage arms across the same eval set.

The axis is either the retriever (one judge, two rankers) or the judge (one
ranker, two graders). Both produce the same paired-by-case result; what differs
is which half of the pipeline is allowed to move.
"""

from __future__ import annotations

from collections import Counter
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
class Arm:
    """One side of a comparison: a retriever and the judge grading it."""

    label: str
    retriever: Retriever
    judge: object


@dataclass(frozen=True)
class CaseComparison:
    case_id: str
    a: CaseResult
    b: CaseResult
    evidence_shared: bool = True

    @property
    def verdict_changed(self) -> bool:
        return self.a.verdict is not self.b.verdict

    @property
    def rank_delta(self) -> int | None:
        if self.a.best_support_rank is None or self.b.best_support_rank is None:
            return None
        return self.b.best_support_rank - self.a.best_support_rank


@dataclass(frozen=True)
class ClaimDivergence:
    """One sentence of the answer the two arms graded differently.

    `None` means that arm never ran a sentence check at all: the per-sentence pass
    only runs on an answer the arm has already called ungrounded, so an arm that was
    happy with the whole answer has nothing to say about its individual sentences.
    """

    text: str
    a_supported: bool | None
    b_supported: bool | None


def claim_divergences(comparison: CaseComparison) -> list[ClaimDivergence]:
    """Sentences where the two arms disagree, which is where a judge split shows up.

    A sentence only one arm graded is reported when that arm called it unsupported.
    The other direction - one arm is happy with a sentence and the other never looked
    at it - is agreement expressed two different ways, and listing it would bury the
    real disagreements under every sentence of every answer.
    """
    a_claims = {check.text: check.supported for check in comparison.a.claims}
    b_claims = {check.text: check.supported for check in comparison.b.claims}
    ordered = list(a_claims) + [text for text in b_claims if text not in a_claims]

    divergences = []
    for text in ordered:
        a = a_claims.get(text)
        b = b_claims.get(text)
        if a == b:
            continue
        if (a is None and b is not False) or (b is None and a is not False):
            continue
        divergences.append(ClaimDivergence(text=text, a_supported=a, b_supported=b))
    return divergences


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
    masked: list[CaseComparison]
    false_alarms: list[CaseComparison]
    shifts: list[tuple[str, str, int]]

    @property
    def agreement(self) -> float:
        """Share of cases the two arms give the same verdict, 0.0 when there are none."""
        total = self.changed + self.unchanged
        if total == 0:
            return 0.0
        return self.unchanged / total


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
    masked = [
        c for c in comparisons if c.a.verdict is Verdict.OK and c.b.verdict is not Verdict.OK
    ]
    false_alarms = [
        c for c in comparisons if c.b.verdict is Verdict.OK and c.a.verdict is not Verdict.OK
    ]
    shift_counts = Counter(
        (c.a.verdict.value, c.b.verdict.value) for c in comparisons if c.verdict_changed
    )
    shifts = [
        (pair[0], pair[1], count)
        for pair, count in sorted(shift_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return ComparisonSummary(
        changed=changed,
        unchanged=unchanged,
        a_retrieved=a_retrieved,
        b_retrieved=b_retrieved,
        gains=gains,
        regressions=regressions,
        rank_gains=rank_gains,
        rank_regressions=rank_regressions,
        masked=masked,
        false_alarms=false_alarms,
        shifts=shifts,
    )


def compare(
    cases: list[EvalCase],
    arm_a: Arm,
    arm_b: Arm,
    config: TriageConfig | None = None,
) -> list[CaseComparison]:
    """Run the same eval set through two arms and pair the results up by case.

    Evidence location is shared between the arms only when they use the same judge.
    Whether the gold answer is backed by the corpus at all is a fact about the corpus,
    so it must not move just because the retriever under test moved - otherwise a weak
    embedding reports an ingestion bug instead of its own recall problem. Locating it
    once also halves the judge calls a retriever comparison costs.

    When the judges differ that rule inverts, which is why it is a rule and not a
    constant: see the branch below.
    """
    config = config or TriageConfig()
    shared = arm_a.judge is arm_b.judge
    comparisons = []
    for case in cases:
        if shared:
            evidence = find_support(arm_a.retriever, arm_a.judge, case.gold, config)
        else:
            # With the judge as the axis, "does this chunk back the gold answer" is
            # the very question being compared. Freezing it with one judge would hide
            # the disagreement, so each arm locates its own evidence.
            evidence = None
        comparisons.append(
            CaseComparison(
                case_id=case.id,
                a=triage_case(case, arm_a.retriever, arm_a.judge, config, evidence),
                b=triage_case(case, arm_b.retriever, arm_b.judge, config, evidence),
                evidence_shared=shared,
            )
        )
    return comparisons

