"""The decision tree that turns a wrong answer into a root cause."""

from __future__ import annotations

from dataclasses import dataclass

from .claims import check_claims, unsupported
from .models import CaseResult, ClaimCheck, EvalCase, Hit, Verdict
from .retriever import Retriever


@dataclass
class TriageConfig:
    k: int = 5
    support_scan: int = 25
    max_support: int = 3
    claim_detail: bool = True


def _rank_map(hits: list[Hit]) -> dict[str, int]:
    return {hit.chunk.id: hit.rank for hit in hits}


def find_support(
    retriever: Retriever,
    judge,
    gold: str,
    config: TriageConfig,
) -> list[Hit]:
    """Chunks anywhere in the corpus that back up the gold answer.

    Judging every chunk would be correct but costs one judge call per chunk, which
    is unaffordable once the judge is an LLM. BM25 over the gold answer is used as a
    cheap recall filter first; only the top `support_scan` candidates get judged.
    """
    candidates = retriever.score_all(gold)[: config.support_scan]
    support = []
    for hit in candidates:
        if judge.supports(gold, hit.chunk.searchable).value:
            support.append(hit)
        if len(support) >= config.max_support:
            break
    return support


def triage_case(
    case: EvalCase,
    retriever: Retriever,
    judge,
    config: TriageConfig | None = None,
) -> CaseResult:
    config = config or TriageConfig()
    notes: list[str] = []

    question_ranking = retriever.score_all(case.question)
    ranks = _rank_map(question_ranking)

    recorded = case.retrieved_ids is not None
    if recorded:
        retrieved = _hits_for_ids(case.retrieved_ids, retriever)
        notes.append("scored against the chunk ids recorded in the eval set, not a fresh search")
    else:
        retrieved = question_ranking[: config.k]

    support = find_support(retriever, judge, case.gold, config)
    support_ids = {hit.chunk.id for hit in support}
    best_rank = min(
        (ranks[cid] for cid in support_ids if cid in ranks),
        default=None,
    )

    if not support:
        return CaseResult(case, Verdict.MISSING_FROM_CORPUS, support, retrieved, best_rank, notes)

    retrieved_ids = {hit.chunk.id for hit in retrieved}
    if not (support_ids & retrieved_ids):
        notes.append(_rank_note(best_rank, len(retrieved), recorded, retriever.name))
        return CaseResult(case, Verdict.RETRIEVAL_MISS, support, retrieved, best_rank, notes)

    if not (case.answer or "").strip():
        return CaseResult(case, Verdict.NO_ANSWER, support, retrieved, best_rank, notes)

    passages = [hit.chunk.searchable for hit in retrieved]
    match = judge.equivalent(case.gold, case.answer)
    grounded = judge.grounded(case.answer, passages)

    # Per-claim checks cost one judge call per sentence per chunk, so they only run
    # when the answer is not fully grounded - which is exactly when they are read.
    claims = []
    if not grounded.value and config.claim_detail:
        claims = check_claims(case.answer, retrieved, judge)

    if match.value:
        if not grounded.value:
            notes.append(
                "answer matches the gold answer but no retrieved chunk supports it - "
                "the model may be answering from memory rather than from context"
            )
            notes.extend(_claim_notes(claims))
        return CaseResult(case, Verdict.OK, support, retrieved, best_rank, notes, claims)

    notes.append(f"answer mismatch: {match.reason}")
    if not grounded.value:
        notes.append(f"groundedness: {grounded.reason}")
        notes.extend(_claim_notes(claims))
        return CaseResult(case, Verdict.UNGROUNDED, support, retrieved, best_rank, notes, claims)

    return CaseResult(case, Verdict.GENERATION_MISS, support, retrieved, best_rank, notes)


def triage_all(
    cases: list[EvalCase],
    retriever: Retriever,
    judge,
    config: TriageConfig | None = None,
) -> list[CaseResult]:
    return [triage_case(case, retriever, judge, config) for case in cases]


def summarize(results: list[CaseResult]) -> dict[str, int]:
    counts = {verdict.value: 0 for verdict in Verdict}
    for result in results:
        counts[result.verdict.value] += 1
    return counts


def _claim_notes(claims: list[ClaimCheck]) -> list[str]:
    """Name the sentences that nothing supports, rather than just saying "ungrounded"."""
    loose = unsupported(claims)
    if not loose or len(loose) == len(claims):
        return []
    return [f"unsupported sentence: {check.text!r}" for check in loose]


def _rank_note(best_rank: int | None, k: int, recorded: bool, backend: str = "bm25") -> str:
    if best_rank is None:
        return (
            "no supporting chunk scores at all for this question - the wording of the "
            "question and the wording of the document have nothing in common"
        )
    if recorded:
        return (
            f"the retriever under test missed it, but the {backend} baseline puts the "
            f"supporting chunk at #{best_rank} for this question"
        )
    return f"the supporting chunk ranks #{best_rank} for this question; k would have to be at least {best_rank}, it is {k}"


def _hits_for_ids(ids: list[str], retriever: Retriever) -> list[Hit]:
    by_id = {chunk.id: chunk for chunk in retriever.chunks}
    hits = []
    for position, chunk_id in enumerate(ids, start=1):
        chunk = by_id.get(chunk_id)
        if chunk is None:
            raise ValueError(f"eval set refers to unknown chunk id {chunk_id!r}")
        hits.append(Hit(chunk=chunk, score=0.0, rank=position))
    return hits
