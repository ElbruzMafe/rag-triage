"""Core data types shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of the corpus."""

    id: str
    doc_id: str
    index: int
    heading: str
    text: str

    @property
    def searchable(self) -> str:
        return f"{self.heading}\n{self.text}" if self.heading else self.text


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int


@dataclass
class EvalCase:
    """One question from the eval set, plus whatever the system under test produced."""

    id: str
    question: str
    gold: str
    answer: str | None = None
    retrieved_ids: list[str] | None = None
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Assessment:
    """A judge's yes/no call, with the reasoning kept for the report."""

    value: bool
    confidence: float
    reason: str


@dataclass(frozen=True)
class ClaimCheck:
    """One sentence of an answer, and the retrieved chunk that supports it."""

    text: str
    supported: bool
    confidence: float
    chunk_id: str | None
    reason: str


class Verdict(str, Enum):
    OK = "ok"
    MISSING_FROM_CORPUS = "missing_from_corpus"
    RETRIEVAL_MISS = "retrieval_miss"
    GENERATION_MISS = "generation_miss"
    UNGROUNDED = "ungrounded"
    NO_ANSWER = "no_answer"


VERDICT_FIX = {
    Verdict.OK: "nothing to fix",
    Verdict.MISSING_FROM_CORPUS: "the evidence is not in the corpus - fix ingestion, not the prompt",
    Verdict.RETRIEVAL_MISS: "evidence exists but never reached the model - fix chunking, embeddings or k",
    Verdict.GENERATION_MISS: "the model had the evidence and still got it wrong - fix the prompt or the model",
    Verdict.UNGROUNDED: "the answer states things no retrieved chunk supports - tighten the prompt or add a citation check",
    Verdict.NO_ANSWER: "no answer recorded for this case - run the system under test first",
}


@dataclass
class CaseResult:
    case: EvalCase
    verdict: Verdict
    support: list[Hit]
    retrieved: list[Hit]
    best_support_rank: int | None
    notes: list[str] = field(default_factory=list)
    claims: list[ClaimCheck] = field(default_factory=list)

    @property
    def fix_hint(self) -> str:
        return VERDICT_FIX[self.verdict]
