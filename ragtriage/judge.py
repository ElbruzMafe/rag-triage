"""Judges answer the semantic questions triage needs: is this claim supported,
is this answer the same as the gold answer, is it grounded in the context."""

from __future__ import annotations

import re
from typing import Protocol

from .models import Assessment
from .retriever import tokenize

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


class Judge(Protocol):
    name: str

    def supports(self, claim: str, passage: str) -> Assessment: ...

    def equivalent(self, gold: str, actual: str) -> Assessment: ...

    def grounded(self, answer: str, passages: list[str]) -> Assessment: ...


class LexicalJudge:
    """Token-overlap judge: deterministic, offline, no API key.

    Good enough to triage a corpus whose wording is close to the gold answers,
    which is the usual case for product documentation. Swap in ClaudeJudge when
    the eval set is paraphrased heavily.
    """

    name = "lexical"

    def __init__(self, support_threshold: float = 0.6, match_threshold: float = 0.55):
        self.support_threshold = support_threshold
        self.match_threshold = match_threshold

    def supports(self, claim: str, passage: str) -> Assessment:
        claim_terms = set(tokenize(claim))
        if not claim_terms:
            return Assessment(False, 0.0, "claim is empty")
        if not passage.strip():
            return Assessment(False, 0.0, "passage is empty")

        present = claim_terms & set(tokenize(passage))
        ratio = len(present) / len(claim_terms)
        return Assessment(
            ratio >= self.support_threshold,
            round(ratio, 3),
            f"{len(present)}/{len(claim_terms)} claim terms present in passage",
        )

    def equivalent(self, gold: str, actual: str) -> Assessment:
        gold_terms = set(tokenize(gold))
        actual_terms = set(tokenize(actual or ""))
        if not gold_terms:
            return Assessment(False, 0.0, "gold answer is empty")
        if not actual_terms:
            return Assessment(False, 0.0, "answer is empty")

        overlap = len(gold_terms & actual_terms) / min(len(gold_terms), len(actual_terms))

        # A swapped figure is the most common RAG failure and survives token overlap
        # almost untouched, so numbers get their own hard check.
        gold_numbers = _numbers(gold)
        actual_numbers = _numbers(actual)
        if gold_numbers and actual_numbers and gold_numbers != actual_numbers:
            return Assessment(
                False,
                round(overlap, 3),
                f"figures disagree: gold has {sorted(gold_numbers)}, answer has {sorted(actual_numbers)}",
            )

        return Assessment(
            overlap >= self.match_threshold,
            round(overlap, 3),
            f"{overlap:.0%} term overlap with the gold answer",
        )

    def grounded(self, answer: str, passages: list[str]) -> Assessment:
        if not passages:
            return Assessment(False, 0.0, "no passages retrieved")
        if not (answer or "").strip():
            return Assessment(False, 0.0, "answer is empty")

        result = self.supports(answer, "\n".join(passages))
        return Assessment(
            result.value,
            result.confidence,
            f"{result.reason} (across {len(passages)} retrieved passages)",
        )


def _numbers(text: str) -> set[str]:
    return {match.group().replace(",", ".").rstrip(".") for match in _NUMBER.finditer(text)}
