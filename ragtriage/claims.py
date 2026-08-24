"""Splitting an answer into claims and locating the chunk that backs each one."""

from __future__ import annotations

import re

from .models import Assessment, ClaimCheck, Hit

# Periods that end one of these never end a sentence. Single letters are here too,
# so "J. Smith" and "U. S." stay in one piece.
ABBREVIATIONS = frozenset(
    "approx cf dept dr e.g eg etc fig i.e ie inc jr ltd mr mrs ms prof sr st vs".split()
)

_BOUNDARY = re.compile(r'([.!?]["\')\]]*)(\s+)')
_LAST_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)$")


def split_claims(text: str) -> list[str]:
    """Split English prose into sentences.

    Deliberately simple: sentence-final punctuation followed by whitespace, minus the
    abbreviations that would otherwise cut a sentence in half. Decimals need no special
    case - "5.5" has no whitespace after the period, so it never looks like a boundary.
    """
    text = (text or "").strip()
    if not text:
        return []

    claims = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        if _false_boundary(text, match):
            continue
        piece = text[start : match.end(1)].strip()
        if piece:
            claims.append(piece)
        start = match.end(2)

    tail = text[start:].strip()
    if tail:
        claims.append(tail)
    return claims


def check_claims(answer: str, retrieved: list[Hit], judge) -> list[ClaimCheck]:
    """Ask the judge which retrieved chunk supports each sentence of the answer.

    An answer usually fails on one sentence, not all of them. Reporting that sentence
    turns "ungrounded" into something a person can act on.
    """
    claims = split_claims(answer)
    if not claims:
        return []

    cache: dict[tuple[str, str], Assessment] = {}
    checks = []
    for claim in claims:
        best = None
        closest = None
        for hit in retrieved:
            key = (claim, hit.chunk.id)
            if key not in cache:
                cache[key] = judge.supports(claim, hit.chunk.searchable)
            assessment = cache[key]

            if assessment.value and (best is None or assessment.confidence > best[0].confidence):
                best = (assessment, hit.chunk.id)
            if closest is None or assessment.confidence > closest.confidence:
                closest = assessment

        if best is not None:
            assessment, chunk_id = best
            checks.append(
                ClaimCheck(claim, True, assessment.confidence, chunk_id, assessment.reason)
            )
        elif closest is not None:
            checks.append(ClaimCheck(claim, False, closest.confidence, None, closest.reason))
        else:
            checks.append(ClaimCheck(claim, False, 0.0, None, "no retrieved passages"))
    return checks


def unsupported(checks: list[ClaimCheck]) -> list[ClaimCheck]:
    return [check for check in checks if not check.supported]


def _false_boundary(text: str, match: re.Match) -> bool:
    if match.group(1)[0] != ".":
        return False

    word = _LAST_WORD.search(text[: match.start(1)])
    if word and word.group(1).rstrip(".").lower() in ABBREVIATIONS:
        return True
    if word and len(word.group(1).rstrip(".")) == 1:
        return True

    nxt = text[match.end(2) : match.end(2) + 1]
    return not (nxt.isupper() or nxt.isdigit() or nxt in "\"'([")
