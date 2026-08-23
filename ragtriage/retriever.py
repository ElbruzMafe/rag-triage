"""A dependency-free Okapi BM25 index over corpus chunks."""

from __future__ import annotations

import math
import re
from collections import Counter

from .models import Chunk, Hit

STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for from by
    with without as is are was were be been being do does did doing have has had having
    it its it's i you he she they we me him her them my your our their there here
    how what when where which who whom why can could should would will shall may might
    not no nor so such only own same too very just about into over under again
    """.split()
)

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, drop stopwords, then stem."""
    return [
        stem(t)
        for t in _WORD.findall(text.lower())
        if (len(t) > 1 or t.isdigit()) and t not in STOPWORDS
    ]


def stem(word: str) -> str:
    """A deliberately small suffix stripper.

    Without it "refunds" in a document never matches "refund" in a question, and both
    retrieval and the lexical judge report failures that are really just word endings.
    A full Porter stemmer is more accurate but not worth the dependency here.
    """
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) > 3 and not word.endswith(("ss", "us", "is")) and word.endswith("s"):
        word = word[:-1]

    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    return word


class BM25Retriever:
    """Classic BM25 ranking. Small corpora only - everything stays in memory."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize(chunk.searchable) for chunk in self.chunks]
        self._freqs = [Counter(tokens) for tokens in self._tokens]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0

        doc_freq: Counter[str] = Counter()
        for tokens in self._tokens:
            doc_freq.update(set(tokens))
        n = len(self.chunks)
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }

    @property
    def size(self) -> int:
        return len(self.chunks)

    def score_all(self, query: str) -> list[Hit]:
        """Every chunk with a non-zero score, best first."""
        terms = tokenize(query)
        scored = []
        for i, chunk in enumerate(self.chunks):
            score = self._score(terms, i)
            if score > 0:
                scored.append((score, chunk))

        # id as the tiebreaker keeps output stable across runs
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return [
            Hit(chunk=chunk, score=round(score, 4), rank=rank)
            for rank, (score, chunk) in enumerate(scored, start=1)
        ]

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return self.score_all(query)[:k]

    def _score(self, terms: list[str], doc: int) -> float:
        if not self._avg_len:
            return 0.0
        freqs = self._freqs[doc]
        length = self._lengths[doc]
        norm = self.k1 * (1 - self.b + self.b * length / self._avg_len)
        total = 0.0
        for term in terms:
            tf = freqs.get(term)
            if not tf:
                continue
            total += self._idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return total
