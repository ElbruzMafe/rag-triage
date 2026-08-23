"""rag-triage: root-cause classification for retrieval-augmented generation failures."""

from .corpus import chunk_document, load_corpus, load_evalset
from .judge import Judge, LexicalJudge
from .models import Assessment, CaseResult, Chunk, EvalCase, Hit, Verdict
from .retriever import BM25Retriever
from .triage import TriageConfig, triage_all, triage_case

__version__ = "0.1.0"

__all__ = [
    "Assessment",
    "BM25Retriever",
    "CaseResult",
    "Chunk",
    "EvalCase",
    "Hit",
    "Judge",
    "LexicalJudge",
    "TriageConfig",
    "Verdict",
    "chunk_document",
    "load_corpus",
    "load_evalset",
    "triage_all",
    "triage_case",
]
