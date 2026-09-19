"""rag-triage: root-cause classification for retrieval-augmented generation failures."""

from .claims import check_claims, split_claims, unsupported
from .compare import CaseComparison, ComparisonSummary, compare, summarize_comparison
from .corpus import chunk_document, load_corpus, load_evalset
from .judge import Judge, LexicalJudge
from .models import Assessment, CaseResult, Chunk, ClaimCheck, EvalCase, Hit, Verdict
from .retriever import BM25Retriever, Retriever
from .triage import TriageConfig, triage_all, triage_case
from .vectors import VectorRetriever, embed_inputs, load_vectors

__version__ = "0.1.0"

__all__ = [
    "Assessment",
    "BM25Retriever",
    "CaseComparison",
    "CaseResult",
    "Chunk",
    "ClaimCheck",
    "ComparisonSummary",
    "EvalCase",
    "Hit",
    "Judge",
    "LexicalJudge",
    "Retriever",
    "TriageConfig",
    "VectorRetriever",
    "Verdict",
    "check_claims",
    "chunk_document",
    "compare",
    "embed_inputs",
    "load_corpus",
    "load_evalset",
    "load_vectors",
    "split_claims",
    "summarize_comparison",
    "triage_all",
    "triage_case",
    "unsupported",
]
