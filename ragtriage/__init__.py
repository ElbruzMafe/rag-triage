"""rag-triage: root-cause classification for retrieval-augmented generation failures."""

from .claims import check_claims, split_claims, unsupported
from .corpus import chunk_document, load_corpus, load_evalset
from .judge import Judge, LexicalJudge
from .models import Assessment, CaseResult, ClaimCheck, Chunk, EvalCase, Hit, Verdict
from .retriever import BM25Retriever, Retriever
from .triage import TriageConfig, triage_all, triage_case
from .vectors import VectorRetriever, embed_inputs, load_vectors

__version__ = "0.1.0"

__all__ = [
    "Assessment",
    "BM25Retriever",
    "CaseResult",
    "ClaimCheck",
    "Chunk",
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
    "embed_inputs",
    "load_corpus",
    "load_evalset",
    "load_vectors",
    "split_claims",
    "triage_all",
    "triage_case",
    "unsupported",
]
