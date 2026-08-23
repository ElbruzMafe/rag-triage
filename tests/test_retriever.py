from ragtriage.models import Chunk
from ragtriage.retriever import BM25Retriever, tokenize


def chunk(cid, text, heading=""):
    return Chunk(id=cid, doc_id=cid.split("#")[0], index=0, heading=heading, text=text)


CHUNKS = [
    chunk("billing.md#0", "Refunds land on the original card within 5 business days.", "Refunds"),
    chunk("api.md#0", "The API allows 600 requests per minute per token.", "Rate limits"),
    chunk("ops.md#0", "Backups run daily and are retained for 30 days.", "Backups"),
]


def test_tokenize_drops_stopwords_and_short_tokens():
    assert tokenize("How long is a refund?") == ["long", "refund"]


def test_tokenize_stems_word_endings():
    assert tokenize("refunds retained invoices") == tokenize("refund retain invoice")


def test_tokenize_keeps_digits():
    assert "5" in tokenize("within 5 days")


def test_search_ranks_the_relevant_chunk_first():
    hits = BM25Retriever(CHUNKS).search("how long do refunds take", k=2)
    assert hits[0].chunk.id == "billing.md#0"
    assert hits[0].rank == 1


def test_search_respects_k():
    assert len(BM25Retriever(CHUNKS).search("days", k=1)) == 1


def test_zero_scoring_chunks_are_excluded():
    hits = BM25Retriever(CHUNKS).score_all("refunds")
    assert [h.chunk.id for h in hits] == ["billing.md#0"]


def test_ties_break_on_chunk_id():
    same = [chunk("b#0", "identical text here"), chunk("a#0", "identical text here")]
    assert [h.chunk.id for h in BM25Retriever(same).search("identical text")] == ["a#0", "b#0"]


def test_empty_index_returns_nothing():
    retriever = BM25Retriever([])
    assert retriever.size == 0
    assert retriever.search("anything") == []


def test_headings_are_searchable():
    hits = BM25Retriever(CHUNKS).search("rate limits")
    assert hits[0].chunk.id == "api.md#0"
