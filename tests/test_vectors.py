import json

import pytest

from ragtriage.models import Chunk, EvalCase
from ragtriage.vectors import VectorRetriever, embed_inputs, load_vectors


def chunk(cid: str, text: str = "body") -> Chunk:
    doc, index = cid.split("#")
    return Chunk(id=cid, doc_id=doc, index=int(index), heading="H", text=text)


CHUNKS = [chunk("a.md#0"), chunk("b.md#0"), chunk("c.md#0")]
VECTORS = {
    "a.md#0": [1.0, 0.0],
    "b.md#0": [0.6, 0.8],
    "c.md#0": [-1.0, 0.0],
}
QUERIES = {"east": [1.0, 0.0], "north": [0.0, 1.0]}


def build(chunks=CHUNKS, vectors=VECTORS, queries=QUERIES, model=None) -> VectorRetriever:
    return VectorRetriever(chunks, vectors, queries, model=model)


def test_ranks_by_cosine_similarity():
    hits = build().score_all("east")
    assert [hit.chunk.id for hit in hits] == ["a.md#0", "b.md#0"]
    assert hits[0].score == 1.0
    assert hits[1].score == 0.6
    assert [hit.rank for hit in hits] == [1, 2]


def test_non_positive_similarity_is_left_out():
    # a.md#0 is orthogonal to "north" and c.md#0 points away from it; neither is a hit.
    hits = build().score_all("north")
    assert [hit.chunk.id for hit in hits] == ["b.md#0"]


def test_ties_break_on_chunk_id():
    chunks = [chunk("z.md#0"), chunk("a.md#0")]
    vectors = {"z.md#0": [1.0, 1.0], "a.md#0": [2.0, 2.0]}
    hits = VectorRetriever(chunks, vectors, {"q": [1.0, 1.0]}).score_all("q")
    assert [hit.chunk.id for hit in hits] == ["a.md#0", "z.md#0"]


def test_search_truncates_to_k():
    assert len(build().search("east", k=1)) == 1
    assert build().search("east", k=10) == build().score_all("east")


def test_name_reports_the_model():
    assert build().name == "vectors"
    assert build(model="text-embedding-3-small").name == "vectors:text-embedding-3-small"
    assert build().size == 3


def test_chunk_without_a_vector_is_an_error():
    with pytest.raises(ValueError) as exc:
        VectorRetriever(CHUNKS, {"a.md#0": [1.0, 0.0]}, QUERIES)
    assert "2 of 3 chunks have no vector" in str(exc.value)
    assert "--embed-inputs" in str(exc.value)


def test_zero_vector_is_an_error():
    with pytest.raises(ValueError, match="all zeros"):
        VectorRetriever([chunk("a.md#0")], {"a.md#0": [0.0, 0.0]}, QUERIES)


def test_mismatched_chunk_dimensions_are_an_error():
    chunks = [chunk("a.md#0"), chunk("b.md#0")]
    with pytest.raises(ValueError, match="dimensions"):
        VectorRetriever(chunks, {"a.md#0": [1.0, 0.0], "b.md#0": [1.0]}, QUERIES)


def test_unknown_query_text_is_a_lookup_error():
    with pytest.raises(LookupError) as exc:
        build().score_all("a question nobody embedded")
    assert "a question nobody embedded" in str(exc.value)
    assert "--embed-inputs" in str(exc.value)


def test_query_vector_of_the_wrong_size_is_an_error():
    retriever = VectorRetriever(CHUNKS, VECTORS, {"q": [1.0, 0.0, 0.0]})
    with pytest.raises(ValueError, match="3 dimensions"):
        retriever.score_all("q")


def write(tmp_path, payload):
    path = tmp_path / "vectors.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_vectors_reads_a_valid_file(tmp_path):
    path = write(tmp_path, {"model": "m", "chunks": VECTORS, "queries": QUERIES})
    chunks, queries, model = load_vectors(path)
    assert chunks == VECTORS
    assert queries == QUERIES
    assert model == "m"


def test_load_vectors_allows_a_file_without_queries(tmp_path):
    path = write(tmp_path, {"chunks": VECTORS})
    chunks, queries, model = load_vectors(path)
    assert chunks == VECTORS
    assert queries == {}
    assert model is None


def test_load_vectors_rejects_mixed_dimensions(tmp_path):
    path = write(tmp_path, {"chunks": {"a.md#0": [1.0, 0.0], "b.md#0": [1.0, 0.0, 0.0]}})
    with pytest.raises(ValueError, match="different dimensions"):
        load_vectors(path)


def test_load_vectors_rejects_a_query_of_the_wrong_dimension(tmp_path):
    path = write(tmp_path, {"chunks": {"a.md#0": [1.0, 0.0]}, "queries": {"q": [1.0]}})
    with pytest.raises(ValueError, match="different dimensions"):
        load_vectors(path)


def test_load_vectors_rejects_empty_and_non_numeric(tmp_path):
    with pytest.raises(ValueError, match="is empty"):
        load_vectors(write(tmp_path, {"chunks": {}}))
    with pytest.raises(ValueError, match="non-empty list"):
        load_vectors(write(tmp_path, {"chunks": {"a.md#0": []}}))
    with pytest.raises(ValueError, match="non-numeric"):
        load_vectors(write(tmp_path, {"chunks": {"a.md#0": ["x"]}}))
    with pytest.raises(ValueError, match="must be an object"):
        load_vectors(write(tmp_path, {"chunks": [1, 2]}))


def test_load_vectors_rejects_broken_json(tmp_path):
    path = tmp_path / "vectors.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_vectors(path)


def test_embed_inputs_lists_every_chunk_question_and_gold():
    cases = [
        EvalCase(id="one", question="q1", gold="g1"),
        EvalCase(id="two", question="q2", gold="g1"),
    ]
    payload = embed_inputs(CHUNKS, cases)
    assert payload["chunks"]["a.md#0"] == "H\nbody"
    # gold answers are queries too: evidence location searches with the gold answer
    assert payload["queries"] == ["q1", "g1", "q2"]
