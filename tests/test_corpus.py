import pytest

from ragtriage.corpus import chunk_document, load_corpus, load_evalset

DOC = """preamble text

# Billing

Refunds take 5 business days.

## Invoices

Invoices go out on the first.
"""


def test_headings_become_chunk_boundaries():
    chunks = chunk_document("billing.md", DOC)
    assert [c.heading for c in chunks] == ["", "Billing", "Invoices"]
    assert [c.id for c in chunks] == ["billing.md#0", "billing.md#1", "billing.md#2"]


def test_searchable_includes_the_heading():
    chunk = chunk_document("billing.md", DOC)[1]
    assert chunk.searchable.startswith("Billing\n")


def test_long_paragraph_is_hard_split_with_overlap():
    body = "x" * 500
    chunks = chunk_document("a.md", f"# H\n\n{body}", max_chars=200, overlap=50)
    assert [len(c.text) for c in chunks] == [200, 200, 200, 50]
    assert chunks[1].text[:50] == chunks[0].text[-50:]


def test_paragraphs_are_packed_up_to_max_chars():
    body = "\n\n".join(["p" * 80] * 5)
    chunks = chunk_document("a.md", f"# H\n\n{body}", max_chars=200)
    assert len(chunks) == 3
    assert all(len(c.text) <= 200 for c in chunks)


def test_blank_sections_produce_no_chunks():
    assert chunk_document("a.md", "# One\n\n\n# Two\n\ntext") == chunk_document(
        "a.md", "# One\n# Two\n\ntext"
    )
    assert len(chunk_document("a.md", "# One\n\n\n# Two\n\ntext")) == 1


def test_load_corpus_reads_nested_directories(tmp_path):
    (tmp_path / "guides").mkdir()
    (tmp_path / "guides" / "a.md").write_text("# A\n\nalpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("bravo", encoding="utf-8")
    (tmp_path / "ignore.json").write_text("{}", encoding="utf-8")

    chunks = load_corpus(tmp_path)
    assert [c.doc_id for c in chunks] == ["b.txt", "guides/a.md"]


def test_load_corpus_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nope")


def _write(tmp_path, text):
    path = tmp_path / "eval.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_evalset_accepts_a_bare_list(tmp_path):
    path = _write(tmp_path, "- id: a\n  question: q\n  gold: g\n")
    assert load_evalset(path)[0].id == "a"


def test_load_evalset_defaults(tmp_path):
    path = _write(tmp_path, "cases:\n  - id: a\n    question: q\n    gold: g\n")
    case = load_evalset(path)[0]
    assert case.answer is None and case.retrieved_ids is None and case.tags == []


def test_load_evalset_rejects_missing_field(tmp_path):
    path = _write(tmp_path, "cases:\n  - id: a\n    question: q\n")
    with pytest.raises(ValueError, match="'gold' is required"):
        load_evalset(path)


def test_load_evalset_rejects_duplicate_ids(tmp_path):
    path = _write(
        tmp_path, "cases:\n  - id: a\n    question: q\n    gold: g\n  - id: a\n    question: q2\n    gold: g2\n"
    )
    with pytest.raises(ValueError, match="duplicate case id"):
        load_evalset(path)


def test_load_evalset_rejects_empty_file(tmp_path):
    with pytest.raises(ValueError, match="non-empty list"):
        load_evalset(_write(tmp_path, "cases: []\n"))
