import json
import sys
from pathlib import Path

import pytest

from ragtriage.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
ARGS = ["--corpus", str(EXAMPLES / "corpus"), "--evalset", str(EXAMPLES / "evalset.yaml")]


def test_runs_the_bundled_example(capsys):
    assert main(ARGS) == 0
    out = capsys.readouterr().out
    assert "rag-triage  9 cases" in out
    assert "what to fix first" in out


def test_strict_fails_when_a_case_is_not_ok():
    assert main(ARGS + ["--strict"]) == 1


def test_writes_json_and_markdown(tmp_path, capsys):
    json_path = tmp_path / "out.json"
    md_path = tmp_path / "out.md"
    main(ARGS + ["--json", str(json_path), "--markdown", str(md_path)])
    capsys.readouterr()

    data = json.loads(json_path.read_text())
    verdicts = {case["id"]: case["verdict"] for case in data["cases"]}
    assert verdicts["scim-provisioning"] == "missing_from_corpus"
    assert verdicts["webhook-retry"] == "retrieval_miss"
    assert verdicts["refund-window"] == "generation_miss"
    assert verdicts["rate-limit"] == "ok"
    assert sum(data["summary"].values()) == 9
    assert "# RAG triage report" in md_path.read_text()


def test_explain_prints_one_case_only(capsys):
    assert main(ARGS + ["--explain", "incident-sla"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("case     incident-sla")
    assert "unsupported] Our on-call engineer" in out
    assert "refund-window" not in out


def test_explain_with_an_unknown_case_id_exits_with_two(capsys):
    assert main(ARGS + ["--explain", "nope"]) == 2
    assert "Known ids:" in capsys.readouterr().err


def test_writes_a_self_contained_html_report(tmp_path, capsys):
    html_path = tmp_path / "out.html"
    main(ARGS + ["--html", str(html_path)])
    capsys.readouterr()

    html = html_path.read_text()
    assert html.startswith("<!doctype html>")
    assert "src=" not in html and "http" not in html
    assert html.count("<details") == 9


def test_claim_detail_can_be_switched_off(tmp_path, capsys):
    json_path = tmp_path / "out.json"
    main(ARGS + ["--no-claim-detail", "--json", str(json_path)])
    capsys.readouterr()

    data = json.loads(json_path.read_text())
    assert all(case["claims"] == [] for case in data["cases"])


def test_zero_k_is_rejected(capsys):
    assert main(ARGS + ["-k", "0"]) == 2
    assert "-k must be at least 1" in capsys.readouterr().err


def test_unknown_retrieved_id_is_reported_not_raised(tmp_path, capsys):
    evalset = tmp_path / "eval.yaml"
    evalset.write_text(
        "cases:\n"
        "  - id: c\n"
        "    question: how long does a refund take\n"
        "    gold: Refunds arrive within 5 business days.\n"
        "    retrieved_ids: [nosuch.md#0]\n"
    )
    code = main(["--corpus", str(EXAMPLES / "corpus"), "--evalset", str(evalset)])
    assert code == 2
    assert "unknown chunk id" in capsys.readouterr().err


def test_missing_corpus_exits_with_two(tmp_path, capsys):
    code = main(["--corpus", str(tmp_path / "nope"), "--evalset", str(EXAMPLES / "evalset.yaml")])
    assert code == 2
    assert "rag-triage:" in capsys.readouterr().err


def test_empty_corpus_directory_exits_with_two(tmp_path, capsys):
    code = main(["--corpus", str(tmp_path), "--evalset", str(EXAMPLES / "evalset.yaml")])
    assert code == 2
    assert "no .md or .txt documents" in capsys.readouterr().err


def test_embed_inputs_then_vectors_round_trip(tmp_path, capsys):
    """The documented path: export the texts, embed them, run against the vectors."""
    inputs = tmp_path / "inputs.json"
    assert main(ARGS + ["--embed-inputs", str(inputs)]) == 0
    assert "16 chunk texts" in capsys.readouterr().out

    payload = json.loads(inputs.read_text())
    assert len(payload["chunks"]) == 16
    assert "What is the API rate limit?" in payload["queries"]

    sys.path.insert(0, str(EXAMPLES))
    from hash_vectors import embed

    vectors = tmp_path / "vectors.json"
    vectors.write_text(
        json.dumps(
            {
                "model": "test",
                "chunks": {cid: embed(text) for cid, text in payload["chunks"].items()},
                "queries": {text: embed(text) for text in payload["queries"]},
            }
        )
    )

    assert main(ARGS + ["--vectors", str(vectors)]) == 0
    out = capsys.readouterr().out
    assert "retriever vectors:test" in out
    assert "9 cases" in out


def test_vectors_built_from_a_different_chunking_is_rejected(tmp_path, capsys):
    inputs = tmp_path / "inputs.json"
    main(ARGS + ["--embed-inputs", str(inputs)])
    payload = json.loads(inputs.read_text())
    vectors = tmp_path / "vectors.json"
    vectors.write_text(
        json.dumps({"chunks": {cid: [1.0, 2.0] for cid in payload["chunks"]}})
    )

    assert main(ARGS + ["--vectors", str(vectors), "--max-chars", "300"]) == 2
    assert "different chunking" in capsys.readouterr().err


def test_missing_query_vector_reports_the_text(tmp_path, capsys):
    inputs = tmp_path / "inputs.json"
    main(ARGS + ["--embed-inputs", str(inputs)])
    payload = json.loads(inputs.read_text())
    vectors = tmp_path / "vectors.json"
    vectors.write_text(
        json.dumps({"chunks": {cid: [1.0, 2.0] for cid in payload["chunks"]}})
    )

    assert main(ARGS + ["--vectors", str(vectors)]) == 2
    assert "use --embed-inputs" in capsys.readouterr().err


def test_missing_vectors_file_is_an_error(capsys, tmp_path):
    assert main(ARGS + ["--vectors", str(tmp_path / "nope.json")]) == 2
    assert "vectors file does not exist" in capsys.readouterr().err


@pytest.fixture
def vectors_file(tmp_path, capsys):
    inputs_path = tmp_path / "inputs.json"
    main(ARGS + ["--embed-inputs", str(inputs_path)])
    capsys.readouterr()
    payload = json.loads(inputs_path.read_text())

    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    from hash_vectors import embed

    vectors_path = tmp_path / "vectors.json"
    vectors_path.write_text(
        json.dumps(
            {
                "model": "hash-demo",
                "chunks": {cid: embed(text) for cid, text in payload["chunks"].items()},
                "queries": {text: embed(text) for text in payload["queries"]},
            }
        )
    )
    return vectors_path


def test_compare_without_vectors_exits_with_two(capsys):
    assert main(ARGS + ["--compare"]) == 2
    assert "--vectors" in capsys.readouterr().err


def test_compare_with_explain_exits_with_two(vectors_file, capsys):
    args = ARGS + ["--compare", "--vectors", str(vectors_file), "--explain", "rate-limit"]
    assert main(args) == 2
    assert "--explain" in capsys.readouterr().err


def test_compare_with_markdown_exits_with_two(vectors_file, tmp_path, capsys):
    out = ["--markdown", str(tmp_path / "out.md")]
    args = ARGS + ["--compare", "--vectors", str(vectors_file)] + out
    assert main(args) == 2
    assert "--markdown is a single-run report" in capsys.readouterr().err


def test_compare_writes_html_report(vectors_file, tmp_path):
    html_path = tmp_path / "comparison.html"
    args = ARGS + ["--compare", "--vectors", str(vectors_file), "--html", str(html_path)]
    assert main(args) == 0

    html = html_path.read_text(encoding="utf-8")
    assert "rag-triage retriever comparison" in html
    assert "vectors:hash-demo" in html
    assert "support rank" in html
    assert "<script" not in html
    assert "http://" not in html and "https://" not in html


def test_compare_judge_writes_html_report(tmp_path):
    html_path = tmp_path / "judges.html"
    args = ARGS + ["--compare", "judge", "--judge-b", "lexical@0.4", "--html", str(html_path)]
    assert main(args) == 0

    html = html_path.read_text(encoding="utf-8")
    assert "rag-triage judge comparison" in html
    assert "evidence chunks" in html
    assert "support rank" not in html


def test_compare_prints_side_by_side_table(vectors_file, capsys):
    assert main(ARGS + ["--compare", "--vectors", str(vectors_file)]) == 0
    out = capsys.readouterr().out
    assert "rag-triage retriever comparison" in out
    assert "bm25" in out
    assert "vectors:hash-demo" in out


def test_compare_writes_json_report(vectors_file, tmp_path, capsys):
    json_path = tmp_path / "comparison.json"
    args = ARGS + ["--compare", "--vectors", str(vectors_file), "--json", str(json_path)]
    assert main(args) == 0
    capsys.readouterr()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(data.keys()) == {"axis", "arms", "evidence_shared", "summary", "cases"}
    assert data["axis"] == "retriever"
    assert data["arms"] == {"a": "bm25", "b": "vectors:hash-demo"}
    assert data["evidence_shared"] is True
    assert len(data["cases"]) == 9


def test_compare_strict_passes_when_no_regressions(vectors_file, capsys):
    assert main(ARGS + ["--compare", "--vectors", str(vectors_file), "--strict"]) == 0
    capsys.readouterr()



def test_compare_judge_needs_a_second_judge(capsys):
    assert main(ARGS + ["--compare", "judge"]) == 2
    assert "--judge-b" in capsys.readouterr().err


def test_judge_b_without_the_judge_axis_exits_with_two(capsys):
    assert main(ARGS + ["--judge-b", "lexical@0.8"]) == 2
    assert "only does something with --compare judge" in capsys.readouterr().err


def test_comparing_a_judge_with_itself_exits_with_two(capsys):
    assert main(ARGS + ["--compare", "judge", "--judge-b", "lexical"]) == 2
    assert "measures nothing" in capsys.readouterr().err


def test_a_bad_judge_spec_exits_with_two(capsys):
    assert main(ARGS + ["--judge", "lexical@nope"]) == 2
    assert "not a number in judge spec" in capsys.readouterr().err


def test_compare_judge_prints_the_support_columns(capsys):
    assert main(ARGS + ["--compare", "judge", "--judge-b", "lexical@0.85"]) == 0
    out = capsys.readouterr().out
    assert "rag-triage judge comparison" in out
    assert "retriever bm25" in out
    assert "A support" in out and "B support" in out
    assert "agreement" in out


def test_compare_judge_writes_json_with_per_arm_evidence(tmp_path, capsys):
    json_path = tmp_path / "judges.json"
    args = ARGS + ["--compare", "judge", "--judge-b", "lexical@0.85", "--json", str(json_path)]
    assert main(args) == 0
    capsys.readouterr()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["axis"] == "judge"
    assert data["arms"] == {"a": "lexical", "b": "lexical@0.85"}
    # Each judge decides for itself what backs the gold answer, so the evidence is not shared.
    assert data["evidence_shared"] is False
    assert all("evidence_chunks" in case["a"] for case in data["cases"])


@pytest.fixture
def paraphrased_set(tmp_path):
    """One case whose gold answer is a paraphrase: supported at 0.6, not at 0.9."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "billing.md").write_text(
        "# Refunds\n\nRefunds are returned to the original payment card within "
        "5 business days.\n",
        encoding="utf-8",
    )
    gold = "A refund lands on the original payment card within 5 business days of approval."
    evalset = tmp_path / "eval.yaml"
    evalset.write_text(
        "cases:\n"
        "  - id: refund-window\n"
        "    question: how long does a refund take\n"
        f"    gold: {gold}\n"
        f"    answer: {gold}\n",
        encoding="utf-8",
    )
    return ["--corpus", str(corpus), "--evalset", str(evalset)]


def test_compare_judge_strict_fails_on_a_masked_failure(paraphrased_set, capsys):
    # The stricter judge stops recognising the evidence, so a case A calls ok turns red.
    args = paraphrased_set + ["--compare", "judge", "--judge-b", "lexical@0.9"]
    assert main(args + ["--strict"]) == 1
    out = capsys.readouterr().out
    assert "masked failures" in out
    assert "the cheaper judge is not reporting" in out


def test_compare_judge_strict_passes_when_the_judges_agree(paraphrased_set, capsys):
    args = paraphrased_set + ["--compare", "judge", "--judge-b", "lexical@0.7", "--strict"]
    assert main(args) == 0
    assert "the two judges agree on every case" in capsys.readouterr().out


def test_compare_judge_uses_the_vector_retriever_when_given_one(vectors_file, capsys):
    args = ARGS + ["--compare", "judge", "--judge-b", "lexical@0.85", "--vectors",
                   str(vectors_file)]
    assert main(args) == 0
    assert "retriever vectors:hash-demo" in capsys.readouterr().out
