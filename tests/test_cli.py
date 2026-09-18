import sys
import json
from pathlib import Path

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
