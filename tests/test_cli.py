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
