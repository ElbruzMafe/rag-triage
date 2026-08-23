import json
from pathlib import Path

from ragtriage.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
ARGS = ["--corpus", str(EXAMPLES / "corpus"), "--evalset", str(EXAMPLES / "evalset.yaml")]


def test_runs_the_bundled_example(capsys):
    assert main(ARGS) == 0
    out = capsys.readouterr().out
    assert "rag-triage  8 cases" in out
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
    assert sum(data["summary"].values()) == 8
    assert "# RAG triage report" in md_path.read_text()


def test_missing_corpus_exits_with_two(tmp_path, capsys):
    code = main(["--corpus", str(tmp_path / "nope"), "--evalset", str(EXAMPLES / "evalset.yaml")])
    assert code == 2
    assert "rag-triage:" in capsys.readouterr().err


def test_empty_corpus_directory_exits_with_two(tmp_path, capsys):
    code = main(["--corpus", str(tmp_path), "--evalset", str(EXAMPLES / "evalset.yaml")])
    assert code == 2
    assert "no .md or .txt documents" in capsys.readouterr().err
