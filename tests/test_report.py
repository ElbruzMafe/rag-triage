from html.parser import HTMLParser

import pytest

from ragtriage.models import CaseResult, ClaimCheck, Chunk, EvalCase, Hit, Verdict
from ragtriage.report import render_case, render_html, render_markdown, render_text, to_dict

CHUNK = Chunk("ops.md#2", "ops.md", 2, "Status & incidents",
              "Incidents are published on the status page within 15 minutes of detection.")
HIT = Hit(chunk=CHUNK, score=6.77, rank=1)


def result(verdict=Verdict.UNGROUNDED, claims=(), notes=(), support=(HIT,)):
    case = EvalCase(
        id="incident-sla",
        question="How quickly are incidents published?",
        gold="Incidents are published within 15 minutes.",
        answer="Incidents are published within 60 minutes. An engineer calls you first.",
        tags=["operations"],
    )
    return CaseResult(case, verdict, list(support), [HIT], 1, list(notes), list(claims))


CLAIMS = [
    ClaimCheck("Incidents are published within 60 minutes.", True, 0.87, "ops.md#2", "fake"),
    ClaimCheck("An engineer calls you first.", False, 0.11, None, "fake"),
]


class Wellformed(HTMLParser):
    """Enough of a check to catch an unbalanced tag in the generated report."""

    VOID = {"meta", "br", "link", "img", "hr", "input"}

    def __init__(self):
        super().__init__()
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.errors.append(tag)


def test_text_report_lists_every_case_and_the_fix_order():
    text = render_text([result(), result(Verdict.OK)], chunks=16, judge="lexical", k=5)
    assert "rag-triage  2 cases  |  16 chunks  |  judge lexical  |  k=5" in text
    assert text.index("ungrounded") < text.index("what to fix first")


def test_explain_shows_the_evidence_and_the_unsupported_sentence():
    text = render_case(result(claims=CLAIMS, notes=["answer mismatch: figures disagree"]))
    assert "verdict  ungrounded  (stage: generation)" in text
    assert "evidence for the gold answer" in text
    assert "<- supporting" in text
    assert "!!  [unsupported] An engineer calls you first." in text
    assert "ok  [ops.md#2] Incidents are published within 60 minutes." in text


def test_explain_says_so_when_the_corpus_has_no_evidence():
    text = render_case(result(Verdict.MISSING_FROM_CORPUS, support=()))
    assert "none found in the corpus" in text
    assert "(stage: corpus)" in text


def test_markdown_calls_out_unsupported_sentences():
    md = render_markdown([result(claims=CLAIMS)])
    assert "**Sentences no retrieved chunk supports:**" in md
    assert "- An engineer calls you first." in md
    assert "Incidents are published within 60 minutes." in md.split("supports:**")[0]


def test_html_is_wellformed_and_escapes_the_corpus():
    html = render_html([result(claims=CLAIMS)], chunks=16, judge="lexical", k=5)
    parser = Wellformed()
    parser.feed(html)
    assert parser.errors == []
    assert parser.stack == []

    # the heading contains an ampersand; a raw one would break the document
    assert "Status &amp; incidents" in html
    assert "flex:1 1 120px" in html  # the CSS survived str.format
    assert 'class="chip ungrounded"' in html


def test_json_carries_the_stage_and_the_claims():
    data = to_dict([result(claims=CLAIMS)])
    case = data["cases"][0]
    assert case["stage"] == "generation"
    assert case["claims"][1] == {
        "text": "An engineer calls you first.",
        "supported": False,
        "confidence": 0.11,
        "chunk_id": None,
    }


@pytest.mark.parametrize("verdict", list(Verdict))
def test_every_verdict_renders_in_every_format(verdict):
    one = [result(verdict)]
    assert render_text(one, chunks=1, judge="lexical", k=5)
    assert render_case(one[0])
    assert render_markdown(one)
    assert render_html(one, chunks=1, judge="lexical", k=5)
    assert to_dict(one)["cases"][0]["verdict"] == verdict.value
