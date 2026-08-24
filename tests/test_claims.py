import pytest

from ragtriage.claims import check_claims, split_claims, unsupported
from ragtriage.judge import LexicalJudge
from ragtriage.models import Assessment, Chunk, Hit

CHUNKS = [
    Chunk("ops.md#0", "ops.md", 0, "Backups",
          "Database backups run daily at 02:00 UTC and are retained for 30 days."),
    Chunk("api.md#0", "api.md", 0, "Rate limits",
          "The API allows 600 requests per minute per token."),
]
RETRIEVED = [Hit(chunk=chunk, score=1.0, rank=i) for i, chunk in enumerate(CHUNKS, start=1)]


class CountingJudge:
    """Says yes to whatever it is told to, and counts how often it is asked."""

    name = "counting"

    def __init__(self, supported_claims=()):
        self.supported_claims = set(supported_claims)
        self.calls = 0

    def supports(self, claim, passage):
        self.calls += 1
        yes = claim in self.supported_claims
        return Assessment(yes, 1.0 if yes else 0.0, "fake")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("One thing. Another thing.", ["One thing.", "Another thing."]),
        ("Is it up? Yes! It is.", ["Is it up?", "Yes!", "It is."]),
        ("Single sentence with no full stop", ["Single sentence with no full stop"]),
        ("   ", []),
        ("", []),
    ],
)
def test_split_claims_basic(text, expected):
    assert split_claims(text) == expected


def test_split_claims_keeps_abbreviations_intact():
    text = "Use a cursor, e.g. next_cursor. Offset paging is not supported."
    assert split_claims(text) == [
        "Use a cursor, e.g. next_cursor.",
        "Offset paging is not supported.",
    ]


def test_split_claims_does_not_break_decimals_or_versions():
    text = "The window is 5.5 days. SCIM 2.0 is supported."
    assert split_claims(text) == ["The window is 5.5 days.", "SCIM 2.0 is supported."]


def test_split_claims_keeps_the_closing_quote_with_its_sentence():
    assert split_claims('He said "no." Then he left.') == ['He said "no."', "Then he left."]


def test_supported_sentence_names_the_chunk_that_backs_it():
    answer = "Backups run daily at 02:00 UTC and are retained for 30 days."
    checks = check_claims(answer, RETRIEVED, LexicalJudge())
    assert len(checks) == 1
    assert checks[0].supported
    assert checks[0].chunk_id == "ops.md#0"


def test_invented_sentence_is_reported_unsupported():
    answer = (
        "Backups run daily at 02:00 UTC and are retained for 30 days. "
        "An engineer telephones every affected customer beforehand."
    )
    checks = check_claims(answer, RETRIEVED, LexicalJudge())
    assert [check.supported for check in checks] == [True, False]

    loose = unsupported(checks)
    assert len(loose) == 1
    assert loose[0].chunk_id is None
    assert loose[0].text.startswith("An engineer telephones")


def test_no_retrieved_context_means_nothing_is_supported():
    checks = check_claims("Anything at all. Really anything.", [], LexicalJudge())
    assert [check.supported for check in checks] == [False, False]
    assert {check.reason for check in checks} == {"no retrieved passages"}


def test_empty_answer_produces_no_checks():
    assert check_claims("", RETRIEVED, LexicalJudge()) == []
    assert check_claims(None, RETRIEVED, LexicalJudge()) == []


def test_the_same_claim_and_chunk_are_judged_once():
    judge = CountingJudge()
    check_claims("A thing. A thing.", RETRIEVED, judge)
    assert judge.calls == len(RETRIEVED)


def test_the_highest_confidence_supporting_chunk_wins():
    class Graded:
        name = "graded"

        def supports(self, claim, passage):
            score = 0.9 if "backups" in passage.lower() else 0.7
            return Assessment(True, score, "fake")

    checks = check_claims("Anything.", RETRIEVED, Graded())
    assert checks[0].chunk_id == "ops.md#0"
    assert checks[0].confidence == 0.9
