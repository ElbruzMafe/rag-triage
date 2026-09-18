import pytest

from ragtriage.judge import LexicalJudge
from ragtriage.models import Assessment, Chunk, EvalCase, Verdict
from ragtriage.retriever import BM25Retriever
from ragtriage.triage import TriageConfig, summarize, triage_all, triage_case

CHUNKS = [
    Chunk("billing.md#0", "billing.md", 0, "Refunds",
          "Refunds are returned to the original payment card within 5 business days."),
    Chunk("api.md#0", "api.md", 0, "Rate limits",
          "The API allows 600 requests per minute per token."),
    Chunk("ops.md#0", "ops.md", 0, "Backups",
          "Database backups run daily at 02:00 UTC and are retained for 30 days."),
]


@pytest.fixture
def retriever():
    return BM25Retriever(CHUNKS)


@pytest.fixture
def judge():
    return LexicalJudge()


def run(case, retriever, judge, **kwargs):
    return triage_case(case, retriever, judge, TriageConfig(**kwargs))


def test_correct_answer_is_ok(retriever, judge):
    case = EvalCase(
        id="c",
        question="what is the API rate limit",
        gold="The API allows 600 requests per minute per token.",
        answer="Each token gets 600 requests per minute.",
    )
    assert run(case, retriever, judge).verdict is Verdict.OK


def test_evidence_outside_the_corpus(retriever, judge):
    case = EvalCase(
        id="c",
        question="does it support SCIM provisioning",
        gold="Users are provisioned through SCIM 2.0 with Okta.",
        answer="Yes, SCIM is supported.",
    )
    result = run(case, retriever, judge)
    assert result.verdict is Verdict.MISSING_FROM_CORPUS
    assert result.support == []


def test_evidence_exists_but_was_not_retrieved(retriever, judge):
    case = EvalCase(
        id="c",
        question="how long does a refund take",
        gold="Refunds are returned to the original payment card within 5 business days.",
        answer="About two weeks.",
        retrieved_ids=["api.md#0", "ops.md#0"],
    )
    result = run(case, retriever, judge)
    assert result.verdict is Verdict.RETRIEVAL_MISS
    assert result.best_support_rank == 1
    assert any("bm25 baseline" in note for note in result.notes)


def test_k_too_small_is_a_retrieval_miss(retriever, judge):
    case = EvalCase(
        id="c",
        question="backups retained days",
        gold="Refunds are returned to the original payment card within 5 business days.",
        answer="whatever",
    )
    result = run(case, retriever, judge, k=1)
    assert result.verdict is Verdict.RETRIEVAL_MISS
    assert "k would have to be at least" in result.notes[-1]


def test_wrong_number_with_the_evidence_in_context(retriever, judge):
    case = EvalCase(
        id="c",
        question="how long does a refund take",
        gold="Refunds are returned to the original payment card within 5 business days.",
        answer="Refunds are returned to the original payment card within 14 business days.",
    )
    result = run(case, retriever, judge)
    assert result.verdict is Verdict.GENERATION_MISS
    assert "figures disagree" in result.notes[0]


def test_answer_nothing_in_context_supports(retriever, judge):
    case = EvalCase(
        id="c",
        question="how often do backups run",
        gold="Database backups run daily at 02:00 UTC and are retained for 30 days.",
        answer="Continuous point in time snapshots stream into a bucket you own.",
    )
    assert run(case, retriever, judge).verdict is Verdict.UNGROUNDED


def test_missing_answer_is_reported_not_guessed(retriever, judge):
    case = EvalCase(
        id="c",
        question="how long does a refund take",
        gold="Refunds are returned to the original payment card within 5 business days.",
    )
    result = run(case, retriever, judge)
    assert result.verdict is Verdict.NO_ANSWER
    assert result.support


def test_a_right_answer_the_context_does_not_support_is_flagged(retriever, judge):
    """Right answer, unsupported by the context: the model knew it without the corpus."""

    class MemorisingJudge(LexicalJudge):
        def grounded(self, answer, passages):
            return Assessment(False, 0.0, "nothing in context")

    case = EvalCase(
        id="c",
        question="how long does a refund take",
        gold="Refunds are returned to the original payment card within 5 business days.",
        answer="Refunds reach the original card within 5 business days.",
    )
    result = run(case, retriever, MemorisingJudge())
    assert result.verdict is Verdict.OK
    assert any("from memory" in note for note in result.notes)


def test_unknown_chunk_id_in_the_eval_set(retriever, judge):
    case = EvalCase(id="c", question="q", gold="g", answer="a", retrieved_ids=["nope.md#9"])
    with pytest.raises(ValueError, match="unknown chunk id"):
        run(case, retriever, judge)


def test_support_scan_bounds_the_number_of_judge_calls(retriever):
    calls = []

    class CountingJudge(LexicalJudge):
        def supports(self, claim, passage):
            calls.append(passage)
            return super().supports(claim, passage)

    case = EvalCase(id="c", question="q", gold="backups run daily retained 30 days", answer="a")
    run(case, retriever, CountingJudge(), support_scan=1)
    assert len(calls) == 1


def test_summarize_counts_every_verdict(retriever, judge):
    cases = [
        EvalCase("a", "what is the API rate limit", "The API allows 600 requests per minute per token.",
                 "600 requests per minute per token."),
        EvalCase("b", "does it support SCIM", "Users are provisioned through SCIM 2.0.", "yes"),
    ]
    counts = summarize(triage_all(cases, retriever, judge))
    assert counts["ok"] == 1
    assert counts["missing_from_corpus"] == 1
    assert sum(counts.values()) == 2


def test_ungrounded_names_the_sentence_nothing_supports(retriever, judge):
    case = EvalCase(
        id="c",
        question="how often do backups run",
        gold="Database backups run daily at 02:00 UTC and are retained for 30 days.",
        answer=(
            "Database backups run daily at 02:00 UTC. "
            "A courier delivers an encrypted tape to your office every Friday."
        ),
    )
    result = run(case, retriever, judge)
    assert result.verdict is Verdict.UNGROUNDED
    assert [check.supported for check in result.claims] == [True, False]
    assert result.claims[0].chunk_id == "ops.md#0"
    assert any(note.startswith("unsupported sentence: 'A courier") for note in result.notes)


def test_claim_detail_can_be_turned_off(retriever, judge):
    case = EvalCase(
        id="c",
        question="how often do backups run",
        gold="Database backups run daily at 02:00 UTC and are retained for 30 days.",
        answer="A courier delivers an encrypted tape to your office every Friday.",
    )
    assert run(case, retriever, judge, claim_detail=False).claims == []


def test_a_grounded_answer_costs_no_claim_checks(retriever, judge):
    """Per-sentence checks are only worth their judge calls when something is unsupported."""
    case = EvalCase(
        id="c",
        question="how long does a refund take",
        gold="Refunds are returned to the original payment card within 5 business days.",
        answer="Refunds are returned to the original payment card within 14 business days.",
    )
    result = run(case, retriever, judge)
    assert result.verdict is Verdict.GENERATION_MISS
    assert result.claims == []
