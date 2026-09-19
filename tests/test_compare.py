import json

import pytest

from ragtriage.compare import (
    Arm,
    CaseComparison,
    compare,
    retrieved_evidence,
    summarize_comparison,
)
from ragtriage.judge import LexicalJudge, build_judge
from ragtriage.models import CaseResult, Chunk, EvalCase, Hit, Verdict
from ragtriage.report import comparison_to_dict, render_comparison
from ragtriage.retriever import BM25Retriever
from ragtriage.triage import TriageConfig, triage_all
from ragtriage.vectors import VectorRetriever

CHUNKS = [
    Chunk("billing.md#0", "billing.md", 0, "Refunds",
          "Refunds are returned to the original payment card within 5 business days."),
    Chunk("api.md#0", "api.md", 0, "Rate limits",
          "The API allows 600 requests per minute per token."),
    Chunk("ops.md#0", "ops.md", 0, "Backups",
          "Database backups run daily at 02:00 UTC and are retained for 30 days."),
]


class CountingJudge:
    """A judge that counts its supports() calls, which is what evidence location spends."""

    def __init__(self, inner=None):
        self.inner = inner or LexicalJudge()
        self.name = f"counting:{self.inner.name}"
        self.supports_calls = 0

    def supports(self, claim, passage):
        self.supports_calls += 1
        return self.inner.supports(claim, passage)

    def equivalent(self, gold, actual):
        return self.inner.equivalent(gold, actual)

    def grounded(self, answer, passages):
        return self.inner.grounded(answer, passages)


def retriever_arms(retriever_a, retriever_b, judge):
    """Two arms that differ only in the retriever: the same judge object on both sides,
    which is what tells compare() the evidence may be located once and shared."""
    return Arm(retriever_a.name, retriever_a, judge), Arm(retriever_b.name, retriever_b, judge)


@pytest.fixture
def judge():
    return LexicalJudge()


def test_identical_retrievers_produce_no_changes(judge):
    retriever_a = BM25Retriever(CHUNKS)
    retriever_b = BM25Retriever(CHUNKS)
    cases = [
        EvalCase(
            id="rate-limit",
            question="what is the API rate limit",
            gold="The API allows 600 requests per minute per token.",
            answer="The API allows 600 requests per minute.",
        ),
        EvalCase(
            id="scim",
            question="does it support SCIM provisioning",
            gold="Users are provisioned through SCIM 2.0 with Okta.",
            answer="Yes, SCIM is supported.",
        ),
        EvalCase(
            id="refund-window",
            question="how long does a refund take",
            gold="Refunds are returned to the original payment card within 5 business days.",
            answer="About two weeks.",
        ),
    ]

    comparisons = compare(cases, *retriever_arms(retriever_a, retriever_b, judge))
    summary = summarize_comparison(comparisons)

    assert summary.changed == 0
    assert summary.unchanged == 3
    assert summary.gains == []
    assert summary.regressions == []
    assert summary.a_retrieved == summary.b_retrieved
    assert summary.rank_gains == []
    assert summary.rank_regressions == []
    assert all(not c.verdict_changed for c in comparisons)
    assert all(c.rank_delta == 0 for c in comparisons if c.rank_delta is not None)


def test_regression_guard_evidence_located_with_baseline_not_candidate(judge):
    chunks = [
        Chunk("doc.md#0", "doc.md", 0, "Refunds",
              "Refunds are returned to the original payment card within 5 business days."),
        Chunk("doc.md#1", "doc.md", 1, "Backups",
              "Database backups run daily at 02:00 UTC and are retained for 30 days."),
    ]
    retriever_a = BM25Retriever(chunks)

    question = "how long do refunds take"
    gold = "Refunds are returned to the original payment card within 5 business days."
    chunk_vectors = {
        "doc.md#0": [-1.0, 0.0],
        "doc.md#1": [0.0, 1.0],
    }
    query_vectors = {
        question: [1.0, 0.0],
        gold: [1.0, 0.0],
    }
    retriever_b = VectorRetriever(chunks, chunk_vectors, query_vectors, model="anti-gold")

    case = EvalCase(
        id="refund-case",
        question=question,
        gold=gold,
        answer="Refunds are returned to the original payment card within 5 business days.",
    )

    comparisons = compare([case], *retriever_arms(retriever_a, retriever_b, judge))
    comp = comparisons[0]

    # Whether the evidence exists in the corpus is a property of the corpus,
    # so it must not change just because the retriever under test changed.
    assert comp.b.verdict != Verdict.MISSING_FROM_CORPUS
    assert comp.b.verdict is Verdict.RETRIEVAL_MISS
    assert comp.a.verdict is Verdict.OK


def test_both_sides_report_same_evidence_chunks(judge):
    retriever_a = BM25Retriever(CHUNKS)
    chunk_vectors = {
        "billing.md#0": [0.1, 0.9],
        "api.md#0": [0.9, 0.1],
        "ops.md#0": [0.5, 0.5],
    }
    q1 = "what is the API rate limit"
    g1 = "The API allows 600 requests per minute per token."
    q2 = "does it support SCIM provisioning"
    g2 = "Users are provisioned through SCIM 2.0 with Okta."
    query_vectors = {
        q1: [1.0, 0.0],
        g1: [1.0, 0.0],
        q2: [0.0, 1.0],
        g2: [0.0, 1.0],
    }
    retriever_b = VectorRetriever(CHUNKS, chunk_vectors, query_vectors, model="test-vectors")

    cases = [
        EvalCase(id="rate-limit", question=q1, gold=g1, answer="600 rpm"),
        EvalCase(id="scim", question=q2, gold=g2, answer="yes"),
    ]

    comparisons = compare(cases, *retriever_arms(retriever_a, retriever_b, judge))
    assert len(comparisons) == 2
    for c in comparisons:
        assert {h.chunk.id for h in c.a.support} == {h.chunk.id for h in c.b.support}


def test_compare_calls_evidence_location_once_per_case():
    retriever_a = BM25Retriever(CHUNKS)
    retriever_b = BM25Retriever(CHUNKS)
    cases = [
        EvalCase(
            id="rate-limit",
            question="what is the API rate limit",
            gold="The API allows 600 requests per minute per token.",
            answer="600 requests per minute.",
        ),
        EvalCase(
            id="backups",
            question="how often do backups run",
            gold="Database backups run daily at 02:00 UTC and are retained for 30 days.",
            answer="daily",
        ),
    ]

    judge_compare = CountingJudge()
    compare(cases, *retriever_arms(retriever_a, retriever_b, judge_compare))

    judge_single = CountingJudge()
    triage_all(cases, retriever_a, judge_single)

    assert judge_compare.supports_calls == judge_single.supports_calls
    assert judge_compare.supports_calls > 0


def test_rank_delta_properties():
    case = EvalCase(id="c", question="q", gold="g")

    def make_comp(cid, a_rank, b_rank):
        res_a = CaseResult(case, Verdict.OK, [], [], best_support_rank=a_rank)
        res_b = CaseResult(case, Verdict.OK, [], [], best_support_rank=b_rank)
        return CaseComparison(case_id=cid, a=res_a, b=res_b)

    assert make_comp("none_a", None, 1).rank_delta is None
    assert make_comp("none_b", 1, None).rank_delta is None
    assert make_comp("none_both", None, None).rank_delta is None

    better = make_comp("better", 5, 2)
    assert better.rank_delta == -3
    assert better.rank_delta < 0

    worse = make_comp("worse", 2, 5)
    assert worse.rank_delta == 3
    assert worse.rank_delta > 0

    same = make_comp("same", 3, 3)
    assert same.rank_delta == 0


def test_gains_and_regressions_direction(judge):
    chunks = [
        Chunk("doc.md#0", "doc.md", 0, "Refunds",
              "Refunds are returned within 5 business days."),
        Chunk("doc.md#1", "doc.md", 1, "Backups",
              "Database backups run daily at 02:00 UTC."),
    ]
    retriever_a = BM25Retriever(chunks)
    question = "how long do refunds take"
    gold = "Refunds are returned within 5 business days."
    chunk_vectors = {
        "doc.md#0": [0.1, 1.0],
        "doc.md#1": [1.0, 0.0],
    }
    query_vectors = {
        question: [1.0, 0.0],
        gold: [1.0, 0.0],
    }
    retriever_b = VectorRetriever(chunks, chunk_vectors, query_vectors, model="test")
    case = EvalCase(id="refund-case", question=question, gold=gold, answer="5 business days")

    comparisons = compare(
        [case], *retriever_arms(retriever_a, retriever_b, judge), config=TriageConfig(k=1)
    )
    summary = summarize_comparison(comparisons)

    assert len(summary.regressions) == 1
    assert summary.regressions[0].case_id == "refund-case"
    assert summary.gains == []


def test_summarize_comparison_empty_list():
    summary = summarize_comparison([])
    assert summary.changed == 0
    assert summary.unchanged == 0
    assert summary.a_retrieved == 0
    assert summary.b_retrieved == 0
    assert summary.gains == []
    assert summary.regressions == []
    assert summary.rank_gains == []
    assert summary.rank_regressions == []


def test_render_comparison_output():
    case_same = EvalCase(id="case-same", question="q1", gold="g1", answer="a1")
    case_diff = EvalCase(id="case-diff", question="q2", gold="g2", answer="a2")

    res_same_a = CaseResult(case_same, Verdict.OK, [], [], best_support_rank=1)
    res_same_b = CaseResult(case_same, Verdict.OK, [], [], best_support_rank=1)
    comp_same = CaseComparison(case_id="case-same", a=res_same_a, b=res_same_b)

    res_diff_a = CaseResult(case_diff, Verdict.OK, [], [], best_support_rank=1)
    res_diff_b = CaseResult(case_diff, Verdict.RETRIEVAL_MISS, [], [], best_support_rank=4)
    comp_diff = CaseComparison(case_id="case-diff", a=res_diff_a, b=res_diff_b)

    comparisons = [comp_same, comp_diff]
    summary = summarize_comparison(comparisons)

    out = render_comparison(
        comparisons,
        summary,
        axis="retriever",
        chunks=10,
        k=3,
        name_a="bm25",
        name_b="vectors:custom",
        fixed="lexical",
    )

    assert "bm25" in out
    assert "vectors:custom" in out
    assert "case-same" in out
    assert "case-diff" in out

    lines = out.splitlines()
    same_line = next(line for line in lines if "case-same" in line)
    diff_line = next(line for line in lines if "case-diff" in line)

    assert diff_line.endswith("*")
    assert not same_line.endswith("*")


def test_comparison_to_dict_round_trip():
    chunk = Chunk("doc.md#0", "doc.md", 0, "Doc", "Passage")
    hit = Hit(chunk=chunk, score=1.0, rank=1)
    case = EvalCase(id="case-1", question="q", gold="g")
    res_a = CaseResult(case, Verdict.OK, support=[hit], retrieved=[hit], best_support_rank=1)
    res_b = CaseResult(
        case, Verdict.RETRIEVAL_MISS, support=[hit], retrieved=[], best_support_rank=4
    )
    comp = CaseComparison(case_id="case-1", a=res_a, b=res_b)
    comparisons = [comp]
    summary = summarize_comparison(comparisons)

    data = comparison_to_dict(
        comparisons, summary, axis="retriever", name_a="bm25", name_b="vectors:model"
    )
    roundtripped = json.loads(json.dumps(data))
    assert roundtripped == data

    assert data["axis"] == "retriever"
    assert data["arms"] == {"a": "bm25", "b": "vectors:model"}
    assert data["evidence_shared"] is True
    assert data["summary"]["changed"] == 1
    assert data["summary"]["unchanged"] == 0
    assert data["summary"]["a_retrieved"] == 1
    assert data["summary"]["b_retrieved"] == 0
    assert data["summary"]["regressions"] == ["case-1"]
    assert data["summary"]["shifts"] == [{"from": "ok", "to": "retrieval_miss", "count": 1}]

    assert len(data["cases"]) == 1
    case_data = data["cases"][0]
    assert case_data["id"] == "case-1"
    assert case_data["a"] == {
        "verdict": "ok",
        "support_rank": 1,
        "evidence_chunks": ["doc.md#0"],
    }
    assert case_data["b"] == {
        "verdict": "retrieval_miss",
        "support_rank": 4,
        "evidence_chunks": ["doc.md#0"],
    }
    assert case_data["verdict_changed"] is True
    assert case_data["rank_delta"] == 3


def test_retrieved_evidence_verdicts():
    case = EvalCase(id="c", question="q", gold="g")

    for verdict in (Verdict.OK, Verdict.GENERATION_MISS, Verdict.UNGROUNDED, Verdict.NO_ANSWER):
        res = CaseResult(case, verdict, [], [], best_support_rank=1)
        assert retrieved_evidence(res) is True

    for verdict in (Verdict.MISSING_FROM_CORPUS, Verdict.RETRIEVAL_MISS):
        res = CaseResult(case, verdict, [], [], best_support_rank=None)
        assert retrieved_evidence(res) is False


# --- the judge axis -------------------------------------------------------

PARAPHRASE = [
    Chunk("billing.md#0", "billing.md", 0, "Refunds",
          "Refunds are returned to the original payment card within 5 business days."),
]
PARAPHRASE_CASE = EvalCase(
    id="refund-window",
    question="how long does a refund take",
    gold="A refund lands on the original payment card within 5 business days of approval.",
    answer="A refund lands on the original payment card within 5 business days of approval.",
)


def judge_arms(retriever, judge_a, judge_b):
    return Arm(judge_a.name, retriever, judge_a), Arm(judge_b.name, retriever, judge_b)


def test_the_judge_axis_lets_each_arm_locate_its_own_evidence():
    retriever = BM25Retriever(PARAPHRASE)
    lenient = build_judge("lexical@0.6")
    strict = build_judge("lexical@0.9")

    comp = compare([PARAPHRASE_CASE], *judge_arms(retriever, lenient, strict))[0]

    # Sharing evidence is right when the retriever moves and wrong when the judge does:
    # "does this chunk back the gold answer" is the judge's own question.
    assert comp.evidence_shared is False
    assert [hit.chunk.id for hit in comp.a.support] == ["billing.md#0"]
    assert comp.b.support == []
    assert comp.a.verdict is Verdict.OK
    assert comp.b.verdict is Verdict.MISSING_FROM_CORPUS


def test_the_retriever_axis_still_shares_its_evidence(judge):
    retriever_a = BM25Retriever(CHUNKS)
    retriever_b = BM25Retriever(CHUNKS)
    case = EvalCase(
        id="rate-limit",
        question="what is the API rate limit",
        gold="The API allows 600 requests per minute per token.",
        answer="600 requests per minute per token.",
    )

    comp = compare([case], *retriever_arms(retriever_a, retriever_b, judge))[0]
    assert comp.evidence_shared is True


def test_the_judge_axis_pays_for_two_evidence_scans_on_purpose():
    retriever = BM25Retriever(CHUNKS)
    cases = [
        EvalCase(
            id="rate-limit",
            question="what is the API rate limit",
            gold="The API allows 600 requests per minute per token.",
            answer="600 requests per minute per token.",
        ),
        EvalCase(
            id="backups",
            question="how often do backups run",
            gold="Database backups run daily at 02:00 UTC and are retained for 30 days.",
            answer="Database backups run daily at 02:00 UTC and are retained for 30 days.",
        ),
    ]

    shared = CountingJudge()
    compare(cases, *judge_arms(retriever, shared, shared))

    left, right = CountingJudge(), CountingJudge()
    compare(cases, *judge_arms(retriever, left, right))

    # Same judge object on both arms: one scan. Two objects: one scan each, and the
    # second one is the whole point rather than waste.
    assert left.supports_calls == shared.supports_calls
    assert right.supports_calls == shared.supports_calls
    assert shared.supports_calls > 0


def _pair(case_id, verdict_a, verdict_b):
    case = EvalCase(id=case_id, question="q", gold="g")
    return CaseComparison(
        case_id=case_id,
        a=CaseResult(case, verdict_a, [], [], best_support_rank=1),
        b=CaseResult(case, verdict_b, [], [], best_support_rank=1),
        evidence_shared=False,
    )


def test_masked_failures_and_false_alarms_point_opposite_ways():
    summary = summarize_comparison(
        [
            _pair("masked", Verdict.OK, Verdict.UNGROUNDED),
            _pair("alarm", Verdict.GENERATION_MISS, Verdict.OK),
            _pair("agreed", Verdict.OK, Verdict.OK),
            _pair("both-fail", Verdict.UNGROUNDED, Verdict.RETRIEVAL_MISS),
        ]
    )

    assert [c.case_id for c in summary.masked] == ["masked"]
    assert [c.case_id for c in summary.false_alarms] == ["alarm"]


def test_shifts_count_verdict_moves_commonest_first():
    summary = summarize_comparison(
        [
            _pair("one", Verdict.UNGROUNDED, Verdict.GENERATION_MISS),
            _pair("two", Verdict.UNGROUNDED, Verdict.GENERATION_MISS),
            _pair("three", Verdict.RETRIEVAL_MISS, Verdict.MISSING_FROM_CORPUS),
            _pair("same", Verdict.OK, Verdict.OK),
        ]
    )

    assert summary.shifts == [
        ("ungrounded", "generation_miss", 2),
        ("retrieval_miss", "missing_from_corpus", 1),
    ]


def test_agreement_is_the_share_of_unchanged_verdicts():
    summary = summarize_comparison(
        [
            _pair("a", Verdict.OK, Verdict.OK),
            _pair("b", Verdict.OK, Verdict.OK),
            _pair("c", Verdict.OK, Verdict.OK),
            _pair("d", Verdict.OK, Verdict.UNGROUNDED),
        ]
    )
    assert summary.agreement == 0.75
    assert summarize_comparison([]).agreement == 0.0


def test_render_judge_comparison_counts_evidence_instead_of_ranks():
    case = EvalCase(id="refund-window", question="q", gold="g", answer="a")
    chunk = Chunk("billing.md#0", "billing.md", 0, "Refunds", "text")
    hit = Hit(chunk=chunk, score=1.0, rank=1)
    comp = CaseComparison(
        case_id="refund-window",
        a=CaseResult(case, Verdict.OK, [hit], [hit], best_support_rank=1),
        b=CaseResult(case, Verdict.MISSING_FROM_CORPUS, [], [hit], best_support_rank=None),
        evidence_shared=False,
    )
    summary = summarize_comparison([comp])

    out = render_comparison(
        [comp],
        summary,
        axis="judge",
        chunks=1,
        k=5,
        name_a="lexical",
        name_b="lexical@0.9",
        fixed="bm25",
    )

    assert "rag-triage judge comparison" in out
    assert "retriever bm25" in out
    assert "A support" in out and "B rank" not in out
    assert "masked failures" in out
    assert "ok -> missing_from_corpus" in out
    assert out.splitlines()[-1].strip().startswith("each judge located its own evidence")


def test_judge_advice_names_a_stage_only_disagreement():
    comps = [
        _pair("incident-sla", Verdict.UNGROUNDED, Verdict.GENERATION_MISS),
        _pair("rate-limit", Verdict.OK, Verdict.OK),
    ]
    out = render_comparison(
        comps,
        summarize_comparison(comps),
        axis="judge",
        chunks=4,
        k=5,
        name_a="lexical",
        name_b="lexical@0.4",
        fixed="bm25",
    )
    # Neither judge calls a passing case failing, so the risk is a misdirected fix.
    assert "masked failures" not in out
    assert "which stage to blame" in out
