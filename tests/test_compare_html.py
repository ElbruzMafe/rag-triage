"""The HTML comparison report, and the sentence diff it is built on."""

from ragtriage.compare import CaseComparison, claim_divergences, summarize_comparison
from ragtriage.models import CaseResult, Chunk, ClaimCheck, EvalCase, Hit, Verdict
from ragtriage.report import render_comparison_html

CHUNK_A = Chunk("billing.md#0", "billing.md", 0, "Refunds", "Refunds take 5 business days.")
CHUNK_B = Chunk("ops.md#2", "ops.md", 2, "Backups", "Backups run daily at 02:00 UTC.")


def claim(text, supported, chunk_id=None):
    return ClaimCheck(
        text=text, supported=supported, confidence=0.8, chunk_id=chunk_id, reason="because"
    )


def pair(
    case_id="a-case",
    verdict_a=Verdict.UNGROUNDED,
    verdict_b=Verdict.GENERATION_MISS,
    *,
    claims_a=(),
    claims_b=(),
    support_a=(),
    support_b=(),
    retrieved_a=(),
    retrieved_b=(),
    notes_a=(),
    notes_b=(),
    rank_a=1,
    rank_b=1,
    evidence_shared=True,
):
    case = EvalCase(
        id=case_id,
        question="how long does a refund take",
        gold="Refunds take 5 business days.",
        answer="Refunds take 14 days. We also phone every customer.",
        tags=["billing"],
    )
    return CaseComparison(
        case_id=case_id,
        a=CaseResult(
            case, verdict_a, list(support_a), list(retrieved_a), rank_a,
            notes=list(notes_a), claims=list(claims_a),
        ),
        b=CaseResult(
            case, verdict_b, list(support_b), list(retrieved_b), rank_b,
            notes=list(notes_b), claims=list(claims_b),
        ),
        evidence_shared=evidence_shared,
    )


def render(comparisons, **kwargs):
    options = {
        "axis": "judge",
        "chunks": 12,
        "k": 5,
        "name_a": "lexical",
        "name_b": "lexical@0.4",
        "fixed": "bm25",
    }
    options.update(kwargs)
    return render_comparison_html(comparisons, summarize_comparison(comparisons), **options)


def test_divergence_reports_a_sentence_the_two_judges_grade_differently():
    comp = pair(
        claims_a=[claim("Refunds take 14 days.", False), claim("We phone you.", True, "ops.md#2")],
        claims_b=[
            claim("Refunds take 14 days.", True, "billing.md#0"),
            claim("We phone you.", True),
        ],
    )
    divergences = claim_divergences(comp)

    assert [d.text for d in divergences] == ["Refunds take 14 days."]
    assert divergences[0].a_supported is False
    assert divergences[0].b_supported is True


def test_divergence_reports_a_sentence_only_one_arm_called_unsupported():
    # B called the whole answer grounded, so it never ran a sentence check at all.
    comp = pair(claims_a=[claim("We phone you.", False)], claims_b=[])
    divergences = claim_divergences(comp)

    assert [d.text for d in divergences] == ["We phone you."]
    assert divergences[0].b_supported is None


def test_divergence_ignores_a_supported_sentence_the_other_arm_never_checked():
    # "A is happy with this sentence and B never looked" is agreement, said two ways.
    comp = pair(claims_a=[claim("Refunds take 14 days.", True, "billing.md#0")], claims_b=[])

    assert claim_divergences(comp) == []


def test_divergence_is_empty_when_both_arms_agree_sentence_by_sentence():
    claims = [claim("Refunds take 14 days.", False), claim("We phone you.", True, "ops.md#2")]

    assert claim_divergences(pair(claims_a=claims, claims_b=list(claims))) == []


def test_html_names_both_arms_and_the_axis():
    html = render([pair()])

    assert "rag-triage judge comparison" in html
    assert "lexical@0.4" in html
    assert "retriever bm25" in html
    assert "<script" not in html
    assert "http://" not in html and "https://" not in html


def test_html_shows_the_sentence_the_judges_split_on():
    comp = pair(claims_a=[claim("We phone every customer.", False)], claims_b=[])
    html = render([comp])

    assert "where the judges split" in html
    assert "We phone every customer." in html
    assert "not checked" in html


def test_html_leaves_out_the_split_section_when_the_arms_agree():
    assert "where the judges split" not in render([pair()])


def test_identical_cells_span_both_columns_and_differing_ones_are_marked():
    same_stage = pair(verdict_a=Verdict.UNGROUNDED, verdict_b=Verdict.GENERATION_MISS)
    html = render([same_stage])

    # Both verdicts blame generation, so the stage row collapses into one cell.
    assert '<tr><th>stage</th><td colspan="2" class="agree">generation</td></tr>' in html
    assert '<td class="diff"><span class="chip ungrounded">ungrounded</span></td>' in html


def test_the_retriever_axis_reports_ranks_and_the_judge_axis_reports_evidence_counts():
    comp = pair(rank_a=1, rank_b=4, support_a=[Hit(CHUNK_A, 1.0, 1)], support_b=[])

    ranks = render([comp], axis="retriever", name_a="bm25", name_b="vectors:mine", fixed="lexical")
    counts = render([comp])

    assert "support rank" in ranks and "evidence chunks" not in ranks
    assert "evidence chunks" in counts and "support rank" not in counts
    assert "nothing in the corpus" in counts


def test_retrieved_context_lists_only_the_chunks_one_arm_saw():
    comp = pair(
        retrieved_a=[Hit(CHUNK_A, 3.0, 1)],
        retrieved_b=[Hit(CHUNK_A, 3.0, 1), Hit(CHUNK_B, 1.0, 2)],
    )
    html = render([comp])

    assert "<dt>only B</dt>" in html
    assert "ops.md#2" in html
    assert "<dt>only A</dt>" not in html


def test_retrieved_context_says_so_when_both_arms_saw_the_same_chunks():
    hits = [Hit(CHUNK_A, 3.0, 1), Hit(CHUNK_B, 1.0, 2)]
    html = render([pair(retrieved_a=hits, retrieved_b=list(hits))])

    assert "both arms retrieved the same 2 chunks" in html
    assert "only A" not in html and "only B" not in html


def test_notes_shared_by_both_arms_are_left_out():
    shared = "scored against the chunk ids recorded in the eval set, not a fresh search"
    html = render([pair(notes_a=[shared, "answer mismatch: figures disagree"], notes_b=[shared])])

    assert "answer mismatch: figures disagree" in html
    assert html.count(shared) == 0


def test_filters_only_offer_the_buckets_that_have_cases():
    changed = pair("moved", Verdict.OK, Verdict.UNGROUNDED)
    html = render([changed], axis="judge")

    # One masked failure, no unchanged case: no "same" tile, but a masked one.
    assert 'id="f-changed"' in html
    assert 'id="f-masked"' in html
    assert 'id="f-same"' not in html
    assert "c-masked" in html


def test_a_retriever_regression_gets_its_own_filter_not_a_masked_one():
    lost = pair("lost", Verdict.OK, Verdict.RETRIEVAL_MISS)
    html = render([lost], axis="retriever", name_a="bm25", name_b="vectors:mine", fixed="lexical")

    assert 'id="f-regression"' in html
    assert 'id="f-masked"' not in html
    assert "c-regression" in html


def test_every_filter_tile_has_a_rule_that_hides_the_other_cards():
    moved = pair("moved", Verdict.OK, Verdict.UNGROUNDED)
    still = pair("still", Verdict.GENERATION_MISS, Verdict.GENERATION_MISS)
    html = render([moved, still])

    for slug in ("changed", "same", "masked"):
        assert f"#f-{slug}:checked ~ .cards .case:not(.c-{slug})" in html
    # "all" is the default and hides nothing.
    assert "#f-all:checked ~ .cards" not in html


def test_the_reading_it_advice_is_carried_over_from_the_text_report():
    moved = pair("moved", Verdict.UNGROUNDED, Verdict.GENERATION_MISS)
    still = pair("still", Verdict.OK, Verdict.OK)
    html = render([moved, still])

    assert "reading it" in html
    assert "which stage to blame" in html


def test_corpus_text_is_escaped():
    case = EvalCase(id="x", question="q", gold="<script>alert(1)</script>", answer="a & b")
    comp = CaseComparison(
        case_id="x",
        a=CaseResult(case, Verdict.OK, [], [], 1),
        b=CaseResult(case, Verdict.OK, [], [], 1),
    )
    html = render([comp])

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "a &amp; b" in html
