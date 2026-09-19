import pytest

from ragtriage.judge import LexicalJudge, build_judge

judge = LexicalJudge()


def test_supports_counts_claim_terms_in_the_passage():
    result = judge.supports("refunds take 5 days", "Refunds take 5 business days on the card.")
    assert result.value and result.confidence == 1.0


def test_supports_rejects_an_unrelated_passage():
    assert not judge.supports("refunds take 5 days", "Backups run nightly.").value


def test_equivalent_accepts_a_paraphrase():
    assert judge.equivalent(
        "The API allows 600 requests per minute per token.",
        "Each token is allowed 600 requests per minute.",
    ).value


def test_equivalent_rejects_a_swapped_number():
    result = judge.equivalent("refunds take 5 business days", "refunds take 14 business days")
    assert not result.value
    assert "figures disagree" in result.reason


def test_equivalent_ignores_numbers_when_only_one_side_has_them():
    assert judge.equivalent("refunds take 5 business days", "refunds take a few business days").value


def test_grounded_without_passages():
    result = judge.grounded("anything", [])
    assert not result.value and result.reason == "no passages retrieved"


def test_grounded_across_several_passages():
    result = judge.grounded("backups run daily", ["unrelated text", "Backups run daily at 02:00."])
    assert result.value and "2 retrieved passages" in result.reason


def test_empty_input_does_not_crash():
    assert judge.supports("", "text").reason == "claim is empty"
    assert judge.supports("claim", "   ").reason == "passage is empty"
    assert judge.equivalent("gold", "").reason == "answer is empty"
    assert judge.grounded("", ["ctx"]).reason == "answer is empty"


def test_build_judge_defaults_to_the_lexical_thresholds():
    built = build_judge("lexical")
    assert built.name == "lexical"
    assert (built.support_threshold, built.match_threshold) == (0.6, 0.55)


def test_build_judge_moves_only_the_support_threshold():
    built = build_judge("lexical@0.9")
    assert built.support_threshold == 0.9
    assert built.match_threshold == LexicalJudge().match_threshold


def test_a_tuned_judge_is_named_after_its_threshold():
    # Two arms of a comparison have to be distinguishable in the report.
    assert build_judge("lexical@0.8").name == "lexical@0.8"
    assert build_judge("lexical@0.80").name == "lexical@0.8"


def test_a_higher_threshold_rejects_partial_support():
    passage = "Refunds are returned to the original payment card within 5 business days."
    claim = "refunds are returned to the card in five days by our billing team"
    assert build_judge("lexical@0.5").supports(claim, passage).value
    assert not build_judge("lexical@0.95").supports(claim, passage).value


@pytest.mark.parametrize(
    "spec, message",
    [
        ("", "judge spec is empty"),
        ("   ", "judge spec is empty"),
        ("lexical@abc", "not a number in judge spec"),
        ("lexical@0", "between 0 and 1"),
        ("lexical@1.5", "between 0 and 1"),
        ("lexical@-0.2", "between 0 and 1"),
        ("claude@0.5", "no threshold"),
        ("Lexical", "unknown judge"),
        ("bm25", "unknown judge"),
    ],
)
def test_build_judge_rejects_a_bad_spec(spec, message):
    with pytest.raises(ValueError, match=message):
        build_judge(spec)
