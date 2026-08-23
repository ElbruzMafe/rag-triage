from ragtriage.judge import LexicalJudge

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
