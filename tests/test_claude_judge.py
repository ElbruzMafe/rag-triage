import json
from types import SimpleNamespace

import pytest

from ragtriage.claude_judge import ClaudeJudge, Usage


class FakeMessages:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(self.payload))],
            usage=SimpleNamespace(input_tokens=120, output_tokens=30),
        )


class FakeClient:
    def __init__(self, payload):
        self.messages = FakeMessages(payload)


@pytest.fixture
def judge():
    payload = {"verdict": True, "confidence": 0.9, "reason": "the passage states it"}
    return ClaudeJudge(client=FakeClient(payload))


def test_supports_parses_the_structured_response(judge):
    result = judge.supports("refunds take 5 days", "Refunds take 5 business days.")
    assert result.value and result.confidence == 0.9
    assert result.reason == "the passage states it"


def test_the_request_pins_the_json_schema(judge):
    judge.supports("claim", "passage")
    request = judge.client.messages.calls[0]
    assert request["model"] == "claude-opus-5"
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["output_config"]["format"]["schema"]["required"] == [
        "verdict",
        "confidence",
        "reason",
    ]


def test_repeated_questions_are_answered_from_the_cache(judge):
    judge.supports("claim", "passage")
    judge.supports("claim", "passage")
    assert len(judge.client.messages.calls) == 1


def test_different_methods_do_not_share_cache_entries(judge):
    judge.supports("a", "b")
    judge.equivalent("a", "b")
    assert len(judge.client.messages.calls) == 2


def test_empty_input_never_reaches_the_api(judge):
    assert not judge.supports("", "passage").value
    assert not judge.equivalent("gold", "").value
    assert not judge.grounded("answer", []).value
    assert judge.client.messages.calls == []


def test_grounded_joins_the_passages(judge):
    judge.grounded("answer", ["one", "two"])
    prompt = judge.client.messages.calls[0]["messages"][0]["content"]
    assert "one\n\n---\n\ntwo" in prompt


def test_usage_counts_calls_tokens_and_cache_hits(judge):
    judge.supports("claim", "passage")
    judge.supports("claim", "passage")
    judge.equivalent("gold", "actual")

    assert judge.usage.calls == 2
    assert judge.usage.cached == 1
    assert judge.usage.input_tokens == 240
    assert judge.usage.output_tokens == 60
    assert judge.usage.cost == pytest.approx((240 * 5.0 + 60 * 25.0) / 1_000_000)


def test_usage_survives_a_response_without_a_usage_block():
    class NoUsage(FakeMessages):
        def create(self, **kwargs):
            response = super().create(**kwargs)
            del response.usage
            return response

    judge = ClaudeJudge(client=SimpleNamespace(messages=NoUsage({
        "verdict": True, "confidence": 1.0, "reason": "r"
    })))
    judge.supports("claim", "passage")
    assert judge.usage.calls == 1
    assert judge.usage.input_tokens == 0


def test_an_unknown_model_reports_no_cost():
    usage = Usage(model="some-future-model", calls=1, input_tokens=10, output_tokens=2)
    assert usage.cost is None
    assert "$" not in usage.summary()
