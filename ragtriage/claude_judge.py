"""A Claude-backed judge, for eval sets whose wording differs from the corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .models import Assessment

MODEL = "claude-opus-5"

# USD per million tokens, input then output. Unknown models simply report no cost.
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

SYSTEM = (
    "You grade a retrieval-augmented question answering system. "
    "Answer only the question asked, strictly from the text you are given. "
    "Never use outside knowledge: if the passage does not state something, it is not supported. "
    "A different number, date or name means the claim is not supported."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "confidence", "reason"],
    "additionalProperties": False,
}


@dataclass
class Usage:
    """What the judge spent. Triage asks a lot of small questions, so the call count
    matters as much as the bill."""

    model: str
    calls: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost(self) -> float | None:
        price = PRICES.get(self.model)
        if price is None:
            return None
        return (self.input_tokens * price[0] + self.output_tokens * price[1]) / 1_000_000

    def summary(self) -> str:
        parts = [
            f"{self.calls} judge calls ({self.cached} served from cache)",
            f"{self.input_tokens} in / {self.output_tokens} out tokens",
        ]
        if self.cost is not None:
            parts.append(f"~${self.cost:.4f}")
        return "  |  ".join(parts)


class ClaudeJudge:
    """Same interface as LexicalJudge, one API call per question.

    Calls are memoised per process because triage asks the same question about the
    same chunk more than once when several eval cases share a gold answer.
    """

    name = "claude"

    def __init__(self, client=None, model: str = MODEL, effort: str = "low"):
        if client is None:
            import anthropic  # imported lazily so the offline judge needs no dependency

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.usage = Usage(model=model)
        self._cache: dict[tuple[str, str, str], Assessment] = {}

    def supports(self, claim: str, passage: str) -> Assessment:
        if not claim.strip() or not passage.strip():
            return Assessment(False, 0.0, "claim or passage is empty")
        return self._ask(
            "supports",
            claim,
            passage,
            f"Passage:\n{passage}\n\nClaim:\n{claim}\n\n"
            "Does the passage state, or directly entail, the claim?",
        )

    def equivalent(self, gold: str, actual: str) -> Assessment:
        if not gold.strip() or not (actual or "").strip():
            return Assessment(False, 0.0, "gold answer or answer is empty")
        return self._ask(
            "equivalent",
            gold,
            actual,
            f"Reference answer:\n{gold}\n\nAnswer under test:\n{actual}\n\n"
            "Do these convey the same facts? Wording may differ; facts and figures may not.",
        )

    def grounded(self, answer: str, passages: list[str]) -> Assessment:
        if not passages:
            return Assessment(False, 0.0, "no passages retrieved")
        if not (answer or "").strip():
            return Assessment(False, 0.0, "answer is empty")
        context = "\n\n---\n\n".join(passages)
        return self._ask(
            "grounded",
            answer,
            context,
            f"Retrieved context:\n{context}\n\nAnswer:\n{answer}\n\n"
            "Is every factual statement in the answer supported by the context?",
        )

    def _ask(self, kind: str, left: str, right: str, prompt: str) -> Assessment:
        key = (kind, left, right)
        if key in self._cache:
            self.usage.cached += 1
            return self._cache[key]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": SCHEMA},
            },
        )
        self._record(response)
        text = next(block.text for block in response.content if block.type == "text")
        data = json.loads(text)
        assessment = Assessment(
            value=bool(data["verdict"]),
            confidence=float(data["confidence"]),
            reason=str(data["reason"]),
        )
        self._cache[key] = assessment
        return assessment

    def _record(self, response) -> None:
        self.usage.calls += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.usage.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.usage.output_tokens += getattr(usage, "output_tokens", 0) or 0
