from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas import RetrievedChunk


AMBIGUOUS_CANCEL = re.compile(r"\b(cancel|stop)\s+(my\s+)?(learnforge|payment|it|this)\b", re.I)
PROMO_EXCEPTION = re.compile(r"\b(promo|promotion|voucher|30[- ]?day|guarantee|special offer|terms when i bought)\b", re.I)
ACCOUNT_SPECIFIC = re.compile(
    r"\b(my order|my account|my charge|my payment|transaction|receipt|restore|manually|transfer|refund me|cancel it)\b",
    re.I,
)


@dataclass
class ConfidenceDecision:
    score: float
    escalate: bool
    reason: str | None
    clarification: bool = False


class ConfidenceEngine:
    def __init__(self, min_similarity: float = 0.34, min_confidence: float = 0.58):
        self.min_similarity = min_similarity
        self.min_confidence = min_confidence

    def assess(self, query: str, results: list[RetrievedChunk]) -> ConfidenceDecision:
        if not results:
            return ConfidenceDecision(0.0, True, "No supporting knowledge-base evidence was retrieved.")

        top = results[0]
        sims = [max(0.0, r.similarity) for r in results[:3]]
        mean_top = sum(sims) / len(sims)
        authority = sum(r.authority for r in results[:3]) / min(3, len(results))
        score = max(0.0, min(1.0, 0.72 * mean_top + 0.28 * authority))

        if top.similarity < self.min_similarity:
            return ConfidenceDecision(score, True, "Retrieved evidence is too weak to answer reliably.")

        if AMBIGUOUS_CANCEL.search(query):
            return ConfidenceDecision(
                min(score, 0.52),
                True,
                "The cancellation request is ambiguous; the assistant should clarify the user's intent.",
                clarification=True,
            )

        if PROMO_EXCEPTION.search(query):
            return ConfidenceDecision(
                min(score, 0.56),
                True,
                "Purchase-specific promotional terms may override the standard policy and require human review.",
            )

        escalated_ticket = any(
            (r.status and "escalat" in r.status.lower()) or "policy ambiguity" in r.content.lower()
            for r in results[:3]
        )
        if escalated_ticket and ACCOUNT_SPECIFIC.search(query):
            return ConfidenceDecision(
                min(score, 0.55),
                True,
                "The retrieved history shows this type of account-specific case requires human verification.",
            )

        if top.source_type == "ticket" and not any(r.source_type in {"policy", "faq"} for r in results[:3]):
            return ConfidenceDecision(
                min(score, 0.54), True, "Only historical ticket evidence was retrieved; current policy should be verified."
            )

        if score < self.min_confidence:
            return ConfidenceDecision(score, True, "Overall retrieval confidence is below the answer threshold.")

        return ConfidenceDecision(score, False, None)
