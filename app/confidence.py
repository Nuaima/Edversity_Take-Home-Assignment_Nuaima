from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas import RetrievedChunk


AMBIGUOUS_CANCEL = re.compile(r"\b(cancel|stop)\s+(my\s+)?(learnforge|payment|it|this)\b", re.I)
PROMO_EXCEPTION = re.compile(
    r"\b(promo|promotion|voucher|30[- ]?day|guarantee|special offer|"
    r"terms when i bought|page said|website said|checkout said|cancellation page)\b",
    re.I,
)
ACCOUNT_SPECIFIC = re.compile(
    r"\b(my order|my account|my charge|my payment|transaction|receipt|restore|"
    r"manually|transfer|refund me|cancel it|my subscription|my purchase)\b",
    re.I,
)
POLICY_SENSITIVE = re.compile(
    r"\b(refund|subscription|cancel|cancellation|charge|payment|transfer|billing)\b",
    re.I,
)
SECURITY_SECRET = re.compile(
    r"\b(cvv|cvc|pin|password|authentication code|auth code|full card|complete card|"
    r"card number|banking password)\b",
    re.I,
)
SECURITY_PROHIBITION = re.compile(
    r"\b(never|must not|do not|don't|should not)\b.{0,120}"
    r"\b(cvv|cvc|pin|password|authentication code|auth code|full card|complete card|card number)\b",
    re.I | re.S,
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

        # Strongest evidence dominates; rerank quality and authority provide
        # smaller calibration signals. The score is a routing heuristic, not a
        # probability of factual correctness.
        score = max(
            0.0,
            min(
                1.0,
                0.55 * max(0.0, top.similarity)
                + 0.15 * mean_top
                + 0.15 * max(0.0, top.rerank_score)
                + 0.15 * top.authority,
            ),
        )

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
                "Purchase-specific wording may override the general policy and requires human verification.",
            )

        escalated_ticket = any(
            (r.status and "escalat" in r.status.lower())
            or "policy ambiguity" in r.content.lower()
            for r in results[:3]
        )
        if escalated_ticket and ACCOUNT_SPECIFIC.search(query):
            return ConfidenceDecision(
                min(score, 0.55),
                True,
                "The retrieved history shows this account-specific case requires human verification.",
            )

        # Security credential questions are a special high-certainty case:
        # if authoritative current sources explicitly prohibit requesting the
        # secret, answer the prohibition instead of escalating merely because
        # the embedding score is modest. This is evidence-based, not a keyword
        # bypass: the retrieved source text itself must contain the prohibition.
        authoritative_security_evidence = [
            r for r in results[:3]
            if r.source_type in {"policy", "faq"} and SECURITY_PROHIBITION.search(r.content)
        ]
        if SECURITY_SECRET.search(query) and authoritative_security_evidence:
            return ConfidenceDecision(max(score, 0.66), False, None)

        # Policy-style questions supported only by tickets are unsafe because
        # tickets are historical examples, not authoritative policy.
        if (
            POLICY_SENSITIVE.search(query)
            and top.source_type == "ticket"
            and not any(r.source_type in {"policy", "faq"} for r in results[:3])
        ):
            return ConfidenceDecision(
                min(score, 0.54),
                True,
                "Only historical ticket evidence was retrieved for a policy-sensitive question.",
            )

        if score < self.min_confidence:
            return ConfidenceDecision(score, True, "Overall retrieval confidence is below the answer threshold.")

        return ConfidenceDecision(score, False, None)
