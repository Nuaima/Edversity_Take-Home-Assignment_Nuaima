from app.confidence import ConfidenceEngine
from app.schemas import RetrievedChunk


def _r(document_id="POLICY-02", source_type="policy", similarity=0.74, status=None, content="Current policy evidence."):
    authority = 1.0 if source_type == "policy" else 0.9 if source_type == "faq" else 0.72
    return RetrievedChunk(
        chunk_id=document_id.lower() + "-001",
        document_id=document_id,
        source_type=source_type,
        title="Example",
        content=content,
        updated_at="2026",
        status=status,
        authority=authority,
        freshness=1.0 if source_type != "ticket" else 0.7,
        contains_deprecation_notice=False,
        similarity=similarity,
        rerank_score=similarity,
    )


def test_purchase_specific_page_wording_escalates():
    d = ConfidenceEngine().assess(
        "The cancellation page said I get a full refund within 14 days. Refund my subscription.",
        [_r(), _r("TICKET-08", "ticket", 0.68, "Escalated due to policy ambiguity")],
    )
    assert d.escalate


def test_policy_question_with_only_ticket_evidence_escalates():
    d = ConfidenceEngine().assess(
        "What is the refund policy?",
        [_r("TICKET-03", "ticket", 0.62, "Escalated")],
    )
    assert d.escalate


def test_authoritative_security_prohibition_does_not_escalate():
    d = ConfidenceEngine().assess(
        "Support asked me to send my full card number and CVV. Should I?",
        [
            _r(
                "POLICY-10",
                "policy",
                0.41,
                content="Support agents must not request complete card numbers, CVV codes, PINs, banking passwords, or authentication codes.",
            ),
            _r(
                "POLICY-07",
                "policy",
                0.39,
                content="LearnForge Support will never request a user's complete password, payment-card security code, or authentication code.",
            ),
        ],
    )
    assert d.escalate is False
    assert d.score >= 0.62
