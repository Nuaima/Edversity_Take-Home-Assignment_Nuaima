from __future__ import annotations

from pathlib import Path

from app.confidence import ConfidenceEngine
from app.generator import AnswerGenerator
from app.query import build_retrieval_query
from app.retriever import KnowledgeRetriever, RetrieverConfig
from app.schemas import ChatRequest, ChatResponse, Citation


class RAGService:
    def __init__(
        self,
        data_dir: Path,
        storage_dir: Path,
        embedding_model: str,
        top_k: int,
        min_similarity: float,
        min_confidence: float,
        groq_api_key: str | None,
        groq_model: str,
        groq_base_url: str,
    ):
        self.retriever = KnowledgeRetriever(
            data_dir=data_dir,
            storage_dir=storage_dir,
            config=RetrieverConfig(embedding_model=embedding_model, top_k=top_k),
        )
        self.confidence = ConfidenceEngine(min_similarity=min_similarity, min_confidence=min_confidence)
        self.generator = AnswerGenerator(groq_api_key, groq_model, groq_base_url)

    def answer(self, request: ChatRequest) -> ChatResponse:
        retrieval_query = build_retrieval_query(request.message, request.history)
        results = self.retriever.search(retrieval_query)
        decision = self.confidence.assess(request.message, results)

        if decision.clarification:
            answer = (
                "Do you want to stop future subscription renewals, request a refund for a payment already made, "
                "cancel a course enrollment, or delete your LearnForge account?"
            )
        else:
            answer = self.generator.generate(request.message, results, decision.reason if decision.escalate else None)

        citations = [
            Citation(document_id=r.document_id, title=r.title, source_type=r.source_type)
            for r in results[:3]
        ]
        label = "HIGH" if decision.score >= 0.75 and not decision.escalate else "MEDIUM" if decision.score >= 0.55 else "LOW"

        return ChatResponse(
            answer=answer,
            confidence=round(decision.score, 3),
            confidence_label=label,
            escalated=decision.escalate,
            escalation_reason=decision.reason,
            citations=citations,
            retrieval_query=retrieval_query,
        )
