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
        gemini_api_key: str | None,
        gemini_model: str,
    ):
        self.retriever = KnowledgeRetriever(
            data_dir=data_dir,
            storage_dir=storage_dir,
            config=RetrieverConfig(embedding_model=embedding_model, top_k=top_k),
        )
        self.confidence = ConfidenceEngine(min_similarity=min_similarity, min_confidence=min_confidence)
        self.generator = AnswerGenerator(gemini_api_key, gemini_model)

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

        # Do not display weak neighboring results as citations merely because
        # FAISS returned them in top-k. Keep sources close to the best semantic hit.
        if results:
            citation_floor = max(
                self.confidence.min_similarity,
                results[0].similarity - 0.10,
            )
            citation_results = [r for r in results if r.similarity >= citation_floor][:3]
        else:
            citation_results = []

        citations = [
            Citation(document_id=r.document_id, title=r.title, source_type=r.source_type)
            for r in citation_results
        ]

        if decision.escalate:
            label = "LOW"
        elif decision.score >= 0.62:
            label = "HIGH"
        else:
            label = "MEDIUM"

        return ChatResponse(
            answer=answer,
            confidence=round(decision.score, 3),
            confidence_label=label,
            escalated=decision.escalate,
            escalation_reason=decision.reason,
            citations=citations,
            retrieval_query=retrieval_query,
        )
