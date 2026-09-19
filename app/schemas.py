from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


SourceType = Literal["policy", "faq", "ticket"]


class KnowledgeChunk(BaseModel):
    chunk_id: str
    document_id: str
    source_type: SourceType
    title: str
    content: str
    updated_at: str | None = None
    status: str | None = None
    authority: float = Field(ge=0.0, le=1.0)
    freshness: float = Field(ge=0.0, le=1.0)
    contains_deprecation_notice: bool = False


class RetrievedChunk(KnowledgeChunk):
    similarity: float
    rerank_score: float


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatTurn] = Field(default_factory=list)


class Citation(BaseModel):
    document_id: str
    title: str
    source_type: SourceType


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    confidence_label: Literal["HIGH", "MEDIUM", "LOW"]
    escalated: bool
    escalation_reason: str | None = None
    citations: list[Citation]
    retrieval_query: str
