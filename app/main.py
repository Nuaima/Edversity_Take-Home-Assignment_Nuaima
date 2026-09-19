from functools import lru_cache

from fastapi import FastAPI

from app.config import settings
from app.rag import RAGService
from app.schemas import ChatRequest, ChatResponse


app = FastAPI(title=settings.app_name, version="1.0.0")


@lru_cache(maxsize=1)
def get_service() -> RAGService:
    return RAGService(
        data_dir=settings.data_dir,
        storage_dir=settings.storage_dir,
        embedding_model=settings.embedding_model,
        top_k=settings.top_k,
        min_similarity=settings.min_similarity,
        min_confidence=settings.min_confidence,
        gemini_api_key=settings.gemini_api_key,
        gemini_model=settings.gemini_model,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return get_service().answer(request)
