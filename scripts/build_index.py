from app.config import settings
from app.retriever import KnowledgeRetriever, RetrieverConfig


if __name__ == "__main__":
    retriever = KnowledgeRetriever(
        data_dir=settings.data_dir,
        storage_dir=settings.storage_dir,
        config=RetrieverConfig(embedding_model=settings.embedding_model, top_k=settings.top_k),
    )
    print(f"Indexed {len(retriever.chunks)} LearnForge records into {settings.storage_dir / 'learnforge.faiss'}")
