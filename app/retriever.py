from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from app.ingestion import load_corpus, write_chunks_jsonl
from app.schemas import KnowledgeChunk, RetrievedChunk


@dataclass
class RetrieverConfig:
    embedding_model: str
    top_k: int = 5


class KnowledgeRetriever:
    """Local FAISS retriever with metadata-aware re-ranking.

    Semantic similarity gets most of the weight, but current policies/FAQs are
    deliberately preferred over historical tickets when scores are close.
    """

    def __init__(self, data_dir: Path, storage_dir: Path, config: RetrieverConfig):
        self.data_dir = data_dir
        self.storage_dir = storage_dir
        self.config = config
        self.model = SentenceTransformer(config.embedding_model)
        self.chunks: list[KnowledgeChunk] = load_corpus(data_dir)
        self.index = self._build_index()

    def _build_index(self) -> faiss.IndexFlatIP:
        texts = [f"{c.title}\n{c.content}" for c in self.chunks]
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.storage_dir / "learnforge.faiss"))
        write_chunks_jsonl(self.chunks, self.storage_dir / "chunks.jsonl")
        return index

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = min(top_k or self.config.top_k, len(self.chunks))
        query_vector = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")
        similarities, indices = self.index.search(query_vector, min(k * 2, len(self.chunks)))

        candidates: list[RetrievedChunk] = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            rerank = 0.78 * float(similarity) + 0.14 * chunk.authority + 0.08 * chunk.freshness
            candidates.append(
                RetrievedChunk(
                    **chunk.model_dump(),
                    similarity=float(similarity),
                    rerank_score=float(rerank),
                )
            )

        candidates.sort(key=lambda x: x.rerank_score, reverse=True)
        return candidates[:k]
