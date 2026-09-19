from __future__ import annotations

import math
import re
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


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do",
    "for", "from", "how", "i", "if", "in", "is", "it", "my", "of", "on",
    "or", "the", "to", "was", "what", "when", "where", "why", "with", "you",
    "your",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _lexical_overlap(query: str, chunk: KnowledgeChunk) -> float:
    q = _tokens(query)
    d = _tokens(f"{chunk.title} {chunk.content}")
    if not q or not d:
        return 0.0
    return len(q & d) / math.sqrt(len(q) * len(d))


class KnowledgeRetriever:
    """Dense FAISS retrieval plus transparent lexical/metadata re-ranking.

    Dense similarity remains the dominant signal. Small lexical, authority and
    freshness priors improve exact policy/FAQ matches and make the ranking more
    robust to the intentionally stale historical tickets in the assignment.
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

        # Pull a wider dense candidate set, then re-rank locally. This is a
        # lightweight hybrid approach without adding BM25 infrastructure.
        candidate_k = min(max(k * 4, 12), len(self.chunks))
        similarities, indices = self.index.search(query_vector, candidate_k)

        candidates: list[RetrievedChunk] = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            lexical = _lexical_overlap(query, chunk)
            rerank = (
                0.72 * float(similarity)
                + 0.12 * lexical
                + 0.10 * chunk.authority
                + 0.06 * chunk.freshness
            )
            candidates.append(
                RetrievedChunk(
                    **chunk.model_dump(),
                    similarity=float(similarity),
                    rerank_score=float(rerank),
                )
            )

        candidates.sort(key=lambda x: x.rerank_score, reverse=True)
        return candidates[:k]
