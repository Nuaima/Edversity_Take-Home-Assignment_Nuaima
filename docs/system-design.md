# System Design

User / Streamlit Chat
  -> Conversation History
  -> Context-aware Query Builder
  -> SentenceTransformer Embedding
  -> FAISS Vector Index
  -> Top-K Semantic Retrieval
  -> Metadata Re-ranker (Authority + Freshness)
  -> Confidence / Risk Gate
      -> Ambiguous: Ask Clarifying Question
      -> Weak/conflicting/account-specific: Human Escalation
      -> Sufficient evidence: Grounded Groq Generation
  -> Answer + Sources + Confidence

Knowledge Base (FAQs + Policies + Tickets) -> Markdown Ingestion -> Embeddings + Metadata -> FAISS

## Query flow
1. Preserve conversation turns.
2. Resolve short follow-ups using the previous user message.
3. Embed the retrieval query with all-MiniLM-L6-v2.
4. Retrieve candidates with cosine similarity in FAISS.
5. Re-rank using similarity, source authority, and freshness.
6. Apply a confidence/risk gate before generation.
7. Clarify ambiguous intent instead of guessing.
8. Escalate weak or purchase-specific cases.
9. Generate only from retrieved context.
10. Return sources and confidence with the answer.
