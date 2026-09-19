# Data Schema

Each supplied FAQ, policy article, or historical ticket is stored as one semantic record because the sample items are already short and topically coherent.

| Field | Type | Purpose |
|---|---|---|
| chunk_id | string | Stable vector-record ID |
| document_id | string | Original source ID such as POLICY-02 |
| source_type | enum | policy, faq, or ticket |
| title | string | Human-readable title |
| content | text | Text embedded and supplied to the LLM |
| updated_at | string/null | Effective/review/update date when present |
| status | string/null | Ticket status |
| authority | float | Source-priority prior |
| freshness | float | Currentness prior |
| contains_deprecation_notice | bool | Source explicitly mentions outdated/retired guidance |
| embedding | float[] | Normalized vector stored in FAISS |

FAISS stores vectors while storage/chunks.jsonl stores aligned metadata. Both are reproducible build artifacts and are gitignored.

## Production version
I would move this to PostgreSQL + pgvector with filterable document_id, source_type, updated_at, version, jurisdiction, product, and is_current fields. Old versions would remain auditable but be excluded from normal retrieval.
