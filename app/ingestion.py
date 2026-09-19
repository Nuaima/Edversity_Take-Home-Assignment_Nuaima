from __future__ import annotations

import json
import re
from pathlib import Path

from app.schemas import KnowledgeChunk


SOURCE_CONFIG = {
    "policies.md": ("policy", 1.00, 1.00),
    "faqs.md": ("faq", 0.90, 0.95),
    "tickets.md": ("ticket", 0.72, 0.70),
}

ENTRY_RE = re.compile(
    r"(?ms)^#\s+(?P<id>(?:FAQ|POLICY|TICKET)-\d+)\s+—\s+(?P<title>.+?)\n(?P<body>.*?)(?=^---\s*$|\Z)"
)

DATE_PATTERNS = [
    re.compile(r"(?im)^(?:Last reviewed|Reviewed|Updated|Last updated|Effective date|Effective):\s*(.+)$"),
]


def _extract_date(body: str) -> str | None:
    for pattern in DATE_PATTERNS:
        match = pattern.search(body)
        if match:
            return match.group(1).strip()
    return None


def _extract_ticket_status(body: str) -> str | None:
    match = re.search(r"(?im)^STATUS:\s*(.+)$", body)
    return match.group(1).strip() if match else None


def _deprecation_notice(body: str) -> bool:
    terms = (
        "older", "outdated", "obsolete", "previous version", "archived documentation",
        "no longer", "retired", "old article", "older documentation"
    )
    lowered = body.lower()
    return any(term in lowered for term in terms)


def parse_markdown_file(path: Path) -> list[KnowledgeChunk]:
    source_type, authority, freshness = SOURCE_CONFIG[path.name]
    text = path.read_text(encoding="utf-8")
    chunks: list[KnowledgeChunk] = []

    for match in ENTRY_RE.finditer(text):
        document_id = match.group("id").strip()
        title = match.group("title").strip()
        body = match.group("body").strip()

        ticket_status = _extract_ticket_status(body) if source_type == "ticket" else None
        effective_freshness = freshness
        if ticket_status and "escalat" in ticket_status.lower():
            effective_freshness = 0.64

        chunks.append(
            KnowledgeChunk(
                chunk_id=f"{document_id.lower()}-001",
                document_id=document_id,
                source_type=source_type,  # type: ignore[arg-type]
                title=title,
                content=body,
                updated_at=_extract_date(body),
                status=ticket_status,
                authority=authority,
                freshness=effective_freshness,
                contains_deprecation_notice=_deprecation_notice(body),
            )
        )

    return chunks


def load_corpus(data_dir: Path) -> list[KnowledgeChunk]:
    corpus: list[KnowledgeChunk] = []
    for filename in ("policies.md", "faqs.md", "tickets.md"):
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required knowledge-base file: {path}")
        corpus.extend(parse_markdown_file(path))
    return corpus


def write_chunks_jsonl(chunks: list[KnowledgeChunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
