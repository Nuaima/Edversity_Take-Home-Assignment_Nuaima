from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from app.config import settings
from app.rag import RAGService
from app.schemas import ChatRequest


def _safe_div(num: int, den: int) -> float:
    return num / den if den else 0.0


def main() -> None:
    dataset = json.loads((ROOT / "eval" / "eval_dataset.json").read_text())
    service = RAGService(
        settings.data_dir,
        settings.storage_dir,
        settings.embedding_model,
        settings.top_k,
        settings.min_similarity,
        settings.min_confidence,
        settings.groq_api_key,
        settings.groq_model,
        settings.groq_base_url,
    )

    rows = []
    source_hit_count = 0
    exact_escalation_count = 0
    tp = fp = fn = 0

    for case in dataset:
        response = service.answer(ChatRequest(message=case["question"]))
        returned = [c.document_id for c in response.citations]
        expected = set(case["expected_sources"])
        source_hit = bool(set(returned) & expected)
        escalation_ok = response.escalated == case["should_escalate"]

        source_hit_count += int(source_hit)
        exact_escalation_count += int(escalation_ok)

        if response.escalated and case["should_escalate"]:
            tp += 1
        elif response.escalated and not case["should_escalate"]:
            fp += 1
        elif not response.escalated and case["should_escalate"]:
            fn += 1

        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "expected_sources": case["expected_sources"],
                "returned_sources": returned,
                "source_hit_at_3": source_hit,
                "expected_escalation": case["should_escalate"],
                "actual_escalation": response.escalated,
                "escalation_correct": escalation_ok,
                "confidence": response.confidence,
                "confidence_label": response.confidence_label,
                "reason": response.escalation_reason,
            }
        )

    n = len(dataset)
    metrics = {
        "cases": n,
        "retrieval_hit_at_3": round(_safe_div(source_hit_count, n), 4),
        "escalation_accuracy": round(_safe_div(exact_escalation_count, n), 4),
        "escalation_precision": round(_safe_div(tp, tp + fp), 4),
        "escalation_recall": round(_safe_div(tp, tp + fn), 4),
    }

    report = {"metrics": metrics, "cases": rows}
    out = ROOT / "eval" / "latest_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved detailed report to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
