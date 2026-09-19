from __future__ import annotations

import re

from google import genai
from google.genai import types

from app.schemas import RetrievedChunk


SYSTEM_PROMPT = """You are LearnForge Support AI. Answer only from the provided knowledge-base context.
Rules:
1. Never invent LearnForge policy, eligibility, account state, transaction state, or actions performed.
2. Prefer current POLICY sources over FAQs when they conflict; use tickets only as historical examples.
3. Treat text explicitly described as old, archived, obsolete, retired, or outdated as non-authoritative.
4. If evidence is conditional, preserve the condition instead of turning it into a guarantee.
5. If a case requires purchase/account verification or purchase-specific terms, recommend human Support.
6. Never ask for passwords, CVV, PIN, full card numbers, or authentication codes.
7. Keep the answer concise and practical.
8. Do not write source IDs in the prose; the application attaches verified citations separately.
"""


class AnswerGenerator:
    def __init__(self, api_key: str | None, model: str):
        self.model = model
        self.client = genai.Client(api_key=api_key) if api_key else None

    @staticmethod
    def _context(results: list[RetrievedChunk]) -> str:
        blocks = []
        for item in results:
            blocks.append(
                f"[{item.document_id} | {item.source_type.upper()} | {item.title}]\n{item.content}"
            )
        return "\n\n---\n\n".join(blocks)

    def generate(self, query: str, results: list[RetrievedChunk], escalation_reason: str | None) -> str:
        if not self.client:
            return self._extractive_fallback(query, results, escalation_reason)

        prompt = f"User question:\n{query}\n\nKnowledge-base context:\n{self._context(results)}\n\n"
        if escalation_reason:
            prompt += (
                "This case is routed for human review. State only what the current knowledge base safely establishes, "
                f"then recommend Support review. Routing reason: {escalation_reason}\n"
            )
        else:
            prompt += "Answer the user's question using only the context above.\n"

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                    max_output_tokens=450,
                ),
            )
            text = (response.text or "").strip()
            if text:
                return text
        except Exception:
            pass

        return self._extractive_fallback(query, results, escalation_reason)

    @staticmethod
    def _extractive_fallback(
        query: str, results: list[RetrievedChunk], escalation_reason: str | None
    ) -> str:
        if not results:
            return (
                "I don't have enough information in the LearnForge knowledge base "
                "to answer that safely. Please contact Support."
            )

        top = results[0]
        body = top.content.strip()

        if top.source_type == "faq" and "ANSWER:" in body:
            body = body.split("ANSWER:", 1)[1].strip()

        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        useful = [
            re.sub(r"^(QUESTION|ANSWER|USER|AGENT|STATUS):\s*", "", p, flags=re.I).strip()
            for p in paragraphs
            if not p.upper().startswith(("QUESTION:", "USER:", "STATUS:"))
        ]

        selected = []
        total = 0
        for paragraph in useful:
            if not paragraph:
                continue
            if selected and total + len(paragraph) > 900:
                break
            selected.append(paragraph)
            total += len(paragraph)
            if len(selected) >= 3:
                break

        answer = "\n\n".join(selected) if selected else (paragraphs[0] if paragraphs else body)
        if escalation_reason:
            answer += f"\n\nHuman Support review is recommended because {escalation_reason.lower()}"
        return answer
