from __future__ import annotations

import re

from openai import OpenAI

from app.schemas import RetrievedChunk


SYSTEM_PROMPT = """You are LearnForge Support AI. Answer only from the provided knowledge-base context.
Rules:
1. Never invent LearnForge policy, eligibility, account state, transaction state, or actions performed.
2. Prefer current POLICY sources over FAQs when they conflict; use tickets only as historical examples.
3. Treat text explicitly described as old, archived, obsolete, retired, or outdated as non-authoritative.
4. If the evidence is conditional, preserve the condition instead of turning it into a guarantee.
5. If the case requires purchase/account verification or conflicting purchase-specific terms, recommend human Support.
6. Do not ask for passwords, CVV, PIN, full card numbers, or authentication codes.
7. Keep the answer concise and practical. Do not fabricate citations; source labels are added separately by the application.
"""


class AnswerGenerator:
    def __init__(self, api_key: str | None, model: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None

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
                "The confidence layer marked this case for human review. Explain what the current knowledge base can "
                f"safely say, then clearly recommend Support review. Reason: {escalation_reason}\n"
            )
        else:
            prompt += "Answer the user's question using only the context above.\n"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=450,
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _extractive_fallback(
        query: str, results: list[RetrievedChunk], escalation_reason: str | None
    ) -> str:
        if not results:
            return "I don't have enough information in the LearnForge knowledge base to answer that safely. Please contact Support."

        top = results[0]
        body = top.content.strip()

        # FAQ records contain QUESTION/ANSWER labels. In fallback mode, surface
        # the answer naturally rather than exposing corpus formatting.
        if top.source_type == "faq" and "ANSWER:" in body:
            body = body.split("ANSWER:", 1)[1].strip()

        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        useful = [
            re.sub(r"^(QUESTION|ANSWER|USER|AGENT|STATUS):\s*", "", p, flags=re.I).strip()
            for p in paragraphs
            if not p.upper().startswith(("QUESTION:", "USER:", "STATUS:"))
        ]
        excerpt = next((p for p in useful if p), paragraphs[0] if paragraphs else body)
        excerpt = excerpt[:700].strip()

        answer = excerpt
        if escalation_reason:
            answer += f"\n\nHuman Support review is recommended because {escalation_reason.lower()}"
        return answer
