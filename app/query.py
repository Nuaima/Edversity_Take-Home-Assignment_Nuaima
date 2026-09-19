from __future__ import annotations

from app.schemas import ChatTurn


FOLLOW_UP_MARKERS = {
    "it", "that", "this", "mine", "they", "them", "then", "what if", "how about", "and if", "after that"
}


def build_retrieval_query(message: str, history: list[ChatTurn]) -> str:
    """Resolve short follow-ups without spending another LLM call.

    For a take-home prototype this is deterministic, cheap, and easy to test.
    A production system could replace this with a dedicated query-rewrite model.
    """
    message = message.strip()
    normalized = message.lower()
    needs_context = len(message.split()) <= 10 or any(marker in normalized for marker in FOLLOW_UP_MARKERS)
    if not needs_context:
        return message

    previous_user = next((turn.content.strip() for turn in reversed(history) if turn.role == "user"), None)
    if not previous_user:
        return message
    return f"Previous user question: {previous_user}\nCurrent follow-up: {message}"
