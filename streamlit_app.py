from __future__ import annotations

import streamlit as st

from app.config import settings
from app.rag import RAGService
from app.schemas import ChatRequest, ChatTurn


st.set_page_config(page_title="LearnForge AI Support", page_icon="🎓", layout="centered")
st.title("🎓 LearnForge AI Support")
st.caption("Grounded answers from LearnForge FAQs, policies, and historical support tickets.")


@st.cache_resource
def load_service() -> RAGService:
    return RAGService(
        data_dir=settings.data_dir,
        storage_dir=settings.storage_dir,
        embedding_model=settings.embedding_model,
        top_k=settings.top_k,
        min_similarity=settings.min_similarity,
        min_confidence=settings.min_confidence,
        groq_api_key=settings.groq_api_key,
        groq_model=settings.groq_model,
        groq_base_url=settings.groq_base_url,
    )


service = load_service()
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Try these cases")
    st.markdown(
        "- Can I get a refund after 10 days?\n"
        "- An old article says refunds are only 7 days.\n"
        "- Can I download courses on my laptop?\n"
        "- Cancel my LearnForge.\n"
        "- I bought it 21 days ago but the promotion promised 30 days."
    )
    st.divider()
    st.caption("Low-confidence, ambiguous, and purchase-specific cases are escalated instead of guessed.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            with st.expander("Evidence & confidence"):
                st.json(message["meta"])

prompt = st.chat_input("Ask LearnForge support...")
if prompt:
    history = [ChatTurn(role=m["role"], content=m["content"]) for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = service.answer(ChatRequest(message=prompt, history=history))
    source_text = ", ".join(c.document_id for c in response.citations)
    rendered = response.answer + f"\n\n**Sources:** {source_text}"
    if response.escalated:
        rendered += "\n\n⚠️ **Human Support review recommended.**"

    meta = {
        "confidence": response.confidence,
        "label": response.confidence_label,
        "escalated": response.escalated,
        "reason": response.escalation_reason,
        "retrieval_query": response.retrieval_query,
        "sources": [c.model_dump() for c in response.citations],
    }
    st.session_state.messages.append({"role": "assistant", "content": rendered, "meta": meta})
    with st.chat_message("assistant"):
        st.markdown(rendered)
        with st.expander("Evidence & confidence"):
            st.json(meta)
