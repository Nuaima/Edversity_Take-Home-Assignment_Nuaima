from __future__ import annotations
import json
from pathlib import Path
from app.config import settings
from app.rag import RAGService
from app.schemas import ChatRequest

def main():
    dataset=json.loads(Path('eval/eval_dataset.json').read_text())
    service=RAGService(settings.data_dir,settings.storage_dir,settings.embedding_model,settings.top_k,settings.min_similarity,settings.min_confidence,settings.groq_api_key,settings.groq_model,settings.groq_base_url)
    retrieval_hits=escalation_hits=0
    for case in dataset:
        response=service.answer(ChatRequest(message=case['question']))
        returned={c.document_id for c in response.citations}
        retrieval_hits += int(bool(returned & set(case['expected_sources'])))
        escalation_hits += int(response.escalated == case['should_escalate'])
        print(case['id'], response.confidence_label, response.escalated, sorted(returned))
    n=len(dataset)
    print(f'Retrieval Hit@3: {retrieval_hits/n:.1%}')
    print(f'Escalation accuracy: {escalation_hits/n:.1%}')

if __name__=='__main__':
    main()
